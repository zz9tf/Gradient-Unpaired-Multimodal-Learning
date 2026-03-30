import os
import sys
import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression
import numpy as np
import wandb
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_MULTIBENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _MULTIBENCH_DIR not in sys.path:
    sys.path.insert(0, _MULTIBENCH_DIR)

from gradient_wrapper.grad_wrapper import GradWrapper, default_monitor_block_fn
from utils.checkpoint import save_multibench_checkpoint
from utils.grad_jsonl_log import (
    append_train_jsonl_row,
    build_train_row,
    rename_grad_agg_stats_keys,
    tensor_stats_to_jsonable,
)

modalities = {
    "mosi": [0, 2], # x, y for non-mimic datasets
    "mosei": [0, 2], # x, y for non-mimic datasets
    "sarcasm": [0, 2], # x, y for non-mimic datasets
    "humor": [0, 2], # x, y for non-mimic datasets
    "mimic": [0, 1], # x, y for mimic dataset
}

label_indices = {
    "mosi": 3,
    "mosei": 3,
    "sarcasm": 3,
    "humor": 3,
    "mimic": 2,
}

# Set seed
def set_seed(seed):
    import os
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Critical: disable cuDNN benchmarking
    os.environ['PYTHONHASHSEED'] = str(seed)
    
# Simple Augmentations 
def permute(x):
  # shuffle the sequence order
  idx = torch.randperm(x.shape[0])
  return x[idx]

def noise(x):
  noise = torch.randn(x.shape) * 0.1
  return x + noise.to(x.device)

def drop(x):
  # drop 20% of the sequences
  drop_num = x.shape[0] // 5
  
  x_aug = torch.clone(x)
  drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
  x_aug[drop_idxs] = 0.0
  return x_aug  

def mixup(x, alpha=1.0):
    indices = torch.randperm(x.shape[0])
    lam = np.random.beta(alpha, alpha)
    aug_x = x * lam + x[indices] * (1 - lam)

    return aug_x

def identity(x):
  return x

def augment(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [permute, noise, drop, identity]

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(4, 2, replace=False)
    t1 = transforms[t_idxs[0]]
    t2 = transforms[t_idxs[1]]
    v1[i] = t1(v1[i])
    v2[i] = t2(v2[i])
  
  return v1, v2

def augment_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [permute, noise, drop, identity]

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(4, 1, replace=False)
    t = transforms[t_idxs[0]]
    v2[i] = t(v2[i])
  
  return v2

def augment_embed_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [noise, mixup, identity]

  t_idxs = np.random.choice(3, 1, replace=False)
  t = transforms[t_idxs[0]]
  v2 = t(v2)

  return v2

def augment_mimic(x_batch):
  if x_batch.dim() == 2:
    return augment_embed_single(x_batch)
  else:
    return augment_single(x_batch)

# MOSI/MOSEI Training
def mosi_label(y_batch):
  res = copy.deepcopy(y_batch)
  res[y_batch >= 0] = 1
  res[y_batch < 0] = 0
  return res

# Sarcasm/Humor Training
def sarcasm_label(y_batch):
  res = copy.deepcopy(y_batch)
  res[y_batch == -1] = 0
  return res

# MSE loss
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, x, y):
        return (x - y).pow(2).mean()

# UML model
class UML(nn.Module):
    def __init__(self, xproj_in, yproj_in, shared_encoder, decoders, modality='x'):
        super().__init__()
        self.xproj_in = xproj_in
        self.yproj_in = yproj_in
        self.encoder = shared_encoder
        self.decoders = nn.ModuleList(decoders)
        self.modality = modality
        self.critic = MSE()
        print("Training using MUSE with modality: ", modality)
    
    def forward(self, x, y):
        # pool sequence dim of z
        loss_x = loss_y = torch.tensor(0.0)
        if x is not None:
            x = x.unsqueeze(1).float() if x.ndim == 2 else x
            x_proj = self.xproj_in(x)
            zx = self.encoder(x_proj)
            x_recon = self.decoders[0](zx)
            if x_recon.shape[1] == 1:
                loss_x = self.critic(x_recon[:, 0, :], x[:, 0, :])
            else:
                # next embedding prediction loss
                loss_x = self.critic(x_recon[:, :-1,:], x[:,1:,:])
        if y is not None:
            y = y.unsqueeze(1).float() if y.ndim == 2 else y
            y_proj = self.yproj_in(y)
            zy = self.encoder(y_proj)
            y_recon = self.decoders[1](zy)
            if y_recon.shape[1] == 1:
                loss_y = self.critic(y_recon[:, 0, :], y[:, 0, :])
            else:
                # next embedding prediction loss
                loss_y = self.critic(y_recon[:, :-1,:], y[:,1:,:])
        
        return {'loss_x': loss_x, 'loss_y': loss_y}

    def get_embedding(self, x, y):
        x = x.unsqueeze(1).float() if x.ndim == 2 else x
        y = y.unsqueeze(1).float() if y.ndim == 2 else y
        x = self.xproj_in(x)
        y = self.yproj_in(y)
        return self.encoder(x).mean(dim=1), self.encoder(y).mean(dim=1)
    
    def get_embedding_single(self, data, modality: str):
        data = data.unsqueeze(1).float() if data.ndim == 2 else data
        if modality == 'x':
            data = self.xproj_in(data)
        elif modality == 'y':
            data = self.yproj_in(data)
        else:
            raise ValueError(f"Invalid modality: {modality}")
        return self.encoder(data).mean(dim=1)

def evaluate_single_modality(model, config):
    ds_name = config["ds_name"]
    modality = config["modality"]
    train_loader = config["train"]
    val_loader = config["val"]
    test_loader = config["test"]
    
    modality_idx = modalities[ds_name][0] if modality == 'x' else modalities[ds_name][1]
    label_idx = label_indices[ds_name]
    
    embds_train = np.concatenate([model.get_embedding_single(data[0][modality_idx].cuda(non_blocking=True), modality).detach().cpu().numpy() for data in train_loader])
    embds_val = np.concatenate([model.get_embedding_single(data[0][modality_idx].cuda(non_blocking=True), modality).detach().cpu().numpy() for data in val_loader])
    embds_test = np.concatenate([model.get_embedding_single(data[0][modality_idx].cuda(non_blocking=True), modality).detach().cpu().numpy() for data in test_loader])
    
    labels_train = np.concatenate([data[label_idx].detach().cpu().numpy() for data in train_loader]).reshape(-1).astype(int)
    labels_val = np.concatenate([data[label_idx].detach().cpu().numpy() for data in val_loader]).reshape(-1).astype(int)
    labels_test = np.concatenate([data[label_idx].detach().cpu().numpy() for data in test_loader]).reshape(-1).astype(int)
       
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
    clf.fit(embds_train, labels_train)
    val_score = clf.score(embds_val, labels_val)
    test_score = clf.score(embds_test, labels_test)
    
    return test_score, val_score

def evaluate(model, config):
    model.eval()
    test_score_x, val_score_x = evaluate_single_modality(model, config[0])
    test_score_y, val_score_y = evaluate_single_modality(model, config[1])
    return test_score_x, test_score_y, val_score_x, val_score_y


def primary_val_metric_for_modality(train_mode: str, score_tuple: Tuple[float, ...]) -> float:
    """
    Select the validation metric used for best-checkpoint selection.

    ``evaluate`` returns
    ``(test_x, test_y, test_xy, val_x, val_y, val_xy)``. We maximize the val
    score that matches training modality.

    Args:
        train_mode: ``x``, ``y``, or ``xy`` (same as ``--modality``).
        score_tuple: Six-float tuple from ``evaluate``.

    Returns:
        Scalar validation accuracy to maximize.

    Raises:
        ValueError: If ``train_mode`` is not ``x``, ``y``, or ``xy``.
    """
    if train_mode == "x":
        return float(score_tuple[2])
    if train_mode == "y":
        return float(score_tuple[3])
    if train_mode == "xy":
        return float(score_tuple[3])
    raise ValueError(f"train_mode must be one of x, y, xy; got {train_mode!r}")

def train(
    model,
    train_mode,
    train_weight_pack,
    config,
    optimizer,
    num_epoch=100,
    step_k=30,
    augment=None,
    debug=False,
    grad_wrapper: Optional[GradWrapper] = None,
    jsonl_path: Optional[str] = None,
    loss_jsonl_path: Optional[str] = None,
    stats_jsonl_path: Optional[str] = None,
    best_model_path: Optional[str] = None,
):
    """
    Train UML with optional GradWrapper and JSONL logging for view_log.

    Args:
        model: UML instance on CUDA.
        train_mode: 'x', 'y', or 'xy'.
        config: List of dicts with keys ``modality``, ``ds_name``, ``train``, ``val``, ``test``, ``freq``.
        train_weight_pack: Prebuilt x/y/xy (+ xy warmup) weight specs from ``main`` (CPU tensors).
        optimizer: Torch optimizer.
        modalities: Index pair into batch tensors (non-mimic).
        num_epoch: Epoch count.
        step_k: Warmup epochs for xy mode (y-only when iter <= step_k).
        augment: Unused placeholder (kept for API compatibility).
        debug: If True, skip wandb logging.
        grad_wrapper: Optional pre-built GradWrapper with pre/post monitor and gpop injected.
        jsonl_path: Legacy mixed JSONL path (contains both loss and stats in one row).
        loss_jsonl_path: If set, append one JSON line per step for loss-only records.
        stats_jsonl_path: If set, append one JSON line per step for stats-only records.
        best_model_path: If set and ``config`` is provided, save checkpoint when the
            modality-matched validation score improves (artifact only).

    Returns:
        Six-tuple from ``evaluate()`` on the **latest model** (end of training) when
        ``config`` is provided; otherwise last in-training ``evaluate`` tuple if any,
        or ``None``.
    """
    if config is None:
        config = []

    model.train()
    global_step = 0
    score = None
    best_val = float("-inf")
    steps_per_epoch = max(len(config[0]["train"]), len(config[1]["train"]))
    

    for _iter in range(num_epoch):
        is_warmup = False
        if train_mode == "xy" and int(step_k) >= 0 and _iter <= int(step_k):
            print(
                f"[MultiBench] xy warmup: y-only epoch [{_iter}/{step_k}] / total_epochs={num_epoch}"
            )
            is_warmup = True

        loader_1_iter = iter(config[0]["train"])
        loader_2_iter = iter(config[1]["train"])
        
        for i_batch in range(steps_per_epoch):
            x1_batch = None
            x2_batch = None
            try:
                data_batch_1 = next(loader_1_iter)[0]
                modality_idx = modalities[config[0]["ds_name"]][0]
                x1_batch = data_batch_1[modality_idx].cuda(non_blocking=True)
            except StopIteration:
                x1_batch = None
            try:
                data_batch_2 = next(loader_2_iter)[0]
                modality_idx = modalities[config[1]["ds_name"]][1]
                x2_batch = data_batch_2[modality_idx].cuda(non_blocking=True)
            except StopIteration:
                x2_batch = None
            if x1_batch is None and x2_batch is None:
                raise ValueError("No data batches found")
            
            global_step += 1

            out_loss = model(x1_batch, x2_batch)
            loss_x, loss_y = out_loss["loss_x"], out_loss["loss_y"]
            lr_now = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad()
            last_stats = None
            has_loss_x = x1_batch is not None
            has_loss_y = x2_batch is not None
            if has_loss_x and has_loss_y:
                loss = loss_x + loss_y
            elif has_loss_x:
                loss = loss_x
            else:
                loss = loss_y
            if grad_wrapper is None:
                loss.backward()
            else:
                losses = {}
                if is_warmup:
                    if has_loss_y:
                        losses["loss_y"] = loss_y
                else:
                    if has_loss_x:
                        losses["loss_x"] = loss_x
                    if has_loss_y:
                        losses["loss_y"] = loss_y
                if not losses:
                    raise ValueError("No losses found")

                last_stats = grad_wrapper.backward(
                    losses=losses,
                    task_weights=train_weight_pack.task_weights_cpu,
                    gpop_weights=train_weight_pack.gpop_schema_weights_cpu,
                )
            optimizer.step()

            if jsonl_path:
                row = build_train_row(
                    step=global_step,
                    epoch=_iter,
                    i_batch=i_batch,
                    loss_x=float(loss_x.detach().cpu().item()),
                    loss_y=float(loss_y.detach().cpu().item()),
                    loss_total=float(loss.detach().cpu().item()),
                    lr=lr_now,
                    stats=last_stats,
                )
                append_train_jsonl_row(jsonl_path, row)
            if (loss_jsonl_path or stats_jsonl_path):
                loss_row = {
                    "step": int(global_step),
                    "epoch": int(_iter),
                    "i_batch": int(i_batch),
                    "loss_x": float(loss_x.detach().cpu().item()),
                    "loss_y": float(loss_y.detach().cpu().item()),
                    "loss_weighted": float(loss.detach().cpu().item()),
                    "lr": float(lr_now),
                }
                if loss_jsonl_path:
                    append_train_jsonl_row(loss_jsonl_path, loss_row)

                if stats_jsonl_path:
                    stats_row = {
                        "step": int(global_step),
                        "epoch": int(_iter),
                        "i_batch": int(i_batch),
                        "stats": tensor_stats_to_jsonable(rename_grad_agg_stats_keys(last_stats or {})),
                    }
                    append_train_jsonl_row(stats_jsonl_path, stats_row)

            if config and i_batch % config[0]["freq"] == 0:
                if steps_per_epoch is not None:
                    steps_until_next_epoch_start = int(steps_per_epoch) - (int(i_batch) + 1)
                    print(
                        f"[MultiBench] current_epoch={_iter} current_batch_idx={i_batch}"
                        f" | next_epoch_total_steps={steps_per_epoch}"
                        f" | steps_until_next_epoch_start={steps_until_next_epoch_start}"
                    )
                else:
                    print(
                        f"[MultiBench] current_epoch={_iter} current_batch_idx={i_batch}"
                        " | next_epoch_total_steps=unknown"
                    )
                model.eval()
                score = evaluate(model, config)
                if best_model_path is not None:
                    vm = primary_val_metric_for_modality(train_mode, score)
                    if vm > best_val:
                        best_val = vm
                        save_multibench_checkpoint(
                            best_model_path,
                            model,
                            modality=train_mode,
                            ds_name=config[1]["ds_name"],
                        )
                        print(
                            f"[MultiBench] best val ({train_mode})={vm:.6f} -> saved {best_model_path}"
                        )

                scores = {
                    "score_x": score[0],
                    "score_y": score[1],
                    "val_score_x": score[2],
                    "val_score_y": score[3],
                }
                if has_loss_x and has_loss_y:
                    # x + y are both available.
                    print(
                        "iter: ",
                        _iter,
                        " i_batch: ",
                        i_batch,
                        " loss_x: ",
                        loss_x.item(),
                        " loss_y: ",
                        loss_y.item(),
                        " loss: ",
                        loss.item(),
                        " score_x: ",
                        scores["score_x"],
                        " score_y: ",
                        scores["score_y"],
                    )
                elif has_loss_x and (not has_loss_y):
                    # Only x-branch is available.
                    print(
                        "iter: ",
                        _iter,
                        " i_batch: ",
                        i_batch,
                        " (x-only) ",
                        " loss_x: ",
                        loss_x.item(),
                        " loss: ",
                        loss.item(),
                        " score_x: ",
                        scores["score_x"],
                    )
                elif (not has_loss_x) and has_loss_y:
                    # Only y-branch is available.
                    print(
                        "iter: ",
                        _iter,
                        " i_batch: ",
                        i_batch,
                        " (y-only) ",
                        " loss_y: ",
                        loss_y.item(),
                        " loss: ",
                        loss.item(),
                        " score_y: ",
                        scores["score_y"],
                    )
                else:
                    raise ValueError("Unexpected state: no losses available (has_loss_x=False, has_loss_y=False).")

                if not debug:
                    log_payload = {
                        "loss_x": float(loss_x.detach().cpu().item()),
                        "loss_weighted": float(loss.detach().cpu().item()),
                    }
                    if "score_x" in scores:
                        log_payload["score_x"] = scores["score_x"]
                    if "val_score_x" in scores:
                        log_payload["val_score_x"] = scores["val_score_x"]

                    if has_loss_y:
                        log_payload["loss_y"] = float(loss_y.detach().cpu().item())
                        if "score_y" in scores:
                            log_payload["score_y"] = scores["score_y"]
                        if "val_score_y" in scores:
                            log_payload["val_score_y"] = scores["val_score_y"]

                    if last_stats is not None:
                        for sk, sv in rename_grad_agg_stats_keys(last_stats).items():
                            if torch.is_tensor(sv) and sv.numel() == 1:
                                log_payload["stats." + sk] = float(sv.detach().cpu().item())
                    wandb.log(log_payload)

                model.train()

    if config:
        print("[MultiBench] Final evaluation with latest model parameters.")
        score = evaluate(model, config)
    return score
            
