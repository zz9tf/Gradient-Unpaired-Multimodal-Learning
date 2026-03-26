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


def evaluate(model, config, ds_name='mosi'):
    embds = {'train': {}, 'val': {}, 'test': {}}

    model.eval()
    if ds_name == 'mimic':
        for type in ['train', 'val', 'test']:
            embds[type]['x1'] = np.concatenate([model.get_embedding(data[0].float().cuda(), data[1].float().cuda())[0].detach().cpu().numpy() for data in config[type]])
            embds[type]['x2'] = np.concatenate([model.get_embedding(data[0].float().cuda(), data[1].float().cuda())[1].detach().cpu().numpy() for data in config[type]])
            embds[type]['labels'] = np.concatenate([data[2].detach().cpu().numpy() for data in config[type]])
    else:
        for type in ['train', 'val', 'test']:
            embds[type]['x1'] = np.concatenate([model.get_embedding(data[0][0].cuda(), data[0][2].cuda())[0].detach().cpu().numpy() for data in config[type]])
            embds[type]['x2'] = np.concatenate([model.get_embedding(data[0][0].cuda(), data[0][2].cuda())[1].detach().cpu().numpy() for data in config[type]])
            embds[type]['labels'] = np.concatenate([data[3].detach().cpu().numpy() for data in config[type]])

    for type in ['train', 'val', 'test']:
        if ds_name == 'mosi' or ds_name == 'mosei':
            embds[type]['labels'] = mosi_label(embds[type]['labels'])
        elif ds_name == 'sarcasm' or ds_name == 'humor':
            embds[type]['labels'] = sarcasm_label(embds[type]['labels'])
        else:
            raise NotImplementedError('Dataset not implemented yet')
        
        embds[type]['labels'] = np.asarray(embds[type]['labels']).reshape(-1).astype(int)

    # Train Logistic Classifier on X alone
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
    clf.fit(embds['train']['x1'], embds['train']['labels'])
    val_score_x = clf.score(embds['val']['x1'], embds['val']['labels'])
    score_x = clf.score(embds['test']['x1'], embds['test']['labels'])
    
    # Train Logistic Classifier on Y alone
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
    clf.fit(embds['train']['x2'], embds['train']['labels'])
    val_score_y = clf.score(embds['val']['x2'], embds['val']['labels'])
    score_y = clf.score(embds['test']['x2'], embds['test']['labels'])
    
    # Train Logistic Classifier on XY together
    train_embeds = np.concatenate([embds['train']['x1'], embds['train']['x2']], axis=1)
    val_embeds = np.concatenate([embds['val']['x1'], embds['val']['x2']], axis=1)
    test_embeds = np.concatenate([embds['test']['x1'], embds['test']['x2']], axis=1)
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
    clf.fit(train_embeds, embds['train']['labels'])
    score_xy = clf.score(test_embeds, embds['test']['labels'])
    val_score_xy = clf.score(val_embeds, embds['val']['labels'])
    return score_x, score_y, score_xy, val_score_x, val_score_y, val_score_xy


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
        return float(score_tuple[3])
    if train_mode == "y":
        return float(score_tuple[4])
    if train_mode == "xy":
        return float(score_tuple[5])
    raise ValueError(f"train_mode must be one of x, y, xy; got {train_mode!r}")

def train(
    model,
    train_mode,
    train_loader_1,
    train_loader_2,
    train_weight_pack,
    optimizer,
    modalities=[0, 2],
    num_epoch=100,
    step_k=30,
    ds_name="mosi",
    eval_config=None,
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
        train_loader_1: Loader for modality batch 1.
        train_loader_2: Loader for modality batch 2.
        train_weight_pack: Prebuilt x/y/xy (+ xy warmup) weight specs from ``main`` (CPU tensors).
        optimizer: Torch optimizer.
        modalities: Index pair into batch tensors (non-mimic).
        num_epoch: Epoch count.
        step_k: Warmup epochs for xy mode (y-only when iter <= step_k).
        ds_name: Dataset name for evaluation.
        eval_config: Optional dict with train/val/test loaders and eval freq.
        augment: Unused placeholder (kept for API compatibility).
        debug: If True, skip wandb logging.
        grad_wrapper: Optional pre-built GradWrapper with pre/post monitor and gpop injected.
        jsonl_path: Legacy mixed JSONL path (contains both loss and stats in one row).
        loss_jsonl_path: If set, append one JSON line per step for loss-only records.
        stats_jsonl_path: If set, append one JSON line per step for stats-only records.
        best_model_path: If set and ``eval_config`` is provided, save checkpoint when the
            modality-matched validation score improves (artifact only).

    Returns:
        Six-tuple from ``evaluate()`` on the **latest model** (end of training) when
        ``eval_config`` is provided; otherwise last in-training ``evaluate`` tuple if any,
        or ``None``.
    """
    if eval_config is None:
        eval_config = {}

    model.train()
    global_step = 0
    score = None
    best_val = float("-inf")
    best_saved = False
    steps_per_epoch = max(len(train_loader_1), len(train_loader_2))

    for _iter in range(num_epoch):
        is_warmup = False
        if train_mode == "xy" and int(step_k) >= 0 and _iter <= int(step_k):
            print(
                f"[MultiBench] xy warmup: y-only epoch [{_iter}/{step_k}] / total_epochs={num_epoch}"
            )
            is_warmup = True

        loader_1_iter = iter(train_loader_1)
        loader_2_iter = iter(train_loader_2)
        for i_batch in range(steps_per_epoch):
            data_batch_1 = None
            data_batch_2 = None
            try:
                data_batch_1 = next(loader_1_iter)
            except StopIteration:
                data_batch_1 = None
            try:
                data_batch_2 = next(loader_2_iter)
            except StopIteration:
                data_batch_2 = None

            if data_batch_1 is None and data_batch_2 is None:
                continue
            global_step += 1
            x1_batch = None
            x2_batch = None
            if ds_name != "mimic":
                if (not is_warmup) and (data_batch_1 is not None):
                    x1_batch = data_batch_1[0][modalities[0]].float().cuda()
                if data_batch_2 is not None:
                    x2_batch = data_batch_2[0][modalities[1]].float().cuda()
            else:
                if (not is_warmup) and (data_batch_1 is not None):
                    x1_batch = data_batch_1[0].float().cuda()
                if data_batch_2 is not None:
                    x2_batch = data_batch_2[1].float().cuda()

            if x1_batch is None and x2_batch is None:
                continue

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
                    continue

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

            if eval_config and i_batch % eval_config["freq"] == 0:
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
                score = evaluate(model, eval_config, ds_name)
                if best_model_path is not None:
                    vm = primary_val_metric_for_modality(train_mode, score)
                    if vm > best_val:
                        best_val = vm
                        save_multibench_checkpoint(
                            best_model_path,
                            model,
                            modality=train_mode,
                            ds_name=ds_name,
                        )
                        print(
                            f"[MultiBench] best val ({train_mode})={vm:.6f} -> saved {best_model_path}"
                        )
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
                    score[0],
                    " score_y: ",
                    score[1],
                    " score_xy: ",
                    score[2],
                )
                if not debug:
                    log_payload = {
                        "loss_x": loss_x.item(),
                        "loss_y": loss_y.item(),
                        "loss_weighted": loss.item(),
                        "score_x": score[0],
                        "score_y": score[1],
                        "score_xy": score[2],
                        "val_score_x": score[3],
                        "val_score_y": score[4],
                        "val_score_xy": score[5],
                    }
                    if last_stats is not None:
                        for sk, sv in rename_grad_agg_stats_keys(last_stats).items():
                            if torch.is_tensor(sv) and sv.numel() == 1:
                                log_payload["stats." + sk] = float(sv.detach().cpu().item())
                    wandb.log(log_payload)

                model.train()

    if eval_config:
        print("[MultiBench] Final evaluation with latest model parameters.")
        score = evaluate(model, eval_config, ds_name)
    return score
            

