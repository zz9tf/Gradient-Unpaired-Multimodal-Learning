import argparse
import json
import os
import shutil
import sys
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_MULTIBENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _MULTIBENCH_DIR not in sys.path:
    sys.path.insert(0, _MULTIBENCH_DIR)

from models import Transformer, Linear
from train import UML, train, set_seed
from torch import optim
from datasets.affect.get_data import get_dataloader
import torch
import yaml
from itertools import product
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import wandb
from gradient_wrapper.grad_wrapper import GradWrapper, default_monitor_block_fn, build_common_slices
from gradient_wrapper.grad_block_monitor import GradientMonitor, MonitorConfig
from gradient_wrapper.grad_gpop import CommonGpopConfig, CommonGpopEditor
from utils.checkpoint import save_multibench_checkpoint
from utils.train_weight_pack import build_train_weight_pack


def print_train_block_overview(
    model: torch.nn.Module,
    train_loader_1,
    train_loader_2,
) -> None:
    """
    Print model block ids for monitor/gpop debugging before training starts.

    Args:
        model: UML model.
        train_loader_1: Loader for branch-1 batches.
        train_loader_2: Loader for branch-2 batches.
    """
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    block_ids_default_all = sorted({default_monitor_block_fn(n) for n, _ in trainable_named})
    block_ids_default_shared = sorted(
        {default_monitor_block_fn(n) for n, _ in trainable_named if uml_encoder_common_param_filter(n)}
    )

    layer_block_ids = set()
    for n, p in trainable_named:
        if not uml_encoder_common_param_filter(n):
            continue
        if ".layers." not in n:
            continue
        try:
            left, right = n.split(".layers.", 1)
            idx = right.split(".", 1)[0]
            layer_block_ids.add(f"{left}.layers.{idx}")
        except Exception:
            pass
    layer_block_ids = sorted(layer_block_ids)
    num_model_blocks = len(layer_block_ids)

    print(
        f"[MultiBench] steps_per_epoch={len(train_loader_1)} | {len(train_loader_2)}"
        f" | model_num_blocks(shared_encoder)= {num_model_blocks}"
        f"\n[MultiBench] block_ids_default_all(len={len(block_ids_default_all)})={block_ids_default_all}"
        f"\n[MultiBench] block_ids_default_shared(len={len(block_ids_default_shared)})={block_ids_default_shared}"
        f"\n[MultiBench] layer_block_ids_shared_encoder(len={len(layer_block_ids)})={layer_block_ids}"
    )


def write_run_config_json(results_dir: str, args: argparse.Namespace) -> str:
    """
    Persist parsed CLI config plus command line for reproducibility.

    Args:
        results_dir: Experiment output directory.
        args: Parsed namespace from this file's ``ArgumentParser``.

    Returns:
        Absolute path to the written ``config.json``.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "config.json")
    payload = vars(args)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    return path

def uml_encoder_common_param_filter(name: str) -> bool:
    """
    Select shared Transformer encoder parameters in UML for gpop common-parameter mask.

    Args:
        name: Parameter name from named_parameters().

    Returns:
        True if this parameter belongs to the shared encoder submodule.
    """
    n = name[7:] if name.startswith("module.") else name
    return n.startswith("encoder.")

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='x')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--zdim', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--n_seeds', type=int, default=1)
parser.add_argument('--dataset1', type=str, default='mosi')
parser.add_argument('--dataset2', type=str, default='mosi')
parser.add_argument('--step_k', type=int, default=-1)
parser.add_argument('--pos_embd', action='store_true')
parser.add_argument('--pos_learnable', action='store_true')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--train_jsonl', action='store_true', help='Write view_log-compatible train.jsonl under results_dir')

parser.add_argument('--debug', action='store_true', help='Skip wandb and use verbose-free paths where applicable')

parser.add_argument('--gpop_monitor', action='store_true', help='Enable pre/post GradientMonitor (for view_log block plots)')
parser.add_argument('--gpop_monitor_beta', type=float, default=0.999, help='Gpop beta for GradientMonitor')
parser.add_argument('--gpop_monitor_relation_tau', type=float, default=1e-8, help='Relation tau for GradientMonitor')
parser.add_argument('--gpop_monitor_enable_common_block', action='store_true', help='Enable common block for GradientMonitor')

parser.add_argument('--gpop', action='store_true', help='Enable gpop on shared encoder')
parser.add_argument('--gpop_ref_build_kind', type=str, default='cov', help='Operator kind for gpop')
parser.add_argument('--gpop_unbiased', action='store_true', help='Unbiased gpop')
parser.add_argument('--gpop_cov_center', action='store_true', help='Covariance center gpop')
parser.add_argument('--gpop_damping', type=float, default=1e-3, help='Damping for gpop')
parser.add_argument('--gpop_cg_max_iter', type=int, default=30, help='CG max iter for gpop')
parser.add_argument('--gpop_cg_tol', type=float, default=1e-6, help='CG tol for gpop')
parser.add_argument('--gpop_ema_beta', type=float, default=0.999, help='EMA beta for gpop')
parser.add_argument('--gpop_eps', type=float, default=1e-8, help='Epsilon for gpop')
parser.add_argument('--gpop_edit_kind', type=str, default='project', help='Edit kind for batch gradient on common dims')
parser.add_argument('--gpop_weights', type=str, default="loss_x=1.0,loss_y=1.0", help='Weights for gpop, e.g. "loss_x=1.0,loss_y=1.0"')

parser.add_argument('--show_layers', action='store_true', help='Print model.named_modules() and pause for manual inspection')
parser.add_argument('--augment', action='store_true', help='Compatibility flag; currently unused')
parser.add_argument('--eval_freq', type=int, default=100, help='Run linear-probe evaluate every this many train batches')
parser.add_argument('--x_random_noise', action='store_true', help='Replace x input with random Gaussian noise in train/eval')


def build_config(dataset1_bundle: dict, dataset2_bundle: dict, eval_freq: int) -> list:
    """
    Build ``eval_config`` for ``train.evaluate`` / ``train.train`` (matches train._normalize_eval_config).

    Each split maps to ``[loader_dataset1_branch, loader_dataset2_branch]`` (unpaired eval loaders).

    Args:
        dataset1_bundle: Output of ``build_dataset_loaders`` for branch 1 (x).
        dataset2_bundle: Output of ``build_dataset_loaders`` for branch 2 (y).
        eval_freq: In-training evaluation stride in batches.

    Returns:
        List of dicts with keys ``modality``, ``ds_name``, ``train``, ``val``, ``test``, ``freq``.
    """
    return [
        {
            "modality": "x",
            "ds_name": dataset1_bundle["ds_name"],
            "train":dataset1_bundle["eval_train_loader"],
            "val": dataset1_bundle["eval_valid_loader"],
            "test": dataset1_bundle["eval_test_loader"],
            "freq": int(eval_freq),
        },
        {
            "modality": "y",
            "ds_name": dataset2_bundle["ds_name"],
            "train":dataset2_bundle["eval_train_loader"],
            "val": dataset2_bundle["eval_valid_loader"],
            "test": dataset2_bundle["eval_test_loader"],
            "freq": int(eval_freq),
        }
    ]


def build_dataset_loaders(dataset_name: str) -> dict:
    """
    Build train/eval loaders and metadata for a given dataset name.

    Args:
        dataset_name: Dataset key. Supported values: mosi, sarcasm, humor, mimic, mosei.

    Returns:
        A dict with keys:
            - ds_name: str
            - batch_size: int
            - indims: list[int] with [xdim, ydim]
            - train_loader: train dataloader
            - eval_train_loader: eval-train dataloader
            - eval_valid_loader: eval-validation dataloader
            - eval_test_loader: eval-test dataloader
    """
    if dataset_name == 'mosi':
        batch_size = 32
        indims = [20, 300]
        train_loader, *_ = get_dataloader(
            './data_files/mosi_data.pkl',
            robust_test=False,
            batch_size=batch_size,
            train_shuffle=True,
        )
        eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader(
            './data_files/mosi_data.pkl',
            robust_test=False,
            batch_size=batch_size,
            train_shuffle=False,
        )
    elif dataset_name == 'sarcasm':
        batch_size = 128
        indims = [371, 300]
        train_loader, *_ = get_dataloader(
            './data_files/sarcasm.pkl',
            batch_size=batch_size,
            data_type='sarcasm',
            vision_norm=True,
        )
        eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader(
            './data_files/sarcasm.pkl',
            batch_size=batch_size,
            data_type='sarcasm',
            train_shuffle=False,
            vision_norm=True,
        )
    elif dataset_name == 'humor':
        batch_size = 128
        indims = [371, 300]
        train_loader, *_ = get_dataloader(
            './data_files/humor.pkl',
            batch_size=batch_size,
            data_type='humor',
        )
        eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader(
            './data_files/humor.pkl',
            batch_size=batch_size,
            data_type='humor',
            train_shuffle=False,
        )
    elif dataset_name == 'mimic':
        from datasets.mimic.get_data import get_dataloader as get_mimic_dataloader
        batch_size = 128
        indims = [5, 12]
        train_loader, *_ = get_mimic_dataloader(7, batch_size=batch_size, imputed_path='./data_files/im.pk')
        eval_train_loader, eval_valid_loader, eval_test_loader = get_mimic_dataloader(
            7,
            imputed_path='./data_files/im.pk',
            train_shuffle=False,
        )
        eval_test_loader = eval_valid_loader  # as per FACTOR-CL codebase
    elif dataset_name == 'mosei':
        batch_size = 32
        indims = [35, 300]
        train_loader, *_ = get_dataloader(
            './data_files/mosei_senti_data.pkl',
            robust_test=False,
            batch_size=batch_size,
            data_type='mosei',
            train_shuffle=True,
        )
        eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader(
            './data_files/mosei_senti_data.pkl',
            robust_test=False,
            batch_size=batch_size,
            data_type='mosei',
            train_shuffle=False,
        )
    else:
        raise NotImplementedError(f'Dataset not implemented yet: {dataset_name}')

    return {
        'ds_name': dataset_name,
        'batch_size': batch_size,
        'indims': indims,
        'train_loader': train_loader,
        'eval_train_loader': eval_train_loader,
        'eval_valid_loader': eval_valid_loader,
        'eval_test_loader': eval_test_loader,
    }


def main(args):
    log_dir = args.log_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = (
        f"log_{timestamp}_{args.run_name}{args.dataset1}_{args.dataset2}_mod{args.modality}"
        f"_zdim{args.zdim}_epochs{args.num_epochs}_pos_embd_{args.pos_embd}"
        f"_learnable_{args.pos_learnable}_step_k{args.step_k}_n_seeds{args.n_seeds}"
    )
    results_dir = os.path.join(args.results_dir, fname)
    cfg_path = write_run_config_json(results_dir, args)
    print(f"[MultiBench] Wrote run config JSON to {cfg_path}")

    if not args.debug:  
        os.makedirs(log_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = log_dir + "/wandb"
        wandb.init(entity="<USENAME>", project="<PROJECTNAME>", name = fname)
        wandb.config.update(args)

    print("Command-line arguments:", sys.argv)
    print("Parsed arguments:", args)

    seeds = [i for i in range(args.n_seeds)]
    outs = {'score_x': [], 'score_y': [], 'val_score_x': [], 'val_score_y': []}
    for seed in seeds:
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"[MultiBench] Seed {seed} outputs -> {seed_dir}")
        set_seed(seed)
        dataset1_bundle = build_dataset_loaders(args.dataset1)
        dataset2_bundle = build_dataset_loaders(args.dataset2)
        train_loader = dataset1_bundle["train_loader"]
        batch_size = dataset2_bundle["batch_size"]
        indims1 = dataset1_bundle["indims"]
        indims2 = dataset2_bundle["indims"]
        train_loader_2 = dataset2_bundle["train_loader"]
        config = build_config(dataset1_bundle, dataset2_bundle, args.eval_freq)

        # Dataset stats
        print("Dataset1: ", args.dataset1)
        print("Dataset2: ", args.dataset2)
        print("Batch size(ds1): ", dataset1_bundle["batch_size"])
        print("Batch size(ds2): ", dataset2_bundle["batch_size"])
        print("Train dataset: ", len(train_loader))
        print("Eval train batches (ds1 / ds2): ", len(dataset1_bundle["eval_train_loader"]), len(dataset2_bundle["eval_train_loader"]))
        print("Eval test batches (ds1 / ds2): ", len(dataset1_bundle["eval_test_loader"]), len(dataset2_bundle["eval_test_loader"]))
        print(f"Modality Info: xdim: {indims1[0]}, ydim: {indims2[1]}, zdim: {args.zdim}")


        # Initialize model and optimizer
        xproj_in = Linear(indims1[0], args.zdim)
        yproj_in = Linear(indims2[1], args.zdim)
        shared_encoder = Transformer(args.zdim, args.zdim, nhead=5, num_layers=5, conv1d=True, out_last=False, pos_embd=args.pos_embd, pos_learnable=args.pos_learnable, max_len=128)
        decoders = [Linear(args.zdim, indims1[0]), Linear(args.zdim, indims2[1])]
        model = UML(
            xproj_in,
            yproj_in,
            shared_encoder,
            decoders,
            modality=args.modality,
            x_random_noise=args.x_random_noise,
        ).cuda()
        # TODO: Add Adam optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        print_train_block_overview(model, train_loader, train_loader_2)

        if bool(args.show_layers):
            for name, module in model.named_modules():
                print(name, "->", module.__class__.__name__)
            print("--------------------------------")
            input()
        else:
            print("\n[Available module names for --show_layers]")

        train_weight_pack = build_train_weight_pack(args.gpop_weights, args.modality)
        print("[MultiBench] train_weight_pack (CPU):", train_weight_pack)

        grad_wrapper = None
        enable_grad_wrapper = bool(args.gpop or args.gpop_monitor)
        if enable_grad_wrapper:
            trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            common_slices = build_common_slices(trainable_named, uml_encoder_common_param_filter)
            loss_schema = ["loss_x", "loss_y"]

            block_ids_default_all = sorted({default_monitor_block_fn(n) for n, _ in trainable_named})
            block_loss_keys = {bid: list(loss_schema) for bid in block_ids_default_all}
            block_loss_keys["__common__"] = list(loss_schema)

            mon_cfg_pre = MonitorConfig(
                prefix="grad_pre",
                gpop_beta=float(args.gpop_monitor_beta),
                relation_tau=float(args.gpop_monitor_relation_tau),
                enable_common_block=bool(args.gpop_monitor_enable_common_block),
            )
            mon_cfg_post = MonitorConfig(
                prefix="grad_post",
                gpop_beta=float(args.gpop_monitor_beta),
                relation_tau=float(args.gpop_monitor_relation_tau),
                enable_common_block=bool(args.gpop_monitor_enable_common_block),
            )

            monitor_pre = None
            monitor_post = None
            if bool(args.gpop_monitor):
                monitor_pre = GradientMonitor(
                    named_params=trainable_named,
                    block_split_fn=default_monitor_block_fn,
                    block_loss_keys=block_loss_keys,
                    cfg=mon_cfg_pre,
                    common_slices=common_slices,
                )
                monitor_post = GradientMonitor(
                    named_params=trainable_named,
                    block_split_fn=default_monitor_block_fn,
                    block_loss_keys=block_loss_keys,
                    cfg=mon_cfg_post,
                    common_slices=common_slices,
                )

            gpop_editor = None
            if bool(args.gpop):
                gpop_editor = CommonGpopEditor(
                    named_params=trainable_named,
                    common_param_filter=uml_encoder_common_param_filter,
                    cfg=CommonGpopConfig(
                        gpop_keys=list(loss_schema),
                        ref_build_kind=str(args.gpop_ref_build_kind),
                        unbiased=bool(args.gpop_unbiased),
                        cov_center=bool(args.gpop_cov_center),
                        damping=float(args.gpop_damping),
                        cg_max_iter=int(args.gpop_cg_max_iter),
                        cg_tol=float(args.gpop_cg_tol),
                        ema_beta=float(args.gpop_ema_beta),
                        eps=float(args.gpop_eps),
                        edit_kind=str(args.gpop_edit_kind),
                    ),
                )

            grad_wrapper = GradWrapper(
                model=model,
                monitor_pre=monitor_pre,
                monitor_post=monitor_post,
                gpop=gpop_editor,
                verbose=not bool(args.debug),
            )
        jsonl_path = os.path.join(seed_dir, 'train.jsonl') if args.train_jsonl else None
        loss_jsonl_path = os.path.join(seed_dir, 'loss.jsonl')
        stats_jsonl_path = os.path.join(seed_dir, 'stats.jsonl')
        best_model_path = os.path.join(seed_dir, "model_best.pth")

        # Train model (saves best-val checkpoint to best_model_path; final return = test on best ckpt)
        score = train(
            model,
            args.modality,
            train_weight_pack,
            config,
            optimizer,
            num_epoch=args.num_epochs,
            step_k=args.step_k,
            augment=args.augment,
            debug=args.debug,
            grad_wrapper=grad_wrapper,
            jsonl_path=jsonl_path,
            loss_jsonl_path=loss_jsonl_path,
            stats_jsonl_path=stats_jsonl_path,
            best_model_path=best_model_path,
        )
        latest_model_path = os.path.join(seed_dir, "model_latest.pth")
        save_multibench_checkpoint(
            latest_model_path,
            model,
            modality=args.modality,
            ds_name=args.dataset2,
        )
        print(f"[MultiBench] Saved latest checkpoint to {latest_model_path}")

        print('seed: ', seed, ' score (best-val checkpoint, test+val metrics): ', score)
        print('=====================================')
        if score is None:
            raise RuntimeError("train() returned no scores; eval_config may be missing or empty.")
        outs['score_x'].append(100*score[0])
        outs['score_y'].append(100*score[1])
        outs['val_score_x'].append(100*score[2])
        outs['val_score_y'].append(100*score[3])

    print(outs)
    
    # Mean across seeds
    outs_mean = {k: np.mean(v) for k, v in outs.items()}
    outs_std = {k: np.std(v) for k, v in outs.items()}
    print("Final scores (mean): ", outs_mean)
    print("Final scores (std): ", outs_std)
    
    if not args.debug:
        wandb.log({f'final_{k}': v for k, v in outs_mean.items()})
        wandb.log({f'final_std_{k}': v for k, v in outs_std.items()})

    with open(os.path.join(results_dir, "outputs.txt"), "w") as f:
        f.write(f"Final scores (mean): {outs_mean}\n")
        f.write(f"Final scores std: {outs_std}\n")


if __name__ == "__main__":
    outer_parser = argparse.ArgumentParser(description="MultiBench Experiment")
    outer_parser.add_argument("-c", "--config", type=str, default="config.json", help="Configuration file")
    outer_parser.add_argument("-d", "--outer_debug", action="store_true", help="Debug mode")
    outer_parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing experiments directory")
    outer_args, remaining_args = outer_parser.parse_known_args()

    if outer_args.outer_debug:
        print("Running command-line arguments...")
        args = parser.parse_args(remaining_args)
        args.overwrite = outer_args.overwrite
        args.debug = True
        main(args)
        sys.exit(0)
    
    with open(outer_args.config, "r") as f:
        sweep_args = yaml.load(f, Loader=yaml.FullLoader)
    
    keys, values = zip(*sweep_args.items())
    combinations = [dict(zip(keys, v)) for v in product(*[v if isinstance(v, list) else [v] for v in values])]

    print("Total combinations:", len(combinations))
    for i, combo in enumerate(combinations):
        print(f"Combination {i}: {combo}")
    for i, combination in enumerate(combinations):
        print(f"=> Running combination {i}: {combination}")
        args = parser.parse_args([], argparse.Namespace(**combination))
        args.overwrite = outer_args.overwrite
        print("=> Parsed arguments:", args)
        main(args)
