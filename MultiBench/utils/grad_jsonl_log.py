# English-only: training JSONL helpers for view_log (plot_all.py / plot_repr.py).
import json
import os
from typing import Any, Dict, Optional

import torch

# Root-level keys from GradAggregator.last_stats that collide with training semantics or are cryptic.
_GRAD_AGG_STATS_RENAMES = {
    "step": "grad_agg_step",  # aggregator inner counter; not the global training step
    "T": "n_tasks",  # number of task losses (e.g. 2); not temperature
}


def rename_grad_agg_stats_keys(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Rename top-level keys in the flat stats dict from GradAggregator before JSON / wandb.

    Args:
        stats: last_stats-style dict (keys like 'mode_id', 'step', 'T', 'pre.xxx', ...).

    Returns:
        New dict with only 'step' and 'T' renamed when they appear as root keys.
    """
    if not stats:
        return {}
    out: Dict[str, Any] = {}
    for k, v in stats.items():
        nk = _GRAD_AGG_STATS_RENAMES.get(k, k)
        out[nk] = v
    return out


def tensor_stats_to_jsonable(stats: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a flat stats dict (scalars only) to JSON-serializable floats.

    Args:
        stats: Mapping from string keys to 0-dim tensors or numeric scalars.

    Returns:
        Same keys with Python float values.

    Raises:
        ValueError: If a tensor is not scalar.
    """
    out: Dict[str, float] = {}
    for k, v in stats.items():
        if torch.is_tensor(v):
            if v.numel() != 1:
                raise ValueError(f"stats[{k}] must be scalar for jsonl, got shape {tuple(v.shape)}")
            out[k] = float(v.detach().cpu().item())
        elif isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            raise TypeError(f"stats[{k}] has unsupported type {type(v)}")
    return out


def append_train_jsonl_row(path: str, row: Dict[str, Any]) -> None:
    """
    Append one JSON object as a single line (JSONL).

    Args:
        path: Output file path; parent directories are created if needed.
        row: Must be JSON-serializable (use tensor_stats_to_jsonable for stats).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_train_row(
    *,
    step: int,
    epoch: int,
    i_batch: int,
    loss_x: float,
    loss_y: float,
    loss_total: float,
    lr: float,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build one record compatible with view_log TrainStatsHub (top-level + nested stats).

    Args:
        step: Global optimization step (x-axis in view_log; not grad_agg_step inside stats).
        epoch: Current epoch index (0-based).
        i_batch: Index within the epoch.
        loss_x: Task-x reconstruction loss.
        loss_y: Task-y reconstruction loss.
        loss_total: Weighted sum alpha_x*loss_x + alpha_y*loss_y (stored as loss_weighted).
        lr: Current learning rate.
        stats: Optional flat dict from GradAggregator.last_stats (tensor values ok).

    Returns:
        Dict ready for json.dumps after stats tensors are converted.
    """
    st: Dict[str, Any] = {}
    if stats:
        st = tensor_stats_to_jsonable(rename_grad_agg_stats_keys(stats))
    return {
        "step": int(step),
        "epoch": int(epoch),
        "i_batch": int(i_batch),
        "loss_x": float(loss_x),
        "loss_y": float(loss_y),
        "loss_weighted": float(loss_total),
        "lr": float(lr),
        "stats": st,
    }
