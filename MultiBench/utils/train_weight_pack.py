"""
Train-time weight specs for MultiBench UML (x / y / xy + xy warmup).

All tensors are stored on CPU (float32). The training loop moves them to the
current device and dtype each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

LOSS_SCHEMA_KEYS: Tuple[str, ...] = ("loss_x", "loss_y")


def parse_kv_weights(s: str) -> Dict[str, float]:
    """
    Parse a comma-separated ``key=float`` list, e.g. ``loss_x=1.0,loss_y=1.0``.

    Args:
        s: Raw CLI string.

    Returns:
        Mapping containing every key in ``LOSS_SCHEMA_KEYS``.

    Raises:
        ValueError: On invalid tokens or missing keys.
    """
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f'Invalid weight token (expected key=float): "{part}"')
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        out[k] = float(v)
    for key in LOSS_SCHEMA_KEYS:
        if key not in out:
            raise ValueError(f'Missing weight for "{key}" in "{s}"')
    return out


@dataclass(frozen=True)
class TrainWeightPack:
    """
    One training regime: which losses are active, row weights, and full gpop schema weights.

    Attributes:
        task_keys: Loss keys in ``losses`` dict order (must match ``task_weights_cpu``).
        task_weights_cpu: Shape ``[len(task_keys)]``, multiplies each task gradient row and
            the weighted scalar loss.
        gpop_schema_weights_cpu: Shape ``[len(LOSS_SCHEMA_KEYS)]``, aligned with
            ``CommonGpopConfig.gpop_keys`` (same order as ``LOSS_SCHEMA_KEYS``).
    """

    task_keys: Tuple[str, ...]
    task_weights_cpu: torch.Tensor
    gpop_schema_weights_cpu: torch.Tensor
    
    def __str__(self):
        return f"TrainWeightPack(task_keys={self.task_keys}, task_weights_cpu={self.task_weights_cpu.tolist()}, gpop_schema_weights_cpu={self.gpop_schema_weights_cpu.tolist()})"


def build_train_weight_pack(gpop_weights: str, modality: str) -> TrainWeightPack:
    """
    Build four specs from user-provided per-loss weights (same scale as gpop schema).

    Args:
        gpop_loss_weight: Must contain ``loss_x`` and ``loss_y`` (e.g. from ``parse_kv_weights``).
        modality: Must be ``x``, ``y``, or ``xy``.

    Returns:
        Frozen pack used by ``train()`` to select the active spec each step.
    """
    gpop_loss_weight = parse_kv_weights(gpop_weights)
    if modality == "x":
        return TrainWeightPack(
            task_keys=("loss_x",),
            task_weights_cpu=torch.tensor([1.0, 0.0], dtype=torch.float32),
            gpop_schema_weights_cpu=torch.tensor([1.0, 0.0], dtype=torch.float32),
        )
    elif modality == "y":
        return TrainWeightPack(
            task_keys=("loss_y",),
            task_weights_cpu=torch.tensor([1.0, 0.0], dtype=torch.float32),
            gpop_schema_weights_cpu=torch.tensor([0.0, 1.0], dtype=torch.float32),
        )
    elif modality == "xy":
        return TrainWeightPack(
            task_keys=("loss_x", "loss_y"),
            task_weights_cpu=torch.tensor([1.0, 1.0], dtype=torch.float32),
            gpop_schema_weights_cpu=torch.tensor([gpop_loss_weight["loss_x"], gpop_loss_weight["loss_y"]], dtype=torch.float32),
        )