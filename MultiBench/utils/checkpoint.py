"""
MultiBench training checkpoint format (single supported style: ``multibench_ckpt``).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

MULTIBENCH_CKPT_FORMAT = "multibench_ckpt"


def save_multibench_checkpoint(
    path: str,
    model: nn.Module,
    modality: str,
    ds_name: str,
) -> None:
    """
    Save ``state_dict`` plus metadata so checkpoints are self-describing.

    Args:
        path: Output ``.pth`` path.
        model: Model to serialize.
        modality: Same as CLI ``--modality`` (``x``, ``y``, or ``xy``).
        ds_name: Dataset name (e.g. ``mosi``).
    """
    torch.save(
        {
            "format": MULTIBENCH_CKPT_FORMAT,
            "modality": modality,
            "ds_name": ds_name,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_multibench_checkpoint(
    path: str,
    map_location,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a MultiBench checkpoint (dict with ``format``, ``state_dict``, metadata).

    Args:
        path: Checkpoint path.
        map_location: Device for loading tensors.

    Returns:
        ``(state_dict, meta)`` where ``meta`` is all keys except ``state_dict``.

    Raises:
        ValueError: If the file is not a valid ``multibench_ckpt`` checkpoint.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt).__name__} from {path!r}")
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint missing 'state_dict' key: {path!r}")
    if ckpt.get("format") != MULTIBENCH_CKPT_FORMAT:
        raise ValueError(
            f"Checkpoint format must be {MULTIBENCH_CKPT_FORMAT!r}, got {ckpt.get('format')!r} in {path!r}"
        )
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return ckpt["state_dict"], meta
