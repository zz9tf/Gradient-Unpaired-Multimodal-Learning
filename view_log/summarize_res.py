#!/usr/bin/env python3
"""Summarize experiment outputs into a Markdown table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


EXPERIMENT_PATTERN = re.compile(
    r"^log_(?:(?P<timestamp_prefix>\d{8}_\d{6})_)?(?P<dataset>.+?)_mod(?P<modality>.+?)_zdim(?P<zdim>[^_]+)"
    r"_epochs(?P<epochs>[^_]+)_pos_embd_(?P<pos_embd>[^_]+)"
    r"_learnable_(?P<learnable>[^_]+)_step_k(?P<step_k>[^_]+)"
    r"_n_seeds(?P<n_seeds>[^_]+)(?:_(?P<timestamp>\d{8}_\d{6}))?$"
)

FLOAT64_PATTERN = re.compile(r"np\.float64\(([-+0-9.eE]+)\)")

# Config JSON keys that duplicate folder-derived row columns; values must match.
CONFIG_KEY_TO_ROW_KEY: Dict[str, str] = {
    "ds_name": "dataset",
    "modality": "modality",
    "zdim": "zdim",
    "num_epochs": "epochs",
    "pos_embd": "pos_embd",
    "pos_learnable": "learnable",
    "step_k": "step_k",
    "n_seeds": "n_seeds",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate MultiBench outputs.txt into a Markdown report."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that contains experiment folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "summary.md",
        help="Output markdown file path.",
    )
    return parser.parse_args()


def parse_experiment_name(folder_name: str) -> Dict[str, str]:
    """Parse metadata from one experiment folder name."""
    match = EXPERIMENT_PATTERN.match(folder_name)
    if match is None:
        raise ValueError(f"Unsupported folder name format: {folder_name}")
    meta = match.groupdict()
    suffix_timestamp = meta.get("timestamp")
    prefix_timestamp = meta.get("timestamp_prefix")
    resolved_timestamp = suffix_timestamp or prefix_timestamp
    if resolved_timestamp is None:
        raise ValueError(
            "Missing timestamp in folder name. Expected either front or tail timestamp: "
            f"{folder_name}"
        )
    meta["timestamp"] = resolved_timestamp
    meta.pop("timestamp_prefix", None)
    return meta


def _folder_value_matches_config_row(row_key: str, folder_str: str, cfg_val: Any) -> bool:
    """Return True if folder-parsed row value is consistent with config JSON value."""
    if row_key in ("zdim", "epochs", "step_k", "n_seeds"):
        return str(folder_str) == str(cfg_val)
    if row_key in ("pos_embd", "learnable"):
        f_true = str(folder_str).lower() in ("true", "1")
        if isinstance(cfg_val, bool):
            return f_true == cfg_val
        return str(folder_str).lower() == str(cfg_val).lower()
    return str(folder_str) == str(cfg_val)


def config_value_to_cell(v: Any) -> str:
    """Serialize one config JSON value to a single table cell string."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, str)):
        return str(v)
    return json.dumps(v, sort_keys=True)


def load_and_merge_config(
    cfg_path: Path, row: Dict[str, str], experiment: str
) -> Dict[str, str]:
    """
    Load config.json and return extra columns (no keys duplicated by folder metadata).

    :param cfg_path: Path to config.json.
    :param row: Row dict already containing folder-derived fields.
    :param experiment: Experiment folder name (for error messages).
    :raises ValueError: If config contradicts folder-derived metadata.
    """
    if not cfg_path.exists():
        return {}

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config.json must be a JSON object: {cfg_path}")

    extras: Dict[str, str] = {}
    for key, cfg_val in payload.items():
        if key in CONFIG_KEY_TO_ROW_KEY:
            row_key = CONFIG_KEY_TO_ROW_KEY[key]
            folder_str = row[row_key]
            if not _folder_value_matches_config_row(row_key, folder_str, cfg_val):
                raise ValueError(
                    f"{experiment}: config '{key}'={cfg_val!r} does not match "
                    f"folder-derived '{row_key}'={folder_str!r}"
                )
            continue
        extras[key] = config_value_to_cell(cfg_val)
    return extras


def parse_metrics_line(line: str, prefix: str) -> Dict[str, float]:
    """Parse one metric dictionary line from outputs.txt."""
    if not line.startswith(prefix):
        raise ValueError(f"Line does not start with '{prefix}': {line}")

    metrics_raw = line[len(prefix) :].strip()
    pairs = re.findall(r"'([^']+)':\s*np\.float64\(([-+0-9.eE]+)\)", metrics_raw)
    if not pairs:
        cleaned = FLOAT64_PATTERN.sub(r"\1", metrics_raw)
        pairs = re.findall(r"'([^']+)':\s*([-+0-9.eE]+)", cleaned)

    if not pairs:
        raise ValueError(f"Cannot parse metrics from line: {line}")

    return {key: float(value) for key, value in pairs}


def parse_output_file(path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Parse mean and std metrics from outputs.txt."""
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    mean_prefixes = (
        "Final scores (mean):",
        "Test/val metrics from latest model evaluation (see train):",
    )
    score_line = next((line for line in lines if any(line.startswith(prefix) for prefix in mean_prefixes)), None)
    std_line = next(
        (line for line in lines if line.startswith("Final scores std:")), None
    )

    if score_line is None:
        supported = ", ".join(repr(prefix) for prefix in mean_prefixes)
        raise ValueError(f"Mean metrics line not found in {path}. Supported prefixes: {supported}")
    if std_line is None:
        raise ValueError(f"'Final scores std:' not found in {path}")

    mean_prefix = next(prefix for prefix in mean_prefixes if score_line.startswith(prefix))
    means = parse_metrics_line(score_line, mean_prefix)
    stds = parse_metrics_line(std_line, "Final scores std:")
    return means, stds


def format_value(mean: float, std: float) -> str:
    """Format mean and std as 'xx.xx +- yy.yy'."""
    return f"{mean:.2f} +- {std:.2f}"


def collect_extra_config_keys(results_dir: Path) -> Set[str]:
    """Collect all config.json keys that are not aliased to folder-derived columns."""
    folders = sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("log_")]
    )
    keys: Set[str] = set()
    for folder in folders:
        cfg_path = folder / "config.json"
        if not cfg_path.exists():
            continue
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"config.json must be a JSON object: {cfg_path}")
        for key in payload:
            if key not in CONFIG_KEY_TO_ROW_KEY:
                keys.add(key)
    return keys


def collect_rows(results_dir: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Collect experiment rows, metric columns, and extra config.json columns.

    Config keys that duplicate folder metadata (e.g. ``ds_name`` vs ``dataset``) are
    validated for consistency but not added as separate columns.
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    folders = sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("log_")]
    )
    extra_config_keys = collect_extra_config_keys(results_dir)
    rows: List[Dict[str, str]] = []
    metric_keys: set[str] = set()

    for folder in folders:
        meta = parse_experiment_name(folder.name)
        try:
            means, stds = parse_output_file(folder / "outputs.txt")

            row: Dict[str, str] = {
                "experiment": folder.name,
                "dataset": meta["dataset"],
                "modality": meta["modality"],
                "zdim": meta["zdim"],
                "epochs": meta["epochs"],
                "pos_embd": meta["pos_embd"],
                "learnable": meta["learnable"],
                "step_k": meta["step_k"],
                "n_seeds": meta["n_seeds"],
                "timestamp": meta["timestamp"],
            }

            extras = load_and_merge_config(folder / "config.json", row, folder.name)
            for key in sorted(extra_config_keys):
                row[key] = extras.get(key, "")

            for key in sorted(means.keys()):
                if key not in stds:
                    raise ValueError(f"Metric '{key}' missing std in {folder}")
                metric_keys.add(key)
                row[key] = format_value(means[key], stds[key])

            rows.append(row)
        except FileNotFoundError:
            print(f"No outputs.txt found in {folder}")
            continue

    if not rows:
        raise ValueError(f"No experiment folders found under {results_dir}")

    fixed_columns = [
        "dataset",
        "modality",
        "zdim",
        "epochs",
        "pos_embd",
        "learnable",
        "step_k",
        "n_seeds",
        "timestamp",
        "experiment",
    ]
    sorted_extra = sorted(extra_config_keys)
    columns = fixed_columns + sorted_extra + sorted(metric_keys)
    return columns, rows


def render_markdown_table(columns: List[str], rows: List[Dict[str, str]]) -> str:
    """Render rows into one Markdown table string."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines = [header, separator]
    for row in rows:
        values = [row.get(col, "") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Entry point."""
    args = parse_args()
    columns, rows = collect_rows(args.results_dir)
    table = render_markdown_table(columns, rows)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = "# MultiBench Results Summary\n\n"
    output_path.write_text(title + table + "\n", encoding="utf-8")
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
