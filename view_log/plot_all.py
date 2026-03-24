import os, json, argparse, math, re
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# IO
# -------------------------
def read_jsonl_rows(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_rows_to_df(path: str, stats_key="stats") -> pd.DataFrame:
    rows = read_jsonl_rows(os.path.abspath(path))
    out = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != stats_key}
        st = r.get(stats_key, {}) or {}
        for k, v in st.items():
            base[f"stats.{k}"] = v
        out.append(base)
    if not out:
        raise ValueError("Empty train.jsonl")
    df = pd.DataFrame(out)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df


ROOT_BLOCKS_PRIORITY = [
    "__common__",
    "xproj_in.fc",
    "yproj_in.fc",
    "encoder.conv",
    "encoder.transformer",
    "encoder.pos_embedding",
    "decoders.0",
    "decoders.1",
]

SINGLE_LOSSES = ["loss_x", "loss_y"]
SINGLE_METRICS = [
    ("gpop_self_align", "self align", "value"),
    ("gpop_norm_ratio", "norm ratio", "ratio"),
    ("gpop_drift", "drift", "value"),
    ("gpop_std", "std", "value"),
    ("gpop_pos_frac", "pos frac", "ratio"),
    ("gpop_neg_frac", "neg frac", "ratio"),
]
PAIR_METRICS = [
    ("gpop_signed_js", "signed JS", "value"),
    ("gpop_mag_gap", "mag gap", "value"),
    ("gpop_conflict_mass", "conflict mass", "ratio"),
    ("gpop_cos", "cos", "value"),
    ("gpop_sign_disagree", "sign disagree", "ratio"),
]
PAIR_MEAN_METRICS = [
    ("gpop_pair_signed_js_mean", "pair signed JS mean", "value"),
    ("gpop_pair_mag_gap_mean", "pair mag gap mean", "value"),
    ("gpop_pair_conflict_mass_mean", "pair conflict mass mean", "ratio"),
    ("gpop_pair_cos_mean", "pair cos mean", "value"),
    ("gpop_pair_sign_disagree_mean", "pair sign disagree mean", "ratio"),
]


class TrainStatsHub:
    def __init__(self, path: str, stats_prefix="stats."):
        self.path = os.path.abspath(path)
        self.run_dir = os.path.dirname(self.path)
        self.df = normalize_rows_to_df(self.path)

        self.stats_prefix = stats_prefix
        self.base_cols = [c for c in self.df.columns if not c.startswith(stats_prefix)]
        self.stats_cols = [c for c in self.df.columns if c.startswith(stats_prefix)]
        self.stats_keys = sorted([c[len(stats_prefix):] for c in self.stats_cols])
        self.x_default = "step" if "step" in self.df.columns else ("iter" if "iter" in self.df.columns else None)

        self.block_metrics = self._discover_block_metrics()

    def _discover_block_metrics(self):
        out: Dict[str, Dict[str, Dict[str, set]]] = {}
        for key in self.stats_keys:
            if not (key.startswith("grad_pre.") or key.startswith("grad_post.")):
                continue
            parts = key.split(".")
            phase = parts[0].replace("grad_", "")  # pre/post
            rem = parts[1:]
            root, kind, metric = self._parse_grad_tail(rem)
            if root is None:
                continue
            out.setdefault(root, {
                "single": {"pre": {"loss_x": set(), "loss_y": set()}, "post": {"loss_x": set(), "loss_y": set()}},
                "pair": {"pre": set(), "post": set()},
                "pair_mean": {"pre": set(), "post": set()},
            })
            if kind == "single":
                loss_name, metric_name = metric
                out[root]["single"][phase][loss_name].add(metric_name)
            elif kind == "pair":
                out[root]["pair"][phase].add(metric)
            elif kind == "pair_mean":
                out[root]["pair_mean"][phase].add(metric)
        return out

    @staticmethod
    def _parse_grad_tail(parts: List[str]):
        # formats:
        # <root>.loss_x.<metric>
        # <root>.loss_x.loss_y.<metric>
        # <root>.gpop_pair_*_mean
        n = len(parts)
        for i in range(1, n + 1):
            if i < n - 1 and parts[i] in {"loss_x", "loss_y"} and parts[i + 1] in {"loss_x", "loss_y"}:
                root = ".".join(parts[:i])
                metric = parts[i + 2]
                return root, "pair", metric
            if i < n and parts[i] in {"loss_x", "loss_y"}:
                root = ".".join(parts[:i])
                metric = parts[i + 1]
                return root, "single", (parts[i], metric)
            if parts[i - 1].startswith("gpop_pair_"):
                root = ".".join(parts[:i - 1])
                metric = parts[i - 1]
                return root, "pair_mean", metric
        return None, None, None

    def has(self, col: str) -> bool:
        return col in self.df.columns

    def resolve_col(self, y: str) -> str:
        if y in self.df.columns:
            return y
        if not y.startswith(self.stats_prefix) and (self.stats_prefix + y) in self.df.columns:
            return self.stats_prefix + y
        return y

    def get_series(self, y: str, x: Optional[str] = None, dropna=True):
        if x is None:
            if self.x_default is None:
                raise KeyError("Missing x-axis column")
            x = self.x_default
        y = self.resolve_col(y)
        if x not in self.df.columns or y not in self.df.columns:
            raise KeyError(f"Missing column: x='{x}' or y='{y}'")
        d = self.df[[x, y]].copy()
        if dropna:
            d = d.dropna()
        return d[x].to_numpy(), d[y].to_numpy(dtype=float)

    def block_list(self) -> List[str]:
        blocks = list(self.block_metrics.keys())
        keep = []
        for b, d in self.block_metrics.items():
            size = 0
            size += sum(len(d["single"][ph][loss]) for ph in ["pre", "post"] for loss in SINGLE_LOSSES)
            size += len(d["pair"]["pre"] | d["pair"]["post"])
            size += len(d["pair_mean"]["pre"] | d["pair_mean"]["post"])
            if size >= 4:
                keep.append(b)
        blocks = keep

        ordered = [b for b in ROOT_BLOCKS_PRIORITY if b in blocks]
        remainder = sorted([b for b in blocks if b not in ROOT_BLOCKS_PRIORITY])
        return ordered + remainder


def moving_avg(y, win=11):
    y = np.asarray(y, dtype=float)
    if win is None or win <= 1 or len(y) < 2:
        return y
    win = min(int(win), len(y))
    return pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def save_fig(fig, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_blockwise] Saved: {outpath}")


def _set_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _maybe_set_ratio_ylim(ax, ylabel: Optional[str], ys: List[np.ndarray]):
    if ylabel != "ratio":
        return
    vals = np.concatenate([np.asarray(y, dtype=float) for y in ys if len(y) > 0], axis=0) if ys else np.array([])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    lo = max(-0.05, float(np.nanmin(vals)) - 0.05)
    hi = min(1.25 if np.nanmax(vals) <= 1.05 else float(np.nanmax(vals)) + 0.05, 1.25)
    if hi > lo:
        ax.set_ylim(lo, hi)


def plot_line(ax, hub: TrainStatsHub, y: str, *, x=None, label=None, smooth_win=0, linewidth=1.7, alpha=0.95):
    try:
        xs, ys = hub.get_series(y=y, x=x, dropna=True)
    except KeyError:
        return False, None
    if smooth_win and smooth_win > 1:
        ys = moving_avg(ys, smooth_win)
    ax.plot(xs, ys, label=(label or y), linewidth=linewidth, alpha=alpha)
    return True, ys


def plot_group(ax, hub: TrainStatsHub, specs: List[Tuple[str, str]], *, title: str, ylabel: Optional[str] = None, smooth_win=0):
    ok = False
    all_ys = []
    for y, label in specs:
        cur_ok, cur_ys = plot_line(ax, hub, y, label=label, smooth_win=smooth_win)
        ok |= cur_ok
        if cur_ok and cur_ys is not None:
            all_ys.append(cur_ys)
    _set_ax(ax, title=title, xlabel=hub.x_default or "x", ylabel=ylabel)
    if ok:
        ax.legend(fontsize=8)
        _maybe_set_ratio_ylim(ax, ylabel, all_ys)
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
    return ok


# -------------------------
# Legacy schema helpers
# -------------------------
def bcol_legacy(phase: str, block: str, metric: str) -> str:
    return f"stats.{phase}.{block}.{metric}"


def block_metric_pairs_legacy(block: str):
    return [
        ("norm_mean", f"{block}: norm mean", "value"),
        ("norm_cv", f"{block}: norm cv", "value"),
        ("norm_max_frac", f"{block}: norm max frac", "ratio"),
        ("trace", f"{block}: trace", "value"),
        ("erank", f"{block}: erank", "value"),
        ("condish", f"{block}: condish", "value"),
        ("lambda1_ratio", f"{block}: lambda1 ratio", "ratio"),
        ("lambda2_ratio", f"{block}: lambda2 ratio", "ratio"),
        ("gmean_drift", f"{block}: gmean drift", "value"),
        ("gpop_drift", f"{block}: gpop drift", "value"),
        ("gpop_norm_ratio", f"{block}: gpop norm ratio", "ratio"),
        ("viol_frac", f"{block}: viol frac", "ratio"),
        ("gpop_neg_frac", f"{block}: gpop neg frac", "ratio"),
        ("gpop_rho_mean", f"{block}: gpop rho mean", "value"),
        ("eff_sum", f"{block}: eff sum", "value"),
        ("sum_norm", f"{block}: sum norm", "value"),
        ("sum_vec_norm", f"{block}: sum vec norm", "value"),
    ]


def available_block_panels_legacy(hub: TrainStatsHub, block: str):
    panels = []
    for metric, title, ylabel in block_metric_pairs_legacy(block):
        pre = bcol_legacy("pre", block, metric)
        post = bcol_legacy("post", block, metric)
        if hub.has(pre) or hub.has(post):
            panels.append((metric, title, ylabel))
    return panels


def fig_block_summary_legacy(hub: TrainStatsHub, block: str, outpath: str, smooth_win=11, ncols=3):
    panels = available_block_panels_legacy(hub, block)
    n = len(panels)
    if n == 0:
        return
    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(axes, panels):
        plot_group(
            ax,
            hub,
            [
                (bcol_legacy("pre", block, metric), f"pre {metric}"),
                (bcol_legacy("post", block, metric), f"post {metric}"),
            ],
            title=title,
            ylabel=ylabel,
            smooth_win=smooth_win,
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Block Summary: {block}", fontsize=16, y=1.01)
    save_fig(fig, outpath)


# -------------------------
# Grad schema helpers
# -------------------------
def gcol(phase: str, block: str, *tail: str) -> str:
    return "stats." + ".".join([f"grad_{phase}", block, *tail])


def fig_block_summary_grad(hub: TrainStatsHub, block: str, outpath: str, smooth_win=11):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), squeeze=False)
    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(axes[:4], SINGLE_METRICS[:4]):
        specs = []
        for phase in ["pre", "post"]:
            for loss_name in SINGLE_LOSSES:
                col = gcol(phase, block, loss_name, metric)
                if hub.has(col):
                    specs.append((col, f"{phase}/{loss_name}"))
        plot_group(ax, hub, specs, title=f"{block}: {title}", ylabel=ylabel, smooth_win=smooth_win)

    specs = []
    for phase in ["pre", "post"]:
        for loss_name in SINGLE_LOSSES:
            for metric in ["gpop_pos_frac", "gpop_neg_frac"]:
                col = gcol(phase, block, loss_name, metric)
                if hub.has(col):
                    short = "pos" if metric.endswith("pos_frac") else "neg"
                    specs.append((col, f"{phase}/{loss_name}/{short}"))
    plot_group(axes[4], hub, specs, title=f"{block}: sign balance", ylabel="ratio", smooth_win=smooth_win)

    specs = []
    for phase in ["pre", "post"]:
        for metric, title, ylabel in PAIR_METRICS:
            col = gcol(phase, block, "loss_x", "loss_y", metric)
            if hub.has(col):
                specs.append((col, f"{phase}/{metric}"))
    plot_group(axes[5], hub, specs, title=f"{block}: pair conflict bundle", ylabel="value", smooth_win=smooth_win)

    fig.suptitle(f"Gradient Monitor Summary: {block}", fontsize=16, y=1.01)
    save_fig(fig, outpath)


def _block_metric_value_matrix(hub: TrainStatsHub, blocks: List[str], metric_cols: List[Tuple[str, str]], reducer="mean"):
    data = np.full((len(blocks), len(metric_cols)), np.nan, dtype=float)
    for i, block in enumerate(blocks):
        for j, (_, col) in enumerate(metric_cols):
            col = hub.resolve_col(col)
            if col not in hub.df.columns:
                continue
            vals = pd.to_numeric(hub.df[col], errors="coerce").astype(float).dropna().to_numpy()
            if len(vals) == 0:
                continue
            if reducer == "last":
                data[i, j] = vals[-1]
            else:
                data[i, j] = np.nanmean(vals)
    return data


def _heatmap(fig, ax, data, row_labels, col_labels, title, cmap="viridis", center=None):
    vals = np.array(data, dtype=float)
    if center is not None:
        finite = vals[np.isfinite(vals)]
        vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
        vmin, vmax = -vmax, vmax
    else:
        vmin = np.nanmin(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 0.0
        vmax = np.nanmax(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def fig_conflict_heatmaps_grad(hub: TrainStatsHub, outpath: str, reducer="mean"):
    blocks = hub.block_list()
    if not blocks:
        return
    pair_cols_pre = [(m[1], gcol("pre", b, "loss_x", "loss_y", m[0])) for m in PAIR_METRICS for b in blocks]
    pair_cols_post = [(m[1], gcol("post", b, "loss_x", "loss_y", m[0])) for m in PAIR_METRICS for b in blocks]
    # build block x metric matrices directly
    metric_defs = [(title, metric) for metric, title, _ in PAIR_METRICS]
    pre_matrix = _block_metric_value_matrix(hub, blocks, [(title, gcol("pre", b, "loss_x", "loss_y", metric)) for title, metric in metric_defs for b in []])
    # explicit matrices, easier to read
    pre_matrix = np.full((len(blocks), len(metric_defs)), np.nan)
    post_matrix = np.full((len(blocks), len(metric_defs)), np.nan)
    for i, block in enumerate(blocks):
        for j, (title, metric) in enumerate(metric_defs):
            for phase, mat in [("pre", pre_matrix), ("post", post_matrix)]:
                col = hub.resolve_col(gcol(phase, block, "loss_x", "loss_y", metric))
                if col not in hub.df.columns:
                    continue
                vals = pd.to_numeric(hub.df[col], errors="coerce").astype(float).dropna().to_numpy()
                if len(vals) == 0:
                    continue
                mat[i, j] = vals[-1] if reducer == "last" else np.nanmean(vals)

    diff_matrix = post_matrix - pre_matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, 0.55 * len(blocks) + 2)))
    metric_titles = [title for title, _ in metric_defs]
    _heatmap(fig, axes[0], pre_matrix, blocks, metric_titles, f"Pre pair metrics ({reducer})")
    _heatmap(fig, axes[1], post_matrix, blocks, metric_titles, f"Post pair metrics ({reducer})")
    _heatmap(fig, axes[2], diff_matrix, blocks, metric_titles, "Post - Pre", cmap="coolwarm", center=0.0)
    fig.suptitle("Cross-loss conflict heatmaps", fontsize=16, y=1.01)
    save_fig(fig, outpath)


def fig_task_heatmaps_grad(hub: TrainStatsHub, outpath: str, reducer="mean"):
    blocks = hub.block_list()
    if not blocks:
        return
    metric_defs = [("self align", "gpop_self_align"), ("norm ratio", "gpop_norm_ratio"), ("drift", "gpop_drift"), ("std", "gpop_std")]
    fig, axes = plt.subplots(2, 2, figsize=(16, max(8, 0.55 * len(blocks) + 2)))
    axes = axes.ravel()
    for ax, loss_name in zip(axes[:2], ["loss_x", "loss_y"]):
        pre = np.full((len(blocks), len(metric_defs)), np.nan)
        post = np.full((len(blocks), len(metric_defs)), np.nan)
        for i, block in enumerate(blocks):
            for j, (title, metric) in enumerate(metric_defs):
                for phase, mat in [("pre", pre), ("post", post)]:
                    col = hub.resolve_col(gcol(phase, block, loss_name, metric))
                    if col not in hub.df.columns:
                        continue
                    vals = pd.to_numeric(hub.df[col], errors="coerce").astype(float).dropna().to_numpy()
                    if len(vals) == 0:
                        continue
                    mat[i, j] = vals[-1] if reducer == "last" else np.nanmean(vals)
        _heatmap(fig, ax, post - pre, blocks, [t for t, _ in metric_defs], f"{loss_name}: post - pre", cmap="coolwarm", center=0.0)
    for ax, metric_name in zip(axes[2:], ["gpop_pair_cos_mean", "gpop_pair_conflict_mass_mean"]):
        mat = np.full((len(blocks), 2), np.nan)
        for i, block in enumerate(blocks):
            for j, phase in enumerate(["pre", "post"]):
                col = hub.resolve_col(gcol(phase, block, metric_name))
                if col not in hub.df.columns:
                    continue
                vals = pd.to_numeric(hub.df[col], errors="coerce").astype(float).dropna().to_numpy()
                if len(vals) == 0:
                    continue
                mat[i, j] = vals[-1] if reducer == "last" else np.nanmean(vals)
        _heatmap(fig, ax, mat, blocks, ["pre", "post"], metric_name, cmap="viridis")
    fig.suptitle("Task and aggregate block heatmaps", fontsize=16, y=1.01)
    save_fig(fig, outpath)


def fig_common_focus_grad(hub: TrainStatsHub, outpath: str, smooth_win=11):
    block = "__common__"
    if block not in hub.block_list():
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_group(
        axes[0, 0], hub,
        [
            (gcol("pre", block, "loss_x", "gpop_self_align"), "pre/loss_x self_align"),
            (gcol("post", block, "loss_x", "gpop_self_align"), "post/loss_x self_align"),
            (gcol("pre", block, "loss_y", "gpop_self_align"), "pre/loss_y self_align"),
            (gcol("post", block, "loss_y", "gpop_self_align"), "post/loss_y self_align"),
        ],
        title="Common block self-alignment",
        ylabel="value",
        smooth_win=smooth_win,
    )
    plot_group(
        axes[0, 1], hub,
        [
            (gcol("pre", block, "loss_x", "loss_y", "gpop_cos"), "pre pair cos"),
            (gcol("post", block, "loss_x", "loss_y", "gpop_cos"), "post pair cos"),
            (gcol("pre", block, "loss_x", "loss_y", "gpop_conflict_mass"), "pre conflict mass"),
            (gcol("post", block, "loss_x", "loss_y", "gpop_conflict_mass"), "post conflict mass"),
        ],
        title="Common block cross-loss interaction",
        ylabel="value",
        smooth_win=smooth_win,
    )
    plot_group(
        axes[1, 0], hub,
        [
            (gcol("pre", block, "gpop_pair_signed_js_mean"), "pre pair JS mean"),
            (gcol("post", block, "gpop_pair_signed_js_mean"), "post pair JS mean"),
            (gcol("pre", block, "gpop_pair_sign_disagree_mean"), "pre sign disagree mean"),
            (gcol("post", block, "gpop_pair_sign_disagree_mean"), "post sign disagree mean"),
        ],
        title="Common block aggregate pair summaries",
        ylabel="value",
        smooth_win=smooth_win,
    )
    plot_group(
        axes[1, 1], hub,
        [
            ("stats.grad_pre.monitor.num_losses_cur", "pre num_losses_cur"),
            ("stats.grad_post.monitor.num_losses_cur", "post num_losses_cur"),
        ],
        title="Monitor runtime signals",
        ylabel="value",
        smooth_win=0,
    )
    fig.suptitle("Common block overview", fontsize=16, y=1.02)
    save_fig(fig, outpath)


# -------------------------
# Generic summary dump
# -------------------------
def dump_summary(hub: TrainStatsHub, outpath: str):
    lines = []
    lines.append(f"Input: {hub.path}")
    lines.append(f"Rows: {len(hub.df)}")
    lines.append(f"x_default: {hub.x_default}")
    lines.append("")
    lines.append("Blocks discovered:")
    for b in hub.block_list():
        d = hub.block_metrics[b]
        lines.append(f"- {b}")
        for phase in ["pre", "post"]:
            for loss_name in SINGLE_LOSSES:
                vals = sorted(d["single"][phase][loss_name])
                lines.append(f"    {phase}.{loss_name}: {', '.join(vals)}")
            lines.append(f"    {phase}.pair: {', '.join(sorted(d['pair'][phase]))}")
            lines.append(f"    {phase}.pair_mean: {', '.join(sorted(d['pair_mean'][phase]))}")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"[plot_blockwise] Saved: {outpath}")


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="train.jsonl")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--smooth", type=int, default=100)
    ap.add_argument("--block-cols", type=int, default=3)
    ap.add_argument("--heatmap-reducer", type=str, default="mean", choices=["mean", "last"])
    args = ap.parse_args()

    hub = TrainStatsHub(args.input)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.input)), "plots_blockwise_auto_split")
    os.makedirs(outdir, exist_ok=True)

    print(f"[plot_blockwise] Input: {hub.path}")
    print(f"[plot_blockwise] Rows: {len(hub.df)} | stats cols: {len(hub.stats_cols)} | x={hub.x_default}")
    print(f"[plot_blockwise] Blocks: {hub.block_list()}")

    for block in hub.block_list():
        safe = block.replace('.', '_')
        fig_block_summary_grad(
            hub,
            block,
            os.path.join(outdir, f"block_{safe}_summary.png"),
            smooth_win=args.smooth,
        )
    fig_common_focus_grad(hub, os.path.join(outdir, "common_focus.png"), smooth_win=args.smooth)
    for heatmap_reducer in ["mean", "last"]:
        fig_conflict_heatmaps_grad(hub, os.path.join(outdir, f"conflict_heatmaps_{heatmap_reducer}.png"), reducer=heatmap_reducer)
        fig_task_heatmaps_grad(hub, os.path.join(outdir, f"task_heatmaps_{heatmap_reducer}.png"), reducer=heatmap_reducer)

    dump_summary(hub, os.path.join(outdir, "summary.txt"))
    print("[plot_blockwise] Done.")


if __name__ == "__main__":
    main()
