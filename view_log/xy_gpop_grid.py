import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle


# =========================================================
# parsing
# =========================================================
def parse_markdown_table(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError(f"No markdown table found in {path}")

    header = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
    rows = []
    for ln in table_lines[2:]:
        parts = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(parts) != len(header):
            continue
        rows.append(parts)

    return pd.DataFrame(rows, columns=header)


def split_pm(cell: str):
    """
    '52.80 +- 2.48' -> (52.80, 2.48)
    """
    m = re.match(
        r"\s*([+-]?[0-9]*\.?[0-9]+)\s*\+-\s*([+-]?[0-9]*\.?[0-9]+)\s*$",
        str(cell),
    )
    if not m:
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def parse_bool_col(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )


def build_df(md_path: Path) -> pd.DataFrame:
    df = parse_markdown_table(md_path)

    # numeric cols
    for c in ["zdim", "epochs", "step_k", "n_seeds"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # bool cols
    for c in ["pos_embd", "learnable", "gpop"]:
        if c in df.columns:
            df[c] = parse_bool_col(df[c])

    # metric mean/std cols
    metric_cols = [
        "score_x", "score_xy", "score_y",
        "val_score_x", "val_score_xy", "val_score_y",
    ]
    for c in metric_cols:
        if c in df.columns:
            vals = df[c].apply(split_pm)
            df[c + "_mean"] = vals.str[0]
            df[c + "_std"] = vals.str[1]

    return df


# =========================================================
# gpop weights helpers
# =========================================================
def parse_weights_str(s: str) -> dict:
    """
    'loss_x=0.2,loss_y=0.8' -> {'loss_x': 0.2, 'loss_y': 0.8}
    """
    if pd.isna(s):
        return {}

    out = {}
    for item in str(s).split(","):
        item = item.strip()
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except ValueError:
            pass
    return out


def weight_sort_key(s: str):
    """
    stable ordering by numeric weights
    """
    d = parse_weights_str(s)
    if not d:
        return (999, str(s))

    keys = sorted(d.keys())
    return tuple(d[k] for k in keys) + (str(s),)


def short_weight_label(s: str) -> str:
    """
    'loss_x=0.2,loss_y=0.8' -> 'x=0.2\ny=0.8'
    """
    d = parse_weights_str(s)
    if not d:
        return str(s)

    parts = []
    for k in sorted(d.keys()):
        kk = k.replace("loss_", "")
        vv = d[k]
        parts.append(f"{kk}={vv:g}")
    return "\n".join(parts)


# =========================================================
# layout specs: heatmap-like ordering
# =========================================================
def row_specs_xy_only():
    """
    y-axis of subplot grid:
    rows = (pos_embd, learnable)
    """
    return [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]


def col_specs_from_df(df: pd.DataFrame):
    """
    x-axis of subplot grid:
    cols = zdim × step_k
    and step_k=-1 first
    """
    zdim_order = sorted([x for x in df["zdim"].dropna().unique()])
    step_order = sorted(
        [x for x in df["step_k"].dropna().unique()],
        key=lambda x: (0 if x == -1 else 1, x)
    )
    col_specs = [(z, k) for z in zdim_order for k in step_order]
    return col_specs, zdim_order, step_order


def get_group_df(
    df: pd.DataFrame,
    modality: str,
    pos_embd: bool,
    learnable: bool,
    zdim: int,
    step_k: int,
    dataset: str | None = None,
    epochs: int | None = None,
) -> pd.DataFrame:
    sub = df[
        (df["modality"] == modality)
        & (df["pos_embd"] == pos_embd)
        & (df["learnable"] == learnable)
        & (df["zdim"] == zdim)
        & (df["step_k"] == step_k)
    ].copy()

    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
    if epochs is not None:
        sub = sub[sub["epochs"] == epochs]

    return sub.reset_index(drop=True)


# =========================================================
# color helpers
# =========================================================
def compute_global_range(df: pd.DataFrame, metric_col: str):
    vals = df[metric_col].dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return 0.0, 1.0

    vmin = float(vals.min())
    vmax = float(vals.max())

    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6
    return vmin, vmax


def best_text_color(value: float, norm):
    nv = norm(value)
    return "black" if nv > 0.62 else "white"


# =========================================================
# draw one cell-subplot using single-group gpop logic
# =========================================================
def draw_single_group_gpop_on_ax(
    ax,
    subdf: pd.DataFrame,
    metric_col: str,
    cmap,
    show_weight_labels: bool = True,
):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#d8d8d8")

    if len(subdf) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9, color="gray")
        return

    baseline_df = subdf[subdf["gpop"] == False].copy()
    gpop_df = subdf[subdf["gpop"] == True].copy()

    if len(baseline_df) == 0 or len(gpop_df) == 0:
        ax.text(0.5, 0.5, "no gpop", ha="center", va="center", fontsize=9, color="gray")
        return

    baseline_value = float(baseline_df.iloc[0][metric_col])

    gpop_df = gpop_df.copy()
    gpop_df["_label"] = gpop_df["gpop_weights"].apply(short_weight_label)
    gpop_df = gpop_df.sort_values(
        by="gpop_weights",
        key=lambda s: s.map(weight_sort_key)
    ).reset_index(drop=True)

    gpop_values = gpop_df[metric_col].astype(float).to_numpy()

    # =========================================================
    # 🔥 核心改动：local normalization
    # =========================================================
    all_vals = np.concatenate([[baseline_value], gpop_values])

    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6
        
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # =========================================================

    # layout
    top_h = 0.42
    gap = 0.05
    bottom_y = top_h + gap
    bottom_h = 1.0 - bottom_y

    # baseline
    ax.add_patch(
        Rectangle(
            (0, 0), 1, top_h,
            facecolor=cmap(norm(baseline_value)),
            edgecolor="white",
            linewidth=1.8,
        )
    )

    ax.text(0.5, top_h/2, f"{baseline_value:.2f}",
            ha="center", va="center", color="white", fontsize=10)

    # gpop
    n = len(gpop_values)
    pad = 0.02
    cell_w = (1 - pad*(n+1)) / n
    cell_h = bottom_h - 2*pad

    for i, (val, lab) in enumerate(zip(gpop_values, gpop_df["_label"])):
        x = pad + i*(cell_w + pad)
        y = bottom_y + pad

        ax.add_patch(
            Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=cmap(norm(val)),
                edgecolor="white",
                linewidth=1.3,
            )
        )

        txt = f"{val:.2f}"
        if show_weight_labels:
            txt += f"\n{lab}"

        ax.text(
            x + cell_w/2,
            y + cell_h/2,
            txt,
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
        )

# =========================================================
# main figure
# =========================================================
def make_xy_gpop_heatmap_subplots(
    df: pd.DataFrame,
    outdir: Path,
    metric_col: str = "val_score_y_mean",
    dataset: str | None = None,
    epochs: int | None = None,
    modality: str = "xy",
    show_weight_labels: bool = True,
):
    use_df = df.copy()

    if dataset is not None:
        use_df = use_df[use_df["dataset"] == dataset]
    if epochs is not None:
        use_df = use_df[use_df["epochs"] == epochs]
    use_df = use_df[use_df["modality"] == modality].copy()

    if len(use_df) == 0:
        raise ValueError("No rows left after filtering.")

    rows = row_specs_xy_only()
    cols, zdim_order, step_order = col_specs_from_df(use_df)

    nrows = len(rows)
    ncols = len(cols)

    vmin, vmax = compute_global_range(use_df, metric_col)
    cmap = plt.get_cmap("viridis")

    fig_w = max(12, ncols * 2.9 + 1.5)
    fig_h = max(7, nrows * 2.5 + 1.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    summary_rows = []

    for r, (pos_embd, learnable) in enumerate(rows):
        for c, (zdim, step_k) in enumerate(cols):
            ax = axes[r, c]

            sub = get_group_df(
                use_df,
                modality=modality,
                pos_embd=pos_embd,
                learnable=learnable,
                zdim=zdim,
                step_k=step_k,
                dataset=dataset,
                epochs=epochs,
            )

            n_baseline = int((sub["gpop"] == False).sum()) if len(sub) else 0
            n_gpop = int((sub["gpop"] == True).sum()) if len(sub) else 0
            summary_rows.append({
                "modality": modality,
                "pos_embd": pos_embd,
                "learnable": learnable,
                "zdim": zdim,
                "step_k": step_k,
                "n_rows": len(sub),
                "n_baseline": n_baseline,
                "n_gpop": n_gpop,
            })

            draw_single_group_gpop_on_ax(
                ax=ax,
                subdf=sub,
                metric_col=metric_col,
                cmap=cmap,
                show_weight_labels=show_weight_labels,
            )

    # column titles
    for c, (zdim, step_k) in enumerate(cols):
        axes[0, c].set_title(
            f"z={int(zdim)}\nk={int(step_k)}",
            fontsize=10,
            pad=8,
        )

    # row labels
    for r, (pos_embd, learnable) in enumerate(rows):
        axes[r, 0].set_ylabel(
            f"{modality}\npos={int(pos_embd)} learn={int(learnable)}",
            rotation=0,
            labelpad=42,
            va="center",
            fontsize=10,
        )

    # vertical separators between zdim blocks
    for g in range(1, len(zdim_order)):
        c_sep = g * len(step_order)
        if c_sep < ncols:
            x = axes[0, c_sep].get_position().x0
            fig.add_artist(
                plt.Line2D(
                    [x, x],
                    [axes[-1, 0].get_position().y0, axes[0, 0].get_position().y1],
                    transform=fig.transFigure,
                    linewidth=2.0,
                    color="black",
                    alpha=0.22,
                )
            )

    title_bits = [f"{modality} gpop heatmap-style subplots", metric_col]
    if dataset is not None:
        title_bits.append(f"dataset={dataset}")
    if epochs is not None:
        title_bits.append(f"epochs={epochs}")

    fig.suptitle(" | ".join(title_bits), fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.04, 0.03, 0.96, 0.97])

    out_png = outdir / f"{modality}_gpop_heatmap_subplots_{metric_col}.png"
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        outdir / f"{modality}_gpop_heatmap_subplots_summary.csv",
        index=False,
    )

    print(f"saved: {out_png}")


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="summary.md")
    parser.add_argument("--outdir", type=str, default="xy_gpop_heatmap_subplots_out")
    parser.add_argument("--metric", type=str, default="score_y_mean")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--modality", type=str, default="xy")
    parser.add_argument("--hide_weight_labels", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = build_df(Path(args.input))
    df.to_csv(outdir / "parsed_results.csv", index=False)

    make_xy_gpop_heatmap_subplots(
        df=df,
        outdir=outdir,
        metric_col=args.metric,
        dataset=args.dataset,
        epochs=args.epochs,
        modality=args.modality,
        show_weight_labels=not args.hide_weight_labels,
    )


if __name__ == "__main__":
    main()