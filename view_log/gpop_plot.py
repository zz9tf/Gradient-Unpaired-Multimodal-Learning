import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import pandas as pd
import re
from pathlib import Path

def plot_gpop_with_baseline(df, metric_col, title=None, save_path=None):
    """
    df: 一个已经筛选好的 dataframe
        应该只包含同一组实验设置下的：
        - 1 行 non-gpop
        - 多行 gpop
    metric_col: 比如 "score_xy_mean"
    """

    # 取 baseline
    baseline_df = df[df["gpop"].astype(str).str.lower() == "false"]
    gpop_df = df[df["gpop"].astype(str).str.lower() == "true"].copy()

    if len(baseline_df) == 0:
        raise ValueError("没找到 non-gpop 行")
    if len(gpop_df) == 0:
        raise ValueError("没找到 gpop 行")

    baseline_value = float(baseline_df.iloc[0][metric_col])

    # 自动生成 gpop label
    gpop_df["_label"] = gpop_df.apply(
        lambda r: f"{r['gpop_weights']}",
        axis=1
    )

    gpop_df = gpop_df.sort_values(["gpop_weights"]).reset_index(drop=True)

    gpop_values = gpop_df[metric_col].astype(float).to_numpy()
    gpop_labels = gpop_df["_label"].tolist()

    n = len(gpop_values)

    # 统一颜色范围
    all_vals = np.concatenate([[baseline_value], gpop_values])
    vmin, vmax = all_vals.min(), all_vals.max()
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n), 3.2))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()

    # 上面：baseline 长条
    ax.add_patch(
        Rectangle(
            (0, 0), n, 1,
            facecolor=cmap(norm(baseline_value)),
            edgecolor="white",
            linewidth=2
        )
    )
    ax.text(
        n / 2, 0.5,
        f"{baseline_value:.3f}",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="white"
    )

    # 下面：gpop 小格子
    for i, val in enumerate(gpop_values):
        ax.add_patch(
            Rectangle(
                (i, 1), 1, 1,
                facecolor=cmap(norm(val)),
                edgecolor="white",
                linewidth=2
            )
        )
        ax.text(
            i + 0.5, 1.5,
            f"{val:.3f}",
            ha="center", va="center",
            fontsize=10, color="white"
        )

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(gpop_labels, rotation=45, ha="right")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["non-gpop", "gpop"])

    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(metric_col)

    ax.set_title(title or metric_col)
    plt.tight_layout()

    if save_path is None:
        save_path = f"{title or metric_col}.png"

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close() 
    

def parse_markdown_table(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    table_lines = [ln for ln in lines if ln.strip().startswith('|')]
    if len(table_lines) < 3:
        raise ValueError('No markdown table found')
    header = [c.strip() for c in table_lines[0].strip().strip('|').split('|')]
    rows = []
    for ln in table_lines[2:]:
        parts = [c.strip() for c in ln.strip().strip('|').split('|')]
        if len(parts) != len(header):
            continue
        rows.append(parts)
    return pd.DataFrame(rows, columns=header)


def split_pm(cell: str):
    m = re.match(r'\s*([0-9.]+)\s*\+-\s*([0-9.]+)\s*$', str(cell))
    if not m:
        return float('nan'), float('nan')
    return float(m.group(1)), float(m.group(2))


def build_df(md_path: Path) -> pd.DataFrame:
    df = parse_markdown_table(md_path)
    for c in ['zdim', 'epochs', 'step_k', 'n_seeds']:
        df[c] = pd.to_numeric(df[c])
    for c in ['pos_embd', 'learnable']:
        df[c] = df[c].map({'True': True, 'False': False})
    metric_cols = ['score_x', 'score_xy', 'score_y', 'val_score_x', 'val_score_xy', 'val_score_y']
    for c in metric_cols:
        vals = df[c].apply(split_pm)
        df[c + '_mean'] = vals.str[0]
        df[c + '_std'] = vals.str[1]
    return df

if __name__ == "__main__":
    df = build_df(Path("/home/zz/zheng/Unpaired-Multimodal-Learning/MultiBench/results/freeze/try/summary.md"))

    plot_gpop_with_baseline(df, "score_y_mean", title="gpop_plot")