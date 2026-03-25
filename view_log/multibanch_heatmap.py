import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


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


def make_heatmap(df: pd.DataFrame, outdir: Path):
    # First dimension is modality (x/xy/y); metric is unified as test_y accuracy.
    row_specs = [
        ("x", "score_y_mean", False, False),
        ("xy", "score_y_mean", False, False),
        ("y", "score_y_mean", False, False),
        ("x", "score_y_mean", False, True),
        ("xy", "score_y_mean", False, True),
        ("y", "score_y_mean", False, True),
        ("x", "score_y_mean", True, False),
        ("xy", "score_y_mean", True, False),
        ("y", "score_y_mean", True, False),
        ("x", "score_y_mean", True, True),
        ("xy", "score_y_mean", True, True),
        ("y", "score_y_mean", True, True),
    ]
    zdim_order = sorted(df['zdim'].unique())
    step_order = sorted(df['step_k'].unique(), key=lambda x: (x == -1, x))
    # make -1 first
    step_order = sorted(df['step_k'].unique(), key=lambda x: (0 if x == -1 else 1, x))
    columns = [(z, k) for z in zdim_order for k in step_order]
    matrix = []
    row_labels = []
    for modality, metric_col, pos, learn in row_specs:
        row = []
        for z, k in columns:
            sub = df[
                (df['modality'] == modality)
                & (df['zdim'] == z)
                & (df['step_k'] == k)
                & (df['pos_embd'] == pos)
                & (df['learnable'] == learn)
            ]
            row.append(float(sub.iloc[0][metric_col]) if len(sub) else float('nan'))
        matrix.append(row)
        row_labels.append(f'{modality} | pos={int(pos)} learn={int(learn)}')

    mat_df = pd.DataFrame(matrix, index=row_labels, columns=[f'z{z}\nk={k}' for z, k in columns])
    mat_df.to_csv(outdir / 'combined_test_heatmap_matrix.csv')

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(mat_df.values, aspect='auto')
    ax.set_xticks(range(len(mat_df.columns)))
    ax.set_xticklabels(mat_df.columns, rotation=0, fontsize=9)
    ax.set_yticks(range(len(mat_df.index)))
    ax.set_yticklabels(mat_df.index, fontsize=10)
    ax.set_title('Combined Test Heatmap')

    # annotate cells
    for i in range(mat_df.shape[0]):
        for j in range(mat_df.shape[1]):
            v = mat_df.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8)

    # add separators between zdim groups and between config groups
    for g in range(1, len(zdim_order)):
        ax.axvline(g * len(step_order) - 0.5, linewidth=2)
    for r in [3, 6, 9]:
        ax.axhline(r - 0.5, linewidth=2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Score')
    fig.tight_layout()
    fig.savefig(outdir / 'combined_test_heatmap.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='summary.md')
    parser.add_argument('--outdir', default='combined_heatmap_out')
    args = parser.parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = build_df(input_path)
    df.to_csv(outdir / 'parsed_results.csv', index=False)
    make_heatmap(df, outdir)
    print(f'Saved outputs to {outdir}')


if __name__ == '__main__':
    main()
