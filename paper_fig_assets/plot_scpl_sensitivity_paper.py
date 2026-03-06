import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAG_RE = re.compile(r"(?P<split>.+)_q(?P<q>\d+)_qs(?P<qs>\d+(?:\.\d+)?)$")


def read_final_metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return {
        "mAP50": float(row.get("mAP50", np.nan)),
        "mAP50_95": float(row.get("mAP50-95", np.nan)),
        "P": float(row.get("P", np.nan)),
        "R": float(row.get("R", np.nan)),
    }


def paper_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def metric_label(metric_key: str) -> str:
    mapping = {
        "mAP50_95": r"mAP@0.5:0.95",
        "mAP50": r"mAP@0.5",
        "P": r"Precision",
        "R": r"Recall",
    }
    return mapping.get(metric_key, metric_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="deeppcb_yolo root")
    ap.add_argument("--split_tag", type=str, default="r10_seed0")
    ap.add_argument("--runs_dir", type=str, default="runs_ssod_paper3_one")
    ap.add_argument("--metric", type=str, default="mAP50_95",
                    choices=["mAP50_95", "mAP50", "P", "R"])
    ap.add_argument("--fixed_eta0", type=float, default=None,
                    help="curve slice at given eta0, e.g. 0.30; default uses best eta0")
    ap.add_argument("--no_point_labels", action="store_true",
                    help="do not annotate curve points with numeric values")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    runs = root / args.runs_dir
    if not runs.exists():
        raise FileNotFoundError(f"runs not found: {runs}")

    out_dir = root / "paper_fig_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------
    # Collect records
    # ----------------------
    records = []
    for final_dir in runs.glob(f"{args.split_tag}_q*_qs*_final_val"):
        csv_path = final_dir / "final_metrics.csv"
        if not csv_path.exists():
            continue

        tag = final_dir.name.replace("_final_val", "")
        m = TAG_RE.match(tag)
        if not m:
            continue

        metrics = read_final_metrics(csv_path)
        if metrics is None:
            continue

        q = int(m.group("q"))
        eta0 = float(m.group("qs"))  # qs is the SCPL base threshold in your paper
        records.append({
            "tag": tag,
            "q": q,
            "eta0": eta0,
            **metrics,
        })

    if not records:
        raise RuntimeError("No final_metrics.csv found. Check runs_dir / split_tag naming.")

    df = pd.DataFrame(records).sort_values(["eta0", "q"]).reset_index(drop=True)

    metric = args.metric
    ylab = metric_label(metric)

    # pivot for heatmap
    piv = df.pivot_table(index="eta0", columns="q", values=metric, aggfunc="max").sort_index()
    Z = piv.values.astype(float)

    # best overall (for caption, not for drawing)
    best_row = df.iloc[df[metric].idxmax()]
    best_q = int(best_row["q"])
    best_eta0 = float(best_row["eta0"])
    best_val = float(best_row[metric])

    # curve slice eta0
    slice_eta0 = float(args.fixed_eta0) if args.fixed_eta0 is not None else best_eta0
    df_fix = df[np.isclose(df["eta0"].values, slice_eta0)].sort_values("q")
    if df_fix.empty:
        raise RuntimeError(
            f"No records found for eta0={slice_eta0}. Available eta0: {sorted(df['eta0'].unique())}"
        )

    paper_rc()

    # ==========================================================
    # (A) HEATMAP ONLY -> PDF (NO best box, NO stars)
    # ==========================================================
    fig, ax = plt.subplots(figsize=(3.6, 2.9), dpi=300, constrained_layout=True)

    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    im = ax.imshow(Z, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

    ax.set_title("SCPL sensitivity")
    ax.set_xlabel(r"SCPL percentile $q$")
    ax.set_ylabel(r"SCPL base $\eta_0$")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels([f"{r:.2f}" for r in piv.index])

    cbar = fig.colorbar(im, ax=ax, fraction=0.060, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(ylab, fontsize=8)

    heat_pdf = out_dir / f"fig_scpl_heatmap_{args.split_tag}.pdf"
    fig.savefig(heat_pdf)
    plt.close(fig)
    print("Saved heatmap:", heat_pdf)

    # ==========================================================
    # (B) CURVE ONLY -> PDF
    # ==========================================================
    fig, ax = plt.subplots(figsize=(3.6, 2.9), dpi=300, constrained_layout=True)

    xs = df_fix["q"].values.astype(int)
    ys = df_fix[metric].values.astype(float)

    ax.plot(xs, ys, marker="o", linewidth=1.6)
    ax.set_title(fr"Effect of $q$ (fixed $\eta_0={slice_eta0:.2f}$)")
    ax.set_xlabel(r"SCPL percentile $q$")
    ax.set_ylabel(ylab)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    y_min = float(np.nanmin(ys))
    y_max = float(np.nanmax(ys))
    pad = max(0.003, (y_max - y_min) * 0.25)
    ax.set_ylim(y_min - pad * 0.6, y_max + pad)

    if not args.no_point_labels:
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.0015, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    curve_pdf = out_dir / f"fig_scpl_curve_{args.split_tag}.pdf"
    fig.savefig(curve_pdf)
    plt.close(fig)
    print("Saved curve  :", curve_pdf)

    # Print best for your log (NOT on figure)
    print(f"Best overall (for caption): q={best_q}, eta0={best_eta0:.2f}, {ylab}={best_val:.6f}")


if __name__ == "__main__":
    main()
