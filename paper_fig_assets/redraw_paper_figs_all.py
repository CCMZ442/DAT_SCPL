# redraw_paper_figs_all.py
# Re-draw paper figures in a consistent journal style (vector PDF).
# Outputs: PDF files into deeppcb_yolo/paper_fig_assets

import json
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


# =========================
# YOU MAY EDIT THESE PATHS
# =========================
ROOT = Path(r"D:\ultralytics2\ultralytics-8.3.235\DeepPCB-master\DeepPCB-master\deeppcb_yolo").resolve()
OUT_DIR = ROOT / "paper_fig_assets"
SUMMARY_JSON = OUT_DIR / "summary.json"  # used for pseudo stats (Fig.5) if exists

# main SSOD results (Fig.6/7): where final_metrics.csv are stored
RUNS = {
    "FixedTau": ROOT / "runs_ssod_paper3_one_fixedtau",
    "DAT-only": ROOT / "runs_ssod_paper3_one_datonly",
    "DAT-SCPL": ROOT / "runs_ssod_paper3_one",
}

RATIOS = [("r01", 1), ("r05", 5), ("r10", 10), ("r20", 20)]
SEEDS = [0, 1, 2]

# Sensitivity data (Fig.8) - optional
SENS_HEATMAP_JSON = OUT_DIR / "q_qs_heatmap_r10_seed0.json"   # optional
SENS_CURVE_JSON = OUT_DIR / "q_qs_curve_r10_seed0.json"       # optional
# ==============================================


def paper_rc():
    """Paper-like rcParams: Times New Roman first, compact sizes, vector PDF."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def safe_stdev(xs):
    return stdev(xs) if len(xs) >= 2 else 0.0


def read_final_metrics(csv_path: Path) -> dict:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            return row
    raise ValueError(f"Empty CSV: {csv_path}")


def collect_main_results():
    """
    Returns dict: method -> dict ratio_tag -> dict(metric -> (mean, std, n))
    """
    out = {}
    for method, runs_root in RUNS.items():
        out[method] = {}
        for ratio_tag, _ratio_x in RATIOS:
            vals = {"mAP50-95": [], "mAP50": [], "P": [], "R": []}
            for seed in SEEDS:
                tag = f"{ratio_tag}_seed{seed}"
                csv_path = Path(runs_root) / f"{tag}_final_val" / "final_metrics.csv"
                if not csv_path.exists():
                    continue
                row = read_final_metrics(csv_path)
                for k in vals.keys():
                    if k in row and row[k] != "":
                        vals[k].append(float(row[k]))

            if len(vals["mAP50-95"]) == 0:
                continue

            out[method][ratio_tag] = {
                k: (mean(v), safe_stdev(v), len(v)) for k, v in vals.items()
            }

    out = {m: d for m, d in out.items() if len(d) > 0}
    return out


def _despine(ax):
    """Light journal-like axes (keep left/bottom, soften top/right)."""
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


# -------------------------
# Fig.5: pseudo stats (more paper, less "report")
# -------------------------
def plot_fig5_pseudo_overall():
    if not SUMMARY_JSON.exists():
        print(f"[SKIP] Fig.5 summary.json not found: {SUMMARY_JSON}")
        return

    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    split = "train"
    if "by_split" not in data or split not in data["by_split"]:
        print(f"[SKIP] Fig.5 summary.json has unexpected format. Need by_split/{split}.")
        return
    s = data["by_split"][split]

    coverage = s["images_with_pseudo"] / max(1, s["images_total"])
    empty_rate = s["pseudo_empty_rate"]
    pseudo_density = s["avg_pseudo_boxes_per_image"]
    gt_density = s["avg_gt_boxes_per_image"]
    mp = s["avg_match_precision_iou50"]
    mr = s["avg_match_recall_iou50"]

    fig = plt.figure(figsize=(7.0, 2.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    vals_a = [coverage * 100.0, empty_rate * 100.0]
    labels_a = ["Coverage", "Empty rate"]
    ax1.bar(labels_a, vals_a, edgecolor="black", linewidth=0.6)
    ax1.set_title("Pseudo-label coverage")
    ax1.set_ylabel("Percent")
    ax1.set_ylim(0, max(5, max(vals_a) * 1.25))
    _despine(ax1)

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    vals_b = [pseudo_density, gt_density]
    labels_b = ["Pseudo", "GT"]
    ax2.bar(labels_b, vals_b, edgecolor="black", linewidth=0.6)
    ax2.set_title("Pseudo-label density")
    ax2.set_ylabel("Boxes / image")
    ax2.set_ylim(0, max(vals_b) * 1.25)
    _despine(ax2)

    # Panel C
    ax3 = fig.add_subplot(gs[0, 2])
    vals_c = [mp, mr]
    labels_c = ["Precision", "Recall"]  # shorter & cleaner
    ax3.bar(labels_c, vals_c, edgecolor="black", linewidth=0.6)
    ax3.set_title("Pseudo-label quality")
    ax3.set_ylabel("Ratio")
    ax3.set_ylim(0, 1.05)
    _despine(ax3)

    out_pdf = OUT_DIR / "fig_pseudo_overall.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# -------------------------
# Fig.6: main bar (NO error bars, paper-like)
# -------------------------
def plot_fig6_main_bar(results):
    """
    Fig.6: compact grouped bar chart for mAP@0.5:0.95 using MEAN only.
    Std is reported in the main table instead (avoids "iron-bar" look at r01).
    """
    methods = ["FixedTau", "DAT-only", "DAT-SCPL"]
    labels = [str(x) for _r, x in RATIOS]  # 1 5 10 20
    x = list(range(len(labels)))
    group_width = 0.78
    bar_w = group_width / len(methods)

    fig, ax = plt.subplots(figsize=(6.2, 2.7), constrained_layout=True)

    for i, m in enumerate(methods):
        ys = []
        for ratio_tag, _xv in RATIOS:
            if m not in results or ratio_tag not in results[m]:
                ys.append(float("nan"))
            else:
                mu, _sd, _n = results[m][ratio_tag]["mAP50-95"]
                ys.append(mu)

        xpos = [xi - group_width/2 + (i + 0.5)*bar_w for xi in x]
        ax.bar(
            xpos, ys,
            width=bar_w,
            edgecolor="black",
            linewidth=0.4,
            label=m
        )

    ax.set_xlabel("Labeled ratio (%)")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 0.82)
    ax.margins(x=0.03)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    _despine(ax)

    out_pdf = OUT_DIR / "fig_main_bar_map5095.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# -------------------------
# Fig.7: main curve (keep std, but very subtle)
# -------------------------
def plot_fig7_main_curve(results):
    """
    Fig.7: curve chart for mAP@0.5:0.95 (mean±std), journal-style.
    """
    methods = ["DAT-SCPL", "DAT-only", "FixedTau"]
    xs = [x for _r, x in RATIOS]

    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)

    for m in methods:
        ys, yerr = [], []
        for ratio_tag, _x in RATIOS:
            if m not in results or ratio_tag not in results[m]:
                ys.append(float("nan"))
                yerr.append(0.0)
            else:
                mu, sd, _n = results[m][ratio_tag]["mAP50-95"]
                ys.append(mu)
                yerr.append(sd)

        ax.errorbar(
            xs, ys,
            yerr=yerr,
            marker="o",
            capsize=2,
            elinewidth=0.8,
            label=m
        )

    ax.set_xlabel("Labeled ratio (%)")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(0.15, 0.78)

    # lighter grid to keep paper look
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(frameon=True, loc="lower right")
    _despine(ax)

    out_pdf = OUT_DIR / "fig_main_curve_map5095.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# -------------------------
# Fig.8: sensitivity (optional)
# -------------------------
def plot_fig8_sensitivity():
    if not (SENS_HEATMAP_JSON.exists() and SENS_CURVE_JSON.exists()):
        print("[SKIP] Fig.8 sensitivity json not found (optional).")
        return

    h = json.loads(SENS_HEATMAP_JSON.read_text(encoding="utf-8"))
    c = json.loads(SENS_CURVE_JSON.read_text(encoding="utf-8"))

    q_vals = h["q_values"]
    qs_vals = h["qs_values"]
    z = h["map"]

    fig = plt.figure(figsize=(7.0, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(z, aspect="auto", origin="lower")
    ax1.set_title("Sensitivity (mAP@0.5:0.95)")
    ax1.set_xlabel("DAT percentile q")
    ax1.set_ylabel("SCPL percentile q_s")
    ax1.set_xticks(range(len(q_vals)))
    ax1.set_xticklabels([str(v) for v in q_vals])
    ax1.set_yticks(range(len(qs_vals)))
    ax1.set_yticklabels([str(v) for v in qs_vals])
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    _despine(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(c["x"], c["y"], marker="o")
    ax2.set_title(c.get("title", "Effect of q or q_s"))
    ax2.set_xlabel(c.get("xlabel", "Percentile"))
    ax2.set_ylabel("mAP@0.5:0.95")
    ax2.grid(True, linewidth=0.5, alpha=0.25)
    _despine(ax2)

    out_pdf = OUT_DIR / "fig_scpl_sensitivity.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def main():
    paper_rc()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fig.5
    plot_fig5_pseudo_overall()

    # Fig.6/7
    results = collect_main_results()
    if len(results) == 0:
        print("[WARN] No SSOD final_metrics.csv found. Check RUNS paths.")
    else:
        plot_fig6_main_bar(results)
        plot_fig7_main_curve(results)

    # Fig.8 (optional)
    plot_fig8_sensitivity()

    print("DONE. PDFs are in:", OUT_DIR)


if __name__ == "__main__":
    main()
