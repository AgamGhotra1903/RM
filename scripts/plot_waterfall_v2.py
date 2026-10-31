"""
scripts/plot_waterfall_v2.py
============================
Generates the Gate 2 Waterfall plot showing step-by-step reduction
in validation RMSE from v1 through Baseline v2 ablations (S1, S2, S3),
benchmarked against Persistence and Diurnal Climatology.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def plot_waterfall(summary_csv="results/summary.csv", output_png="results/plots/waterfall_baseline_v2.png"):
    csv_p = Path(summary_csv)
    if not csv_p.exists():
        print(f"Error: {summary_csv} not found.")
        return

    df = pd.read_csv(csv_p)
    out_dir = Path(output_png).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute mean Val RMSE per stage
    stages = ["v1_checkpoint", "S1_seed42", "S2_seed42", "S3_seed42"]
    stage_labels = [
        "v1 Checkpoint\n(Absolute, 4ep)",
        "S1: Residual Head\n(T0 + Delta T)",
        "S2: Cyclical Time\n(sin/cos hour+doy)",
        "S3: Rebalanced Loss\n(Inv-Pers Weighting)"
    ]

    mean_rmses = []
    available_labels = []

    for s, lbl in zip(stages, stage_labels):
        sub = df[df["run_id"] == s]
        if not sub.empty:
            mean_rmse = sub["val_rmse"].astype(float).mean()
            mean_rmses.append(mean_rmse)
            available_labels.append(lbl)

    # Extract Persistence baseline mean val RMSE
    # From phase0_metrics_val.csv or hardcoded empirical val baseline
    # Val persistence RMSEs: H1: 2.205, H2: 3.690, H4: 5.667, H8: 5.741, H12: 2.098 -> Mean: 3.880 K
    pers_val_mean = 3.880
    # Climatology val mean RMSE: ~2.825 K
    clim_val_mean = 2.825

    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Set up waterfall bars
    n = len(mean_rmses)
    x = np.arange(n)
    colors = ["#d9534f"] + ["#337ab7" if i < n-1 else "#5cb85c" for i in range(1, n)]

    bars = plt.bar(x, mean_rmses, color=colors, width=0.55, edgecolor="black", linewidth=1.2, zorder=3)

    # Add connecting lines and annotations
    for i in range(n):
        val = mean_rmses[i]
        plt.text(i, val + 0.15, f"{val:.3f} K", ha="center", va="bottom", fontsize=11, fontweight="bold")
        if i > 0:
            diff = val - mean_rmses[i-1]
            pct = (diff / mean_rmses[i-1]) * 100
            diff_str = f"{diff:+.2f} K ({pct:+.1f}%)"
            mid_y = (val + mean_rmses[i-1]) / 2.0
            plt.annotate(
                diff_str,
                xy=(i - 0.5, mean_rmses[i-1]),
                xytext=(i - 0.5, mid_y + 0.6),
                arrowprops=dict(arrowstyle="->", color="#c0392b" if diff > 0 else "#27ae60", lw=1.5),
                ha="center", fontsize=9, fontweight="bold",
                color="#c0392b" if diff > 0 else "#27ae60",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", alpha=0.9)
            )

    # Horizontal benchmark lines
    plt.axhline(pers_val_mean, color="#e67e22", linestyle="--", linewidth=2.0, zorder=4,
                label=f"Persistence Baseline (Mean: {pers_val_mean:.3f} K)")
    plt.axhline(clim_val_mean, color="#8e44ad", linestyle=":", linewidth=2.0, zorder=4,
                label=f"Diurnal Climatology Baseline (Mean: {clim_val_mean:.3f} K)")

    plt.xticks(x, available_labels, fontsize=10)
    plt.ylabel("Validation Mean RMSE (Kelvin)", fontsize=12, fontweight="bold")
    plt.title("Broad-UNet Optimization Waterfall: Phase 2 Baseline v2 Ablation Progression\nValidation Set (2022–2023) Mean Across All Forecast Horizons (+2h to +24h)", fontsize=13, pad=15)
    plt.ylim(0, max(mean_rmses) * 1.18)
    plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    plt.legend(frameon=True, facecolor="white", edgecolor="#ccc", loc="upper right", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Waterfall plot saved successfully to {output_png}")

if __name__ == "__main__":
    plot_waterfall()

