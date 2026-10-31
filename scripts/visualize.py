"""
scripts/visualize.py
====================
Publication-quality visualization generator for ERA5 Broad-UNet temperature forecasting.
Generates:
1. Horizon comparison plots (MAE, RMSE, R^2 vs Lead Time in Hours: 2, 4, 8, 16, 24h).
2. Geographically registered 4-panel temperature maps (Ground Truth, Broad-UNet, Persistence, Absolute Error).
3. Grid-point temporal trajectories (diurnal cycles across horizons).
4. Error distribution histograms and scatter plots with 1:1 reference line.
5. Spatial MAE heatmaps.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config

# Set styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

def plot_horizon_metrics(csv_path: Path, save_dir: Path):
    """
    Plots MAE, RMSE, and R^2 versus forecast horizon lead time (2, 4, 8, 16, 24 hours).
    """
    df = pd.read_csv(csv_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract distinct models and horizons
    models = df["Model"].unique()
    horizons = config.HORIZONS
    lead_hours = [config.HORIZON_TO_HOURS[h] for h in horizons]
    
    colors = {
        "Broad-UNet": "#1d3557",
        "Persistence Baseline": "#e63946",
        "Climatology Baseline": "#2a9d8f",
        "Plain U-Net Baseline": "#f4a261"
    }
    markers = {
        "Broad-UNet": "o",
        "Persistence Baseline": "s",
        "Climatology Baseline": "^",
        "Plain U-Net Baseline": "D"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # 1. MAE vs Lead Time
    for m in models:
        m_df = df[df["Model"] == m]
        axes[0].plot(
            lead_hours, m_df["mae"],
            label=m, color=colors.get(m, "#457b9d"),
            marker=markers.get(m, "o"), linewidth=2.4, markersize=8
        )
    axes[0].set_xlabel("Forecast Lead Time (Hours)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean Absolute Error (Kelvin)", fontsize=11, fontweight="bold")
    axes[0].set_title("MAE vs. Forecast Lead Time", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xticks(lead_hours)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # 2. RMSE vs Lead Time
    for m in models:
        m_df = df[df["Model"] == m]
        axes[1].plot(
            lead_hours, m_df["rmse"],
            label=m, color=colors.get(m, "#457b9d"),
            marker=markers.get(m, "o"), linewidth=2.4, markersize=8
        )
    axes[1].set_xlabel("Forecast Lead Time (Hours)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Root Mean Squared Error (Kelvin)", fontsize=11, fontweight="bold")
    axes[1].set_title("RMSE vs. Forecast Lead Time", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xticks(lead_hours)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    # 3. R^2 vs Lead Time
    for m in models:
        m_df = df[df["Model"] == m]
        axes[2].plot(
            lead_hours, m_df["r2"],
            label=m, color=colors.get(m, "#457b9d"),
            marker=markers.get(m, "o"), linewidth=2.4, markersize=8
        )
    axes[2].set_xlabel("Forecast Lead Time (Hours)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Coefficient of Determination ($R^2$)", fontsize=11, fontweight="bold")
    axes[2].set_title("$R^2$ vs. Forecast Lead Time", fontsize=13, fontweight="bold", pad=10)
    axes[2].set_xticks(lead_hours)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    out_file = save_dir / "horizon_comparison_curves.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Horizon comparison curves saved to {out_file}")

def plot_spatial_maps(npz_path: Path, save_dir: Path):
    """
    Generates geographically referenced 4-panel spatial maps for representative test samples.
    """
    if not npz_path.exists():
        print(f"Prediction arrays not found at {npz_path}")
        return
        
    data = np.load(npz_path)
    y_true_all = data["y_true"]   # (num_horizons, num_samples, 28, 34)
    y_broad_all = data["y_broad"] # (num_horizons, num_samples, 28, 34)
    y_pers_all = data["y_pers"]   # (num_horizons, num_samples, 28, 34)
    horizons = data["horizons"]
    lead_hours = data["lead_hours"]

    # Coordinates
    lats = np.linspace(30.5, 23.75, 28)
    lons = np.linspace(76.75, 85.0, 34)

    maps_dir = save_dir / "spatial_forecast_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = [5, 15, 25]

    for sample_i in sample_indices:
        for h_idx, h in enumerate(horizons):
            lead_h = lead_hours[h_idx]
            gt = y_true_all[h_idx, sample_i]
            pred = y_broad_all[h_idx, sample_i]
            pers = y_pers_all[h_idx, sample_i]
            err = np.abs(gt - pred)

            vmin = min(gt.min(), pred.min(), pers.min())
            vmax = max(gt.max(), pred.max(), pers.max())

            fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)

            # Panel 1: Ground Truth
            im0 = axes[0].imshow(gt, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=[lons[0], lons[-1], lats[-1], lats[0]])
            axes[0].set_title(f"Ground Truth ($t2m$)\nTarget at +{lead_h}h", fontsize=12, fontweight="bold")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Temperature (K)")

            # Panel 2: Broad-UNet Forecast
            im1 = axes[1].imshow(pred, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=[lons[0], lons[-1], lats[-1], lats[0]])
            axes[1].set_title(f"Broad-UNet Prediction\nLead Time: {lead_h}h", fontsize=12, fontweight="bold")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Temperature (K)")

            # Panel 3: Persistence Baseline
            im2 = axes[2].imshow(pers, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=[lons[0], lons[-1], lats[-1], lats[0]])
            axes[2].set_title(f"Persistence Baseline\nLast Observed ($t_0$)", fontsize=12, fontweight="bold")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Temperature (K)")

            # Panel 4: Broad-UNet Absolute Error
            im3 = axes[3].imshow(err, cmap="inferno", extent=[lons[0], lons[-1], lats[-1], lats[0]])
            axes[3].set_title(f"Broad-UNet Absolute Error\nMean: {np.mean(err):.2f} K | Max: {np.max(err):.2f} K", fontsize=12, fontweight="bold")
            fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="|Error| (K)")

            for ax in axes:
                ax.set_xlabel("Longitude (°E)", fontsize=10)
                ax.set_ylabel("Latitude (°N)", fontsize=10)
                ax.grid(True, linestyle=":", alpha=0.4, color="white")

            fig.suptitle(f"ERA5 Regional Temperature Forecast | Horizon H={h} ({lead_h}-Hour Ahead) | Sample {sample_i}", fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()
            out_img = maps_dir / f"sample_{sample_i}_lead_{lead_h}h.png"
            plt.savefig(out_img, bbox_inches="tight")
            plt.close()

    print(f"Spatial maps successfully generated in {maps_dir}")

def plot_scatter_and_residuals(npz_path: Path, save_dir: Path):
    """
    Plots predicted vs observed scatter plots and residual error distributions.
    """
    if not npz_path.exists():
        return
        
    data = np.load(npz_path)
    y_true_all = data["y_true"]
    y_broad_all = data["y_broad"]
    horizons = data["horizons"]
    lead_hours = data["lead_hours"]

    fig, axes = plt.subplots(2, len(horizons), figsize=(4.5 * len(horizons), 8.5), dpi=300)

    for h_idx, h in enumerate(horizons):
        lead_h = lead_hours[h_idx]
        yt = y_true_all[h_idx].ravel()
        yp = y_broad_all[h_idx].ravel()
        residuals = yp - yt

        # Row 1: Scatter plot
        axes[0, h_idx].scatter(yt, yp, alpha=0.15, s=6, color="#1d3557", rasterized=True)
        min_v = min(yt.min(), yp.min())
        max_v = max(yt.max(), yp.max())
        axes[0, h_idx].plot([min_v, max_v], [min_v, max_v], "r--", lw=1.8, label="1:1 Perfect")
        axes[0, h_idx].set_title(f"Lead +{lead_h}h (H={h})", fontsize=11, fontweight="bold")
        axes[0, h_idx].set_xlabel("Observed (K)", fontsize=10)
        axes[0, h_idx].set_ylabel("Predicted (K)", fontsize=10)
        axes[0, h_idx].grid(True, linestyle="--", alpha=0.5)
        axes[0, h_idx].legend(fontsize=8, loc="upper left")

        # Row 2: Residual distribution
        axes[1, h_idx].hist(residuals, bins=40, density=True, color="#457b9d", edgecolor="black", alpha=0.7)
        axes[1, h_idx].axvline(0, color="red", linestyle="--", lw=1.5)
        axes[1, h_idx].set_title(f"Residuals (+{lead_h}h)\nMean: {np.mean(residuals):.2f} K | Std: {np.std(residuals):.2f} K", fontsize=10, fontweight="bold")
        axes[1, h_idx].set_xlabel("Error (Pred - True, K)", fontsize=10)
        axes[1, h_idx].set_ylabel("Density", fontsize=10)
        axes[1, h_idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = save_dir / "scatter_and_residuals.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Scatter and residual plots saved to {out_file}")

if __name__ == "__main__":
    csv_file = config.REPORTS_DIR / "metrics_summary.csv"
    npz_file = config.OUTPUTS_DIR / "test_predictions" / "sample_eval_arrays.npz"
    figures_dir = config.OUTPUTS_DIR / "figures"
    
    if csv_file.exists():
        plot_horizon_metrics(csv_file, figures_dir)
    if npz_file.exists():
        plot_spatial_maps(npz_file, figures_dir)
        plot_scatter_and_residuals(npz_file, figures_dir)

