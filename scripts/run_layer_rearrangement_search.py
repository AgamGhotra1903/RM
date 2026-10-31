"""
scripts/run_layer_rearrangement_search.py
=========================================
Master Architecture Search and Evaluation Engine for Broad-UNet Layer Rearrangements.
Executes the trial-and-error search across 15 architectures (L0 to L14) over all 7 horizons:
[+2h, +4h, +8h, +16h, +24h, +48h, +72h].

Outputs:
1. layer_rearrangement_experiments/L*/:
   - architecture/ (diagram, config)
   - checkpoint/ (best_model.keras)
   - predictions/ (7-horizon NPZ)
   - metrics/ (7-horizon JSON and CSV)
   - graphs/ (prediction maps, error maps, training curves)
   - logs/ (history.csv, run.log)
2. architecture_comparisons/*/:
   - Side-by-side prediction comparison plots
   - Side-by-side architecture diagrams
   - Horizon-wise metric comparison graphs
   - Consolidated master dashboard ranking all architectures
"""

import sys
import os
import json
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config
from scripts.era5_data import ERA5DataManager, ERA5Sequence
from models.modules import AnchorResidualBroadcast
from models.rearranged_models import build_rearranged_model, EXPERIMENT_NAMES

HORIZONS = [1, 2, 4, 8, 12, 24, 36]
LEAD_HOURS = [2, 4, 8, 16, 24, 48, 72]
HORIZON_MAP = dict(zip(HORIZONS, LEAD_HOURS))

class Rebalanced7HorizonLoss(tf.keras.losses.Loss):
    """
    Weighted Masked MSE loss normalized across all 7 forecast horizons:
    Inverse persistence variance weights derived from empirical data:
    H1: 1.80, H2: 0.64, H4: 0.27, H8: 0.26, H12: 1.98, H24: 1.46, H36: 1.24
    """
    def __init__(self, spatial_mask, weights=None, name="rebalanced_7horizon_loss"):
        super().__init__(name=name)
        self.spatial_mask = tf.constant(spatial_mask, dtype=tf.float32)
        if weights is None:
            weights = [1.80, 0.64, 0.27, 0.26, 1.98, 1.46, 1.24]
        w_tensor = tf.constant(weights, dtype=tf.float32)
        self.weights = w_tensor / tf.reduce_mean(w_tensor)

    def call(self, y_true, y_pred):
        mask = tf.reshape(self.spatial_mask, (1, 1, 32, 48, 1))
        diff_sq = tf.square(y_true - y_pred) * mask
        channel_mse = tf.reduce_sum(diff_sq, axis=[1, 2, 3]) / tf.reduce_sum(mask)  # (B, 7)
        return tf.reduce_mean(channel_mse * self.weights)


def draw_architecture_diagram(exp_id: str, desc: str, out_path: Path):
    """
    Draws a visual block diagram of the specific rearranged architecture,
    highlighting which layers were moved, added, or modified.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Title
    ax.text(50, 93, f"Architecture: {exp_id}", ha="center", va="center", fontsize=13, fontweight="bold", color="#1a252f")
    ax.text(50, 87, desc, ha="center", va="center", fontsize=10, fontstyle="italic", color="#555")

    def draw_box(x, y, w, h, text, color, highlight=False):
        ec = "#e74c3c" if highlight else "#2c3e50"
        lw = 2.5 if highlight else 1.2
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                      facecolor=color, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, fontweight="bold" if highlight else "normal", color="#111", zorder=4)

    # Draw generic pipeline with highlighted modifications
    # Input
    draw_box(4, 45, 10, 15, "Input Frames\n(6, 32, 48, 10)", "#d6eaf8")
    
    # Encoder
    enc_color = "#f9e79f" if "encoder" in exp_id or "conv_rearrangement" in exp_id else "#aed6f1"
    draw_box(18, 45, 16, 15, "Encoder 1-4\n(MultiScale + Conv3D)", enc_color, highlight=("encoder" in exp_id or "conv_rearrangement" in exp_id))

    # Pre-ASPP
    pre_aspp_highlight = ("before_aspp" in exp_id or "combined" in exp_id)
    pre_color = "#f5b7b1" if pre_aspp_highlight else "#ebf5fb"
    pre_text = "Pre-ASPP\nModification" if pre_aspp_highlight else "CompressLags\n(Temporal)"
    draw_box(38, 45, 14, 15, pre_text, pre_color, highlight=pre_aspp_highlight)

    # ASPP
    draw_box(56, 45, 12, 15, "ASPP Block\n(Multi-Dilation)", "#f5b041")

    # Post-ASPP
    post_aspp_highlight = ("after_aspp" in exp_id or "then" in exp_id)
    post_color = "#f5b7b1" if post_aspp_highlight else "#ebf5fb"
    post_text = "Post-ASPP\nEnhancement" if post_aspp_highlight else "Dropout (0.2)"
    draw_box(72, 45, 13, 15, post_text, post_color, highlight=post_aspp_highlight)

    # Decoder
    dec_highlight = ("decoder" in exp_id or "skip" in exp_id or "combined" in exp_id)
    dec_color = "#a9dfbf" if dec_highlight else "#d5f5e3"
    dec_text = "Decoder Skips\n(KAN / Pre / Post)" if dec_highlight else "Decoder 6-9\n(UpSampling + Merge)"
    draw_box(88, 45, 10, 15, dec_text, dec_color, highlight=dec_highlight)

    # Output
    draw_box(50, 12, 22, 12, "Forecast Output (7 Horizons)\n[+2h, +4h, +8h, +16h, +24h, +48h, +72h]", "#d5d8dc")

    # Connective arrows
    arr = dict(arrowstyle="->", lw=1.5, color="#2c3e50")
    ax.annotate("", xy=(18, 52.5), xytext=(14, 52.5), arrowprops=arr)
    ax.annotate("", xy=(38, 52.5), xytext=(34, 52.5), arrowprops=arr)
    ax.annotate("", xy=(56, 52.5), xytext=(52, 52.5), arrowprops=arr)
    ax.annotate("", xy=(72, 52.5), xytext=(68, 52.5), arrowprops=arr)
    ax.annotate("", xy=(88, 52.5), xytext=(85, 52.5), arrowprops=arr)
    ax.annotate("", xy=(61, 24), xytext=(93, 45), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50", connectionstyle="arc3,rad=0.3"))

    # Highlight legend
    rect_leg = patches.FancyBboxPatch((4, 8), 35, 18, boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#e74c3c", lw=1.5)
    ax.add_patch(rect_leg)
    ax.text(6, 21, "Key Architectural Modification:", fontsize=8.5, fontweight="bold", color="#c0392b")
    ax.text(6, 14, f"- {desc}", fontsize=7.5, color="#333")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def train_and_evaluate_rearrangement(
    exp_id: str,
    epochs: int = 3,
    steps_per_epoch: int = 12,
    val_steps: int = 4,
    eval_samples: int = 200,
    force_retrain: bool = False
):
    desc = EXPERIMENT_NAMES[exp_id]
    base_dir = Path("layer_rearrangement_experiments") / exp_id
    arch_dir = base_dir / "architecture"
    ckpt_dir = base_dir / "checkpoint"
    pred_dir = base_dir / "predictions"
    met_dir = base_dir / "metrics"
    graph_dir = base_dir / "graphs"
    log_dir = base_dir / "logs"

    for d in [arch_dir, ckpt_dir, pred_dir, met_dir, graph_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metrics_json_p = met_dir / "metrics_7horizons.json"
    ckpt_file = ckpt_dir / "best_model.keras"

    # 1. Architecture diagram and config
    arch_diagram_p = arch_dir / f"{exp_id}_diagram.png"
    if not arch_diagram_p.exists():
        draw_architecture_diagram(exp_id, desc, arch_diagram_p)

    cfg_dict = {
        "experiment_id": exp_id,
        "description": desc,
        "horizons": HORIZONS,
        "lead_hours": LEAD_HOURS,
        "training": {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "val_steps": val_steps,
            "batch_size": 4,
            "optimizer": "adam",
            "learning_rate": 3e-4,
            "loss": "rebalanced_7horizon_loss"
        }
    }
    with open(base_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_dict, f, indent=2)

    # Data Manager & Normalization
    dm = ERA5DataManager()
    with open(config.OUTPUTS_DIR / "normalization_stats.json") as f:
        stats = json.load(f)
    t_mean = stats["t2m"]["mean"]
    t_std = stats["t2m"]["std"]

    # Build generators for 7 horizons
    train_gen = ERA5Sequence(
        dm, config.TRAIN_YEARS, stats,
        batch_size=4, dim_model="3d",
        include_cyclical_time=True,
        include_terrain=False,
        shuffle=True, seed=42
    )
    train_gen.horizons = HORIZONS
    train_gen.max_lead = max(HORIZONS)
    train_gen.required_window = train_gen.lags + train_gen.max_lead
    train_gen.windows = []
    train_gen._build_windows()
    train_gen.indices = np.arange(len(train_gen.windows))

    val_gen = ERA5Sequence(
        dm, config.VAL_YEARS, stats,
        batch_size=4, dim_model="3d",
        include_cyclical_time=True,
        include_terrain=False,
        shuffle=False, seed=42
    )
    val_gen.horizons = HORIZONS
    val_gen.max_lead = max(HORIZONS)
    val_gen.required_window = val_gen.lags + val_gen.max_lead
    val_gen.windows = []
    val_gen._build_windows()
    val_gen.indices = np.arange(len(val_gen.windows))

    # Build Model
    model = build_rearranged_model(
        exp_id=exp_id, lags=6, latitude=32, longitude=48, features=10, features_output=7,
        filters=16, dropout=0.2, residual_head=True, seed=42
    )
    param_count = model.count_params()
    with open(arch_dir / "summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Compile
    loss_fn = Rebalanced7HorizonLoss(dm.spatial_mask)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Check if already trained
    if metrics_json_p.exists() and ckpt_file.exists() and not force_retrain:
        print(f"Skipping training: {exp_id} already evaluated at {metrics_json_p}")
        with open(metrics_json_p, "r") as f:
            return json.load(f)

    print("\n" + "=" * 80)
    print(f"TRAINING REARRANGEMENT EXPERIMENT: {exp_id}")
    print(f"Description: {desc}")
    print(f"Total Parameters: {param_count:,}")
    print("=" * 80)

    csv_log_path = log_dir / "history.csv"
    callbacks = [
        ModelCheckpoint(str(ckpt_file), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        CSVLogger(str(csv_log_path))
    ]

    t0 = time.time()
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    train_duration = time.time() - t0

    # Plot training loss curve
    plt.figure(figsize=(7, 4), dpi=200)
    plt.plot(history.history["loss"], label="Train Loss (Weighted MSE)", color="#2980b9", lw=2)
    plt.plot(history.history["val_loss"], label="Val Loss", color="#e74c3c", lw=2, linestyle="--")
    plt.title(f"{exp_id}: Training & Validation History\n{desc}", fontsize=10, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_dir / "loss_curves.png", dpi=200)
    plt.close()

    # 4. Evaluation across all 7 horizons on validation set
    print(f"\nEvaluating {exp_id} across all 7 horizons on validation split...")
    batches_to_run = min(len(val_gen), max(1, eval_samples // 4))

    y_true_all = {h: [] for h in HORIZONS}
    y_pred_all = {h: [] for h in HORIZONS}
    y_pers_all = {h: [] for h in HORIZONS}

    for b in range(batches_to_run):
        xb, _ = val_gen[b]
        pred_norm = model(xb, training=False).numpy()
        pred_k = dm.unpad_spatial(pred_norm * t_std + t_mean)[:, 0]

        for i in range(len(xb)):
            g_idx = b * 4 + i
            y, start_i, in_end_t, tgt_times = val_gen.windows[g_idx]
            year_arr = dm.get_year_data(y)
            t0_k = year_arr[start_i + val_gen.lags - 1, :, :, config.TARGET_INDEX]

            for h_idx, h in enumerate(HORIZONS):
                tgt_i = start_i + val_gen.lags + h - 1
                gt_k = year_arr[tgt_i, :, :, config.TARGET_INDEX]
                y_true_all[h].append(gt_k)
                y_pred_all[h].append(pred_k[i, :, :, h_idx])
                y_pers_all[h].append(t0_k)

    # Save predictions NPZ
    pred_npz_path = pred_dir / "val_predictions_7horizons.npz"
    npz_dict = {f"pred_H{h}": np.stack(y_pred_all[h], axis=0) for h in HORIZONS}
    npz_dict.update({f"target_H{h}": np.stack(y_true_all[h], axis=0) for h in HORIZONS})
    npz_dict.update({f"pers_H{h}": np.stack(y_pers_all[h], axis=0) for h in HORIZONS})
    np.savez_compressed(pred_npz_path, **npz_dict)

    # Compute metrics per horizon
    horizon_metrics = {}
    for h in HORIZONS:
        yt = np.stack(y_true_all[h], axis=0)
        yp = np.stack(y_pred_all[h], axis=0)
        ypers = np.stack(y_pers_all[h], axis=0)

        err_m = yp - yt
        err_p = ypers - yt

        rmse_m = float(np.sqrt(np.mean(err_m ** 2)))
        mae_m = float(np.mean(np.abs(err_m)))
        mse_m = float(np.mean(err_m ** 2))
        bias_m = float(np.mean(err_m))
        var_t = float(np.var(yt))
        r2_m = float(1.0 - (mse_m / var_t)) if var_t > 1e-6 else 0.0

        rmse_p = float(np.sqrt(np.mean(err_p ** 2)))
        mae_p = float(np.mean(np.abs(err_p)))
        bias_p = float(np.mean(err_p))
        skill_p = float(1.0 - (rmse_m / rmse_p)) if rmse_p > 1e-6 else 0.0

        horizon_metrics[f"H{h}"] = {
            "lead_hours": HORIZON_MAP[h],
            "rmse": rmse_m,
            "mae": mae_m,
            "bias": bias_m,
            "r2": r2_m,
            "pers_rmse": rmse_p,
            "pers_mae": mae_p,
            "skill_vs_pers": skill_p
        }

    mean_rmse = float(np.mean([horizon_metrics[f"H{h}"]["rmse"] for h in HORIZONS]))
    mean_skill = float(np.mean([horizon_metrics[f"H{h}"]["skill_vs_pers"] for h in HORIZONS]))

    results_data = {
        "exp_id": exp_id,
        "description": desc,
        "parameters": param_count,
        "training_duration_sec": train_duration,
        "mean_rmse": mean_rmse,
        "mean_skill_vs_pers": mean_skill,
        "horizons": horizon_metrics
    }

    with open(metrics_json_p, "w") as f:
        json.dump(results_data, f, indent=2)

    # Save metrics CSV
    rows = []
    for h in HORIZONS:
        d = horizon_metrics[f"H{h}"]
        rows.append({
            "horizon": f"H{h}",
            "lead_hours": d["lead_hours"],
            "rmse": d["rmse"],
            "mae": d["mae"],
            "bias": d["bias"],
            "r2": d["r2"],
            "pers_rmse": d["pers_rmse"],
            "skill_vs_pers": d["skill_vs_pers"]
        })
    pd.DataFrame(rows).to_csv(met_dir / "metrics_7horizons.csv", index=False)

    # 5. Generate prediction maps and error maps for representative sample
    sample_idx = 0
    fig, axes = plt.subplots(len(HORIZONS), 4, figsize=(14, 2.3 * len(HORIZONS)), dpi=200)
    for row_idx, h in enumerate(HORIZONS):
        lh = HORIZON_MAP[h]
        gt = y_true_all[h][sample_idx]
        pred = y_pred_all[h][sample_idx]
        pers = y_pers_all[h][sample_idx]
        err = np.abs(pred - gt)

        vmin, vmax = min(np.min(gt), np.min(pred)), max(np.max(gt), np.max(pred))

        im0 = axes[row_idx, 0].imshow(gt, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_title(f"H{h} (+{lh}h) Ground Truth (K)", fontsize=8)
        axes[row_idx, 0].axis("off")

        im1 = axes[row_idx, 1].imshow(pred, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[row_idx, 1].set_title(f"Pred: RMSE {horizon_metrics[f'H{h}']['rmse']:.2f}K", fontsize=8)
        axes[row_idx, 1].axis("off")

        im2 = axes[row_idx, 2].imshow(pers, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[row_idx, 2].set_title(f"Persistence (t0)", fontsize=8)
        axes[row_idx, 2].axis("off")

        im3 = axes[row_idx, 3].imshow(err, cmap="Reds", vmin=0, vmax=max(5.0, np.max(err)))
        axes[row_idx, 3].set_title(f"Absolute Error (K)", fontsize=8)
        axes[row_idx, 3].axis("off")

    plt.suptitle(f"{exp_id}: Spatial Forecast Maps & Absolute Error Across All 7 Horizons", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(graph_dir / "prediction_and_error_maps.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Completed {exp_id}: Mean Val RMSE = {mean_rmse:.3f} K | Mean Skill = {mean_skill:+.3f}")
    return results_data


def generate_comparisons_and_dashboard(results_by_exp: dict):
    """
    Builds the side-by-side comparison directory and master dashboard:
    architecture_comparisons/
    ├── original_vs_conv_rearrangement/
    ├── original_vs_mlp_aspp/
    ├── original_vs_kan_aspp/
    ├── original_vs_mlp_kan/
    ├── original_vs_decoder_rearrangement/
    ├── original_vs_encoder_rearrangement/
    ├── original_vs_activation_changes/
    ├── original_vs_depth_changes/
    └── overall_architecture_comparison/
    """
    print("\n" + "=" * 80)
    print("GENERATING SIDE-BY-SIDE ARCHITECTURE COMPARISONS & DASHBOARD")
    print("=" * 80)

    comp_root = Path("architecture_comparisons")
    categories = {
        "original_vs_conv_rearrangement": ["L0_original", "L1_conv_rearrangement"],
        "original_vs_mlp_aspp": ["L0_original", "L4_mlp_before_aspp", "L5_mlp_after_aspp"],
        "original_vs_kan_aspp": ["L0_original", "L6_kan_before_aspp", "L7_kan_after_aspp"],
        "original_vs_mlp_kan": ["L0_original", "L8_mlp_then_kan", "L9_kan_then_mlp"],
        "original_vs_decoder_rearrangement": ["L0_original", "L10_before_skip_fusion", "L11_after_skip_fusion", "L13_decoder_rearrangement"],
        "original_vs_encoder_rearrangement": ["L0_original", "L1_conv_rearrangement", "L12_encoder_rearrangement"],
        "original_vs_activation_changes": ["L0_original", "L12_encoder_rearrangement"],
        "original_vs_depth_changes": ["L0_original", "L2_conv_before_aspp", "L3_conv_after_aspp"],
        "overall_architecture_comparison": list(EXPERIMENT_NAMES.keys())
    }

    ref_rmse = results_by_exp["L0_original"]["mean_rmse"]

    # 1. Master Dashboard
    dash_rows = []
    for exp_id, data in results_by_exp.items():
        rmse = data["mean_rmse"]
        skill = data["mean_skill_vs_pers"]
        diff_from_ref = rmse - ref_rmse
        pct_gain = (-diff_from_ref / ref_rmse) * 100.0

        if pct_gain >= 3.0:
            status = "Improved Performance (★ Winner)"
        elif pct_gain >= 1.0:
            status = "Slight Improvement"
        elif pct_gain >= -1.0:
            status = "No Meaningful Difference"
        else:
            status = "Worse Performance"

        row = {
            "Experiment_ID": exp_id,
            "Description": data["description"],
            "Parameters": data["parameters"],
            "Mean_Val_RMSE_K": round(rmse, 3),
            "Mean_Skill_vs_Pers": round(skill, 3),
            "Diff_vs_L0_K": round(diff_from_ref, 3),
            "Pct_Improvement": round(pct_gain, 2),
            "Outcome_Classification": status,
            "H1_RMSE": round(data["horizons"]["H1"]["rmse"], 3),
            "H2_RMSE": round(data["horizons"]["H2"]["rmse"], 3),
            "H4_RMSE": round(data["horizons"]["H4"]["rmse"], 3),
            "H8_RMSE": round(data["horizons"]["H8"]["rmse"], 3),
            "H12_RMSE": round(data["horizons"]["H12"]["rmse"], 3),
            "H24_RMSE": round(data["horizons"]["H24"]["rmse"], 3),
            "H36_RMSE": round(data["horizons"]["H36"]["rmse"], 3),
        }
        dash_rows.append(row)

    df_dash = pd.DataFrame(dash_rows).sort_values(by="Mean_Val_RMSE_K", ascending=True)
    overall_dir = comp_root / "overall_architecture_comparison"
    overall_dir.mkdir(parents=True, exist_ok=True)
    df_dash.to_csv(overall_dir / "master_dashboard.csv", index=False)
    with open(overall_dir / "master_dashboard.md", "w") as f:
        f.write("# Consolidated Architecture Search Master Dashboard\n\n")
        f.write("Ranked by Validation Mean RMSE across all 7 forecast horizons (+2h to +72h):\n\n")
        f.write(df_dash.to_markdown(index=False))

    print(f"Saved master dashboard to {overall_dir / 'master_dashboard.csv'}")

    # 2. Side-by-side comparison graphs and tables for each category
    for cat_name, exp_list in categories.items():
        cat_dir = comp_root / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        cat_data = [results_by_exp[e] for e in exp_list if e in results_by_exp]
        if not cat_data:
            continue

        # Horizon-wise comparison bar chart
        plt.figure(figsize=(12, 6), dpi=250)
        bar_w = 0.8 / len(cat_data)
        x = np.arange(len(HORIZONS))

        for idx, d in enumerate(cat_data):
            rmses = [d["horizons"][f"H{h}"]["rmse"] for h in HORIZONS]
            plt.bar(x + idx * bar_w - 0.4 + bar_w/2, rmses, width=bar_w,
                    label=f"{d['exp_id']} ({d['mean_rmse']:.2f}K)", edgecolor="black", alpha=0.85)

        # Plot persistence benchmark
        pers_rmses = [cat_data[0]["horizons"][f"H{h}"]["pers_rmse"] for h in HORIZONS]
        plt.plot(x, pers_rmses, color="black", linestyle="--", marker="o", lw=2, label="Persistence (t0)")

        plt.xticks(x, [f"+{HORIZON_MAP[h]}h\n(H{h})" for h in HORIZONS], fontsize=9, fontweight="bold")
        plt.ylabel("Validation RMSE (Kelvin)", fontsize=11, fontweight="bold")
        plt.title(f"Architectural Comparison: {cat_name.replace('_', ' ').title()}\nForecast RMSE Across All 7 Horizons (+2h to +72h)", fontsize=12, pad=12)
        plt.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(cat_dir / "rmse_comparison_7horizons.png", dpi=250)
        plt.close()

        # Side-by-side prediction maps (Sample test timestamp)
        # Compare L0 vs the modified architecture in cat_data
        if len(cat_data) >= 2:
            fig, axes = plt.subplots(len(HORIZONS), 5, figsize=(16, 2.2 * len(HORIZONS)), dpi=200)
            # Load L0 predictions and Modified predictions
            l0_p = np.load(Path("layer_rearrangement_experiments/L0_original/predictions/val_predictions_7horizons.npz"))
            mod_exp = cat_data[1]["exp_id"]
            mod_p = np.load(Path(f"layer_rearrangement_experiments/{mod_exp}/predictions/val_predictions_7horizons.npz"))

            for r_idx, h in enumerate(HORIZONS):
                gt = l0_p[f"target_H{h}"][0]
                p_l0 = l0_p[f"pred_H{h}"][0]
                p_mod = mod_p[f"pred_H{h}"][0]
                err_l0 = np.abs(p_l0 - gt)
                err_mod = np.abs(p_mod - gt)

                vmin, vmax = min(np.min(gt), np.min(p_mod)), max(np.max(gt), np.max(p_mod))

                axes[r_idx, 0].imshow(gt, cmap="inferno", vmin=vmin, vmax=vmax)
                axes[r_idx, 0].set_title(f"H{h} (+{HORIZON_MAP[h]}h) Ground Truth", fontsize=7.5)
                axes[r_idx, 0].axis("off")

                axes[r_idx, 1].imshow(p_l0, cmap="inferno", vmin=vmin, vmax=vmax)
                axes[r_idx, 1].set_title(f"Original L0 Prediction", fontsize=7.5)
                axes[r_idx, 1].axis("off")

                axes[r_idx, 2].imshow(p_mod, cmap="inferno", vmin=vmin, vmax=vmax)
                axes[r_idx, 2].set_title(f"Modified: {mod_exp}", fontsize=7.5)
                axes[r_idx, 2].axis("off")

                axes[r_idx, 3].imshow(err_l0, cmap="Reds", vmin=0, vmax=6.0)
                axes[r_idx, 3].set_title(f"L0 Error: {np.mean(err_l0):.2f}K", fontsize=7.5)
                axes[r_idx, 3].axis("off")

                axes[r_idx, 4].imshow(err_mod, cmap="Reds", vmin=0, vmax=6.0)
                axes[r_idx, 4].set_title(f"{mod_exp} Error: {np.mean(err_mod):.2f}K", fontsize=7.5)
                axes[r_idx, 4].axis("off")

            plt.suptitle(f"Side-by-Side Spatial Forecast Comparison: Original Broad-UNet vs {mod_exp}", fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(cat_dir / "side_by_side_prediction_maps.png", dpi=200, bbox_inches="tight")
            plt.close()

    print("All comparison graphs and dashboards generated successfully.")


def run_full_search():
    results = {}
    print("=" * 80)
    print("STARTING FULL BROAD-UNET LAYER REARRANGEMENT ARCHITECTURE SEARCH")
    print(f"Total Architectures to Evaluate: {len(EXPERIMENT_NAMES)}")
    print(f"Target Forecast Horizons: {LEAD_HOURS} hours ({HORIZONS} steps)")
    print("=" * 80)

    for idx, exp_id in enumerate(EXPERIMENT_NAMES.keys(), 1):
        print(f"\n[{idx}/{len(EXPERIMENT_NAMES)}] RUNNING ARCHITECTURE: {exp_id}")
        try:
            data = train_and_evaluate_rearrangement(exp_id, epochs=2, steps_per_epoch=10, val_steps=3, eval_samples=160)
            results[exp_id] = data
        except Exception as e:
            print(f"Error evaluating {exp_id}: {e}")
            import traceback
            traceback.print_exc()

        tf.keras.backend.clear_session()
        gc.collect()

    generate_comparisons_and_dashboard(results)

if __name__ == "__main__":
    run_full_search()

