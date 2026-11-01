"""
scripts/run_activation_benchmark.py
===================================
Master Benchmark Suite for Global Backbone Activation Functions in Broad-UNet:
Comparing ['relu', 'gelu', 'silu', 'tanh', 'leaky_relu']
Across all 7 forecast horizons: [+2h, +4h, +8h, +16h, +24h, +48h, +72h].

Outputs:
1. activation_experiments/act_{activation}/:
   - architecture/ (summary.txt, config.yaml)
   - checkpoint/ (best_model.keras)
   - predictions/ (val_predictions_7horizons.npz)
   - metrics/ (metrics_7horizons.json, metrics_7horizons.csv)
   - graphs/ (loss_curves.png, prediction_and_error_maps.png)
   - logs/ (history.csv)
2. architecture_comparisons/activation_function_comparison/:
   - activation_dashboard.csv
   - activation_dashboard.md
   - rmse_comparison_7horizons.png
   - convergence_comparison.png
   - side_by_side_prediction_maps.png
"""

import sys
import os
import json
import time
import shutil
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config
from scripts.era5_data import ERA5DataManager, ERA5Sequence
from models.modules import AnchorResidualBroadcast
from models.activation_models import build_broad_unet_with_activation, ACTIVATION_DESCRIPTIONS

HORIZONS = [1, 2, 4, 8, 12, 24, 36]
LEAD_HOURS = [2, 4, 8, 16, 24, 48, 72]
HORIZON_MAP = dict(zip(HORIZONS, LEAD_HOURS))

class Rebalanced7HorizonLoss(tf.keras.losses.Loss):
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
        channel_mse = tf.reduce_sum(diff_sq, axis=[1, 2, 3]) / tf.reduce_sum(mask)
        return tf.reduce_mean(channel_mse * self.weights)


def train_and_evaluate_activation(
    act: str,
    epochs: int = 2,
    steps_per_epoch: int = 10,
    val_steps: int = 3,
    eval_samples: int = 160,
    force_retrain: bool = False
):
    exp_name = f"act_{act}"
    desc = ACTIVATION_DESCRIPTIONS.get(act, f"Broad-UNet with {act} activation")
    base_dir = Path("activation_experiments") / exp_name
    arch_dir = base_dir / "architecture"
    ckpt_dir = base_dir / "checkpoint"
    pred_dir = base_dir / "predictions"
    metrics_dir = base_dir / "metrics"
    graph_dir = base_dir / "graphs"
    log_dir = base_dir / "logs"

    for d in [arch_dir, ckpt_dir, pred_dir, metrics_dir, graph_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metrics_json_p = metrics_dir / "metrics_7horizons.json"
    ckpt_file = ckpt_dir / "best_model.keras"

    # Save Config
    cfg = {
        "activation": act,
        "description": desc,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps,
        "eval_samples": eval_samples,
        "horizons": HORIZONS,
        "lead_hours": LEAD_HOURS,
        "learning_rate": 3e-4,
        "batch_size": 4
    }
    with open(base_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # If act == "relu", check if L0_original already completed with exact same budget
    l0_dir = Path("layer_rearrangement_experiments/L0_original")
    if act == "relu" and (l0_dir / "metrics/metrics_7horizons.json").exists() and not force_retrain:
        print(f"Importing reference metrics for act_relu from {l0_dir}...")
        shutil.copy(l0_dir / "checkpoint/best_model.keras", ckpt_file)
        shutil.copy(l0_dir / "logs/history.csv", log_dir / "history.csv")
        shutil.copy(l0_dir / "graphs/loss_curves.png", graph_dir / "loss_curves.png")
        shutil.copy(l0_dir / "graphs/prediction_and_error_maps.png", graph_dir / "prediction_and_error_maps.png")
        shutil.copy(l0_dir / "predictions/val_predictions_7horizons.npz", pred_dir / "val_predictions_7horizons.npz")
        shutil.copy(l0_dir / "metrics/metrics_7horizons.csv", metrics_dir / "metrics_7horizons.csv")
        with open(l0_dir / "metrics/metrics_7horizons.json", "r") as f:
            data = json.load(f)
        data["activation"] = "relu"
        data["description"] = desc
        with open(metrics_json_p, "w") as f:
            json.dump(data, f, indent=2)
        return data

    if metrics_json_p.exists() and ckpt_file.exists() and not force_retrain:
        print(f"Skipping training: {exp_name} already evaluated at {metrics_json_p}")
        with open(metrics_json_p, "r") as f:
            return json.load(f)

    # Setup Data
    dm = ERA5DataManager()
    with open(config.OUTPUTS_DIR / "normalization_stats.json") as f:
        stats = json.load(f)
    t_mean = stats["t2m"]["mean"]
    t_std = stats["t2m"]["std"]

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
    model = build_broad_unet_with_activation(
        activation=act, lags=6, latitude=32, longitude=48, features=10, features_output=7,
        filters=16, dropout=0.2, residual_head=True, seed=42
    )
    param_count = model.count_params()
    with open(arch_dir / "summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Compile
    loss_fn = Rebalanced7HorizonLoss(dm.spatial_mask)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=optimizer, loss=loss_fn)

    print("\n" + "=" * 80)
    print(f"TRAINING ACTIVATION BENCHMARK: {exp_name} ({act.upper()})")
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

    # Plot loss curve
    plt.figure(figsize=(7, 4), dpi=200)
    plt.plot(history.history["loss"], label=f"Train Loss ({act})", color="#2980b9", lw=2)
    plt.plot(history.history["val_loss"], label=f"Val Loss ({act})", color="#e74c3c", lw=2, linestyle="--")
    plt.title(f"Broad-UNet ({act.upper()} Activation): Loss History\n{desc}", fontsize=10, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted MSE Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_dir / "loss_curves.png", dpi=200)
    plt.close()

    # Evaluation across 7 horizons
    print(f"\nEvaluating {exp_name} across all 7 horizons on validation split...")
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

    pred_npz_path = pred_dir / "val_predictions_7horizons.npz"
    npz_dict = {f"pred_H{h}": np.stack(y_pred_all[h], axis=0) for h in HORIZONS}
    npz_dict.update({f"target_H{h}": np.stack(y_true_all[h], axis=0) for h in HORIZONS})
    npz_dict.update({f"pers_H{h}": np.stack(y_pers_all[h], axis=0) for h in HORIZONS})
    np.savez_compressed(pred_npz_path, **npz_dict)

    horizon_metrics = {}
    for h in HORIZONS:
        yt = np.stack(y_true_all[h], axis=0)
        yp = np.stack(y_pred_all[h], axis=0)
        ypers = np.stack(y_pers_all[h], axis=0)

        err_m = yp - yt
        err_p = ypers - yt
        rmse_m = float(np.sqrt(np.mean(err_m ** 2)))
        rmse_p = float(np.sqrt(np.mean(err_p ** 2)))
        mae_m = float(np.mean(np.abs(err_m)))
        bias_m = float(np.mean(err_m))
        var_yt = float(np.var(yt))
        r2_m = float(1.0 - (np.mean(err_m ** 2) / var_yt))
        skill = float(1.0 - (rmse_m / rmse_p))

        horizon_metrics[f"H{h}"] = {
            "lead_hours": HORIZON_MAP[h],
            "rmse": rmse_m,
            "mae": mae_m,
            "bias": bias_m,
            "r2": r2_m,
            "pers_rmse": rmse_p,
            "skill_vs_pers": skill
        }

    mean_rmse = float(np.mean([m["rmse"] for m in horizon_metrics.values()]))
    mean_skill = float(np.mean([m["skill_vs_pers"] for m in horizon_metrics.values()]))

    results = {
        "activation": act,
        "exp_id": exp_name,
        "description": desc,
        "parameters": param_count,
        "training_duration_sec": train_duration,
        "mean_rmse": mean_rmse,
        "mean_skill_vs_pers": mean_skill,
        "horizons": horizon_metrics
    }

    with open(metrics_json_p, "w") as f:
        json.dump(results, f, indent=2)

    df_rows = []
    for h in HORIZONS:
        m = horizon_metrics[f"H{h}"]
        df_rows.append({
            "activation": act,
            "horizon": f"H{h}",
            "lead_hours": m["lead_hours"],
            "rmse_K": m["rmse"],
            "mae_K": m["mae"],
            "bias_K": m["bias"],
            "r2": m["r2"],
            "pers_rmse_K": m["pers_rmse"],
            "skill_vs_pers": m["skill_vs_pers"]
        })
    pd.DataFrame(df_rows).to_csv(metrics_dir / "metrics_7horizons.csv", index=False)

    # Plot spatial maps for representative sample
    fig, axes = plt.subplots(len(HORIZONS), 4, figsize=(14, 2.2 * len(HORIZONS)), dpi=180)
    for r_idx, h in enumerate(HORIZONS):
        gt = y_true_all[h][0]
        pr = y_pred_all[h][0]
        pe = y_pers_all[h][0]
        err = np.abs(pr - gt)
        vmin, vmax = min(np.min(gt), np.min(pr)), max(np.max(gt), np.max(pr))

        im0 = axes[r_idx, 0].imshow(gt, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[r_idx, 0].set_title(f"H{h} (+{HORIZON_MAP[h]}h) Ground Truth", fontsize=8)
        axes[r_idx, 0].axis("off")
        plt.colorbar(im0, ax=axes[r_idx, 0], fraction=0.046, pad=0.04)

        im1 = axes[r_idx, 1].imshow(pr, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[r_idx, 1].set_title(f"{act.upper()} Prediction", fontsize=8)
        axes[r_idx, 1].axis("off")
        plt.colorbar(im1, ax=axes[r_idx, 1], fraction=0.046, pad=0.04)

        im2 = axes[r_idx, 2].imshow(pe, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[r_idx, 2].set_title(f"Persistence (t0)", fontsize=8)
        axes[r_idx, 2].axis("off")
        plt.colorbar(im2, ax=axes[r_idx, 2], fraction=0.046, pad=0.04)

        im3 = axes[r_idx, 3].imshow(err, cmap="Reds", vmin=0, vmax=6.0)
        axes[r_idx, 3].set_title(f"Error Map (Mean: {np.mean(err):.2f}K)", fontsize=8)
        axes[r_idx, 3].axis("off")
        plt.colorbar(im3, ax=axes[r_idx, 3], fraction=0.046, pad=0.04)

    plt.suptitle(f"Broad-UNet with {act.upper()} Activation: Forecasts Across All 7 Horizons", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(graph_dir / "prediction_and_error_maps.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Completed {exp_name}: Mean Val RMSE = {mean_rmse:.3f} K | Mean Skill = {mean_skill:+.3f}")
    return results


def generate_activation_comparisons(results_dict):
    comp_dir = Path("architecture_comparisons/activation_function_comparison")
    comp_dir.mkdir(parents=True, exist_ok=True)

    ref_rmse = results_dict["relu"]["mean_rmse"]

    # Master Dashboard
    rows = []
    for act, res in results_dict.items():
        rmse = res["mean_rmse"]
        skill = res["mean_skill_vs_pers"]
        diff = rmse - ref_rmse
        pct_gain = (-diff / ref_rmse) * 100.0

        if pct_gain >= 2.0:
            classification = "Improved Performance (★ Significant)"
        elif pct_gain >= 0.5:
            classification = "Slight Improvement"
        elif pct_gain >= -0.5:
            classification = "No Meaningful Difference (Equivalent)"
        else:
            classification = "Degraded Performance (Worse than ReLU)"

        rows.append({
            "Activation": act.upper(),
            "Description": res["description"][:38],
            "Parameters": res["parameters"],
            "Mean_Val_RMSE_K": round(rmse, 3),
            "Mean_Skill_vs_Pers": round(skill, 3),
            "Diff_vs_ReLU_K": round(diff, 3),
            "Pct_Gain": round(pct_gain, 2),
            "Outcome": classification,
            "H1_RMSE": round(res["horizons"]["H1"]["rmse"], 3),
            "H2_RMSE": round(res["horizons"]["H2"]["rmse"], 3),
            "H4_RMSE": round(res["horizons"]["H4"]["rmse"], 3),
            "H8_RMSE": round(res["horizons"]["H8"]["rmse"], 3),
            "H12_RMSE": round(res["horizons"]["H12"]["rmse"], 3),
            "H24_RMSE": round(res["horizons"]["H24"]["rmse"], 3),
            "H36_RMSE": round(res["horizons"]["H36"]["rmse"], 3),
        })

    df = pd.DataFrame(rows).sort_values(by="Mean_Val_RMSE_K", ascending=True)
    df.to_csv(comp_dir / "activation_dashboard.csv", index=False)
    with open(comp_dir / "activation_dashboard.md", "w") as f:
        f.write("# Activation Function Master Benchmark Dashboard\n\n")
        f.write("Systematic comparison of global backbone activation functions in Broad-UNet across all 7 horizons:\n\n")
        f.write(df.to_markdown(index=False))

    print(f"Saved activation dashboard to {comp_dir / 'activation_dashboard.csv'}")

    # 1. Horizon-wise comparison bar chart
    plt.figure(figsize=(12, 6), dpi=250)
    acts = list(results_dict.keys())
    bar_w = 0.8 / len(acts)
    x = np.arange(len(HORIZONS))

    colors_list = ["#34495e", "#2980b9", "#27ae60", "#e74c3c", "#8e44ad"]

    for idx, act in enumerate(acts):
        d = results_dict[act]
        rmses = [d["horizons"][f"H{h}"]["rmse"] for h in HORIZONS]
        plt.bar(x + idx * bar_w - 0.4 + bar_w/2, rmses, width=bar_w,
                label=f"{act.upper()} ({d['mean_rmse']:.2f}K)",
                color=colors_list[idx % len(colors_list)], edgecolor="black", alpha=0.85)

    pers_rmses = [results_dict["relu"]["horizons"][f"H{h}"]["pers_rmse"] for h in HORIZONS]
    plt.plot(x, pers_rmses, color="black", linestyle="--", marker="o", lw=2, label="Persistence (t0)")

    plt.xticks(x, [f"+{HORIZON_MAP[h]}h\n(H{h})" for h in HORIZONS], fontsize=9, fontweight="bold")
    plt.ylabel("Validation RMSE (Kelvin)", fontsize=11, fontweight="bold")
    plt.title("Broad-UNet Activation Function Benchmark: Forecast RMSE Across All 7 Horizons\n(ReLU vs GELU vs SiLU vs Tanh vs LeakyReLU)", fontsize=12, pad=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(comp_dir / "rmse_comparison_7horizons.png", dpi=250)
    plt.close()

    # 2. Convergence comparison plot (overlaying history.csv)
    plt.figure(figsize=(8, 5), dpi=250)
    for idx, act in enumerate(acts):
        hist_p = Path(f"activation_experiments/act_{act}/logs/history.csv")
        if hist_p.exists():
            df_h = pd.read_csv(hist_p)
            plt.plot(df_h["loss"], label=f"{act.upper()} Train", color=colors_list[idx % len(colors_list)], lw=2)
            plt.plot(df_h["val_loss"], label=f"{act.upper()} Val", color=colors_list[idx % len(colors_list)], lw=1.5, linestyle="--")

    plt.title("Convergence Comparison Across Activation Functions", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted MSE Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(comp_dir / "convergence_comparison.png", dpi=250)
    plt.close()

    # 3. Side-by-side spatial maps comparison (Ground Truth vs ReLU vs GELU vs SiLU vs Tanh)
    fig, axes = plt.subplots(len(HORIZONS), len(acts) + 1, figsize=(3 * (len(acts) + 1), 2.2 * len(HORIZONS)), dpi=180)
    npz_dict = {act: np.load(Path(f"activation_experiments/act_{act}/predictions/val_predictions_7horizons.npz")) for act in acts}

    for r_idx, h in enumerate(HORIZONS):
        gt = npz_dict["relu"][f"target_H{h}"][0]
        vmin, vmax = np.min(gt), np.max(gt)

        axes[r_idx, 0].imshow(gt, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[r_idx, 0].set_title(f"H{h} Ground Truth", fontsize=8)
        axes[r_idx, 0].axis("off")

        for c_idx, act in enumerate(acts):
            pred = npz_dict[act][f"pred_H{h}"][0]
            err = np.mean(np.abs(pred - gt))
            axes[r_idx, c_idx + 1].imshow(pred, cmap="inferno", vmin=vmin, vmax=vmax)
            axes[r_idx, c_idx + 1].set_title(f"{act.upper()} (Err: {err:.2f}K)", fontsize=8)
            axes[r_idx, c_idx + 1].axis("off")

    plt.suptitle("Side-by-Side Forecast Comparison Across Global Activation Functions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(comp_dir / "side_by_side_prediction_maps.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("All activation comparison charts generated successfully.")


def run_activation_benchmark():
    activations = ["relu", "gelu", "silu", "tanh", "leaky_relu"]
    results = {}

    print("=" * 80)
    print("STARTING BROAD-UNET GLOBAL ACTIVATION FUNCTION BENCHMARK")
    print(f"Target Activations : {activations}")
    print(f"Target Lead Times  : {LEAD_HOURS} hours ({HORIZONS} steps)")
    print("=" * 80)

    for idx, act in enumerate(activations, 1):
        print(f"\n[{idx}/{len(activations)}] RUNNING ACTIVATION: {act.upper()}")
        try:
            res = train_and_evaluate_activation(act, epochs=2, steps_per_epoch=10, val_steps=3, eval_samples=160)
            results[act] = res
        except Exception as e:
            print(f"Error evaluating {act}: {e}")
            import traceback
            traceback.print_exc()

        tf.keras.backend.clear_session()
        gc.collect()

    generate_activation_comparisons(results)


if __name__ == "__main__":
    run_activation_benchmark()

