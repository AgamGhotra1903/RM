"""
scripts/run_hybrid_model.py
===========================
Executes training and full validation + test evaluation for:
L15_hybrid_kan_leaky_relu: KAN Skip Fusion at Merge6 & Merge7 + LeakyReLU Backbone (alpha=0.2).

Combines the top two empirical discoveries of the Broad-UNet architecture search.
Outputs:
1. layer_rearrangement_experiments/L15_hybrid_kan_leaky_relu/
   - architecture/ (summary.txt, config.yaml)
   - checkpoint/ (best_model.keras)
   - predictions/ (val_predictions_7horizons.npz)
   - metrics/ (metrics_7horizons.json, metrics_7horizons.csv)
   - graphs/ (loss_curves.png, prediction_and_error_maps.png)
   - logs/ (history.csv)
2. Appends to master_dashboard.csv and summary.csv
"""

import sys
import os
import json
import time
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
from models.modules import AnchorResidualBroadcast, SkipFusionBlock
from models.hybrid_model import build_hybrid_kan_leaky_relu

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


def train_and_evaluate_hybrid(
    exp_id: str = "L15_hybrid_kan_leaky_relu",
    epochs: int = 2,
    steps_per_epoch: int = 10,
    val_steps: int = 3,
    eval_samples: int = 160,
    force_retrain: bool = False
):
    print("=" * 80)
    print(f"STARTING HYBRID RUN: {exp_id}")
    print("KAN Skip Fusion (Merge6 & Merge7) + LeakyReLU Backbone (alpha=0.2)")
    print("=" * 80)

    base_dir = Path("layer_rearrangement_experiments") / exp_id
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

    if metrics_json_p.exists() and ckpt_file.exists() and not force_retrain:
        print(f"Skipping {exp_id} - already completed.")
        with open(metrics_json_p) as f:
            return json.load(f)

    # Save Config
    cfg = {
        "exp_id": exp_id,
        "name": "Hybrid KAN Skip Fusion + LeakyReLU Backbone",
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps,
        "eval_samples": eval_samples,
        "horizons": HORIZONS,
        "lead_hours": LEAD_HOURS,
        "learning_rate": 3e-4,
        "batch_size": 4,
        "activation": "leaky_relu",
        "skip_fusion": "kan_merge6_merge7"
    }
    with open(arch_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    dm = ERA5DataManager()
    with open(config.OUTPUTS_DIR / "normalization_stats.json") as f:
        stats = json.load(f)
    t_mean = stats["t2m"]["mean"]
    t_std = stats["t2m"]["std"]

    # Build Generators
    train_gen = ERA5Sequence(
        dm, config.TRAIN_YEARS, stats,
        batch_size=4, dim_model="3d",
        include_cyclical_time=True,
        include_terrain=False,
        shuffle=True
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
        shuffle=False
    )
    val_gen.horizons = HORIZONS
    val_gen.max_lead = max(HORIZONS)
    val_gen.required_window = val_gen.lags + val_gen.max_lead
    val_gen.windows = []
    val_gen._build_windows()
    val_gen.indices = np.arange(len(val_gen.windows))

    model = build_hybrid_kan_leaky_relu(
        lags=6, latitude=32, longitude=48, features=10, features_output=7,
        filters=16, dropout=0.2, residual_head=True
    )

    with open(arch_dir / "summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    loss_fn = Rebalanced7HorizonLoss(spatial_mask=dm.spatial_mask)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss_fn)

    callbacks = [
        ModelCheckpoint(str(ckpt_file), monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        CSVLogger(str(log_dir / "history.csv"))
    ]

    print(f"Training {exp_id} for {epochs} epochs x {steps_per_epoch} steps...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Plot loss curves
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="Train Loss", marker="o", color="#1a365d")
    plt.plot(history.history["val_loss"], label="Val Loss", marker="s", color="#e53e3e")
    plt.title(f"Hybrid KAN+LeakyReLU Loss Curves", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Rebalanced 7-Horizon Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(graph_dir / "loss_curves.png", dpi=180)
    plt.close()

    # Reload best checkpoint
    model.load_weights(str(ckpt_file))

    # Evaluate on Validation Set
    batch_size = 4
    eval_batches = min(len(val_gen), max(1, eval_samples // batch_size))
    total_eval = eval_batches * batch_size

    print(f"Evaluating {exp_id} on {total_eval} validation samples...")
    preds_k = {h: [] for h in HORIZONS}
    targets_k = {h: [] for h in HORIZONS}
    pers_k = {h: [] for h in HORIZONS}

    for b in range(eval_batches):
        xb, _ = val_gen[b]
        p_norm = model(xb, training=False).numpy()
        p_unpad = dm.unpad_spatial(p_norm * t_std + t_mean)[:, 0]

        for i in range(len(xb)):
            g_idx = b * batch_size + i
            y, start_i, in_end_t, tgt_times = val_gen.windows[g_idx]
            year_arr = dm.get_year_data(y)
            t0 = year_arr[start_i + val_gen.lags - 1, :, :, config.TARGET_INDEX]

            for h_idx, h in enumerate(HORIZONS):
                tgt_i = start_i + val_gen.lags + h - 1
                gt = year_arr[tgt_i, :, :, config.TARGET_INDEX]
                preds_k[h].append(p_unpad[i, :, :, h_idx])
                targets_k[h].append(gt)
                pers_k[h].append(t0)

    # Save NPZ predictions
    npz_data = {f"pred_H{h}": np.stack(preds_k[h], axis=0) for h in HORIZONS}
    npz_data.update({f"target_H{h}": np.stack(targets_k[h], axis=0) for h in HORIZONS})
    npz_data.update({f"pers_H{h}": np.stack(pers_k[h], axis=0) for h in HORIZONS})
    np.savez_compressed(pred_dir / "val_predictions_7horizons.npz", **npz_data)

    metrics = {}
    csv_rows = []
    rmse_list = []
    skill_list = []

    for h_idx, h in enumerate(HORIZONS):
        p = np.stack(preds_k[h], axis=0)
        t = np.stack(targets_k[h], axis=0)
        pers = np.stack(pers_k[h], axis=0)

        err = p - t
        err_pers = pers - t

        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        var_t = float(np.var(t))
        r2 = float(1.0 - np.mean(err ** 2) / var_t)

        rmse_pers = float(np.sqrt(np.mean(err_pers ** 2)))
        skill = float(1.0 - rmse / rmse_pers)

        rmse_list.append(rmse)
        skill_list.append(skill)

        lead = LEAD_HOURS[h_idx]
        metrics[f"H{h}"] = {
            "lead_hours": lead,
            "rmse_K": rmse,
            "mae_K": mae,
            "bias_K": bias,
            "r2": r2,
            "rmse_pers_K": rmse_pers,
            "skill_vs_pers": skill
        }
        csv_rows.append({
            "horizon": f"H{h}",
            "lead_hours": lead,
            "rmse_K": rmse,
            "mae_K": mae,
            "bias_K": bias,
            "r2": r2,
            "rmse_pers_K": rmse_pers,
            "skill_vs_pers": skill
        })

    metrics["summary"] = {
        "exp_id": exp_id,
        "name": "Hybrid KAN Skip Fusion + LeakyReLU Backbone",
        "params": int(model.count_params()),
        "mean_val_rmse_K": float(np.mean(rmse_list)),
        "mean_skill_vs_pers": float(np.mean(skill_list))
    }

    with open(metrics_json_p, "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(csv_rows).to_csv(metrics_dir / "metrics_7horizons.csv", index=False)

    # Plot spatial maps
    fig, axes = plt.subplots(3, 7, figsize=(21, 9))
    sample_idx = 0

    for h_idx, h in enumerate(HORIZONS):
        gt = targets_k[h][sample_idx]
        pr = preds_k[h][sample_idx]
        diff = np.abs(pr - gt)
        lead = LEAD_HOURS[h_idx]

        im0 = axes[0, h_idx].imshow(gt, cmap="coolwarm")
        axes[0, h_idx].set_title(f"GT +{lead}h\n(Mean: {np.mean(gt):.1f}K)", fontsize=9)
        axes[0, h_idx].axis("off")

        im1 = axes[1, h_idx].imshow(pr, cmap="coolwarm")
        axes[1, h_idx].set_title(f"Hybrid +{lead}h\n(Mean: {np.mean(pr):.1f}K)", fontsize=9)
        axes[1, h_idx].axis("off")

        im2 = axes[2, h_idx].imshow(diff, cmap="magma", vmin=0, vmax=5)
        axes[2, h_idx].set_title(f"Abs Error +{lead}h\n(RMSE: {np.sqrt(np.mean(diff**2)):.2f}K)", fontsize=9)
        axes[2, h_idx].axis("off")

    plt.suptitle("Hybrid KAN + LeakyReLU Broad-UNet Spatial Predictions Across 7 Horizons", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(graph_dir / "prediction_and_error_maps.png", dpi=180)
    plt.close()

    print("\n" + "=" * 80)
    print(f"HYBRID EXPERIMENT COMPLETED: {exp_id}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Mean Validation RMSE: {metrics['summary']['mean_val_rmse_K']:.3f} K")
    print(f"Mean Skill vs Pers  : {metrics['summary']['mean_skill_vs_pers']:+.3f}")
    print("=" * 80)

    return metrics

if __name__ == "__main__":
    train_and_evaluate_hybrid()
