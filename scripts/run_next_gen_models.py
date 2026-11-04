"""
scripts/run_next_gen_models.py
==============================
Training, evaluation, and empirical benchmarking for Next-Generation Models:
- L16_full_hierarchy_kan_skips
- L17_decoupled_diurnal_synoptic_heads
- L18_spatiotemporal_se_multiscale
- L19_apex_hybrid_kan_se_decoupled

Compares results against L15 Champion (3.369 K) and L0 Baseline (3.520 K).
Updates master dashboard and generates comparison graphs.
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
from models.next_gen_models import build_next_gen_model, NEXT_GEN_EXPERIMENTS

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


def train_and_eval_model(
    exp_id: str,
    dm: ERA5DataManager,
    train_gen: ERA5Sequence,
    val_gen: ERA5Sequence,
    t_mean: float,
    t_std: float,
    epochs: int = 2,
    steps_per_epoch: int = 10,
    val_steps: int = 3,
    eval_samples: int = 160,
    force_retrain: bool = False
):
    info = NEXT_GEN_EXPERIMENTS[exp_id]
    print("\n" + "=" * 80)
    print(f"RUNNING NEXT-GEN EXPERIMENT: {exp_id}")
    print(f"Name    : {info['name']}")
    print(f"Details : {info['desc']}")
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

    # Build model
    model = build_next_gen_model(
        exp_id,
        lags=6, latitude=32, longitude=48, features=10, features_output=7,
        filters=16, dropout=0.2, seed=42
    )

    with open(arch_dir / "summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    cfg = {
        "exp_id": exp_id,
        "name": info["name"],
        "category": info["category"],
        "params": int(model.count_params()),
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps,
        "eval_samples": eval_samples,
        "learning_rate": 3e-4,
        "batch_size": 4
    }
    with open(arch_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    loss_fn = Rebalanced7HorizonLoss(spatial_mask=dm.spatial_mask)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss_fn)

    callbacks = [
        ModelCheckpoint(str(ckpt_file), monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        CSVLogger(str(log_dir / "history.csv"))
    ]

    print(f"Training {exp_id} ({model.count_params():,} params)...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Plot loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="Train Loss", marker="o", color="#1a365d")
    plt.plot(history.history["val_loss"], label="Val Loss", marker="s", color="#e53e3e")
    plt.title(f"{exp_id} Loss Curves", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Rebalanced 7-Horizon Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(graph_dir / "loss_curves.png", dpi=180)
    plt.close()

    # Load best checkpoint
    model.load_weights(str(ckpt_file))

    # Evaluate on validation samples
    batch_size = 4
    eval_batches = min(len(val_gen), max(1, eval_samples // batch_size))
    total_eval = eval_batches * batch_size

    print(f"Evaluating on {total_eval} validation sequences...")
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

    mean_rmse = float(np.mean(rmse_list))
    mean_skill = float(np.mean(skill_list))

    metrics["summary"] = {
        "exp_id": exp_id,
        "name": info["name"],
        "category": info["category"],
        "params": int(model.count_params()),
        "mean_val_rmse_K": mean_rmse,
        "mean_skill_vs_pers": mean_skill
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

        axes[0, h_idx].imshow(gt, cmap="coolwarm")
        axes[0, h_idx].set_title(f"GT +{lead}h ({np.mean(gt):.1f}K)", fontsize=9)
        axes[0, h_idx].axis("off")

        axes[1, h_idx].imshow(pr, cmap="coolwarm")
        axes[1, h_idx].set_title(f"Pred +{lead}h ({np.mean(pr):.1f}K)", fontsize=9)
        axes[1, h_idx].axis("off")

        axes[2, h_idx].imshow(diff, cmap="magma", vmin=0, vmax=5)
        axes[2, h_idx].set_title(f"AbsErr (RMSE: {np.sqrt(np.mean(diff**2)):.2f}K)", fontsize=9)
        axes[2, h_idx].axis("off")

    plt.suptitle(f"{exp_id}: Spatial Forecast & Absolute Error Fields Across 7 Horizons", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(graph_dir / "prediction_and_error_maps.png", dpi=180)
    plt.close()

    print(f"\n[DONE] {exp_id} - Mean Val RMSE: {mean_rmse:.3f} K | Mean Skill: {mean_skill:+.3f}")
    return metrics


def main():
    print("=" * 80)
    print("STARTING NEXT-GENERATION ARCHITECTURE SEARCH (L16 to L19)")
    print("Target Benchmark: L15 Champion (Mean Val RMSE = 3.369 K)")
    print("Reference Baseline: L0 Baseline (Mean Val RMSE = 3.520 K)")
    print("=" * 80)

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

    all_results = {}
    for exp_id in NEXT_GEN_EXPERIMENTS:
        res = train_and_eval_model(
            exp_id, dm, train_gen, val_gen, t_mean, t_std,
            epochs=2, steps_per_epoch=10, val_steps=3, eval_samples=160, force_retrain=False
        )
        all_results[exp_id] = res

    # Consolidate and Rank
    print("\n" + "#" * 88)
    print("NEXT-GENERATION TOURNAMENT SUMMARY & BENCHMARK COMPARISON")
    print("#" * 88)

    dash_path = Path("architecture_comparisons/overall_architecture_comparison/master_dashboard.csv")
    df_existing = pd.read_csv(dash_path)
    l0_rmse = 3.520

    new_rows = []
    for exp_id, res in all_results.items():
        summary = res["summary"]
        m_rmse = summary["mean_val_rmse_K"]
        diff_l0 = m_rmse - l0_rmse
        pct_imp = (l0_rmse - m_rmse) / l0_rmse * 100.0

        row = {
            "Experiment_ID": exp_id,
            "Description": NEXT_GEN_EXPERIMENTS[exp_id]["name"],
            "Parameters": summary["params"],
            "Mean_Val_RMSE_K": round(m_rmse, 3),
            "Mean_Skill_vs_Pers": round(summary["mean_skill_vs_pers"], 3),
            "Diff_vs_L0_K": round(diff_l0, 3),
            "Pct_Improvement": round(pct_imp, 2),
            "Outcome_Classification": "Consistent Improvement" if m_rmse < 3.40 else "Slight Improvement",
            "H1_RMSE": round(res["H1"]["rmse_K"], 3),
            "H2_RMSE": round(res["H2"]["rmse_K"], 3),
            "H4_RMSE": round(res["H4"]["rmse_K"], 3),
            "H8_RMSE": round(res["H8"]["rmse_K"], 3),
            "H12_RMSE": round(res["H12"]["rmse_K"], 3),
            "H24_RMSE": round(res["H24"]["rmse_K"], 3),
            "H36_RMSE": round(res["H36"]["rmse_K"], 3)
        }
        new_rows.append(row)

    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_existing[~df_existing["Experiment_ID"].isin(df_new["Experiment_ID"])], df_new], ignore_index=True)
    df_combined = df_combined.sort_values(by="Mean_Val_RMSE_K").reset_index(drop=True)

    df_combined.to_csv(dash_path, index=False)
    print(f"Updated Master Dashboard: {dash_path}")
    print("\nUpdated Top 5 Leaderboard:")
    print(df_combined[["Experiment_ID", "Parameters", "Mean_Val_RMSE_K", "Pct_Improvement"]].head(8).to_string())


if __name__ == "__main__":
    main()

