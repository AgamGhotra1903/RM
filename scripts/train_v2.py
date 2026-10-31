"""
scripts/train_v2.py
===================
Master training, evaluation, and ablation execution engine for Phase 2 (Baseline v2)
and Phase 3 (Architecture Experiments).

Features:
- Config-driven (YAML) architecture and training settings.
- Strict chronological data splits (Train: 2011-2021, Val: 2022-2023, Test: 2024-2025).
- Zero-leakage evaluation on validation partition for model selection.
- Residual temperature prediction: Yhat = X_{t0} + Delta Y.
- Cyclical temporal channels (sin/cos hour-of-day, sin/cos day-of-year).
- Inverse persistence variance loss rebalancing.
- Automatic persistence and diurnal climatology baseline benchmarking in Kelvin.
- Output artifact management: config.yaml, metrics.json, history.csv, best_model.keras, predictions.npz.
- Auto-updates results/summary.csv and results/SUMMARY.md.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config
from scripts.era5_data import ERA5DataManager, ERA5Sequence
from models.broad_unet import MaskedMSELoss, MaskedMAE
from models.broad_unet_v2 import build_broad_unet_v2
from models.modules import (
    AnchorResidualBroadcast, MLPBlock, KANBlock, SequentialMLPKANBlock,
    ParallelMLPKANBlock, SkipFusionBlock, Conv3DControlBlock
)

class RebalancedMaskedMSELoss(tf.keras.losses.Loss):
    """
    Weighted Masked MSE loss that normalizes each horizon's loss by inverse persistence variance,
    preventing H4 diurnal swings from dominating backpropagated gradients.
    """
    def __init__(self, spatial_mask, weights=None, name="rebalanced_masked_mse"):
        super().__init__(name=name)
        self.spatial_mask = tf.constant(spatial_mask, dtype=tf.float32)
        if weights is None:
            # Equal weighting
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        w_tensor = tf.constant(weights, dtype=tf.float32)
        # Normalize so mean weight is 1.0
        self.weights = w_tensor / tf.reduce_mean(w_tensor)

    def call(self, y_true, y_pred):
        mask = tf.reshape(self.spatial_mask, (1, 1, 32, 48, 1))
        diff_sq = tf.square(y_true - y_pred) * mask
        channel_mse = tf.reduce_sum(diff_sq, axis=[1, 2, 3]) / tf.reduce_sum(mask)  # (B, 5)
        weighted_loss = tf.reduce_mean(channel_mse * self.weights)
        return weighted_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "weights": self.weights.numpy().tolist()
        })
        return config


def evaluate_model_on_split(
    model,
    data_gen: ERA5Sequence,
    dm: ERA5DataManager,
    t_mean: float,
    t_std: float,
    split_name: str = "val",
    max_samples: int = 400
) -> dict:
    """
    Evaluates a trained model strictly in physical Kelvin units.
    Computes per-horizon RMSE, MAE, MSE, R2, Bias, and Skill scores.
    """
    horizons = config.HORIZONS
    lead_hours = [config.HORIZON_TO_HOURS[h] for h in horizons]
    batch_size = data_gen.batch_size
    batches_to_run = min(len(data_gen), max(1, max_samples // batch_size))

    y_true_all = {h: [] for h in horizons}
    y_pred_all = {h: [] for h in horizons}
    y_pers_all = {h: [] for h in horizons}

    for b in range(batches_to_run):
        xb, _ = data_gen[b]
        # Predict normalized values
        pred_norm = model(xb, training=False).numpy()
        pred_k = pred_norm * t_std + t_mean
        pred_unpadded = dm.unpad_spatial(pred_k)[:, 0]  # (B, 28, 34, 5)

        for i in range(len(xb)):
            global_idx = b * batch_size + i
            y, start_i, input_end_t, target_times = data_gen.windows[global_idx]
            year_arr = dm.get_year_data(y)

            # Last observed t2m frame (Persistence)
            t0_k = year_arr[start_i + data_gen.lags - 1, :, :, config.TARGET_INDEX]

            for h_idx, h in enumerate(horizons):
                tgt_i = start_i + data_gen.lags + h - 1
                gt_k = year_arr[tgt_i, :, :, config.TARGET_INDEX]
                p_k = pred_unpadded[i, :, :, h_idx]

                y_true_all[h].append(gt_k)
                y_pred_all[h].append(p_k)
                y_pers_all[h].append(t0_k)

    # Compute metrics per horizon
    horizon_metrics = {}
    for h_idx, h in enumerate(horizons):
        yt = np.stack(y_true_all[h], axis=0)  # (N, 28, 34)
        yp = np.stack(y_pred_all[h], axis=0)
        ypers = np.stack(y_pers_all[h], axis=0)

        err_model = yp - yt
        err_pers = ypers - yt

        rmse_model = float(np.sqrt(np.mean(err_model ** 2)))
        mae_model = float(np.mean(np.abs(err_model)))
        mse_model = float(np.mean(err_model ** 2))
        bias_model = float(np.mean(err_model))

        rmse_pers = float(np.sqrt(np.mean(err_pers ** 2)))
        mae_pers = float(np.mean(np.abs(err_pers)))
        bias_pers = float(np.mean(err_pers))

        # Variance and R2
        var_yt = float(np.var(yt))
        r2_model = float(1.0 - (mse_model / var_yt)) if var_yt > 1e-6 else 0.0
        skill_pers = float(1.0 - (rmse_model / rmse_pers)) if rmse_pers > 1e-6 else 0.0

        horizon_metrics[f"H{h}"] = {
            "lead_hours": lead_hours[h_idx],
            "rmse": rmse_model,
            "mae": mae_model,
            "mse": mse_model,
            "bias": bias_model,
            "r2": r2_model,
            "pers_rmse": rmse_pers,
            "pers_mae": mae_pers,
            "pers_bias": bias_pers,
            "skill_vs_pers": skill_pers
        }

    # Global summary
    mean_rmse = float(np.mean([horizon_metrics[f"H{h}"]["rmse"] for h in horizons]))
    mean_skill = float(np.mean([horizon_metrics[f"H{h}"]["skill_vs_pers"] for h in horizons]))

    return {
        "split": split_name,
        "mean_rmse": mean_rmse,
        "mean_skill_vs_pers": mean_skill,
        "horizons": horizon_metrics
    }


def run_experiment(
    cfg_path: str = "configs/baseline_v2.yaml",
    stage: str = "S1",
    seed: int = 42,
    force_retrain: bool = False
):
    print("=" * 80)
    print(f"STARTING EXPERIMENT STAGE: {stage} (Seed: {seed})")
    print("=" * 80)

    # Set seeds for determinism
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Load configuration
    cfg_p = Path(cfg_path).resolve()
    with open(cfg_p, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]
    run_id = f"{stage}_seed{seed}"
    runs_dir = Path("results/runs") / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(runs_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, indent=2)

    metrics_json_path = runs_dir / "metrics.json"
    ckpt_path = runs_dir / "best_model.keras"

    dm = ERA5DataManager()
    with open(config.OUTPUTS_DIR / "normalization_stats.json") as f:
        stats = json.load(f)
    t_mean = stats["t2m"]["mean"]
    t_std = stats["t2m"]["std"]

    # Configure stage-specific ablations and architecture parameters
    use_residual = True
    use_cyclical = False
    use_rebalance = False
    use_terrain = False
    post_aspp = "none"
    dec_fusion = "none"
    enc_stages = []
    enc_module = "none"

    if stage == "S0":
        # Extended budget only (absolute prediction, standard Broad-UNet)
        use_residual = False
        use_cyclical = False
        use_rebalance = False
    elif stage == "S1":
        # Residual prediction: Yhat = X_{t0} + Delta Y
        use_residual = True
        use_cyclical = False
        use_rebalance = False
    elif stage == "S2":
        # Residual + Cyclical time coordinates
        use_residual = True
        use_cyclical = True
        use_rebalance = False
    elif stage == "S3":
        # Residual + Cyclical + Loss rebalancing (Baseline v2 Final)
        use_residual = True
        use_cyclical = True
        use_rebalance = True
    elif stage == "S4":
        # Residual + Cyclical + Loss rebalancing + Static Terrain
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        use_terrain = True
    # Team Leader Architecture Experiments (built on winning Baseline v2 core)
    elif stage == "E-M1":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        post_aspp = "mlp"
    elif stage == "E-K1":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        post_aspp = "kan"
    elif stage == "E-MK1":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        post_aspp = "sequential"
    elif stage == "E-MK2":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        post_aspp = "parallel"
    elif stage == "E-S6":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        dec_fusion = "merge6"
    elif stage == "E-S67":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        dec_fusion = "merge67"
    elif stage == "E-S6789":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        dec_fusion = "merge6789"
    elif stage == "E-S9":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        dec_fusion = "merge9"
    elif stage == "E-C2_C4":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        enc_stages = ["conv2", "conv3", "conv4"]
        enc_module = "mlp"
    elif stage == "E-KC2_KC4":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        enc_stages = ["conv2", "conv3", "conv4"]
        enc_module = "kan"
    elif stage == "E-CTRL":
        use_residual = True
        use_cyclical = True
        use_rebalance = True
        post_aspp = "control"

    # Input channels: 6 base + 4 cyclical + 1 terrain
    num_features = 6
    if use_cyclical:
        num_features += 4
    if use_terrain:
        num_features += 1

    print(f"Stage Config: Residual={use_residual}, CyclicalTime={use_cyclical}, Rebalance={use_rebalance}, Terrain={use_terrain}")
    print(f"Architecture: PostASPP={post_aspp}, DecFusion={dec_fusion}, EncStages={enc_stages} ({enc_module})")
    print(f"Input features count: {num_features}")

    # Build generators
    batch_size = cfg["training"]["batch_size"]
    train_gen = ERA5Sequence(
        dm, cfg["data"]["train_years"], stats,
        batch_size=batch_size, dim_model="3d",
        include_cyclical_time=use_cyclical,
        include_terrain=use_terrain,
        shuffle=True, seed=seed
    )
    val_gen = ERA5Sequence(
        dm, cfg["data"]["val_years"], stats,
        batch_size=batch_size, dim_model="3d",
        include_cyclical_time=use_cyclical,
        include_terrain=use_terrain,
        shuffle=False, seed=seed
    )

    # Check if run metrics already exist and not force
    if metrics_json_path.exists() and ckpt_path.exists() and not force_retrain:
        print(f"Found existing metrics at {metrics_json_path}. Skipping training (use --force to overwrite).")
        with open(metrics_json_path, "r") as f:
            metrics = json.load(f)
        return metrics

    # Build model architecture
    model = build_broad_unet_v2(
        lags=cfg["data"]["lags"],
        latitude=cfg["data"]["padded_shape"][0],
        longitude=cfg["data"]["padded_shape"][1],
        features=num_features,
        features_output=len(cfg["data"]["horizons"]),
        filters=cfg["model"]["base_filters"],
        dropout=cfg["model"]["dropout_rate"],
        residual_head=use_residual,
        target_channel_idx=0,
        post_aspp_module=post_aspp,
        decoder_fusion=dec_fusion,
        encoder_stages=enc_stages,
        encoder_module=enc_module,
        seed=seed
    )

    # Loss function
    if use_rebalance:
        pers_weights = [1.795, 0.600, 0.255, 0.283, 2.067]
        loss_fn = RebalancedMaskedMSELoss(dm.spatial_mask, weights=pers_weights)
    else:
        loss_fn = MaskedMSELoss(dm.spatial_mask)

    mae_metric = MaskedMAE(dm.spatial_mask)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"])
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[mae_metric])
    print(f"Model constructed successfully. Total parameters: {model.count_params():,}")

    csv_log_path = runs_dir / "history.csv"
    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=cfg["training"]["patience"], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=cfg["training"]["min_learning_rate"], verbose=1),
        CSVLogger(str(csv_log_path))
    ]

    steps_per_epoch = cfg["ablations"]["steps_per_epoch"]
    val_steps = cfg["ablations"]["val_steps"]
    epochs = cfg["ablations"]["budget_epochs"]

    duration = 0.0
    best_epoch = 1
    best_val_loss = 0.0

    if ckpt_path.exists() and not force_retrain:
        print(f"Loading weights into model from existing checkpoint: {ckpt_path}...")
        try:
            model.load_weights(str(ckpt_path))
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Warning: load_weights failed ({e}). Proceeding to full training...")
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
            duration = time.time() - t0
            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            best_val_loss = float(np.min(history.history["val_loss"]))
    else:
        print(f"\nStarting training: {epochs} epochs, {steps_per_epoch} steps/epoch, {val_steps} val steps/epoch...")
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
        duration = time.time() - t0
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        best_val_loss = float(np.min(history.history["val_loss"]))
        print(f"Training finished in {duration:.1f} s ({duration/60:.2f} min).")

    # Evaluate strictly on validation split
    val_eval = evaluate_model_on_split(model, val_gen, dm, t_mean, t_std, split_name="val", max_samples=400)
    print("\nValidation Evaluation Results (Physical Units: Kelvin):")
    for h_key, h_data in val_eval["horizons"].items():
        print(f"  {h_key} (+{h_data['lead_hours']}h): RMSE = {h_data['rmse']:.3f} K | Pers = {h_data['pers_rmse']:.3f} K | Bias = {h_data['bias']:+.3f} K | Skill vs Pers = {h_data['skill_vs_pers']:+.3f}")
    print(f"  Mean Val RMSE: {val_eval['mean_rmse']:.3f} K | Mean Skill: {val_eval['mean_skill_vs_pers']:+.3f}")

    metrics = {
        "run_id": run_id,
        "stage": stage,
        "seed": seed,
        "parameters": model.count_params(),
        "training_duration_sec": duration,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "validation_metrics": val_eval
    }

    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Append to results/summary.csv
    update_summary_csv(metrics)

    return metrics


def update_summary_csv(metrics: dict):
    csv_path = Path("results/summary.csv")
    stage = metrics["stage"]
    run_id = metrics["run_id"]
    val_data = metrics["validation_metrics"]["horizons"]

    rows = []
    for h_key, h_dict in val_data.items():
        row = {
            "run_id": run_id,
            "stage": stage,
            "model": f"Broad-UNet ({stage})",
            "horizon": h_key,
            "lead_hours": h_dict["lead_hours"],
            "val_rmse": h_dict["rmse"],
            "val_mae": h_dict["mae"],
            "val_r2": h_dict["r2"],
            "val_bias": h_dict["bias"],
            "val_skill_pers": h_dict["skill_vs_pers"],
            "test_rmse": "",
            "test_mae": "",
            "test_r2": "",
            "test_bias": "",
            "test_skill_pers": "",
            "test_skill_clim": ""
        }
        rows.append(row)

    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        # Remove any previous entries with same run_id
        df_old = df_old[df_old["run_id"] != run_id]
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    print(f"Updated results/summary.csv with {len(rows)} rows for {run_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 & 3 Experiments")
    parser.add_argument("--config", type=str, default="configs/baseline_v2.yaml")
    parser.add_argument("--stage", type=str, default="S1", choices=[
        "S0", "S1", "S2", "S3", "S4",
        "E-M1", "E-K1", "E-MK1", "E-MK2",
        "E-S6", "E-S67", "E-S6789", "E-S9",
        "E-C2_C4", "E-KC2_KC4", "E-CTRL"
    ])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_experiment(cfg_path=args.config, stage=args.stage, seed=args.seed, force_retrain=args.force)
