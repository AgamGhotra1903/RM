"""
scripts/train.py
================
Full training pipeline for Broad-UNet on ERA5 temperature nowcasting.
Supports both unified multi-horizon training (all 5 horizons) and horizon-specific models.
Utilizes Masked MSE loss to ignore padded boundary cells during backpropagation.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config
from scripts.era5_data import ERA5DataManager, ERA5Sequence
from models.broad_unet import build_broad_unet, build_unet_baseline, MaskedMSELoss, MaskedMAE

def set_seeds(seed: int = config.RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def plot_loss_curves(history, save_path: Path, title: str):
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(history.history["loss"], label="Train Loss (Masked MSE)", color="#1d3557", lw=2.2)
    plt.plot(history.history["val_loss"], label="Val Loss (Masked MSE)", color="#e63946", lw=2.2, linestyle="--")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Normalized Loss", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def train_model(
    horizon: str = "all",  # "all" or specific integer: "1", "2", "4", "8", "12"
    model_type: str = "broad_unet",
    epochs: int = config.EPOCHS,
    steps_per_epoch: int = None,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE
):
    set_seeds()
    print("\n" + "=" * 70)
    print(f"STARTING TRAINING PIPELINE: {model_type.upper()} | HORIZON: {horizon}")
    print("=" * 70)

    dm = ERA5DataManager()
    stats = dm.compute_normalization_stats(config.TRAIN_YEARS)

    h_int = None if horizon == "all" else int(horizon)
    features_output = len(config.HORIZONS) if h_int is None else 1
    
    # Destination directory
    if h_int is None:
        target_dir = config.OUTPUTS_DIR / "multi_horizon"
    else:
        target_dir = config.OUTPUTS_DIR / f"horizon_{h_int}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generators
    train_gen = ERA5Sequence(
        dm, config.TRAIN_YEARS, stats,
        horizon=h_int, batch_size=batch_size,
        dim_model="3d" if model_type == "broad_unet" else "2d",
        shuffle=True
    )
    val_gen = ERA5Sequence(
        dm, config.VAL_YEARS, stats,
        horizon=h_int, batch_size=batch_size,
        dim_model="3d" if model_type == "broad_unet" else "2d",
        shuffle=False
    )

    total_train_steps = len(train_gen)
    total_val_steps = len(val_gen)
    actual_steps = steps_per_epoch if steps_per_epoch else total_train_steps
    val_steps = max(10, int(actual_steps * 0.25)) if steps_per_epoch else total_val_steps
    
    print(f"Training windows:   {len(train_gen.windows):,} ({total_train_steps} batches, running {actual_steps}/epoch)")
    print(f"Validation windows: {len(val_gen.windows):,} ({total_val_steps} batches, evaluating {val_steps}/epoch)")

    # Construct model
    if model_type == "broad_unet":
        model = build_broad_unet(
            lags=config.LAGS,
            latitude=config.PADDED_LAT,
            longitude=config.PADDED_LON,
            features=config.NUM_INPUT_FEATURES,
            features_output=features_output,
            filters=config.BASE_FILTERS,
            dropout=config.DROPOUT_RATE,
            activation=config.ACTIVATION
        )
    else:
        model = build_unet_baseline(
            lags=config.LAGS,
            latitude=config.PADDED_LAT,
            longitude=config.PADDED_LON,
            features=config.NUM_INPUT_FEATURES,
            features_output=features_output,
            filters=config.BASE_FILTERS,
            dropout=config.DROPOUT_RATE,
            activation=config.ACTIVATION
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = MaskedMSELoss(dm.spatial_mask)
    mae_metric = MaskedMAE(dm.spatial_mask)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[mae_metric])
    
    print(f"Model compiled. Total parameters: {model.count_params():,}")

    ckpt_name = f"best_{model_type}_{horizon}.keras"
    ckpt_path = target_dir / ckpt_name
    csv_path = target_dir / f"training_history_{horizon}.csv"
    plot_path = target_dir / f"loss_curve_{horizon}.png"

    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        CSVLogger(str(csv_path))
    ]

    start_time = time.time()
    history = model.fit(
        train_gen,
        steps_per_epoch=actual_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.2f}s ({duration/60:.2f} min). Best model: {ckpt_path}")

    # Plot loss curve
    title = f"{model_type.upper()} ({'All Horizons' if h_int is None else f'Lead H={h_int} ({config.HORIZON_TO_HOURS[h_int]}h)'})"
    plot_loss_curves(history, plot_path, title)
    
    # Save training summary
    summary = {
        "model_type": model_type,
        "horizon": horizon,
        "parameters": model.count_params(),
        "epochs_completed": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "training_duration_sec": duration,
        "checkpoint_path": str(ckpt_path)
    }
    with open(target_dir / f"training_summary_{horizon}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Broad-UNet on ERA5 Weather Data")
    parser.add_argument("--horizon", type=str, default="all", choices=["all", "1", "2", "4", "8", "12"])
    parser.add_argument("--model", type=str, default="broad_unet", choices=["broad_unet", "unet_original"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    train_model(
        horizon=args.horizon,
        model_type=args.model,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

