"""
scripts/evaluate.py
===================
Independent test evaluation on the held-out ERA5 partition (2024-2025).
Evaluates all 5 forecast horizons (1, 2, 4, 8, 12 steps -> 2, 4, 8, 16, 24 hours).
Computes physical metrics strictly in Kelvin for Broad-UNet, Persistence, and Climatology.
"""

import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config
from scripts.era5_data import ERA5DataManager, ERA5Sequence
from models.broad_unet import MaskedMSELoss, MaskedMAE

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes MSE (K^2), RMSE (K), MAE (K), R^2, and Mean Bias (K)
    over 2D physical spatial grids (28 x 34).
    """
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    
    # R^2 calculation
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (sse / sst)) if sst > 1e-6 else 0.0
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias
    }

def run_evaluation(
    checkpoint_path: Path = None,
    samples_to_eval: int = None,
    batch_size: int = config.BATCH_SIZE
):
    print("\n" + "=" * 70)
    print("STARTING TEST EVALUATION ON UNTOUCHED PARTITION (2024 - 2025)")
    print("=" * 70)

    dm = ERA5DataManager()
    with open(config.OUTPUTS_DIR / "normalization_stats.json") as f:
        stats = json.load(f)

    # Climatology baseline dictionary
    clim_map = dm.compute_climatology(config.TRAIN_YEARS)

    # Auto-detect checkpoint if not provided
    if checkpoint_path is None:
        ckpt_candidate = config.OUTPUTS_DIR / "multi_horizon" / "best_broad_unet_all.keras"
        if ckpt_candidate.exists():
            checkpoint_path = ckpt_candidate
        else:
            raise FileNotFoundError("No trained checkpoint found! Specify --checkpoint or train model first.")
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
            checkpoint_path = (config.PROJECT_ROOT / checkpoint_path).resolve()
            
    print(f"Loading model checkpoint from: {checkpoint_path}")
    custom_objects = {
        "MaskedMSELoss": MaskedMSELoss(dm.spatial_mask),
        "MaskedMAE": MaskedMAE(dm.spatial_mask),
        "masked_mse": MaskedMSELoss(dm.spatial_mask),
        "masked_mae": MaskedMAE(dm.spatial_mask)
    }
    model = tf.keras.models.load_model(str(checkpoint_path), custom_objects=custom_objects, compile=False)

    # Build test generator
    test_gen = ERA5Sequence(
        dm, config.TEST_YEARS, stats,
        horizon=None, batch_size=batch_size,
        dim_model="3d", shuffle=False
    )

    total_test_batches = len(test_gen)
    batches_to_run = min(total_test_batches, samples_to_eval // batch_size) if samples_to_eval else total_test_batches
    print(f"Total test windows: {len(test_gen.windows):,} | Evaluating on {batches_to_run} batches ({batches_to_run * batch_size} samples)...")

    # Storage for predictions per horizon
    horizons = config.HORIZONS  # [1, 2, 4, 8, 12]
    all_y_true = {h: [] for h in horizons}
    all_y_broad = {h: [] for h in horizons}
    all_y_pers = {h: [] for h in horizons}
    all_y_clim = {h: [] for h in horizons}
    
    t_mean = stats["t2m"]["mean"]
    t_std = stats["t2m"]["std"]
    t_min = stats["t2m"]["min"]
    t_range = stats["t2m"]["range"]

    sample_metadata = []

    for b in range(batches_to_run):
        xb, yb = test_gen[b]
        
        # Model inference: output shape (batch, 1, 32, 48, 5)
        pred_norm = model.predict(xb, verbose=0)
        
        # Denormalize model predictions to Kelvin and unpad to (28, 34)
        if config.NORM_METHOD == "zscore":
            pred_kelvin = pred_norm * t_std + t_mean
        else:
            pred_kelvin = pred_norm * t_range + t_min
            
        pred_unpadded = dm.unpad_spatial(pred_kelvin)[:, 0]  # shape: (batch, 28, 34, 5)

        for i in range(len(xb)):
            global_sample_idx = b * batch_size + i
            x_raw, y_raw_dict, pers_t2m, end_t, target_t_dict = test_gen.get_sample_raw(global_sample_idx)
            
            if b == 0 and i < 5:
                sample_metadata.append({
                    "sample_idx": global_sample_idx,
                    "input_end_time": str(end_t),
                    "target_times": {h: str(target_t_dict[h]) for h in horizons}
                })

            for h_idx, h in enumerate(horizons):
                gt = y_raw_dict[h]  # (28, 34) in Kelvin
                pred_h = pred_unpadded[i, :, :, h_idx]  # (28, 34) in Kelvin
                pers = pers_t2m  # (28, 34) in Kelvin
                
                # Climatology lookup
                tgt_t = target_t_dict[h]
                clim_key = (tgt_t.month, tgt_t.day, tgt_t.hour)
                clim = clim_map.get(clim_key, np.full((28, 34), t_mean, dtype=np.float32))

                all_y_true[h].append(gt)
                all_y_broad[h].append(pred_h)
                all_y_pers[h].append(pers)
                all_y_clim[h].append(clim)

        if (b + 1) % 50 == 0 or b == batches_to_run - 1:
            print(f"Evaluated {b + 1} / {batches_to_run} batches...")

    # Calculate metrics for each horizon
    summary_rows = []
    full_metrics = {}

    print("\n" + "=" * 90)
    print("DETAILED QUANTITATIVE TEST RESULTS (ORIGINAL PHYSICAL UNITS: KELVIN)")
    print("=" * 90)
    header = f"{'Horizon':<10} | {'Lead Time':<12} | {'Model':<24} | {'RMSE (K)':<10} | {'MAE (K)':<10} | {'MSE (K^2)':<10} | {'R^2':<8} | {'Bias (K)':<8}"
    print(header)
    print("-" * 90)

    for h in horizons:
        lead_str = f"{config.HORIZON_TO_HOURS[h]} Hours"
        h_str = f"H={h}"
        
        y_true_arr = np.stack(all_y_true[h], axis=0)
        y_broad_arr = np.stack(all_y_broad[h], axis=0)
        y_pers_arr = np.stack(all_y_pers[h], axis=0)
        y_clim_arr = np.stack(all_y_clim[h], axis=0)

        m_broad = compute_metrics(y_true_arr, y_broad_arr)
        m_pers = compute_metrics(y_true_arr, y_pers_arr)
        m_clim = compute_metrics(y_true_arr, y_clim_arr)

        full_metrics[f"horizon_{h}"] = {
            "lead_hours": config.HORIZON_TO_HOURS[h],
            "broad_unet": m_broad,
            "persistence": m_pers,
            "climatology": m_clim
        }

        # Add to summary table
        summary_rows.append({"Horizon": h_str, "Lead Time": lead_str, "Model": "Broad-UNet", **m_broad})
        summary_rows.append({"Horizon": h_str, "Lead Time": lead_str, "Model": "Persistence Baseline", **m_pers})
        summary_rows.append({"Horizon": h_str, "Lead Time": lead_str, "Model": "Climatology Baseline", **m_clim})

        print(f"{h_str:<10} | {lead_str:<12} | {'Broad-UNet':<24} | {m_broad['rmse']:<10.4f} | {m_broad['mae']:<10.4f} | {m_broad['mse']:<10.4f} | {m_broad['r2']:<8.4f} | {m_broad['bias']:<8.4f}")
        print(f"{'':<10} | {'':<12} | {'Persistence Baseline':<24} | {m_pers['rmse']:<10.4f} | {m_pers['mae']:<10.4f} | {m_pers['mse']:<10.4f} | {m_pers['r2']:<8.4f} | {m_pers['bias']:<8.4f}")
        print(f"{'':<10} | {'':<12} | {'Climatology Baseline':<24} | {m_clim['rmse']:<10.4f} | {m_clim['mae']:<10.4f} | {m_clim['mse']:<10.4f} | {m_clim['r2']:<8.4f} | {m_clim['bias']:<8.4f}")
        print("-" * 90)

    # Save summary CSV & JSON
    df_metrics = pd.DataFrame(summary_rows)
    csv_path = config.REPORTS_DIR / "metrics_summary.csv"
    json_path = config.REPORTS_DIR / "metrics_summary.json"
    df_metrics.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"\nMetrics summary CSV saved to {csv_path}")
    print(f"Metrics summary JSON saved to {json_path}")

    # Save sample test prediction numpy arrays for visualization
    preds_dir = config.OUTPUTS_DIR / "test_predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preds_dir / "sample_eval_arrays.npz",
        y_true=np.stack([np.stack(all_y_true[h][:50]) for h in horizons]),
        y_broad=np.stack([np.stack(all_y_broad[h][:50]) for h in horizons]),
        y_pers=np.stack([np.stack(all_y_pers[h][:50]) for h in horizons]),
        horizons=horizons,
        lead_hours=[config.HORIZON_TO_HOURS[h] for h in horizons]
    )
    print(f"Evaluation arrays saved to {preds_dir / 'sample_eval_arrays.npz'}")

    return full_metrics, df_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Broad-UNet Multi-Horizon on ERA5 Test Partition")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_evaluation(checkpoint_path=ckpt, samples_to_eval=args.samples, batch_size=args.batch_size)
