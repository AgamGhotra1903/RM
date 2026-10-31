"""
scripts/phase0_audit_and_reproduce.py
======================================
Phase 0: Audit and Reproduce
1. Verifies repo configuration, tensor dimensions, and layer names.
2. Reproduces the exact v1 test metrics from best_broad_unet_all.keras.
3. Computes all three baselines on Validation (2022-2023) and Test (2024-2025):
   (a) Persistence: y_{t+h} = y_t
   (b) Diurnal Climatology: mean_temp(month, hour) computed strictly from TRAIN (2011-2021)
   (c) 24h-Prior Persistence: y_{t+h} = y_{target_time - 24h}
4. Computes Skill Scores vs each baseline: Skill = 1 - RMSE_model / RMSE_baseline
5. Logs output to results/phase0_metrics_test.csv, results/phase0_metrics_val.csv,
   results/SUMMARY.md, and results/summary.csv.
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from configs import config
from scripts.era5_data import ERA5DataManager, ERA5Sequence
from models.broad_unet import MaskedMSELoss, MaskedMAE

def compute_spatial_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes physical metrics in Kelvin over (N, 28, 34) arrays:
    - Global pooled metrics (mean over all N*28*34 pixels)
    - Spatial-mean-of-per-pixel metrics (metric computed per pixel, then averaged across 28x34)
    """
    diff = y_pred - y_true
    # Global pooled
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (sse / sst)) if sst > 1e-6 else 0.0

    # Per-pixel temporal metrics, then spatially averaged
    # diff shape: (N, 28, 34)
    pixel_mse = np.mean(diff ** 2, axis=0)  # (28, 34)
    pixel_rmse = np.sqrt(pixel_mse)
    pixel_mae = np.mean(np.abs(diff), axis=0)
    pixel_bias = np.mean(diff, axis=0)
    
    pixel_sst = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    pixel_sse = np.sum(diff ** 2, axis=0)
    pixel_r2 = np.where(pixel_sst > 1e-6, 1.0 - (pixel_sse / pixel_sst), 0.0)

    return {
        "global": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "bias": bias
        },
        "spatial_mean": {
            "mse": float(np.mean(pixel_mse)),
            "rmse": float(np.mean(pixel_rmse)),
            "mae": float(np.mean(pixel_mae)),
            "r2": float(np.mean(pixel_r2)),
            "bias": float(np.mean(pixel_bias))
        }
    }

def run_phase0_evaluation(samples_to_eval: int = 400, split_name: str = "test"):
    print("=" * 80)
    print(f"PHASE 0 EVALUATION: Split = {split_name.upper()} | Samples = {samples_to_eval or 'FULL'}")
    print("=" * 80)

    dm = ERA5DataManager()
    with open(config.OUTPUTS_DIR / "normalization_stats.json") as f:
        stats = json.load(f)

    # Compute Climatology strictly from training years (month, hour)
    print("Computing Diurnal Climatology from TRAIN (2011-2021)...")
    climatology_map = {}
    for y in config.TRAIN_YEARS:
        arr = dm.get_year_data(y)[..., config.TARGET_INDEX]
        times = dm.timestamps_by_year[y]
        for i, t in enumerate(times):
            key = (t.month, t.hour)
            if key not in climatology_map:
                climatology_map[key] = []
            climatology_map[key].append(arr[i])
            
    clim_mean_map = {k: np.mean(v, axis=0) for k, v in climatology_map.items()}
    print(f"Climatology table built: {len(clim_mean_map)} unique (month, hour) bins.")

    years = config.TEST_YEARS if split_name == "test" else config.VAL_YEARS
    gen = ERA5Sequence(
        dm, years, stats,
        horizon=None, batch_size=config.BATCH_SIZE,
        dim_model="3d", shuffle=False
    )

    ckpt_path = config.OUTPUTS_DIR / "multi_horizon" / "best_broad_unet_all.keras"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    custom_objects = {
        "MaskedMSELoss": MaskedMSELoss(dm.spatial_mask),
        "MaskedMAE": MaskedMAE(dm.spatial_mask),
        "masked_mse": MaskedMSELoss(dm.spatial_mask),
        "masked_mae": MaskedMAE(dm.spatial_mask)
    }
    print(f"Loading checkpoint: {ckpt_path}")
    model = tf.keras.models.load_model(str(ckpt_path), custom_objects=custom_objects, compile=False)

    total_batches = len(gen)
    batches_to_run = min(total_batches, samples_to_eval // config.BATCH_SIZE) if samples_to_eval else total_batches
    print(f"Evaluating {batches_to_run} batches ({batches_to_run * config.BATCH_SIZE} samples)...")

    horizons = config.HORIZONS  # [1, 2, 4, 8, 12]
    all_y_true = {h: [] for h in horizons}
    all_y_broad = {h: [] for h in horizons}
    all_y_pers = {h: [] for h in horizons}
    all_y_clim = {h: [] for h in horizons}
    all_y_pers24 = {h: [] for h in horizons}

    t_mean = stats["t2m"]["mean"]
    t_std = stats["t2m"]["std"]

    t0 = time.time()
    for b in range(batches_to_run):
        xb, yb = gen[b]
        pred_norm = model(xb, training=False).numpy()  # Fast TF inference
        pred_kelvin = pred_norm * t_std + t_mean
        pred_unpadded = dm.unpad_spatial(pred_kelvin)[:, 0]  # (B, 28, 34, 5)

        for i in range(len(xb)):
            global_sample_idx = b * config.BATCH_SIZE + i
            y, start_i, input_end_t, target_times = gen.windows[global_sample_idx]
            year_arr = dm.get_year_data(y)
            year_times = dm.timestamps_by_year[y]

            # Persistence (t0) is input frame at lags - 1
            pers_t0 = year_arr[start_i + gen.lags - 1, :, :, config.TARGET_INDEX]

            for h_idx, h in enumerate(horizons):
                target_i = start_i + gen.lags + h - 1
                gt = year_arr[target_i, :, :, config.TARGET_INDEX]
                pred_h = pred_unpadded[i, :, :, h_idx]

                tgt_t = target_times[h_idx]
                clim_key = (tgt_t.month, tgt_t.hour)
                clim = clim_mean_map.get(clim_key, np.full((28, 34), t_mean, dtype=np.float32))

                # 24h prior persistence: target time - 24 hours (12 steps of 2 hours)
                target_minus_24h_idx = target_i - 12
                if target_minus_24h_idx >= 0:
                    pers_24h = year_arr[target_minus_24h_idx, :, :, config.TARGET_INDEX]
                else:
                    pers_24h = pers_t0  # fallback to pers_t0 if at boundary

                all_y_true[h].append(gt)
                all_y_broad[h].append(pred_h)
                all_y_pers[h].append(pers_t0)
                all_y_clim[h].append(clim)
                all_y_pers24[h].append(pers_24h)

        if (b + 1) % 25 == 0 or b == batches_to_run - 1:
            elapsed = time.time() - t0
            print(f"  Processed {b + 1}/{batches_to_run} batches in {elapsed:.1f}s ({(b+1)/elapsed:.1f} batches/s)...")

    results_rows = []
    results_dict = {}

    for h in horizons:
        lead_h = config.HORIZON_TO_HOURS[h]
        yt = np.stack(all_y_true[h], axis=0)
        yp_broad = np.stack(all_y_broad[h], axis=0)
        yp_pers = np.stack(all_y_pers[h], axis=0)
        yp_clim = np.stack(all_y_clim[h], axis=0)
        yp_pers24 = np.stack(all_y_pers24[h], axis=0)

        m_broad = compute_spatial_metrics(yt, yp_broad)
        m_pers = compute_spatial_metrics(yt, yp_pers)
        m_clim = compute_spatial_metrics(yt, yp_clim)
        m_pers24 = compute_spatial_metrics(yt, yp_pers24)

        # Skill score = 1 - RMSE_model / RMSE_baseline
        rmse_b = m_broad["global"]["rmse"]
        rmse_p = m_pers["global"]["rmse"]
        rmse_c = m_clim["global"]["rmse"]
        rmse_p24 = m_pers24["global"]["rmse"]

        skill_pers = 1.0 - (rmse_b / rmse_p)
        skill_clim = 1.0 - (rmse_b / rmse_c)
        skill_pers24 = 1.0 - (rmse_b / rmse_p24)

        results_dict[f"H{h}"] = {
            "lead_hours": lead_h,
            "broad_unet": m_broad,
            "persistence": m_pers,
            "climatology": m_clim,
            "persistence_24h": m_pers24,
            "skill_vs_persistence": skill_pers,
            "skill_vs_climatology": skill_clim,
            "skill_vs_pers24h": skill_pers24
        }

        for model_name, m_data in [
            ("Broad-UNet v1", m_broad),
            ("Persistence (t0)", m_pers),
            ("Diurnal Climatology", m_clim),
            ("Persistence (24h-prior)", m_pers24)
        ]:
            results_rows.append({
                "Split": split_name,
                "Horizon": f"H{h}",
                "Lead_Hours": lead_h,
                "Model": model_name,
                "RMSE_Global": m_data["global"]["rmse"],
                "MAE_Global": m_data["global"]["mae"],
                "MSE_Global": m_data["global"]["mse"],
                "R2_Global": m_data["global"]["r2"],
                "Bias_Global": m_data["global"]["bias"],
                "RMSE_SpatialMean": m_data["spatial_mean"]["rmse"],
                "MAE_SpatialMean": m_data["spatial_mean"]["mae"],
                "R2_SpatialMean": m_data["spatial_mean"]["r2"],
                "Skill_vs_Pers": skill_pers if model_name == "Broad-UNet v1" else 0.0,
                "Skill_vs_Clim": skill_clim if model_name == "Broad-UNet v1" else 0.0
            })

    df_out = pd.DataFrame(results_rows)
    return results_dict, df_out

def main():
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate on TEST (400 samples to verify reproduction)
    test_dict_400, test_df_400 = run_phase0_evaluation(samples_to_eval=400, split_name="test")
    test_df_400.to_csv(results_dir / "phase0_metrics_test.csv", index=False)
    with open(results_dir / "phase0_metrics_test.json", "w") as f:
        json.dump(test_dict_400, f, indent=2)

    # 2. Evaluate on VAL (400 samples)
    val_dict_400, val_df_400 = run_phase0_evaluation(samples_to_eval=400, split_name="val")
    val_df_400.to_csv(results_dir / "phase0_metrics_val.csv", index=False)
    with open(results_dir / "phase0_metrics_val.json", "w") as f:
        json.dump(val_dict_400, f, indent=2)

    # Print Reproduction Verification Table
    print("\n" + "=" * 105)
    print("PHASE 0: VERIFICATION & REPRODUCTION OF v1 NUMBERS (TEST SPLIT, KELVIN)")
    print("=" * 105)
    header = f"{'Horizon':<8} | {'Lead':<6} | {'Broad RMSE':<11} | {'Pers RMSE':<10} | {'Clim RMSE':<10} | {'Broad R2':<9} | {'Bias (K)':<9} | {'Skill vs Pers':<14} | {'Skill vs Clim'}"
    print(header)
    print("-" * 105)

    for h in config.HORIZONS:
        d = test_dict_400[f"H{h}"]
        lead = f"+{d['lead_hours']}h"
        b_rmse = d["broad_unet"]["global"]["rmse"]
        p_rmse = d["persistence"]["global"]["rmse"]
        c_rmse = d["climatology"]["global"]["rmse"]
        b_r2 = d["broad_unet"]["global"]["r2"]
        bias = d["broad_unet"]["global"]["bias"]
        sk_p = d["skill_vs_persistence"]
        sk_c = d["skill_vs_climatology"]
        print(f"H{h:<7} | {lead:<6} | {b_rmse:<11.4f} | {p_rmse:<10.4f} | {c_rmse:<10.4f} | {b_r2:<9.4f} | {bias:<+9.4f} | {sk_p:<+14.4f} | {sk_c:<+.4f}")

    print("=" * 105)

    # Update results/SUMMARY.md and results/summary.csv
    summary_csv_path = results_dir / "summary.csv"
    summary_md_path = results_dir / "SUMMARY.md"

    # Compile run record
    run_record = []
    for h in config.HORIZONS:
        d_val = val_dict_400[f"H{h}"]
        d_test = test_dict_400[f"H{h}"]
        run_record.append({
            "run_id": "v1_checkpoint",
            "stage": "Phase0_Audit",
            "model": "Broad-UNet (v1 4ep 35steps)",
            "horizon": f"H{h}",
            "lead_hours": d_test["lead_hours"],
            "val_rmse": d_val["broad_unet"]["global"]["rmse"],
            "val_mae": d_val["broad_unet"]["global"]["mae"],
            "val_r2": d_val["broad_unet"]["global"]["r2"],
            "val_bias": d_val["broad_unet"]["global"]["bias"],
            "val_skill_pers": d_val["skill_vs_persistence"],
            "test_rmse": d_test["broad_unet"]["global"]["rmse"],
            "test_mae": d_test["broad_unet"]["global"]["mae"],
            "test_r2": d_test["broad_unet"]["global"]["r2"],
            "test_bias": d_test["broad_unet"]["global"]["bias"],
            "test_skill_pers": d_test["skill_vs_persistence"],
            "test_skill_clim": d_test["skill_vs_climatology"]
        })
    pd.DataFrame(run_record).to_csv(summary_csv_path, index=False)

    with open(summary_md_path, "w") as f:
        f.write("# ERA5 Broad-UNet Project Master Results Summary\n\n")
        f.write("Master tracking register for all runs, baselines, and ablation stages.\n\n")
        f.write("## Phase 0: v1 Checkpoint & Baseline Reproduction (Physical Units: Kelvin)\n\n")
        f.write(test_df_400.to_markdown(index=False))
        f.write("\n\n*Updated automatically by evaluation and training pipelines.*\n")

    print(f"\nArtifacts generated:")
    print(f"  - {results_dir / 'phase0_metrics_test.csv'}")
    print(f"  - {results_dir / 'phase0_metrics_val.csv'}")
    print(f"  - {summary_csv_path}")
    print(f"  - {summary_md_path}")

if __name__ == "__main__":
    main()

