"""
scripts/evaluate_activation_test.py
===================================
Evaluates the Activation Function Finalists ('act_leaky_relu', 'act_tanh', 'act_gelu', 'act_silu', 'act_relu')
strictly on the held-out test split (2024-2025) across all 7 forecast horizons:
[+2h, +4h, +8h, +16h, +24h, +48h, +72h].

Performs 1,000 paired bootstrap resamples comparing LeakyReLU and Tanh against standard ReLU.
Outputs:
- results/activation_test_metrics_7horizons.json
- results/activation_test_metrics_7horizons.csv
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config
from scripts.era5_data import ERA5DataManager, ERA5Sequence
from models.activation_models import build_broad_unet_with_activation

HORIZONS = [1, 2, 4, 8, 12, 24, 36]
LEAD_HOURS = [2, 4, 8, 16, 24, 48, 72]
HORIZON_MAP = dict(zip(HORIZONS, LEAD_HOURS))

def run_activation_test_evaluation(
    activations = ["leaky_relu", "tanh", "gelu", "silu", "relu"],
    ref_act = "relu",
    max_test_samples: int = 400,
    n_bootstraps: int = 1000
):
    print("=" * 80)
    print("HELD-OUT TEST EVALUATION (2024-2025) FOR ACTIVATION FUNCTION BENCHMARK")
    print(f"Activations Tested: {activations} (Reference: {ref_act})")
    print(f"Lead Times Evaluated: {LEAD_HOURS} hours")
    print("=" * 80)

    dm = ERA5DataManager()
    with open(config.OUTPUTS_DIR / "normalization_stats.json") as f:
        stats = json.load(f)
    t_mean = stats["t2m"]["mean"]
    t_std = stats["t2m"]["std"]

    # Build Test Generator with cyclical time features
    test_gen = ERA5Sequence(
        dm, config.TEST_YEARS, stats,
        batch_size=4, dim_model="3d",
        include_cyclical_time=True,
        include_terrain=False,
        shuffle=False
    )
    test_gen.horizons = HORIZONS
    test_gen.max_lead = max(HORIZONS)
    test_gen.required_window = test_gen.lags + test_gen.max_lead
    test_gen.windows = []
    test_gen._build_windows()
    test_gen.indices = np.arange(len(test_gen.windows))

    batch_size = 4
    batches_to_run = min(len(test_gen), max(1, max_test_samples // batch_size))
    total_samples = batches_to_run * batch_size

    # Load all models
    models = {}
    for act in activations:
        ckpt_p = Path(f"activation_experiments/act_{act}/checkpoint/best_model.keras")
        if not ckpt_p.exists():
            print(f"Warning: Checkpoint not found for {act}: {ckpt_p}")
            continue
        print(f"Loading Model for Activation: {act}...")
        m = build_broad_unet_with_activation(
            activation=act, lags=6, latitude=32, longitude=48, features=10, features_output=7,
            filters=16, dropout=0.2, residual_head=True
        )
        m.load_weights(str(ckpt_p))
        models[act] = m

    preds = {act: {h: [] for h in HORIZONS} for act in models}
    targets = {h: [] for h in HORIZONS}
    pers_t0 = {h: [] for h in HORIZONS}
    climatology = {h: [] for h in HORIZONS}
    clim_dict = dm.compute_climatology(config.TRAIN_YEARS)

    print(f"\nRunning test inferences across {total_samples} samples ({batches_to_run} batches)...")
    for b in range(batches_to_run):
        xb, _ = test_gen[b]
        
        # Predict with each model
        preds_k = {}
        for act, m in models.items():
            p_norm = m(xb, training=False).numpy()
            preds_k[act] = dm.unpad_spatial(p_norm * t_std + t_mean)[:, 0]

        for i in range(len(xb)):
            g_idx = b * batch_size + i
            y, start_i, in_end_t, tgt_times = test_gen.windows[g_idx]
            year_arr = dm.get_year_data(y)

            t0 = year_arr[start_i + test_gen.lags - 1, :, :, config.TARGET_INDEX]

            for h_idx, h in enumerate(HORIZONS):
                tgt_i = start_i + test_gen.lags + h - 1
                gt = year_arr[tgt_i, :, :, config.TARGET_INDEX]
                tgt_dt = pd.to_datetime(tgt_times[h_idx])
                clim_val = clim_dict.get((tgt_dt.month, tgt_dt.day, tgt_dt.hour), t0)

                for act in models:
                    preds[act][h].append(preds_k[act][i, :, :, h_idx])
                targets[h].append(gt)
                pers_t0[h].append(t0)
                climatology[h].append(clim_val)

    # Compute metrics & Paired Bootstrap CIs vs ReLU
    results_dict = {}
    csv_rows = []

    for h_idx, h in enumerate(HORIZONS):
        yt = np.stack(targets[h], axis=0)
        pt0 = np.stack(pers_t0[h], axis=0)
        pclim = np.stack(climatology[h], axis=0)

        err_pers = pt0 - yt
        rmse_pers = float(np.sqrt(np.mean(err_pers ** 2)))
        rmse_clim = float(np.sqrt(np.mean((pclim - yt) ** 2)))
        seq_mse_pers = np.mean(err_pers ** 2, axis=(1, 2))

        ref_err = np.stack(preds[ref_act][h], axis=0) - yt
        ref_rmse = float(np.sqrt(np.mean(ref_err ** 2)))
        ref_skill = float(1.0 - (ref_rmse / rmse_pers))
        seq_mse_ref = np.mean(ref_err ** 2, axis=(1, 2))

        def calc_skill(mse_m, mse_p):
            return 1.0 - np.sqrt(np.mean(mse_m)) / np.sqrt(np.mean(mse_p))

        rng = np.random.default_rng(42)

        for act in models:
            err = np.stack(preds[act][h], axis=0) - yt
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            bias = float(np.mean(err))
            var_yt = float(np.var(yt))
            r2 = float(1.0 - np.mean(err ** 2) / var_yt)
            skill = float(1.0 - (rmse / rmse_pers))
            diff_vs_ref = ref_rmse - rmse

            # Bootstrap skill difference vs reference ReLU
            seq_mse_act = np.mean(err ** 2, axis=(1, 2))
            boot_diff = []
            for _ in range(n_bootstraps):
                idx = rng.choice(len(seq_mse_act), size=len(seq_mse_act), replace=True)
                sa = calc_skill(seq_mse_act[idx], seq_mse_pers[idx])
                sref = calc_skill(seq_mse_ref[idx], seq_mse_pers[idx])
                boot_diff.append(sa - sref)

            ci_low = float(np.percentile(boot_diff, 2.5))
            ci_high = float(np.percentile(boot_diff, 97.5))

            row = {
                "horizon": f"H{h}",
                "lead_hours": LEAD_HOURS[h_idx],
                "activation": act,
                "rmse_K": rmse,
                "mae_K": mae,
                "bias_K": bias,
                "r2": r2,
                "skill_vs_pers": skill,
                "diff_vs_relu_K": diff_vs_ref,
                "pct_gain_vs_relu": (diff_vs_ref / ref_rmse) * 100.0 if ref_rmse > 0 else 0.0,
                "persistence_rmse_K": rmse_pers,
                "climatology_rmse_K": rmse_clim,
                "skill_diff_vs_relu": skill - ref_skill,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high
            }
            csv_rows.append(row)

    df_res = pd.DataFrame(csv_rows)
    out_csv = Path("results/activation_test_metrics_7horizons.csv")
    out_json = Path("results/activation_test_metrics_7horizons.json")
    df_res.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(csv_rows, f, indent=2)

    print("\n" + "=" * 80)
    print("ACTIVATION TEST EVALUATION COMPLETE")
    print(f"Results written to {out_csv} and {out_json}")
    print("=" * 80)

    # Print summary table of mean test RMSE
    summary = df_res.groupby("activation").agg(
        Mean_Test_RMSE_K=("rmse_K", "mean"),
        Mean_Test_MAE_K=("mae_K", "mean"),
        Mean_Test_Skill=("skill_vs_pers", "mean"),
        Mean_Gain_vs_ReLU=("pct_gain_vs_relu", "mean")
    ).reset_index().sort_values("Mean_Test_RMSE_K")
    print("\nOVERALL TEST SUMMARY ACROSS ALL 7 HORIZONS:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_activation_test_evaluation()
