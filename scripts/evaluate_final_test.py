"""
scripts/evaluate_final_test.py
==============================
Phase 3 Stage D: Final Held-Out Test Split Evaluation (2024-2025).
Evaluates the winning rearranged architecture against Original Broad-UNet (L0_original)
and persistence/climatology baselines strictly once on the held-out test split across
all 7 forecast horizons: [+2h, +4h, +8h, +16h, +24h, +48h, +72h].

Performs paired bootstrap resampling (1,000 resamples) to compute empirical 95%
confidence intervals for skill score differences.

Outputs:
- results/test_metrics_7horizons_comparison.json
- results/test_metrics_7horizons_comparison.csv
- layer_rearrangement_experiments/{winner_exp}/predictions/test_predictions_7horizons.npz
- Appends final test metrics to results/summary.csv
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
from models.rearranged_models import build_rearranged_model, EXPERIMENT_NAMES

HORIZONS = [1, 2, 4, 8, 12, 24, 36]
LEAD_HOURS = [2, 4, 8, 16, 24, 48, 72]
HORIZON_MAP = dict(zip(HORIZONS, LEAD_HOURS))

ALL_EXP_NAMES = dict(EXPERIMENT_NAMES)
ALL_EXP_NAMES["L15_hybrid_kan_leaky_relu"] = "Hybrid KAN Skip Fusion + LeakyReLU Backbone"

def get_model(exp_id: str):
    if exp_id == "L15_hybrid_kan_leaky_relu":
        from models.hybrid_model import build_hybrid_kan_leaky_relu
        return build_hybrid_kan_leaky_relu(
            lags=6, latitude=32, longitude=48, features=10, features_output=7,
            filters=16, dropout=0.2, residual_head=True
        )
    elif exp_id.startswith("act_"):
        from models.activation_models import build_broad_unet_with_activation
        act = exp_id.replace("act_", "")
        return build_broad_unet_with_activation(
            activation=act, lags=6, latitude=32, longitude=48, features=10, features_output=7,
            filters=16, dropout=0.2, residual_head=True
        )
    else:
        return build_rearranged_model(
            exp_id=exp_id, lags=6, latitude=32, longitude=48, features=10, features_output=7,
            filters=16, dropout=0.2, residual_head=True
        )

def run_test_evaluation(
    winner_exp: str = "L15_hybrid_kan_leaky_relu",
    baseline_exp: str = "L0_original",
    max_test_samples: int = 400,
    n_bootstraps: int = 1000
):
    print("=" * 80)
    print("FINAL HELD-OUT TEST EVALUATION (2024-2025) ACROSS 7 FORECAST HORIZONS")
    print(f"Target Winner Architecture: {winner_exp} ({ALL_EXP_NAMES.get(winner_exp, '')})")
    print(f"Reference Architecture    : {baseline_exp} ({ALL_EXP_NAMES.get(baseline_exp, '')})")
    print(f"Lead Times Evaluated      : {LEAD_HOURS} hours")
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

    # 1. Load Winner Model
    print(f"\nLoading Winner Model Checkpoint: {winner_exp}...")
    if winner_exp.startswith("act_"):
        winner_ckpt = Path(f"activation_experiments/{winner_exp}/checkpoint/best_model.keras")
    else:
        winner_ckpt = Path(f"layer_rearrangement_experiments/{winner_exp}/checkpoint/best_model.keras")
    
    if not winner_ckpt.exists():
        raise FileNotFoundError(f"Winner checkpoint not found: {winner_ckpt}")

    winner_model = get_model(winner_exp)
    winner_model.load_weights(str(winner_ckpt))

    # 2. Load Reference Model (L0_original)
    print(f"Loading Reference Model Checkpoint: {baseline_exp}...")
    if baseline_exp.startswith("act_"):
        ref_ckpt = Path(f"activation_experiments/{baseline_exp}/checkpoint/best_model.keras")
    else:
        ref_ckpt = Path(f"layer_rearrangement_experiments/{baseline_exp}/checkpoint/best_model.keras")

    ref_model = get_model(baseline_exp)
    ref_model.load_weights(str(ref_ckpt))

    # Run inference across held-out test split
    print(f"\nRunning test inferences across {total_samples} samples ({batches_to_run} batches)...")
    preds_winner = {h: [] for h in HORIZONS}
    preds_ref = {h: [] for h in HORIZONS}
    targets = {h: [] for h in HORIZONS}
    pers_t0 = {h: [] for h in HORIZONS}
    pers_24h = {h: [] for h in HORIZONS}
    climatology = {h: [] for h in HORIZONS}

    clim_dict = dm.compute_climatology(config.TRAIN_YEARS)

    for b in range(batches_to_run):
        xb, _ = test_gen[b]

        p_w_norm = winner_model(xb, training=False).numpy()
        p_ref_norm = ref_model(xb, training=False).numpy()

        p_w_k = dm.unpad_spatial(p_w_norm * t_std + t_mean)[:, 0]
        p_ref_k = dm.unpad_spatial(p_ref_norm * t_std + t_mean)[:, 0]

        for i in range(len(xb)):
            g_idx = b * batch_size + i
            y, start_i, in_end_t, tgt_times = test_gen.windows[g_idx]
            year_arr = dm.get_year_data(y)

            t0 = year_arr[start_i + test_gen.lags - 1, :, :, config.TARGET_INDEX]

            for h_idx, h in enumerate(HORIZONS):
                tgt_i = start_i + test_gen.lags + h - 1
                gt = year_arr[tgt_i, :, :, config.TARGET_INDEX]
                tgt_dt = pd.to_datetime(tgt_times[h_idx])

                # 24h prior persistence
                prior_24h_i = tgt_i - 12
                p24 = year_arr[prior_24h_i, :, :, config.TARGET_INDEX] if prior_24h_i >= 0 else t0

                # Diurnal climatology
                clim_val = clim_dict.get((tgt_dt.month, tgt_dt.day, tgt_dt.hour), t0)

                preds_winner[h].append(p_w_k[i, :, :, h_idx])
                preds_ref[h].append(p_ref_k[i, :, :, h_idx])
                targets[h].append(gt)
                pers_t0[h].append(t0)
                pers_24h[h].append(p24)
                climatology[h].append(clim_val)

    # Save test predictions NPZ
    pred_dir = Path(f"layer_rearrangement_experiments/{winner_exp}/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_npz_path = pred_dir / "test_predictions_7horizons.npz"
    npz_data = {f"pred_H{h}": np.stack(preds_winner[h], axis=0) for h in HORIZONS}
    npz_data.update({f"ref_H{h}": np.stack(preds_ref[h], axis=0) for h in HORIZONS})
    npz_data.update({f"target_H{h}": np.stack(targets[h], axis=0) for h in HORIZONS})
    npz_data.update({f"pers_H{h}": np.stack(pers_t0[h], axis=0) for h in HORIZONS})
    np.savez_compressed(save_npz_path, **npz_data)
    print(f"Saved test predictions NPZ to {save_npz_path}")

    # Compute metrics & Paired Bootstrap CIs
    print("\n" + "=" * 80)
    print("TEST EVALUATION RESULTS & STATISTICAL SIGNIFICANCE (BOOTSTRAP 95% CI)")
    print("=" * 80)

    comparison_metrics = {}
    csv_rows = []

    for h_idx, h in enumerate(HORIZONS):
        yt = np.stack(targets[h], axis=0)
        pw = np.stack(preds_winner[h], axis=0)
        pref = np.stack(preds_ref[h], axis=0)
        pt0 = np.stack(pers_t0[h], axis=0)
        pclim = np.stack(climatology[h], axis=0)

        err_w = pw - yt
        err_ref = pref - yt
        err_pers = pt0 - yt
        err_clim = pclim - yt

        rmse_w = float(np.sqrt(np.mean(err_w ** 2)))
        rmse_ref = float(np.sqrt(np.mean(err_ref ** 2)))
        rmse_pers = float(np.sqrt(np.mean(err_pers ** 2)))
        rmse_clim = float(np.sqrt(np.mean(err_clim ** 2)))

        mae_w = float(np.mean(np.abs(err_w)))
        mae_ref = float(np.mean(np.abs(err_ref)))

        bias_w = float(np.mean(err_w))
        bias_ref = float(np.mean(err_ref))

        var_yt = float(np.var(yt))
        r2_w = float(1.0 - np.mean(err_w ** 2) / var_yt)
        r2_ref = float(1.0 - np.mean(err_ref ** 2) / var_yt)

        skill_w_pers = float(1.0 - (rmse_w / rmse_pers))
        skill_ref_pers = float(1.0 - (rmse_ref / rmse_pers))

        # Bootstrap on sequence MSEs
        seq_mse_w = np.mean(err_w ** 2, axis=(1, 2))
        seq_mse_ref = np.mean(err_ref ** 2, axis=(1, 2))
        seq_mse_pers = np.mean(err_pers ** 2, axis=(1, 2))

        def calc_skill(mse_m, mse_p):
            return 1.0 - np.sqrt(np.mean(mse_m)) / np.sqrt(np.mean(mse_p))

        rng = np.random.default_rng(42)
        boot_diff = []
        for _ in range(n_bootstraps):
            idx = rng.choice(len(seq_mse_w), size=len(seq_mse_w), replace=True)
            sw = calc_skill(seq_mse_w[idx], seq_mse_pers[idx])
            sref = calc_skill(seq_mse_ref[idx], seq_mse_pers[idx])
            boot_diff.append(sw - sref)

        ci_diff = [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))]
        skill_diff = skill_w_pers - skill_ref_pers

        print(f"\nHorizon H{h} (+{LEAD_HOURS[h_idx]}h):")
        print(f"  Winner ({winner_exp}) : RMSE = {rmse_w:.3f} K | MAE = {mae_w:.3f} K | Bias = {bias_w:+.3f} K | R2 = {r2_w:.3f} | Skill = {skill_w_pers:+.3f}")
        print(f"  Reference ({baseline_exp}) : RMSE = {rmse_ref:.3f} K | MAE = {mae_ref:.3f} K | Bias = {bias_ref:+.3f} K | R2 = {r2_ref:.3f} | Skill = {skill_ref_pers:+.3f}")
        print(f"  Persistence (t0)     : RMSE = {rmse_pers:.3f} K")
        print(f"  Skill Diff [Winner - Ref]: {skill_diff:+.3f} (95% CI: [{ci_diff[0]:+.3f}, {ci_diff[1]:+.3f}])")

        comparison_metrics[f"H{h}"] = {
            "lead_hours": LEAD_HOURS[h_idx],
            "winner_rmse": rmse_w,
            "winner_mae": mae_w,
            "winner_bias": bias_w,
            "winner_r2": r2_w,
            "winner_skill_pers": skill_w_pers,
            "ref_rmse": rmse_ref,
            "ref_mae": mae_ref,
            "ref_bias": bias_ref,
            "ref_r2": r2_ref,
            "ref_skill_pers": skill_ref_pers,
            "persistence_rmse": rmse_pers,
            "climatology_rmse": rmse_clim,
            "skill_diff_winner_ref": skill_diff,
            "skill_diff_95ci": ci_diff
        }

        csv_rows.append({
            "horizon": f"H{h}",
            "lead_hours": LEAD_HOURS[h_idx],
            "winner_rmse_K": rmse_w,
            "winner_mae_K": mae_w,
            "winner_bias_K": bias_w,
            "winner_r2": r2_w,
            "winner_skill_pers": skill_w_pers,
            "ref_rmse_K": rmse_ref,
            "ref_mae_K": mae_ref,
            "ref_bias_K": bias_ref,
            "ref_r2": r2_ref,
            "ref_skill_pers": skill_ref_pers,
            "persistence_rmse_K": rmse_pers,
            "skill_diff": skill_diff,
            "ci_95_low": ci_diff[0],
            "ci_95_high": ci_diff[1]
        })

    # Save test metrics JSON and CSV
    json_out = Path(f"results/test_metrics_7horizons_{winner_exp}_vs_{baseline_exp}.json")
    with open(json_out, "w") as f:
        json.dump(comparison_metrics, f, indent=2)

    csv_out = Path(f"results/test_metrics_7horizons_{winner_exp}_vs_{baseline_exp}.csv")
    pd.DataFrame(csv_rows).to_csv(csv_out, index=False)

    # Also update master comparison files
    with open(Path("results/test_metrics_7horizons_comparison.json"), "w") as f:
        json.dump(comparison_metrics, f, indent=2)
    pd.DataFrame(csv_rows).to_csv(Path("results/test_metrics_7horizons_comparison.csv"), index=False)

    print(f"\nSaved test comparison JSON to {json_out} and CSV to {csv_out}")

    return comparison_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", type=str, default="L15_hybrid_kan_leaky_relu")
    parser.add_argument("--ref", type=str, default="L0_original")
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    run_test_evaluation(winner_exp=args.winner, baseline_exp=args.ref, max_test_samples=args.max_samples, n_bootstraps=args.bootstraps)
