"""
scripts/run_screening.py
========================
Phase 3 Stage A: Architectural Screening Engine
Executes Stage A screening runs across all 11 experimental configurations on top of Baseline v2.
Filters out variants that do not improve validation RMSE over Baseline v2 by at least 1.0%.
Resilient: automatically skips already completed runs.
"""

import sys
import gc
import json
import time
from pathlib import Path
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_v2 import run_experiment

STAGES = [
    "E-M1",      # 1. MLP after ASPP
    "E-K1",      # 2. KAN after ASPP
    "E-MK1",     # 3. Sequential MLP -> KAN
    "E-MK2",     # 3. Parallel MLP || KAN
    "E-S6",      # 4. KAN in decoder merge6
    "E-S67",     # 4. KAN in decoder merge6+7
    "E-S6789",   # 4. KAN in decoder merge6+7+8+9
    "E-S9",      # 4. KAN in decoder merge9
    "E-C2_C4",   # 5. MLP in encoder conv2..conv4
    "E-KC2_KC4", # 5. KAN in encoder conv2..conv4
    "E-CTRL"     # 6. Parameter-matched Conv3D Control
]

def run_all_screening(seed: int = 42, stages=None):
    if stages is None:
        stages = STAGES

    print("=" * 80)
    print("PHASE 3 STAGE A: ARCHITECTURE SCREENING")
    print(f"Total variants to screen: {len(stages)}")
    print(f"Target list: {stages}")
    print("=" * 80)

    # Baseline v2 benchmark (S3)
    b2_metrics_p = Path("results/runs/S3_seed42/metrics.json")
    if not b2_metrics_p.exists():
        print("Warning: S3_seed42 (Baseline v2) not found! Running S3 first...")
        run_experiment("configs/baseline_v2.yaml", stage="S3", seed=seed)

    with open(b2_metrics_p, "r") as f:
        b2_data = json.load(f)
    b2_mean_rmse = b2_data["validation_metrics"]["mean_rmse"]
    print(f"Baseline v2 (S3) Validation Mean RMSE: {b2_mean_rmse:.3f} K")

    results_table = []

    for idx, stage in enumerate(stages, 1):
        run_id = f"{stage}_seed{seed}"
        metrics_file = Path("results/runs") / run_id / "metrics.json"

        print("\n" + "#" * 80)
        print(f"[{idx}/{len(stages)}] SCREENING STAGE: {stage} (Run ID: {run_id})")
        print("#" * 80)

        t_start = time.time()
        try:
            metrics = run_experiment("configs/baseline_v2.yaml", stage=stage, seed=seed)
            mean_rmse = metrics["validation_metrics"]["mean_rmse"]
            mean_skill = metrics["validation_metrics"]["mean_skill_vs_pers"]
            diff_from_b2 = mean_rmse - b2_mean_rmse
            pct_improvement = (-diff_from_b2 / b2_mean_rmse) * 100.0

            status = "PROMOTED (>= +1.0%)" if pct_improvement >= 1.0 else ("BORDERLINE" if pct_improvement >= 0.0 else "ELIMINATED")

            results_table.append({
                "stage": stage,
                "params": metrics["parameters"],
                "val_rmse": mean_rmse,
                "val_skill": mean_skill,
                "diff_k": diff_from_b2,
                "pct_improvement": pct_improvement,
                "status": status,
                "duration_min": (time.time() - t_start) / 60.0
            })

            print(f"\nCompleted {stage}: Val Mean RMSE = {mean_rmse:.3f} K | vs Baseline v2: {diff_from_b2:+.3f} K ({pct_improvement:+.2f}%) | {status}")

        except Exception as e:
            print(f"Error running stage {stage}: {e}")
            import traceback
            traceback.print_exc()

        # Clean up memory
        tf.keras.backend.clear_session()
        gc.collect()

    print("\n" + "=" * 80)
    print("STAGE A SCREENING COMPLETE: SUMMARY TABLE")
    print("=" * 80)
    df_res = pd.DataFrame(results_table)
    if not df_res.empty:
        df_res = df_res.sort_values(by="val_rmse", ascending=True)
        print(df_res.to_string(index=False))
        df_res.to_csv("results/stage_a_screening_results.csv", index=False)
        print("\nSaved screening rankings to results/stage_a_screening_results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_all_screening(seed=args.seed, stages=args.stages)

