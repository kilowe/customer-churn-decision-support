"""
threshold_gain_check.py

What this tests
----------------
Section 5.1.1 of the thesis states: "threshold optimization reduces the cost
by 60.6% for the baseline model, versus 30.9% for the balanced model, in the
low incentive scenario. In the medium incentive scenario, this reduction
reaches 35.1% for baseline versus 5.4% for balanced. In the high incentive
scenario ... 12.9% for baseline versus 1.6% for balanced."

These six percentages all come from Table 3 (validation set), comparing the
cost at the naive threshold 0.50 against the cost at the threshold selected
by select_cost_based() -- and all of it from the single seed=42 run.

This script recomputes the same six percentages (per model x scenario) for
each of the same 15 seeds already used in robustness_check.py, so we can see
whether "baseline benefits far more from threshold tuning than balanced" is a
general finding or a seed=42 artifact.

How it works
------------
Same approach as robustness_check.py: import validation_pipeline,
threshold_optimization and final_test_evaluation as modules, monkey-patch
RANDOM_STATE, and re-run the pipeline per seed. Nothing on disk is modified.

For each seed, after threshold_optimization.run_threshold_optimization() has
written threshold_grid_validation.csv and selected_thresholds.csv:
- cost_naive(model, scenario)  = CFN*FN + CFP*FP, using the row where
  threshold == 0.50 in threshold_grid_validation.csv for that model
  (validation set, matching how Table 3 was built).
- cost_optimal(model, scenario) = the "total_cost" already stored in
  selected_thresholds.csv for the cost_based selection of that model/scenario
  (also validation set -- these are NOT the test-set costs used in
  Final_Summary.csv / the earlier robustness_check.py).
- pct_reduction = 100 * (cost_naive - cost_optimal) / cost_naive

Usage
-----
Place this file in src/, next to the other pipeline scripts and next to
robustness_check.py, then run:

    python src/threshold_gain_check.py

Output (outputs/reproducibility_supplement/)
---------------------------------------------
- threshold_gain_per_seed.csv    : one row per (seed, model, scenario) with
  cost_naive, cost_optimal, pct_reduction.
- threshold_gain_summary.csv     : one row per (model, scenario), with the
  mean/std/min/max of pct_reduction across the 15 seeds, plus the seed=42
  value shown separately so it's easy to compare against the thesis text.

How to read it
---------------
If the seed=42 pattern (baseline gains much more from tuning than balanced)
holds, you should see, for every scenario, mean_pct_reduction clearly higher
for baseline than for balanced, with a moderate std (i.e. seed=42 is not an
outlier). If instead the two rows are close, or seed=42 sits far from the
mean of the other 14 seeds, that specific mechanism-level claim in 5.1.1 needs
the same softening treatment as the total-cost comparison in 5.1.2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import validation_pipeline as vp
import threshold_optimization as topt
import final_test_evaluation as fte

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility_supplement"

SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 123, 777, 2024]


def compute_gain_for_seed(seed: int) -> pd.DataFrame:
    grid = pd.read_csv(OUTPUT_DIR / "threshold_grid_validation.csv")
    selected = pd.read_csv(OUTPUT_DIR / "selected_thresholds.csv")

    naive = grid[grid["threshold"] == 0.50].set_index("model")

    rows = []
    for model in ["baseline", "balanced"]:
        fn = int(naive.loc[model, "fn"])
        fp = int(naive.loc[model, "fp"])

        for scenario_name, costs in topt.SCENARIOS.items():
            cfn, cfp = costs["cfn"], costs["cfp"]
            cost_naive = cfn * fn + cfp * fp

            opt_row = selected[
                (selected["model"] == model)
                & (selected["selection_type"] == "cost_based")
                & (selected["scenario"] == scenario_name)
            ].iloc[0]
            cost_optimal = float(opt_row["total_cost"])

            pct_reduction = 100 * (cost_naive - cost_optimal) / cost_naive if cost_naive > 0 else float("nan")

            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "scenario": scenario_name,
                    "cost_naive_050": cost_naive,
                    "cost_optimal": cost_optimal,
                    "pct_reduction": pct_reduction,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for seed in SEEDS:
        print(f"\n=== Running pipeline with random_state={seed} ===")
        vp.RANDOM_STATE = seed
        vp.fit_models_and_export_probabilities()
        topt.run_threshold_optimization()
        fte.evaluate_on_test_set()  # not used here, but kept so outputs stay consistent with robustness_check.py

        gain_df = compute_gain_for_seed(seed)
        all_rows.append(gain_df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "threshold_gain_per_seed.csv", index=False)

    seed42 = combined[combined["seed"] == 42].set_index(["model", "scenario"])["pct_reduction"]

    summary = (
        combined.groupby(["model", "scenario"])["pct_reduction"]
        .agg(mean_pct_reduction="mean", std_pct_reduction="std", min_pct_reduction="min", max_pct_reduction="max")
        .reset_index()
    )
    summary["seed42_pct_reduction"] = summary.apply(
        lambda r: seed42.loc[(r["model"], r["scenario"])], axis=1
    )

    summary.to_csv(OUTPUT_DIR / "threshold_gain_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("THRESHOLD-TUNING GAIN -- 15 seeds vs. the seed=42 value quoted in section 5.1.1")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'threshold_gain_per_seed.csv'}")
    print(f"- {OUTPUT_DIR / 'threshold_gain_summary.csv'}")


if __name__ == "__main__":
    main()
