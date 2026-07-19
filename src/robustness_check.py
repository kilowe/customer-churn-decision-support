"""
robustness_check.py

Why this script exists
-----------------------
The thesis pipeline (validation_pipeline.py -> threshold_optimization.py ->
final_test_evaluation.py) is only ever run with random_state=42. Every number
in chapters 4 and 5 -- including which model (baseline vs balanced) wins in
each cost scenario, and the two optimal thresholds 0.37 / 0.62 -- rests on a
single random partition of the 7,043 customers into train/validation/test.

This script reruns the exact same pipeline logic across several different
seeds for the train/val/test split, and aggregates the results, so we can see
whether the conclusions (optimal thresholds, total cost per scenario, which
model wins) are stable, or whether they are sensitive to the particular
partition drawn with seed=42.

How it works
------------
It does NOT modify validation_pipeline.py, threshold_optimization.py, or
final_test_evaluation.py on disk. It imports them as modules and
monkey-patches their module-level RANDOM_STATE constant before calling their
public functions, since none of the three functions accept random_state as an
argument. This works because Python resolves global names referenced inside a
function body at call time, not at definition time -- confirmed by hand before
writing this file (see note at the bottom of this docstring for the
minimal reproduction of that behavior, kept for reference).

Usage
-----
1. Place this file directly in src/, next to validation_pipeline.py,
   threshold_optimization.py and final_test_evaluation.py.
2. From the project root, run:

       python src/robustness_check.py

3. Look at the console output and at the three CSV files it writes to
   outputs/reproducibility_supplement/ (see below). Each full run overwrites
   split_summary.csv, validation_probabilities.csv, test_probabilities.csv,
   threshold_grid_validation.csv, selected_thresholds.csv, Evaluation_Results.csv,
   Business_Cost_Analysis.csv and Final_Summary.csv exactly as the normal
   pipeline does -- this script just captures Final_Summary.csv after each
   seed before the next seed overwrites it.

Output files (outputs/reproducibility_supplement/)
----------------------------------------------------
- robustness_check_summary.csv        : Final_Summary.csv stacked for every seed.
- robustness_check_per_seed.csv       : one row per (seed, scenario), baseline
                                         cost vs balanced cost side by side,
                                         and which model wins.
- robustness_check_winner_stability.csv : one row per scenario, counting how
                                         many of the N seeds each model wins,
                                         plus mean/std of the cost gap.

How to read the result
-----------------------
If "balanced_wins" and "baseline_wins" in robustness_check_winner_stability.csv
are both far from 0 for the low_incentive and medium_incentive scenarios (e.g.
3 vs 12 out of 15), that confirms the seed=42 result is not a fluke: baseline
reliably beats balanced there. If instead the split is close to 50/50, that
means the "baseline wins in low/medium" conclusion in section 5.1.2 of the
thesis is largely an artifact of which customers happened to land in the test
set under seed=42, and the discussion should be reworded to reflect that
uncertainty (e.g. report the win-rate across seeds and the mean cost gap with
its std, rather than a single deterministic verdict).

Note on validation performed before delivering this script
------------------------------------------------------------
Before writing this version, the two mechanisms this script depends on were
verified in isolation:
1. Monkey-patching a module-level constant that is read inside a function
   body (not passed as a default argument) does take effect at call time.
   Minimal check that was run:

       import types
       mod = types.ModuleType("fakemod")
       exec("RANDOM_STATE = 42\\ndef show():\\n    return RANDOM_STATE", mod.__dict__)
       mod.RANDOM_STATE = 7
       assert mod.show() == 7   # passes

2. The pivot_table / groupby aggregation logic below (winner detection, cost
   gap, win counts) was pressure-tested on synthetic Final_Summary-shaped data
   with 5 dummy seeds and produced the expected columns with no errors.

What was NOT possible to verify end-to-end in this environment, and why
--------------------------------------------------------------------------
A full live run of this exact script against the real pipeline could not be
completed here, for two independent reasons, both environment-specific to
where this script was drafted (not to your machine):
- Fetching the raw CSV (WA_Fn-UseC_-Telco-Customer-Churn.csv, ~977 KB,
  7,043 rows) from your GitHub repo hit a hard size cap in the fetch tool
  available here, which returned only the first ~487 rows before cutting off.
- This sandbox has scikit-learn unavailable and no outbound package-installer
  access, so LogisticRegression could not be imported at all here.
Because of this, no numeric result below comes from a real run on your full
dataset -- only the plumbing (imports, monkeypatch, file I/O, aggregation) was
checked. Please run this on your own machine, where both the full CSV and
your normal Python environment (per requirements.txt) are available, and
treat the resulting CSVs as the first real evidence on this question.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import validation_pipeline as vp
import threshold_optimization as topt
import final_test_evaluation as fte

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility_supplement"

# Seeds to test. 42 is kept first as the thesis reference value.
# The other 14 were picked as a simple, unremarkable spread (0-9, then a few
# larger round-ish numbers) -- not chosen after looking at any result, and not
# tuned in any way. Feel free to add more; more seeds only make the
# win-rate estimate in robustness_check_winner_stability.csv more precise.
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 123, 777, 2024]


def run_one_seed(seed: int) -> pd.DataFrame:
    """Run the full validation -> threshold -> test-evaluation pipeline for one seed.

    Returns Final_Summary.csv (tagged with the seed) for that run.
    """
    # This is the single point of control: both successive 80/20 splits inside
    # fit_models_and_export_probabilities() (see validation_pipeline.py,
    # section 3.3.1 of the thesis) read RANDOM_STATE from the module's global
    # scope at call time, so reassigning it here is enough -- no need to edit
    # the file itself.
    vp.RANDOM_STATE = seed

    vp.fit_models_and_export_probabilities()
    topt.run_threshold_optimization()
    fte.evaluate_on_test_set()

    final_summary = pd.read_csv(OUTPUT_DIR / "Final_Summary.csv")
    final_summary.insert(0, "seed", seed)
    return final_summary


def main() -> None:
    all_runs = []
    for seed in SEEDS:
        print(f"\n=== Running pipeline with random_state={seed} ===")
        df = run_one_seed(seed)
        all_runs.append(df)

    combined = pd.concat(all_runs, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "robustness_check_summary.csv", index=False)

    # Focus on the cost-based selections: this is exactly what section 5.1.2
    # of the thesis uses to say "baseline wins in low/medium, balanced wins in
    # high incentive".
    cost_based = combined[combined["selection_type"] == "cost_based"].copy()

    pivot = cost_based.pivot_table(
        index=["seed", "scenario"], columns="model", values="total_cost"
    ).reset_index()
    pivot["winner"] = np.where(pivot["balanced"] < pivot["baseline"], "balanced", "baseline")
    pivot["cost_gap_balanced_minus_baseline"] = pivot["balanced"] - pivot["baseline"]

    stability = (
        pivot.groupby("scenario")
        .agg(
            n_seeds=("winner", "count"),
            balanced_wins=("winner", lambda s: (s == "balanced").sum()),
            baseline_wins=("winner", lambda s: (s == "baseline").sum()),
            mean_cost_gap=("cost_gap_balanced_minus_baseline", "mean"),
            std_cost_gap=("cost_gap_balanced_minus_baseline", "std"),
        )
        .reset_index()
    )

    pivot.to_csv(OUTPUT_DIR / "robustness_check_per_seed.csv", index=False)
    stability.to_csv(OUTPUT_DIR / "robustness_check_winner_stability.csv", index=False)

    print("\n" + "=" * 80)
    print(f"ROBUSTNESS CHECK -- {len(SEEDS)} seeds (seed=42 is the thesis reference value)")
    print("=" * 80)
    print(stability.to_string(index=False))
    print("\nPer-seed detail:")
    print(pivot.to_string(index=False))

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'robustness_check_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'robustness_check_per_seed.csv'}")
    print(f"- {OUTPUT_DIR / 'robustness_check_winner_stability.csv'}")


if __name__ == "__main__":
    main()
