from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility_supplement"
VALIDATION_PATH = OUTPUT_DIR / "validation_probabilities.csv"

THRESHOLD_START = 0.05
THRESHOLD_END = 0.95
THRESHOLD_STEP = 0.01
MIN_RECALL_CONSTRAINT = 0.70

SCENARIOS = {
    "low_incentive": {"cfn": 300, "cfp": 30},
    "medium_incentive": {"cfn": 300, "cfp": 60},
    "high_incentive": {"cfn": 300, "cfp": 120},
}


def build_threshold_grid() -> pd.DataFrame:
    """The Evaluate validation metrics across thresholds for both models.

    Why:
    Threshold selection must be based on validation data only.
    This keeps the test set untouched for final evaluation.
    """
    validation_df = pd.read_csv(VALIDATION_PATH)
    thresholds = np.round(
        np.arange(THRESHOLD_START, THRESHOLD_END + 1e-9, THRESHOLD_STEP), 2
    )

    rows: list[dict] = []
    for model_name, score_col in [
        ("baseline", "baseline_proba"),
        ("balanced", "balanced_proba"),
    ]:
        y_true = validation_df["y_true"].to_numpy()
        y_score = validation_df[score_col].to_numpy()

        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            rows.append(
                {
                    "model": model_name,
                    "threshold": float(threshold),
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )

    return pd.DataFrame(rows)


def select_max_f1(df_model: pd.DataFrame) -> dict:
    """Select threshold with highest F1; tie-break by higher recall then lower threshold."""
    ranked = df_model.sort_values(
        by=["f1", "recall", "threshold"], ascending=[False, False, True]
    )
    best = ranked.iloc[0].to_dict()
    return {
        "selection_type": "max_f1",
        "scenario": "not_applicable",
        "threshold": best["threshold"],
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "tn": int(best["tn"]),
        "fp": int(best["fp"]),
        "fn": int(best["fn"]),
        "tp": int(best["tp"]),
        "cfn": np.nan,
        "cfp": np.nan,
        "total_cost": np.nan,
        "recall_constraint_satisfied": True,
    }


def select_cost_based(
    df_model: pd.DataFrame, scenario_name: str, cfn: int, cfp: int
) -> dict:
    """Select minimum-cost threshold with recall guardrail when possible."""
    working = df_model.copy()
    working["total_cost"] = cfn * working["fn"] + cfp * working["fp"]

    constrained = working[working["recall"] >= MIN_RECALL_CONSTRAINT].copy()

    if constrained.empty:
        print(
            f"[Warning] Model={df_model['model'].iloc[0]}, scenario={scenario_name}: "
            f"no threshold satisfies recall >= {MIN_RECALL_CONSTRAINT:.2f}. "
            "Selecting global minimum-cost threshold without the recall constraint."
        )
        selected_pool = working
        recall_ok = False
    else:
        selected_pool = constrained
        recall_ok = True

    best = selected_pool.sort_values(
        by=["total_cost", "threshold"], ascending=[True, True]
    ).iloc[0]

    return {
        "selection_type": "cost_based",
        "scenario": scenario_name,
        "threshold": float(best["threshold"]),
        "accuracy": float(best["accuracy"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "tn": int(best["tn"]),
        "fp": int(best["fp"]),
        "fn": int(best["fn"]),
        "tp": int(best["tp"]),
        "cfn": int(cfn),
        "cfp": int(cfp),
        "total_cost": float(best["total_cost"]),
        "recall_constraint_satisfied": recall_ok,
    }


def run_threshold_optimization() -> None:
    """Run threshold grid and save selected validation thresholds."""
    threshold_grid = build_threshold_grid()

    selections: list[dict] = []
    for model_name in ["baseline", "balanced"]:
        model_grid = threshold_grid[threshold_grid["model"] == model_name].copy()

        max_f1_result = select_max_f1(model_grid)
        selections.append({"model": model_name, **max_f1_result})

        for scenario_name, costs in SCENARIOS.items():
            cost_result = select_cost_based(
                model_grid,
                scenario_name=scenario_name,
                cfn=costs["cfn"],
                cfp=costs["cfp"],
            )
            selections.append({"model": model_name, **cost_result})

    selected_thresholds = pd.DataFrame(selections)

    threshold_grid.to_csv(OUTPUT_DIR / "threshold_grid_validation.csv", index=False)
    selected_thresholds.to_csv(OUTPUT_DIR / "selected_thresholds.csv", index=False)

    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION SUMMARY (VALIDATION ONLY)")
    print("=" * 80)
    print(selected_thresholds.to_string(index=False))

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'threshold_grid_validation.csv'}")
    print(f"- {OUTPUT_DIR / 'selected_thresholds.csv'}")


if __name__ == "__main__":
    run_threshold_optimization()
