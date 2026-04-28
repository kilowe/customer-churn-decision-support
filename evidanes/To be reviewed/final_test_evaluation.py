from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility_supplement"
TEST_PROB_PATH = OUTPUT_DIR / "test_probabilities.csv"
SELECTED_THRESHOLDS_PATH = OUTPUT_DIR / "selected_thresholds.csv"


def evaluate_on_test_set() -> None:
    """The Evaluate fixed thresholds on the held-out test set exactly once.

    Important:
    - No threshold optimization is performed here.
    - Thresholds are read from validation-based selections only.
    - This supports a clean separation between model selection and final reporting.
    """

    if not TEST_PROB_PATH.exists():
        raise FileNotFoundError(
            f"Missing required file: {TEST_PROB_PATH}\n"
            "Please run validation_pipeline.py before final_test_evaluation.py."
        )

    if not SELECTED_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Missing required file: {SELECTED_THRESHOLDS_PATH}\n"
            "Please run threshold_optimization.py before final_test_evaluation.py."
        )

    test_df = pd.read_csv(TEST_PROB_PATH)
    selected_df = pd.read_csv(SELECTED_THRESHOLDS_PATH)

    model_to_proba_column = {
        "baseline": "baseline_proba",
        "balanced": "balanced_proba",
    }

    rows: list[dict] = []

    y_true = test_df["y_true"].to_numpy()

    for _, selected in selected_df.iterrows():
        model_name = selected["model"]
        threshold = float(selected["threshold"])
        proba_col = model_to_proba_column[model_name]
        y_score = test_df[proba_col].to_numpy()
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        cfn = selected["cfn"]
        cfp = selected["cfp"]
        total_cost = None
        if pd.notna(cfn) and pd.notna(cfp):
            total_cost = float(cfn * fn + cfp * fp)

        rows.append(
            {
                "model": model_name,
                "selection_type": selected["selection_type"],
                "scenario": selected["scenario"],
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_score),
                "pr_auc": average_precision_score(y_true, y_score),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "cfn": cfn,
                "cfp": cfp,
                "total_cost": total_cost,
                "recall_constraint_satisfied": selected["recall_constraint_satisfied"],
            }
        )

    evaluation_df = pd.DataFrame(rows)

    evaluation_df.to_csv(OUTPUT_DIR / "Evaluation_Results.csv", index=False)

    business_cost_df = evaluation_df[
        evaluation_df["selection_type"] == "cost_based"
    ].copy()
    business_cost_df.to_csv(OUTPUT_DIR / "Business_Cost_Analysis.csv", index=False)

    final_summary_df = evaluation_df[
        [
            "model",
            "selection_type",
            "scenario",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "total_cost",
        ]
    ].copy()
    final_summary_df.to_csv(OUTPUT_DIR / "Final_Summary.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION SUMMARY")
    print("=" * 80)
    print(final_summary_df.to_string(index=False))

    max_f1_only = evaluation_df[evaluation_df["selection_type"] == "max_f1"].copy()
    print("\nBaseline vs balanced under max_f1 thresholds:")
    print(
        max_f1_only[
            [
                "model",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    print("\nInterpretation note:")
    print(
        "The best configuration depends on operational priorities (recall, precision, and cost), "
        "so no single model is universally best in every decision context."
    )

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'Evaluation_Results.csv'}")
    print(f"- {OUTPUT_DIR / 'Business_Cost_Analysis.csv'}")
    print(f"- {OUTPUT_DIR / 'Final_Summary.csv'}")


if __name__ == "__main__":
    evaluate_on_test_set()
