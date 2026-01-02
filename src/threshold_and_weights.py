from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from preprocessing import run_preprocessing


RANDOM_STATE = 42
TEST_SIZE = 0.20


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split with stratification to preserve the churn ratio in both sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight=None,
) -> LogisticRegression:
    """
    Train a Logistic Regression model without tuning.

    - We keep the model simple and stable.
    - class_weight can be None (baseline) or 'balanced'.
    - No feature changes and no additional models are introduced.
    """
    model = LogisticRegression(
        max_iter=3000,  # helps convergence without changing the modeling approach
        solver="lbfgs",
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def predict_with_threshold(
    model: LogisticRegression, X: pd.DataFrame, threshold: float
) -> np.ndarray:
    """
    Convert churn probabilities into class predictions using a custom threshold.

    Default threshold is 0.5, but lowering it increases recall (typically) at the cost of precision.
    """
    proba_churn = model.predict_proba(X)[:, 1]
    return (proba_churn >= threshold).astype(int)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Compute the required metrics for the churn class (label=1), plus confusion matrix components.
    """
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=1)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "recall_churn": rec,
        "precision_churn": prec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def print_comparison_table(results: list[dict]) -> None:
    """
    Print a compact comparison table for quick reading in the console.
    """
    df = pd.DataFrame(results)

    # Format a few columns for readability
    for col in ["accuracy", "recall_churn", "precision_churn"]:
        df[col] = df[col].map(lambda v: f"{v:.4f}")

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(
        df[
            [
                "model",
                "threshold",
                "accuracy",
                "recall_churn",
                "precision_churn",
                "fp",
                "fn",
                "tp",
                "tn",
            ]
        ].to_string(index=False)
    )


def explain_tradeoffs(best_base: dict, best_bal: dict) -> None:
    """
    Business-first explanation: focus on false negatives vs false positives.
    """
    print("\n" + "-" * 80)
    print("BUSINESS INTERPRETATION (FP/FN)")
    print("-" * 80)

    print("Key idea:")
    print(
        "- False Negatives (FN): churners missed -> lost customers (usually high cost)"
    )
    print(
        "- False Positives (FP): unnecessary retention actions -> operational/discount cost\n"
    )

    print("Baseline (reference):")
    print(
        f"- threshold={best_base['threshold']}, recall={best_base['recall_churn']:.4f}, precision={best_base['precision_churn']:.4f}, FN={best_base['fn']}, FP={best_base['fp']}"
    )

    print("Balanced (reference):")
    print(
        f"- threshold={best_bal['threshold']}, recall={best_bal['recall_churn']:.4f}, precision={best_bal['precision_churn']:.4f}, FN={best_bal['fn']}, FP={best_bal['fp']}"
    )

    print("\nHow to read this:")
    print(
        "- If the business priority is to reduce missed churners, prefer the configuration that reduces FN the most."
    )
    print(
        "- If retention actions are expensive, watch how much FP increases when lowering the threshold or using balanced weights."
    )


def main() -> None:
    # 1) Build X (fully numeric) and y (0/1). No feature changes here.
    X, y = run_preprocessing(save_csv=True)

    # 2) Train/test split (stratified).
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n" + "=" * 80)
    print("DATA SPLIT CHECK")
    print("=" * 80)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test:  X={X_test.shape},  y={y_test.shape}")
    print("\nChurn ratio (train):")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print("\nChurn ratio (test):")
    print(y_test.value_counts(normalize=True).rename("proportion"))

    # 3) Train two models: baseline and balanced.
    baseline_model = train_logistic_regression(X_train, y_train, class_weight=None)
    balanced_model = train_logistic_regression(
        X_train, y_train, class_weight="balanced"
    )

    # 4) Evaluate different thresholds without retraining (thresholding uses predict_proba).
    thresholds = [0.50, 0.40, 0.30]
    results: list[dict] = []

    for t in thresholds:
        pred_base = predict_with_threshold(baseline_model, X_test, threshold=t)
        m_base = compute_metrics(y_test, pred_base)
        results.append({"model": "logreg_baseline", "threshold": t, **m_base})

        pred_bal = predict_with_threshold(balanced_model, X_test, threshold=t)
        m_bal = compute_metrics(y_test, pred_bal)
        results.append({"model": "logreg_balanced", "threshold": t, **m_bal})

    # 5) Print comparison table
    print_comparison_table(results)

    # 6) Pick one reference config per model (highest recall, tie-breaker: higher precision)
    df = pd.DataFrame(results)

    def pick_reference(model_name: str) -> dict:
        sub = df[df["model"] == model_name].copy()
        sub = sub.sort_values(
            by=["recall_churn", "precision_churn"], ascending=[False, False]
        )
        return sub.iloc[0].to_dict()

    best_base = pick_reference("logreg_baseline")
    best_bal = pick_reference("logreg_balanced")

    explain_tradeoffs(best_base, best_bal)

    # 7) Print confusion matrices for full transparency (one per threshold)
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES (rows=true, cols=pred) labels [0,1]")
    print("=" * 80)
    for t in thresholds:
        pred_base = predict_with_threshold(baseline_model, X_test, threshold=t)
        cm_base = confusion_matrix(y_test, pred_base, labels=[0, 1])

        pred_bal = predict_with_threshold(balanced_model, X_test, threshold=t)
        cm_bal = confusion_matrix(y_test, pred_bal, labels=[0, 1])

        print(f"\nBaseline | threshold={t}")
        print(cm_base)

        print(f"\nBalanced | threshold={t}")
        print(cm_bal)


if __name__ == "__main__":
    main()
