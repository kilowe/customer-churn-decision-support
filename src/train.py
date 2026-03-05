from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Import of the preprocessing function
from preprocessing import run_preprocessing


RANDOM_STATE = 42
TEST_SIZE = 0.20  # 80/20 split


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Decisions:
    - 80% train / 20% test
    - Random split, stratified on y (to preserve churn ratio)
    - No cross-validation (for now)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_baseline_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LogisticRegression:
    """
    Train a simple baseline model.

    Why Logistic Regression here:
    - Strong baseline for tabular classification.
    - Interpretable and stable.
    - No tuning is performed.

    Notes:
    - No class_weight adjustments here for the moment.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """
    Print required metrics:
    - Accuracy (informational only)
    - Recall (on negative class = 1)
    - Precision (negative class = 1)
    - Confusion Matrix

    Also includes a business-first interpretation of FP/FN.
    """
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=1)
    prec = precision_score(y_true, y_pred, pos_label=1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print(f"Accuracy (informational): {acc:.4f}")
    print(f"Recall (Churn=1):         {rec:.4f}")
    print(f"Precision (Churn=1):      {prec:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred) with labels [0, 1]:")
    print(cm)

    print("\nDetailed counts:")
    print(f"TN (0->0): {tn}  | Correctly predicted non-churn")
    print(f"FP (0->1): {fp}  | Non-churn predicted as churn")
    print(f"FN (1->0): {fn}  | Churn predicted as non-churn")
    print(f"TP (1->1): {tp}  | Correctly predicted churn")

    print("\n" + "-" * 80)
    print("BUSINESS-FIRST INTERPRETATION (FP/FN Costs)")
    print("-" * 80)
    print(
        "False Negatives (FN) = real churners we missed.\n"
        "- Business cost: lost customers + lost revenue because we did NOT intervene.\n"
        "- Mitigation (later steps): adjust threshold, class weights, retention strategies.\n"
    )
    print(
        "False Positives (FP) = customers predicted to churn but they would not.\n"
        "- Business cost: unnecessary retention offers/discounts/support workload.\n"
        "- Mitigation (later steps): improve precision via modeling/thresholding.\n"
    )

    # Optional: a compact report (still no tuning/CV)
    print("Classification report (optional, for detail):")
    print(classification_report(y_true, y_pred, digits=4))


def main() -> None:
    """
    End-to-end run:
     - Build X_encoded and y from preprocessing
     - 80/20 stratified split
     - Train a baseline model
     - Print metrics + confusion matrix + business interpretation
    """
    # X_encoded (numeric) and y (0/1)
    X_encoded, y = run_preprocessing(save_csv=True)

    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT")
    print("=" * 80)
    print(f"Full dataset: X={X_encoded.shape}, y={y.shape}")
    print("Target distribution (full):")
    print(y.value_counts(normalize=True).rename("proportion"))

    X_train, X_test, y_train, y_test = split_data(X_encoded, y)

    print("\nSplit shapes:")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    print("\nTarget distribution (train):")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print("\nTarget distribution (test):")
    print(y_test.value_counts(normalize=True).rename("proportion"))

    model = train_baseline_model(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluate_classification(y_test, y_pred)


if __name__ == "__main__":
    main()
