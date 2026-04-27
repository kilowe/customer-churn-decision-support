from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ensure src/ is importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import run_preprocessing


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN_VAL = 0.20
MAX_ITER = 3000
SOLVER = "lbfgs"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility_supplement"


def ensure_output_dir() -> None:
    """Create output folder for reproducibility artifacts.

    Why:
    Keeping all supplement outputs in a dedicated folder supports
    auditability and makes reruns easier to compare.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def churn_prevalence(y: pd.Series) -> float:
    """Return churn prevalence as a simple mean of the binary target."""
    return float(y.mean())


def fit_models_and_export_probabilities() -> None:
    """Run explicit train/validation/test workflow and save probabilities.

    Why this split structure exists:
    - Validation is used only for threshold selection.
    - Test is kept fully held out for final evaluation.
    - This avoids tuning decisions on the final test set.
    """
    ensure_output_dir()

    X_encoded, y = run_preprocessing(save_csv=True)

    # First split: 80% train_val and 20% test, stratified by churn label.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_encoded,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Second split: 80% train and 20% validation, from train_val only.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN_VAL,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    baseline_model = LogisticRegression(
        max_iter=MAX_ITER,
        solver=SOLVER,
        class_weight=None,
    )
    balanced_model = LogisticRegression(
        max_iter=MAX_ITER,
        solver=SOLVER,
        class_weight="balanced",
    )

    baseline_model.fit(X_train, y_train)
    balanced_model.fit(X_train, y_train)

    val_baseline_proba = baseline_model.predict_proba(X_val)[:, 1]
    val_balanced_proba = balanced_model.predict_proba(X_val)[:, 1]

    test_baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    test_balanced_proba = balanced_model.predict_proba(X_test)[:, 1]

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "row_count": len(y_train),
                "feature_count": X_train.shape[1],
                "churn_prevalence": churn_prevalence(y_train),
            },
            {
                "split": "validation",
                "row_count": len(y_val),
                "feature_count": X_val.shape[1],
                "churn_prevalence": churn_prevalence(y_val),
            },
            {
                "split": "test",
                "row_count": len(y_test),
                "feature_count": X_test.shape[1],
                "churn_prevalence": churn_prevalence(y_test),
            },
        ]
    )

    validation_probabilities = pd.DataFrame(
        {
            "row_id": range(len(y_val)),
            "y_true": y_val.to_numpy(),
            "baseline_proba": val_baseline_proba,
            "balanced_proba": val_balanced_proba,
        }
    )

    test_probabilities = pd.DataFrame(
        {
            "row_id": range(len(y_test)),
            "y_true": y_test.to_numpy(),
            "baseline_proba": test_baseline_proba,
            "balanced_proba": test_balanced_proba,
        }
    )

    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
    validation_probabilities.to_csv(
        OUTPUT_DIR / "validation_probabilities.csv", index=False
    )
    test_probabilities.to_csv(OUTPUT_DIR / "test_probabilities.csv", index=False)

    print("\n" + "=" * 80)
    print("VALIDATION PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Train shape:      X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shape:       X={X_test.shape}, y={y_test.shape}")

    print("\nChurn prevalence by split:")
    print(f"- Train:      {churn_prevalence(y_train):.4f}")
    print(f"- Validation: {churn_prevalence(y_val):.4f}")
    print(f"- Test:       {churn_prevalence(y_test):.4f}")

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'split_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'validation_probabilities.csv'}")
    print(f"- {OUTPUT_DIR / 'test_probabilities.csv'}")


if __name__ == "__main__":
    fit_models_and_export_probabilities()
