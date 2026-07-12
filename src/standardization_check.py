from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure src/ is importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import run_preprocessing


"""
Standardization check (supplementary, not part of the main pipeline).

Why this script exists:
- Section 3.2.4 of the thesis explains that numeric variables are not
  standardized, and argues this choice is reasonable given the model used
  (L2-regularized logistic regression, C=1.0, lbfgs solver).
- L2 regularization penalizes coefficients unevenly when variables are on
  different scales: a coefficient is naturally smaller for a large-scale
  variable (e.g. TotalCharges) than for a small-scale variable (e.g. a
  binary 0/1 column), for the same real effect. This script checks how
  much this actually matters for the baseline model of this study, rather
  than assuming an answer.

What this script does NOT do:
- It does not replace validation_pipeline.py or any validated result
  already used in the thesis.
- It does not change class_weight, C, the solver, or any other modeling
  choice already justified in chapter 3.
- It only adds one extra step (standardizing tenure, MonthlyCharges,
  TotalCharges) to the existing baseline model, trained on the exact same
  split logic, so the two runs are comparable side by side.

Output:
- A single CSV comparing, for each of the 23 variables, the coefficient
  and importance rank with and without standardization.
- A short printed summary comparing predicted probabilities between the
  two versions on the validation set (Pearson correlation, mean absolute
  difference, share of customers whose predicted class flips at 0.50).
"""

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN_VAL = 0.20
MAX_ITER = 3000
SOLVER = "lbfgs"

CONTINUOUS_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "standardization_check"


def ensure_output_dir() -> None:
    """Create a dedicated output folder for this check.

    Why:
    Kept separate from outputs/reproducibility_supplement/, which holds
    the files already referenced in the thesis, so this exploratory check
    cannot be confused with a validated pipeline output.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_standardization_check() -> None:
    ensure_output_dir()

    X_encoded, y = run_preprocessing(save_csv=False)

    # Same two-step split as validation_pipeline.py, same random_state,
    # so both versions below are trained on the same rows.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_encoded,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN_VAL,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    # ----------------------------
    # Version A: as in the thesis (no standardization)
    # ----------------------------
    model_a = LogisticRegression(max_iter=MAX_ITER, solver=SOLVER, class_weight=None)
    model_a.fit(X_train, y_train)

    # ----------------------------
    # Version B: tenure, MonthlyCharges, TotalCharges standardized
    # (scaler fit on train only, applied to train and validation)
    # ----------------------------
    scaler = StandardScaler()
    X_train_std = X_train.copy()
    X_val_std = X_val.copy()
    X_train_std[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_val_std[CONTINUOUS_COLS] = scaler.transform(X_val[CONTINUOUS_COLS])

    model_b = LogisticRegression(max_iter=MAX_ITER, solver=SOLVER, class_weight=None)
    model_b.fit(X_train_std, y_train)

    # ----------------------------
    # Coefficient comparison
    # ----------------------------
    coefs_a = pd.Series(model_a.coef_[0], index=X_train.columns)
    coefs_b = pd.Series(model_b.coef_[0], index=X_train.columns)

    rank_a = coefs_a.abs().rank(ascending=False)
    rank_b = coefs_b.abs().rank(ascending=False)

    comparison = pd.DataFrame(
        {
            "variable": X_train.columns,
            "coef_non_standardized": coefs_a.values,
            "rank_non_standardized": rank_a.astype(int).values,
            "coef_standardized": coefs_b.values,
            "rank_standardized": rank_b.astype(int).values,
        }
    ).sort_values("rank_non_standardized")

    comparison.to_csv(OUTPUT_DIR / "standardization_coefficient_comparison.csv", index=False)

    spearman_rho = pd.Series(rank_a.values).corr(pd.Series(rank_b.values), method="spearman")

    # ----------------------------
    # Prediction comparison on the validation set
    # ----------------------------
    proba_a = model_a.predict_proba(X_val)[:, 1]
    proba_b = model_b.predict_proba(X_val_std)[:, 1]

    pearson_r = float(np.corrcoef(proba_a, proba_b)[0, 1])
    mean_abs_diff = float(np.mean(np.abs(proba_a - proba_b)))
    class_flip_mask = (proba_a >= 0.5) != (proba_b >= 0.5)
    class_flip_count = int(class_flip_mask.sum())
    class_flip_share = class_flip_count / len(proba_a)

    prediction_summary = pd.DataFrame(
        [
            {
                "spearman_rank_correlation_coefficients": spearman_rho,
                "pearson_correlation_predicted_probabilities": pearson_r,
                "mean_absolute_difference_predicted_probability": mean_abs_diff,
                "validation_customers": len(proba_a),
                "class_flip_count_at_0_50": class_flip_count,
                "class_flip_share_at_0_50": class_flip_share,
            }
        ]
    )
    prediction_summary.to_csv(OUTPUT_DIR / "standardization_prediction_summary.csv", index=False)

    # ----------------------------
    # Printed summary
    # ----------------------------
    print("\n" + "=" * 80)
    print("STANDARDIZATION CHECK - COEFFICIENT RANKING")
    print("=" * 80)
    print(f"Spearman rank correlation between the two coefficient rankings: {spearman_rho:.4f}")
    print("\nContinuous variables, rank out of", len(X_train.columns), "(1 = most important):")
    for col in CONTINUOUS_COLS:
        print(
            f"- {col}: non-standardized coef={coefs_a[col]:.5f} (rank {int(rank_a[col])})"
            f"  |  standardized coef={coefs_b[col]:.5f} (rank {int(rank_b[col])})"
        )

    print("\n" + "=" * 80)
    print("STANDARDIZATION CHECK - PREDICTED PROBABILITIES (validation set)")
    print("=" * 80)
    print(f"Pearson correlation: {pearson_r:.5f}")
    print(f"Mean absolute difference: {mean_abs_diff:.5f}")
    print(f"Class flips at threshold 0.50: {class_flip_count} / {len(proba_a)} ({100*class_flip_share:.2f}%)")

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'standardization_coefficient_comparison.csv'}")
    print(f"- {OUTPUT_DIR / 'standardization_prediction_summary.csv'}")


if __name__ == "__main__":
    run_standardization_check()
