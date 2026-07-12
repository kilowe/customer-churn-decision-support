from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src/ is importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import run_preprocessing


RANDOM_STATE = 42
TEST_SIZE = 0.20

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility_supplement"
DSS_RISK_OUTPUT_PATH = OUTPUT_DIR / "DSS_Risk_Output.csv"


def rebuild_test_set():
    """Rebuild X_test/y_test with the exact same split used in validation_pipeline.py.

    Why:
    - DSS_Risk_Output.csv only stores row_id, probability and risk_tier.
    - To know which original features (InternetService, Contract, TechSupport)
      correspond to each row_id, we must reproduce the same train/test split,
      with the same random_state and test_size, so that row order matches.
    """
    X_encoded, y = run_preprocessing(save_csv=False)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_encoded,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Reset index so row order becomes 0..n-1, matching row_id in DSS_Risk_Output.csv
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_test, y_test


def compute_high_tier_composition() -> None:
    """Cross-tabulate risk tiers with the key churn drivers identified in 4.5.

    Why:
    - Section 4.6 needs the real composition of the HIGH tier (fiber, month-to-month
      contract, no tech support), not an invented estimate.
    """
    if not DSS_RISK_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {DSS_RISK_OUTPUT_PATH}\n"
            "Run dss_outputs.py first to generate DSS_Risk_Output.csv."
        )

    X_test, y_test = rebuild_test_set()
    dss_df = pd.read_csv(DSS_RISK_OUTPUT_PATH)

    if len(X_test) != len(dss_df):
        raise ValueError(
            f"Row count mismatch: X_test has {len(X_test)} rows, "
            f"DSS_Risk_Output.csv has {len(dss_df)} rows. "
            "The split used here does not match the one used to build DSS_Risk_Output.csv."
        )

    merged = X_test.copy()
    merged["row_id"] = range(len(merged))
    merged["y_true_check"] = y_test.to_numpy()
    merged = merged.merge(dss_df[["row_id", "y_true", "risk_tier"]], on="row_id")

    # Sanity check: y_true from DSS output must match y_test rebuilt here.
    mismatches = int((merged["y_true_check"] != merged["y_true"]).sum())
    if mismatches > 0:
        print(
            f"[Warning] {mismatches} rows have mismatched y_true. "
            "The rebuilt split likely does not match the original one exactly."
        )
    else:
        print("[Check] y_true matches exactly: split successfully reproduced.")

    merged["is_fiber"] = merged["InternetService_Fiber optic"] == 1
    merged["is_month_to_month"] = merged["Contract"] == 0
    merged["no_tech_support"] = merged["TechSupport"] == 0
    merged["profile_all_three"] = (
        merged["is_fiber"] & merged["is_month_to_month"] & merged["no_tech_support"]
    )

    rows: list[dict] = []
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        sub = merged[merged["risk_tier"] == tier]
        n = len(sub)
        rows.append(
            {
                "risk_tier": tier,
                "customer_count": n,
                "pct_fiber": round(100 * sub["is_fiber"].mean(), 2) if n else None,
                "pct_month_to_month": round(100 * sub["is_month_to_month"].mean(), 2)
                if n
                else None,
                "pct_no_tech_support": round(100 * sub["no_tech_support"].mean(), 2)
                if n
                else None,
                "pct_all_three_combined": round(
                    100 * sub["profile_all_three"].mean(), 2
                )
                if n
                else None,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / "risk_tier_composition.csv", index=False)

    print("\n" + "=" * 80)
    print("RISK TIER COMPOSITION (test set)")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'risk_tier_composition.csv'}")


if __name__ == "__main__":
    compute_high_tier_composition()
