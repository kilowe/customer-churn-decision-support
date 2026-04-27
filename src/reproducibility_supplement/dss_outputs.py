from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproducibility_supplement"
TEST_PROB_PATH = OUTPUT_DIR / "test_probabilities.csv"
SELECTED_THRESHOLDS_PATH = OUTPUT_DIR / "selected_thresholds.csv"

HIGH_ACTION = "Immediate retention intervention"
MEDIUM_ACTION = "Light engagement or monitoring"
LOW_ACTION = "Monitor only"


def select_primary_threshold(selected_df: pd.DataFrame) -> float:
    """Pick balanced max_f1 threshold as primary DSS threshold when available."""
    preferred = selected_df[
        (selected_df["model"] == "balanced")
        & (selected_df["selection_type"] == "max_f1")
    ]

    if not preferred.empty:
        return float(preferred.iloc[0]["threshold"])

    fallback = selected_df[selected_df["selection_type"] == "max_f1"]
    if not fallback.empty:
        print("[Warning] Balanced max_f1 threshold missing. Using first available max_f1 threshold.")
        return float(fallback.iloc[0]["threshold"])

    print("[Warning] No max_f1 selection found. Falling back to threshold=0.50.")
    return 0.50


def assign_risk_tier(probability: float, threshold: float) -> str:
    """Assign DSS risk tiers from churn probability and selected threshold."""
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold / 2:
        return "MEDIUM"
    return "LOW"


def risk_action_from_tier(risk_tier: str) -> str:
    """Return recommended operational action for each risk tier."""
    if risk_tier == "HIGH":
        return HIGH_ACTION
    if risk_tier == "MEDIUM":
        return MEDIUM_ACTION
    return LOW_ACTION


def generate_dss_outputs() -> None:
    """Generate DSS-style risk outputs on test probabilities.

    Why:
    This file translates model outputs into simple, auditable decision-support
    tiers that can be reviewed by business stakeholders.
    """
    test_df = pd.read_csv(TEST_PROB_PATH)
    selected_df = pd.read_csv(SELECTED_THRESHOLDS_PATH)

    threshold = select_primary_threshold(selected_df)

    dss_df = pd.DataFrame(
        {
            "row_id": test_df["row_id"],
            "y_true": test_df["y_true"],
            "churn_probability": test_df["balanced_proba"],
        }
    )

    dss_df["risk_tier"] = dss_df["churn_probability"].apply(
        lambda p: assign_risk_tier(float(p), threshold)
    )
    dss_df["recommended_action"] = dss_df["risk_tier"].apply(risk_action_from_tier)

    summary_df = (
        dss_df.groupby("risk_tier", as_index=False)
        .agg(
            customer_count=("row_id", "count"),
            actual_churn_count=("y_true", "sum"),
        )
        .assign(
            actual_churn_rate=lambda df: df["actual_churn_count"] / df["customer_count"]
        )
    )

    tier_order = pd.CategoricalDtype(categories=["HIGH", "MEDIUM", "LOW"], ordered=True)
    summary_df["risk_tier"] = summary_df["risk_tier"].astype(tier_order)
    summary_df = summary_df.sort_values("risk_tier").reset_index(drop=True)

    dss_df.to_csv(OUTPUT_DIR / "DSS_Risk_Output.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "risk_tier_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("DSS RISK TIER OUTPUT SUMMARY")
    print("=" * 80)
    print(f"Primary threshold used: {threshold:.2f}")
    print("\nRisk tier distribution and churn rate:")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'DSS_Risk_Output.csv'}")
    print(f"- {OUTPUT_DIR / 'risk_tier_summary.csv'}")


if __name__ == "__main__":
    generate_dss_outputs()
