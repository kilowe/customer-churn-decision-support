from __future__ import annotations

import pandas as pd

try:
    # If config.py is present, use centralized paths
    from config import RAW_TELCO_PATH, PROCESSED_TELCO_PATH, PROCESSED_DIR
except Exception:
    # Fallback if you don't use config.py (keeps the script executable)
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_TELCO_PATH = (
        PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_TELCO_PATH = PROCESSED_DIR / "telco_clean.csv"

# Constants (explicit to avoid mistakes / target leakage)
TARGET_COL = "Churn"
ID_COL = "customerID"
TOTAL_CHARGES_COL = "TotalCharges"
TENURE_COL = "tenure"

# Encoding decisions (explicit and interpretable)
BINARY_COLS = [
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "Partner",
    "Dependents",
    "StreamingTV",
    "StreamingMovies",
    "PhoneService",
]

NOMINAL_OHE_COLS = [
    "PaymentMethod",
    "InternetService",
    "MultipleLines",
]

CONTRACT_COL = "Contract"

CONTRACT_MAPPING = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}


def ensure_directories() -> None:
    """
    Create required directories if they do not exist.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(path=RAW_TELCO_PATH) -> pd.DataFrame:
    """
    Load the raw Telco dataset CSV without modifying it.

    Why:
    - We want a clean separation between raw data and processed data.
    - This supports reproducibility and auditability.
    """
    return pd.read_csv(path)


def print_basic_summary(df: pd.DataFrame, title: str) -> None:
    """
    Print a compact dataset summary:
    - Shape (rows, columns)
    - Dtypes
    - NaN count per column
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"Shape: {df.shape}")

    print("\nDtypes:")
    print(df.dtypes.sort_index())

    print("\nNaN count per column (descending):")
    na_counts = df.isna().sum().sort_values(ascending=False)
    if (na_counts > 0).any():
        print(na_counts[na_counts > 0])
    else:
        print("No NaN detected.")


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges from text to numeric.

    Telco dataset issue:
    - TotalCharges contains blank strings or spaces (" ").
    - It could be stored as object (text) instead of numeric.

    Approach:
    - Strip whitespace.
    - Convert to numeric with errors='coerce' so invalid values become NaN.

    Why:
    - This is "standardizing types", not "optimizing".
    - It surfaces data quality issues explicitly.
    """
    df = df.copy()

    # Normalize blank/space values
    df[TOTAL_CHARGES_COL] = df[TOTAL_CHARGES_COL].astype(str).str.strip()

    # Convert to numeric; invalid values -> NaN
    df[TOTAL_CHARGES_COL] = pd.to_numeric(df[TOTAL_CHARGES_COL], errors="coerce")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using a simple and business-justified rule.

    Business justification (Telco context):
    - Missing TotalCharges commonly occurs for new customers with tenure == 0.
    - For a brand-new customer, TotalCharges should logically be 0.

    Rule:
    1) If tenure == 0 AND TotalCharges is NaN -> set TotalCharges to 0.
    2) If any NaN values remain after that:
       - Drop those rows, but log how many were removed.

    Why:
    - We are not removing rows to "improve scores".
    - We apply a domain-consistent fix first.
    - Remaining NaNs are treated as data anomalies and removed transparently.
    """
    df = df.copy()

    # Rule 1: new customers (tenure == 0) should have TotalCharges == 0
    mask_new_customers = (df[TENURE_COL] == 0) & (df[TOTAL_CHARGES_COL].isna())
    df.loc[mask_new_customers, TOTAL_CHARGES_COL] = 0.0

    # Rule 2: drop any remaining NaNs (documented)
    remaining_na_total = int(df.isna().sum().sum())
    if remaining_na_total > 0:
        before = df.shape[0]
        df = df.dropna()
        after = df.shape[0]
        print("\n[Missing Values] NaNs remain after business rule.")
        print(
            f"Rows dropped due to remaining NaNs: {before - after} (before={before}, after={after})"
        )
    else:
        print("\n[Missing Values] No remaining NaNs after business rule.")

    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (features) and y (target) without splitting.

    Robust behavior:
    - If 'Churn' is already encoded as 0/1 (numeric), use it directly.
    - Otherwise, map 'Yes'/'No' to 1/0.

    Acceptance criteria enforced:
    - No target leakage: 'Churn' is excluded from X.
    - 'customerID' is excluded from X.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    churn_series = df[TARGET_COL]

    # Case 1: Already numeric (0/1)
    if pd.api.types.is_numeric_dtype(churn_series):
        unique_vals = set(churn_series.dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Unexpected numeric values in '{TARGET_COL}': {unique_vals}. Expected only 0/1."
            )
        y = churn_series.astype("int64")

    # Case 2: Still text (Yes/No)
    else:
        y = churn_series.map({"Yes": 1, "No": 0})
        if y.isna().any():
            bad_vals = set(churn_series.dropna().unique()) - {"Yes", "No"}
            raise ValueError(
                f"Unexpected values in '{TARGET_COL}'. Expected only 'Yes'/'No'. Found: {bad_vals}"
            )
        y = y.astype("int64")

    # Build X by removing target + ID
    drop_cols = [TARGET_COL]
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)

    X = df.drop(columns=drop_cols)

    return X, y


def basic_binary_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a minimal encoding for strictly binary Yes/No columns.

    What we do here:
    - Convert columns with unique values exactly {"Yes", "No"} into 1/0.

    What we do NOT do here (by decision):
    - One-hot encode multi-category columns (Contract, PaymentMethod, etc.)
      because this will be handled later in a proper sklearn Pipeline.

    Why:
    - Keeps preprocessing simple and safe.
    - Avoids premature feature expansion.
    - Still standardizes common binary fields early.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            uniques = set(df[col].dropna().unique())
            if uniques == {"Yes", "No"}:
                df[col] = df[col].map({"Yes": 1, "No": 0}).astype("int64")

    return df


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding (no scaling, no model, no feature selection).

    Goals:
    - Convert selected binary columns to 0/1 (interpretable).
    - One-hot encode selected nominal columns (no artificial order).
    - Encode Contract as an ordinal variable with explicit business mapping.
    - Ensure the output is fully numeric and no rows are lost.

    Notes:
    - For Telco, some "binary-like" columns contain values such as
      "No internet service" or "No phone service". Business-wise, these are still "not subscribed",
      so we map them to 0 for the binary columns in the decision list.
    """
    X_enc = X.copy()

    # ----------------------------
    # 1) Binary columns -> 0/1
    # ----------------------------
    for col in BINARY_COLS:
        if col not in X_enc.columns:
            raise ValueError(f"Expected binary column '{col}' not found in X.")

        # Map "Yes" -> 1, everything else -> 0 (covers "No", "No internet service", "No phone service")
        # This keeps the meaning: service/option present vs not present.
        X_enc[col] = (
            X_enc[col]
            .apply(lambda v: 1 if str(v).strip() == "Yes" else 0)
            .astype("int64")
        )

    # ----------------------------
    # 2) Contract -> ordinal mapping
    # ----------------------------
    if CONTRACT_COL not in X_enc.columns:
        raise ValueError(f"Expected ordinal column '{CONTRACT_COL}' not found in X.")

    X_enc[CONTRACT_COL] = X_enc[CONTRACT_COL].map(CONTRACT_MAPPING)

    # Safety check: if mapping produced NaN, it means unexpected categories exist
    if X_enc[CONTRACT_COL].isna().any():
        bad_vals = set(X[CONTRACT_COL].dropna().unique()) - set(CONTRACT_MAPPING.keys())
        raise ValueError(
            f"Unexpected values in '{CONTRACT_COL}'. Expected {set(CONTRACT_MAPPING.keys())}. Found: {bad_vals}"
        )

    X_enc[CONTRACT_COL] = X_enc[CONTRACT_COL].astype("int64")

    # ----------------------------
    # 3) Nominal columns -> One-Hot Encoding
    # ----------------------------
    for col in NOMINAL_OHE_COLS:
        if col not in X_enc.columns:
            raise ValueError(f"Expected nominal column '{col}' not found in X.")

    X_enc = pd.get_dummies(
        X_enc,
        columns=NOMINAL_OHE_COLS,
        drop_first=False,  # keep full interpretability
        dtype="int64",
    )

    # -------------------------------------------------
    # 5 Drop any remaining non-numeric columns (explicit)
    # -------------------------------------------------
    remaining_non_numeric = X_enc.select_dtypes(exclude=["number"]).columns.tolist()
    if remaining_non_numeric:
        print(
            "\n[Encoding Warning] Dropping non-encoded categorical columns to keep X fully numeric:"
        )
        print(f"- Dropped columns: {remaining_non_numeric}")
        X_enc = X_enc.drop(columns=remaining_non_numeric)

    # ----------------------------
    # Final check: all numeric
    # ----------------------------
    non_numeric = X_enc.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"X_encoded still contains non-numeric columns: {non_numeric}")

    return X_enc


def save_processed(df: pd.DataFrame, path=PROCESSED_TELCO_PATH) -> None:
    """
    Save the cleaned dataset to data/processed.

    Why:
    - Useful for EDA notebooks and future steps.
    - Provides an auditable artifact (what was changed and when).
    """
    df.to_csv(path, index=False)
    print(f"\n[Saved] Clean dataset written to: {path}")


def run_preprocessing(save_csv: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    End-to-end preprocessing runner.

    Sscope (cleaning, no optimization):
    - Load raw CSV
    - Print summary (before)
    - Convert TotalCharges to numeric
    - Handle missing values using a simple business rule
    - Save cleaned dataset (optional)
    - Build X and y (no split)

    Scope (encoding, no scaling, no model):
    - Encode selected binary columns to 0/1
    - One-hot encode selected nominal columns
    - Encode Contract with explicit ordinal mapping
    - Print shapes before/after encoding
    - Ensure no row loss
    - Return X_encoded (fully numeric) and y unchanged
    """
    ensure_directories()

    # ----------------------------
    # 1) Load raw data + pre-clean summary
    # ----------------------------
    df_raw = load_raw_data()
    print_basic_summary(df_raw, "RAW DATA SUMMARY (before cleaning)")

    # ----------------------------
    # 2) Cleaning
    # ----------------------------
    df = clean_total_charges(df_raw)
    df = handle_missing_values(df)

    # IMPORTANT:
    # We do NOT encode everything here; specifies exactly what to encode.
    # Also, we do NOT "optimize" by removing outliers or manipulating extremes.

    print_basic_summary(df, "CLEANED DATA SUMMARY (after cleaning)")

    # Explicit checks (visibility for acceptance criteria)
    if ID_COL in df.columns:
        print(
            f"\n[Check] '{ID_COL}' exists in cleaned dataset (OK for storage), but will be removed from X."
        )
    print(f"[Check] Target '{TARGET_COL}' will be excluded from X (no leakage).")

    # Save cleaned dataset artifact (optional but useful)
    if save_csv:
        save_processed(df)

    # ----------------------------
    # 3) Build X and y (no split)
    # ----------------------------
    X, y = prepare_xy(df)

    # ----------------------------
    # 4) Encoding- X only
    # ----------------------------
    print("\n" + "-" * 80)
    print("ENCODING - BEFORE")
    print("-" * 80)
    print(f"X shape (before encoding): {X.shape}")

    X_encoded = encode_features(X)

    print("\n" + "-" * 80)
    print("ENCODING - AFTER")
    print("-" * 80)
    print(f"X_encoded shape (after encoding): {X_encoded.shape}")

    # Acceptance criteria: no row loss
    if X_encoded.shape[0] != X.shape[0]:
        raise ValueError(
            f"Row loss detected during encoding: before={X.shape[0]}, after={X_encoded.shape[0]}"
        )

    # ----------------------------
    # 5) Final output summary
    # ----------------------------
    print("\n" + "-" * 80)
    print("OUTPUT SUMMARY (X_encoded, y)")
    print("-" * 80)
    print(f"X_encoded shape: {X_encoded.shape}")
    print(f"y shape: {y.shape}")

    # y must remain unchanged
    print("\ny distribution:")
    print(y.value_counts(dropna=False))

    return X_encoded, y


if __name__ == "__main__":
    run_preprocessing(save_csv=True)
