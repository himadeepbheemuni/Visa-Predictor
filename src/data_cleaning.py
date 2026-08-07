"""
data_cleaning.py
-----------------
Loads the raw ingested visa data, cleans it (removes duplicates, handles
missing values, standardizes text fields, fixes obvious data-entry issues),
and writes a clean file ready for modeling.

Run directly: python src/data_cleaning.py
"""
import numpy as np
import pandas as pd

RAW_PATH = "/home/claude/VisaPredict/data/EasyVisa_raw.csv"
CLEAN_PATH = "/home/claude/VisaPredict/data/EasyVisa_clean.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the dataframe and return (clean_df, stats) for reporting."""
    stats = {"start_rows": len(df)}

    # --- 1. Standardize text fields (strip whitespace, fix casing) ---
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
    if "continent" in df.columns:
        df["continent"] = df["continent"].str.title()

    # --- 2. Drop exact duplicate rows ---
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    stats["duplicates_removed"] = int(dup_count)

    # --- 3. Handle missing values column by column ---
    missing_before = df.isna().sum().sum()

    # Categorical columns: impute with the mode (most frequent category)
    cat_cols = ["has_job_experience", "requires_job_training", "region_of_employment"]
    for col in cat_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(dropna=True).iloc[0]
            df[col] = df[col].fillna(mode_val)

    # Numeric columns: impute with the median (robust to outliers/skew)
    num_cols = ["no_of_employees", "prevailing_wage", "yr_of_estab"]
    for col in num_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    missing_after = df.isna().sum().sum()
    stats["missing_values_imputed"] = int(missing_before - missing_after)

    # --- 4. Fix logically impossible values ---
    # no_of_employees shouldn't be negative (data-entry sign errors)
    if "no_of_employees" in df.columns:
        neg_count = (df["no_of_employees"] < 0).sum()
        df["no_of_employees"] = df["no_of_employees"].abs()
        stats["negative_employee_counts_fixed"] = int(neg_count)

    stats["end_rows"] = len(df)
    stats["rows_removed_pct"] = round(
        (stats["start_rows"] - stats["end_rows"]) / stats["start_rows"] * 100, 2
    )
    total_issues_before = stats["duplicates_removed"] + missing_before
    stats["total_issues_resolved"] = int(total_issues_before)

    return df, stats


def main():
    df_raw = load_raw()
    print(f"Raw data: {len(df_raw)} rows, {df_raw.shape[1]} columns")
    print(f"  Duplicate rows: {df_raw.duplicated().sum()}")
    print(f"  Missing cells:  {df_raw.isna().sum().sum()}")

    df_clean, stats = clean(df_raw)

    df_clean.to_csv(CLEAN_PATH, index=False)

    print("\n--- Cleaning Summary ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    pct_row_issue_before = (
        (df_raw.duplicated().sum() + df_raw.isna().any(axis=1).sum()) / len(df_raw) * 100
    )
    pct_row_issue_after = (
        (df_clean.duplicated().sum() + df_clean.isna().any(axis=1).sum()) / len(df_clean) * 100
    )
    print(f"\nRows with a data-quality issue: {pct_row_issue_before:.1f}% -> {pct_row_issue_after:.1f}%")
    print(f"Saved clean dataset -> {CLEAN_PATH}")


if __name__ == "__main__":
    main()
