"""
simulate_raw_data.py
---------------------
The public EasyVisa dataset (OFLC visa applications) is already very clean.
To build and demonstrate a genuine data-cleaning step (as opposed to a no-op),
this script simulates the kind of messiness that shows up when data is
scraped / exported from a real case-management system: duplicate rows and
missing values scattered across columns.

This is documented here on purpose -- when you talk about this project,
be upfront that the *source* data was clean and you intentionally
introduced realistic noise to build + prove out a real cleaning pipeline.

Output: data/EasyVisa_raw.csv (the "as ingested" file that data_cleaning.py
consumes)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SRC = "/home/claude/VisaPredict/data/EasyVisa.csv"
OUT = "/home/claude/VisaPredict/data/EasyVisa_raw.csv"

def main():
    df = pd.read_csv(SRC)
    n = len(df)
    print(f"Loaded clean source: {n} rows, {df.shape[1]} columns")

    # 1) Duplicate ~8% of rows (exact duplicates, as if a batch export ran twice)
    dup_frac = 0.08
    dup_idx = RNG.choice(df.index, size=int(n * dup_frac), replace=False)
    duplicates = df.loc[dup_idx]
    messy = pd.concat([df, duplicates], ignore_index=True)
    print(f"Injected {len(duplicates)} duplicate rows -> {len(messy)} rows")

    # 2) Scatter missing values (NaN) across several columns, at different rates
    # simulating optional/late-filled fields in a real intake form
    missing_rates = {
        "requires_job_training": 0.04,
        "no_of_employees": 0.03,
        "prevailing_wage": 0.05,
        "region_of_employment": 0.02,
        "has_job_experience": 0.03,
        "yr_of_estab": 0.02,
    }
    total_cells = 0
    for col, rate in missing_rates.items():
        idx = RNG.choice(messy.index, size=int(len(messy) * rate), replace=False)
        messy.loc[idx, col] = np.nan
        total_cells += len(idx)
    print(f"Injected {total_cells} missing values across {len(missing_rates)} columns")

    # 3) A few inconsistent string values (whitespace / casing) to test cleaning robustness
    cap_idx = RNG.choice(messy.index, size=int(len(messy) * 0.01), replace=False)
    messy.loc[cap_idx, "continent"] = messy.loc[cap_idx, "continent"].astype(str).str.upper()

    messy = messy.sample(frac=1.0, random_state=42).reset_index(drop=True)
    messy.to_csv(OUT, index=False)

    before_issue_rows = messy.duplicated().sum() + messy.isna().any(axis=1).sum()
    print(f"\nSaved raw/messy dataset -> {OUT}")
    print(f"Rows with an issue (duplicate or missing field): {before_issue_rows} "
          f"({before_issue_rows / len(messy):.1%} of {len(messy)} rows)")

if __name__ == "__main__":
    main()
