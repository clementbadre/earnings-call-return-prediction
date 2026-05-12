"""
JKP Factors preprocessing.

Input  : ../[usa]_[all_factors]_[monthly]_[vw_cap].csv
Output : ../outputs/jkp_wide.parquet

Steps:
  1. Filter 2008-2023
  2. Keep columns [date, name, ret] only (direction ignored — ret already sign-corrected)
  3. Pivot long → wide : index=date, columns=factor_name, values=ret
  4. Verify 0% NaN (expected from prior analysis)
"""

import os
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "[usa]_[all_factors]_[monthly]_[vw_cap].csv")
OUT_PATH = os.path.join(ROOT, "outputs", "jkp_wide.parquet")

START = "2008-01-01"
END   = "2023-12-31"


def load_jkp(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["date", "name", "ret"],
        parse_dates=["date"],
    )
    return df


def filter_period(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["date"] >= START) & (df["date"] <= END)
    return df[mask].copy()


def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="date", columns="name", values="ret")
    wide.columns.name = None
    wide = wide.sort_index()
    return wide


def main():
    print("Loading JKP...")
    df = load_jkp(RAW_PATH)
    print(f"  Raw shape (long) : {df.shape}")

    df = filter_period(df)
    print(f"  After period filter (2008-2023) : {df.shape}")
    print(f"  Unique dates   : {df['date'].nunique()}")
    print(f"  Unique factors : {df['name'].nunique()}")

    wide = pivot_wide(df)
    print(f"\n  Wide shape : {wide.shape}  (dates × factors)")

    nan_total = wide.isnull().sum().sum()
    print(f"  Total NaN  : {nan_total}  (expected 0)")

    print(f"\nDate range : {wide.index.min().date()} → {wide.index.max().date()}")
    print(f"Factors    : {wide.columns.tolist()[:5]} ... ({wide.shape[1]} total)")

    wide.to_parquet(OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
