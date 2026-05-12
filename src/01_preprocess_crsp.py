"""
CRSP monthly returns preprocessing.

Input  : ../monthly_crsp.csv
Output : ../outputs/crsp_clean.parquet

Steps:
  1. Keep only useful columns
  2. Filter 2008-2023
  3. Drop rows with missing MthRet
  4. Winsorise MthRet at [1%, 99%]
  5. Compute excess return = MthRet - sprtrn
"""

import os
import pandas as pd
import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "monthly_crsp.csv")
OUT_PATH = os.path.join(ROOT, "outputs", "crsp_clean.parquet")

COLS = ["PERMNO", "MthCalDt", "MthRet", "sprtrn", "SICCD"]
# MthRet is kept only to compute excess_ret, then dropped before saving
START = "2008-01-01"
END   = "2023-12-31"


def load_crsp(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=COLS,
        parse_dates=["MthCalDt"],
    )
    return df


def filter_period(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["MthCalDt"] >= START) & (df["MthCalDt"] <= END)
    return df.loc[mask].copy()


def clean_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["MthRet", "sprtrn"]).copy()

    p1  = df["MthRet"].quantile(0.01)
    p99 = df["MthRet"].quantile(0.99)
    df["MthRet"]     = df["MthRet"].clip(p1, p99)
    df["excess_ret"] = df["MthRet"] - df["sprtrn"]
    return df


def main():
    print("Loading CRSP...")
    df = load_crsp(RAW_PATH)
    print(f"  Raw shape : {df.shape}")

    df = filter_period(df)
    print(f"  After period filter (2008-2023) : {df.shape}")

    df = clean_returns(df)
    print(f"  After cleaning : {df.shape}")

    print(f"\nStats excess_ret:")
    print(df["excess_ret"].describe().round(4))
    print(f"  % positive : {(df['excess_ret'] > 0).mean()*100:.1f}%")
    print(f"  Unique PERMNO : {df['PERMNO'].nunique()}")

    df = df.drop(columns=["MthRet"])  # keep only excess_ret, not raw return
    df = df.sort_values(["PERMNO", "MthCalDt"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
