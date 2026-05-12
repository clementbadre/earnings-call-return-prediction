"""
Merge CRSP + JKP + FinBERT scores into final dataset.

Inputs  : ../outputs/crsp_clean.parquet
          ../outputs/jkp_wide.parquet
          ../outputs/finbert_scores.parquet
Output  : ../outputs/dataset_final.parquet

Merge logic:
  1. CRSP (base)   LEFT JOIN JKP on MthCalDt = date
     → adds 153 factor columns (same value for all stocks in a given month)
  2. Result        LEFT JOIN FinBERT on (PERMNO, call_date)
     → merge_asof per PERMNO : most recent call STRICTLY before MthCalDt
     → strict anti look-ahead: call_date < MthCalDt (not <=)
  3. Drop rows with no FinBERT score (stock never had an earnings call)
"""

import os
import pandas as pd
import numpy as np

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRSP_PATH    = os.path.join(ROOT, "outputs", "crsp_clean.parquet")
JKP_PATH     = os.path.join(ROOT, "outputs", "jkp_wide.parquet")
SCORES_PATH  = os.path.join(ROOT, "outputs", "finbert_scores.parquet")
OUT_PATH     = os.path.join(ROOT, "outputs", "dataset_final.parquet")


def load_data():
    crsp   = pd.read_parquet(CRSP_PATH)
    jkp    = pd.read_parquet(JKP_PATH)
    scores = pd.read_parquet(SCORES_PATH)
    return crsp, jkp, scores


def merge_jkp(crsp: pd.DataFrame, jkp: pd.DataFrame) -> pd.DataFrame:
    """
    Left join JKP on year-month, with 1-month lag.

    JKP factor return for date t = return from t-1 to t (same period as CRSP MthRet).
    To avoid look-ahead bias, we use factor t-1 to predict CRSP return t.
    Concretely: shift JKP dates forward by 1 month before merging.
    """
    jkp = jkp.reset_index()
    crsp = crsp.copy()
    crsp["_ym"] = crsp["MthCalDt"].dt.to_period("M")
    # shift JKP forward 1 month: factor observed at t becomes available to predict t+1
    jkp["_ym"]  = (jkp["date"] + pd.DateOffset(months=1)).dt.to_period("M")
    jkp = jkp.drop(columns="date")
    merged = crsp.merge(jkp, on="_ym", how="left")
    merged = merged.drop(columns="_ym")
    return merged


def merge_finbert(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    For each (PERMNO, MthCalDt), attach the most recent FinBERT score
    where call_date < MthCalDt (strict look-ahead prevention).
    merge_asof requires left sorted by key column (MthCalDt) globally.
    """
    df     = df.sort_values("MthCalDt")
    scores = scores.rename(columns={"permno": "PERMNO"})
    scores = scores.sort_values("call_date")

    merged = pd.merge_asof(
        df,
        scores[["PERMNO", "call_date", "p_pos", "p_neg", "p_neu"]],
        left_on="MthCalDt",
        right_on="call_date",
        by="PERMNO",
        direction="backward",
        allow_exact_matches=False,  # strict: call_date must be < MthCalDt
    )
    return merged


def main():
    print("Loading datasets...")
    crsp, jkp, scores = load_data()
    print(f"  CRSP   : {crsp.shape}")
    print(f"  JKP    : {jkp.shape}")
    print(f"  Scores : {scores.shape}")

    print("\nMerging JKP...")
    df = merge_jkp(crsp, jkp)
    first_jkp_col = jkp.columns[0]
    jkp_nan = df[first_jkp_col].isnull().sum()
    print(f"  Shape after JKP merge : {df.shape}")
    print(f"  JKP NaN check (first factor '{first_jkp_col}') : {jkp_nan}")

    print("\nMerging FinBERT scores (merge_asof, strict look-ahead prevention)...")
    df = merge_finbert(df, scores)
    print(f"  Shape after FinBERT merge : {df.shape}")

    n_with_scores = df["p_pos"].notna().sum()
    n_total       = len(df)
    print(f"  Rows with FinBERT score : {n_with_scores} / {n_total} ({n_with_scores/n_total*100:.1f}%)")

    df = df.dropna(subset=["p_pos"])
    print(f"  Shape after dropping rows without score : {df.shape}")
    print(f"  Unique PERMNO : {df['PERMNO'].nunique()}")
    print(f"  Date range    : {df['MthCalDt'].min().date()} → {df['MthCalDt'].max().date()}")

    cols_to_drop = ["call_date", "sprtrn"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.reset_index(drop=True)

    print(f"\nFinal dataset shape : {df.shape}")
    print(f"Columns : PERMNO, MthCalDt, excess_ret, SICCD, [153 JKP], p_pos, p_neg, p_neu")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
