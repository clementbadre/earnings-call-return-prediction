"""
FinBERT feature engineering — improves raw sentiment scores.

Input  : ../outputs/finbert_scores.parquet
Output : ../outputs/finbert_features.parquet

New features:
  net_sentiment     = p_pos - p_neg              (composite directional score)
  sentiment_strength= 1 - p_neu                  (how non-neutral the call is)
  delta_net         = net_sentiment - previous    (quarter-over-quarter change)
  delta_pos         = p_pos - previous p_pos      (positive sentiment change)
  months_since_call = months between call and prediction date (freshness)

Cross-sectional normalization (within each prediction month):
  net_sentiment_zscore  = z-score of net_sentiment across all stocks in that month
  delta_net_zscore      = z-score of delta_net across all stocks in that month

Freshness filter:
  Only rows with months_since_call <= 3 are kept (fresh signal only).
  Beyond 3 months, the sentiment signal is empirically shown to be noise.
"""

import os
import pandas as pd
import numpy as np

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_PATH  = os.path.join(ROOT, "outputs", "finbert_scores.parquet")
CRSP_PATH    = os.path.join(ROOT, "outputs", "crsp_clean.parquet")
JKP_PATH     = os.path.join(ROOT, "outputs", "jkp_wide.parquet")
OUT_PATH     = os.path.join(ROOT, "outputs", "dataset_final_v2.parquet")

FRESHNESS_MAX = 3   # max months between call and prediction


def load_data():
    scores = pd.read_parquet(SCORES_PATH)
    crsp   = pd.read_parquet(CRSP_PATH)
    jkp    = pd.read_parquet(JKP_PATH)
    scores["call_date"] = pd.to_datetime(scores["call_date"])
    crsp["MthCalDt"]    = pd.to_datetime(crsp["MthCalDt"])
    return scores, crsp, jkp


def engineer_call_features(scores: pd.DataFrame) -> pd.DataFrame:
    """Add composite and delta features at the call level."""
    scores = scores.sort_values(["permno", "call_date"]).copy()

    scores["net_sentiment"]      = scores["p_pos"] - scores["p_neg"]
    scores["sentiment_strength"] = 1.0 - scores["p_neu"]

    # Quarter-over-quarter change (previous call for same stock)
    prev = scores.groupby("permno")[["net_sentiment", "p_pos", "p_neg"]].shift(1)
    scores["delta_net"] = scores["net_sentiment"] - prev["net_sentiment"]
    scores["delta_pos"] = scores["p_pos"] - prev["p_pos"]
    scores["delta_neg"] = scores["p_neg"] - prev["p_neg"]

    return scores


def merge_jkp(crsp: pd.DataFrame, jkp: pd.DataFrame) -> pd.DataFrame:
    """Left join JKP with 1-month lag on year-month period."""
    jkp = jkp.reset_index()
    crsp = crsp.copy()
    crsp["_ym"] = crsp["MthCalDt"].dt.to_period("M")
    jkp["_ym"]  = (jkp["date"] + pd.DateOffset(months=1)).dt.to_period("M")
    jkp = jkp.drop(columns="date")
    merged = crsp.merge(jkp, on="_ym", how="left")
    merged = merged.drop(columns="_ym")
    return merged


def merge_finbert(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Merge most recent call (strict look-ahead) per stock per month.
    Adds freshness and drops stale signals.
    """
    df     = df.sort_values("MthCalDt")
    scores = scores.rename(columns={"permno": "PERMNO"}).sort_values("call_date")

    feat_cols = ["PERMNO", "call_date",
                 "p_pos", "p_neg", "p_neu",
                 "net_sentiment", "sentiment_strength",
                 "delta_net", "delta_pos", "delta_neg"]

    merged = pd.merge_asof(
        df,
        scores[feat_cols],
        left_on="MthCalDt",
        right_on="call_date",
        by="PERMNO",
        direction="backward",
        allow_exact_matches=False,
    )

    # Freshness in months
    merged["months_since_call"] = (
        (merged["MthCalDt"] - merged["call_date"]) / pd.Timedelta(days=30.5)
    ).round()

    # Apply freshness filter
    before = len(merged)
    merged = merged[merged["months_since_call"] <= FRESHNESS_MAX]
    print(f"  Freshness filter (≤{FRESHNESS_MAX} months): {before:,} → {len(merged):,} rows")

    return merged


def crosssectional_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score each feature within each prediction month (cross-sectional normalization)."""
    df = df.copy()
    for col in cols:
        grp = df.groupby("MthCalDt")[col]
        df[f"{col}_zscore"] = (df[col] - grp.transform("mean")) / (grp.transform("std") + 1e-8)
    return df


def main():
    print("Loading data...")
    scores, crsp, jkp = load_data()
    print(f"  Scores : {scores.shape}")
    print(f"  CRSP   : {crsp.shape}")
    print(f"  JKP    : {jkp.shape}")

    print("\nEngineering call-level features...")
    scores = engineer_call_features(scores)

    print("\nMerging JKP (1-month lag)...")
    df = merge_jkp(crsp, jkp)

    print("\nMerging FinBERT features (strict look-ahead + freshness filter)...")
    df = merge_finbert(df, scores)

    print("\nAdding cross-sectional z-scores...")
    df = crosssectional_zscore(df, ["net_sentiment", "delta_net", "delta_pos"])

    # Drop columns not needed for modeling
    df = df.drop(columns=[c for c in ["call_date", "sprtrn", "MthRet"] if c in df.columns])
    df = df.reset_index(drop=True)

    print(f"\n=== FINAL DATASET ===")
    print(f"Shape        : {df.shape}")
    print(f"Unique PERMNO: {df['PERMNO'].nunique()}")
    print(f"Date range   : {df['MthCalDt'].min().date()} → {df['MthCalDt'].max().date()}")
    print(f"NaN p_pos    : {df['p_pos'].isnull().sum()}")
    print(f"NaN delta_net: {df['delta_net'].isnull().sum()} (NaN = first call for that stock)")
    print(f"\nNew FinBERT features: {[c for c in df.columns if any(k in c for k in ['sentiment','delta','p_pos','p_neg','p_neu','months','zscore'])]}")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
