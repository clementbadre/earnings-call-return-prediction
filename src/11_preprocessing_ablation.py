"""
Preprocessing ablation + Ridge benchmark.

Uses EXACTLY the same pipeline as 06_models.py + 08_backtest.py:
  - same XGBoost hyperparameters (incl. colsample_bytree, tree_method, device)
  - same temporal split and scaling
  - same backtest via 08_backtest.run_strategy()
  - same performance_metrics()
  - "Full setup" row loads saved FinBERT_only predictions from model_results_v2.parquet
    to guarantee exact alignment with Table 3.

Rank IC reported as MONTHLY cross-sectional (mean of per-month Spearman),
which is the metric defined in the methodology section.

Outputs: printed table (copy into LaTeX) + outputs/ablation_preproc.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "outputs", "dataset_final_v2.parquet")
PRED_PATH = os.path.join(ROOT, "outputs", "model_results_v2.parquet")
CRSP_PATH = os.path.join(ROOT, "outputs", "crsp_clean.parquet")
OUT_PATH  = os.path.join(ROOT, "outputs", "ablation_preproc.parquet")

TRAIN_END    = "2014-12-31"
VAL_END      = "2018-12-31"
COST_ONE_WAY = 0.002
SEED         = 42

np.random.seed(SEED)

# ── XGBoost hyperparameters — IDENTICAL to 06_models.py ──────────────────────

XGB_PARAMS = dict(
    n_estimators       = 500,
    learning_rate      = 0.05,
    max_depth          = 4,
    subsample          = 0.8,
    colsample_bytree   = 0.8,   # critical — missing in v1 of ablation script
    reg_lambda         = 1.0,
    early_stopping_rounds = 30,
    eval_metric        = "rmse",
    tree_method        = "hist",
    device             = "cpu",
    random_state       = SEED,
    verbosity          = 0,
)

# ── Feature sets ──────────────────────────────────────────────────────────────

FEAT_SETS = {
    "Raw FinBERT levels":          ["p_pos", "p_neg", "p_neu"],
    "+ Freshness / aggregates":    ["p_pos", "p_neg", "p_neu",
                                    "net_sentiment", "sentiment_strength", "months_since_call"],
    "+ Delta features":            ["p_pos", "p_neg", "p_neu",
                                    "net_sentiment", "sentiment_strength", "months_since_call",
                                    "delta_net", "delta_pos", "delta_neg"],
    "+ Cross-sect. z-scores (full)": ["p_pos", "p_neg", "p_neu",
                                      "net_sentiment", "sentiment_strength", "months_since_call",
                                      "delta_net", "delta_pos", "delta_neg",
                                      "net_sentiment_zscore", "delta_net_zscore", "delta_pos_zscore"],
}

# ── Helpers (same as 08_backtest.py) ─────────────────────────────────────────

def compute_turnover(pos_t: set, pos_t1: set) -> float:
    if not pos_t1:
        return 0.0
    return 1.0 - len(pos_t & pos_t1) / len(pos_t1)


def run_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """LS Decile, identical to 08_backtest.run_strategy()."""
    monthly = []
    prev_long, prev_short = set(), set()
    for date, g in df.groupby("MthCalDt"):
        n = len(g)
        if n < 10:
            continue
        g = g.sort_values("prediction")
        n_long  = max(1, int(n * 0.10))
        n_short = max(1, int(n * 0.10))
        cur_long  = set(g["PERMNO"].iloc[-n_long:])
        cur_short = set(g["PERMNO"].iloc[:n_short])
        ret = g["excess_ret"].iloc[-n_long:].mean() - g["excess_ret"].iloc[:n_short].mean()
        to_l = compute_turnover(prev_long,  cur_long)
        to_s = compute_turnover(prev_short, cur_short)
        turn = (to_l + to_s) / 2
        cost = turn * COST_ONE_WAY * 2
        monthly.append({"MthCalDt": date, "ret_gross": ret,
                        "ret_net": ret - cost, "turnover": turn})
        prev_long, prev_short = cur_long, cur_short
    return pd.DataFrame(monthly)


def performance_metrics(monthly: pd.DataFrame, sprtrn: pd.Series) -> dict:
    """Same as 08_backtest.performance_metrics()."""
    rets     = monthly.set_index("MthCalDt")["ret_net"]
    ann_ret  = rets.mean() * 12 * 100
    ann_vol  = rets.std()  * np.sqrt(12)
    sharpe   = ann_ret / 100 / ann_vol if ann_vol > 0 else float("nan")
    turn     = monthly["turnover"].mean() * 100
    return {"Net": round(ann_ret, 2), "Sharpe": round(sharpe, 2), "Turn": round(turn, 0)}


def monthly_rank_ic(df: pd.DataFrame) -> float:
    """Monthly cross-sectional Rank IC (mean of per-month Spearman)."""
    ics = []
    for _, g in df.groupby("MthCalDt"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g["excess_ret"], g["prediction"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


# ── Training / prediction ─────────────────────────────────────────────────────

def temporal_split(df):
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    tr = df[df["MthCalDt"] <= TRAIN_END]
    va = df[(df["MthCalDt"] > TRAIN_END) & (df["MthCalDt"] <= VAL_END)]
    te = df[df["MthCalDt"] > VAL_END]
    return tr, va, te


def train_xgb(feats, tr, va, te):
    cols  = feats + ["excess_ret", "PERMNO", "MthCalDt"]
    tr2   = tr[cols].dropna()
    va2   = va[cols].dropna()
    te2   = te[cols].dropna()
    sc    = StandardScaler()
    X_tr  = sc.fit_transform(tr2[feats].values)
    X_va  = sc.transform(va2[feats].values)
    X_te  = sc.transform(te2[feats].values)
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, tr2["excess_ret"].values,
              eval_set=[(X_va, va2["excess_ret"].values)],
              verbose=False)
    preds = model.predict(X_te)
    df_pred = te2[["PERMNO", "MthCalDt", "excess_ret"]].copy()
    df_pred["prediction"] = preds
    return df_pred


def train_ridge(feats, tr, te):
    cols  = feats + ["excess_ret", "PERMNO", "MthCalDt"]
    tr2   = tr[cols].dropna()
    te2   = te[cols].dropna()
    sc    = StandardScaler()
    X_tr  = sc.fit_transform(tr2[feats].values)
    X_te  = sc.transform(te2[feats].values)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, tr2["excess_ret"].values)
    preds = model.predict(X_te)
    df_pred = te2[["PERMNO", "MthCalDt", "excess_ret"]].copy()
    df_pred["prediction"] = preds
    return df_pred


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    tr, va, te = temporal_split(df)
    print(f"  Train {len(tr):,}  Val {len(va):,}  Test {len(te):,}")

    # Load saved predictions for "full setup" baseline (same as Table 3)
    print("\nLoading saved FinBERT_only predictions (from model_results_v2.parquet)...")
    saved = pd.read_parquet(PRED_PATH)
    saved["MthCalDt"] = pd.to_datetime(saved["MthCalDt"])
    mask  = saved["MthCalDt"] > VAL_END
    full_pred = saved[saved["model"] == "FinBERT_only"][mask][
        ["PERMNO", "MthCalDt", "excess_ret", "prediction"]
    ].copy()
    print(f"  Loaded {len(full_pred):,} test rows for FinBERT_only.")

    results = {}

    # ── XGBoost preprocessing ablation ───────────────────────────────────────
    feat_list = list(FEAT_SETS.items())
    for name, feats in feat_list[:-1]:   # all except "full" (replaced by saved)
        print(f"\nTraining: {name}...")
        df_pred = train_xgb(feats, tr, va, te)
        ic      = monthly_rank_ic(df_pred)
        m       = run_strategy(df_pred)
        perf    = performance_metrics(m, None)
        results[name] = {"Model": "XGBoost", "IC": ic, **perf}
        print(f"  IC={ic:.3f}  Net={perf['Net']:+.2f}%  Sharpe={perf['Sharpe']:.2f}  Turn={perf['Turn']:.0f}%")

    # Full setup: use SAVED predictions for exact alignment with Table 3
    name_full = list(FEAT_SETS.keys())[-1]
    print(f"\nFull setup (from saved FinBERT_only predictions)...")
    ic_full = monthly_rank_ic(full_pred)
    m_full  = run_strategy(full_pred)
    perf_f  = performance_metrics(m_full, None)
    results[name_full] = {"Model": "XGBoost", "IC": ic_full, **perf_f}
    print(f"  IC={ic_full:.3f}  Net={perf_f['Net']:+.2f}%  Sharpe={perf_f['Sharpe']:.2f}  Turn={perf_f['Turn']:.0f}%")

    # ── Ridge benchmark ───────────────────────────────────────────────────────
    print("\nTraining Ridge benchmark (full 12 features)...")
    ridge_feats = list(FEAT_SETS.values())[-1]
    df_ridge    = train_ridge(ridge_feats, tr, te)
    ic_r        = monthly_rank_ic(df_ridge)
    m_r         = run_strategy(df_ridge)
    perf_r      = performance_metrics(m_r, None)
    results["Ridge (linear benchmark)"] = {"Model": "Ridge", "IC": ic_r, **perf_r}
    print(f"  IC={ic_r:.3f}  Net={perf_r['Net']:+.2f}%  Sharpe={perf_r['Sharpe']:.2f}  Turn={perf_r['Turn']:.0f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*78)
    print("PREPROCESSING ABLATION — XGBoost FinBERT, LS Decile, 2019–2023")
    print("Full setup row = saved FinBERT_only predictions (identical to Table 3)")
    print("="*78)
    print(f"{'Feature set':45s} {'IC':>7} {'Net%':>8} {'Sharpe':>7} {'Turn':>6}")
    print("-"*78)
    for k, v in results.items():
        print(f"{k:45s} {v['IC']:>7.3f} {v['Net']:>8.2f} {v['Sharpe']:>7.2f} {v['Turn']:>5.0f}%")

    print("\n\nLaTeX rows:")
    for k, v in results.items():
        short = k.split("(")[0].strip()
        sign  = "+" if v["Net"] > 0 else ""
        print(f"  {v['Model']:8s} & {short:42s} & ${v['IC']:.3f}$ & ${sign}{v['Net']:.2f}\\%$ & ${v['Sharpe']:.2f}$ & ${v['Turn']:.0f}\\%$ \\\\")


if __name__ == "__main__":
    main()
