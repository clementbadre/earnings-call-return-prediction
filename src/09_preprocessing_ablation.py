"""
Preprocessing ablation + Ridge benchmark.

Tests incremental value of each feature-engineering step on the test set.
Also runs a Ridge regression benchmark on the full FinBERT feature set.

Outputs: printed table (copy into LaTeX)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "outputs", "dataset_final_v2.parquet")
COST_ONE_WAY = 0.002

TRAIN_END = "2014-12-31"
VAL_END   = "2018-12-31"
TEST_END  = "2023-12-31"

SEED = 42
np.random.seed(SEED)

# ── Feature sets (incremental) ────────────────────────────────────────────────

FEAT_SETS = {
    "Raw levels\n(p_pos/neg/neu)": ["p_pos", "p_neg", "p_neu"],
    "Raw + freshness\n(+months_since_call)": ["p_pos", "p_neg", "p_neu", "net_sentiment", "sentiment_strength", "months_since_call"],
    "Raw + freshness\n+ delta features": ["p_pos", "p_neg", "p_neu", "net_sentiment", "sentiment_strength",
                                           "delta_net", "delta_pos", "delta_neg", "months_since_call"],
    "Full (+ z-scores)\n[current setup]": ["p_pos", "p_neg", "p_neu", "net_sentiment", "sentiment_strength",
                                             "delta_net", "delta_pos", "delta_neg", "months_since_call",
                                             "net_sentiment_zscore", "delta_net_zscore", "delta_pos_zscore"],
}

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    early_stopping_rounds=30,
    eval_metric="rmse",
    verbosity=0,
)


def temporal_split(df):
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    tr = df[df["MthCalDt"] <= TRAIN_END]
    va = df[(df["MthCalDt"] > TRAIN_END) & (df["MthCalDt"] <= VAL_END)]
    te = df[(df["MthCalDt"] > VAL_END)  & (df["MthCalDt"] <= TEST_END)]
    return tr, va, te


def monthly_rank_ic(df_te, preds):
    df_te = df_te.copy()
    df_te["pred"] = preds
    ics = []
    for _, g in df_te.groupby("MthCalDt"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g["excess_ret"], g["pred"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)), float(np.std(ics)), len(ics)


def backtest_ls_decile(df_te, preds, long_q=0.10, short_q=0.10):
    df_te = df_te.copy()
    df_te["pred"] = preds
    monthly = []
    prev_long, prev_short = set(), set()
    for date, g in df_te.groupby("MthCalDt"):
        n = len(g)
        if n < 10:
            continue
        g = g.sort_values("pred")
        n_long  = max(1, int(n * long_q))
        n_short = max(1, int(n * short_q))
        long_idx  = set(g["PERMNO"].iloc[-n_long:])
        short_idx = set(g["PERMNO"].iloc[:n_short])
        ret = g["excess_ret"].iloc[-n_long:].mean() - g["excess_ret"].iloc[:n_short].mean()

        to_l = 1.0 - len(long_idx & prev_long)  / max(len(long_idx),  1)
        to_s = 1.0 - len(short_idx & prev_short) / max(len(short_idx), 1)
        turn = (to_l + to_s) / 2
        cost = turn * COST_ONE_WAY * 2
        monthly.append({"date": date, "ret_net": ret - cost, "turn": turn})
        prev_long, prev_short = long_idx, short_idx

    m = pd.DataFrame(monthly)
    if len(m) == 0:
        return float("nan"), float("nan"), float("nan")
    net   = m["ret_net"].mean() * 12 * 100
    vol   = m["ret_net"].std()  * np.sqrt(12)
    sharpe = net / 100 / vol if vol > 0 else float("nan")
    turn  = m["turn"].mean() * 100
    return net, round(sharpe, 2), round(turn, 0)


def run_xgb(feats, tr, va, te):
    sc = StandardScaler()
    # Drop rows where any feature is NaN (delta features are NaN for first call)
    cols = feats + ["excess_ret", "PERMNO", "MthCalDt"]
    tr2  = tr[cols].dropna()
    va2  = va[cols].dropna()
    te2  = te[cols].dropna()

    X_tr = sc.fit_transform(tr2[feats].values)
    X_va = sc.transform(va2[feats].values)
    X_te = sc.transform(te2[feats].values)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, tr2["excess_ret"].values,
              eval_set=[(X_va, va2["excess_ret"].values)],
              verbose=False)
    preds = model.predict(X_te)
    return te2, preds


def run_ridge(feats, tr, va, te):
    cols = feats + ["excess_ret", "PERMNO", "MthCalDt"]
    tr2  = tr[cols].dropna()
    te2  = te[cols].dropna()

    sc = StandardScaler()
    X_tr = sc.fit_transform(tr2[feats].values)
    X_te = sc.transform(te2[feats].values)

    model = Ridge(alpha=1.0, random_state=SEED)
    model.fit(X_tr, tr2["excess_ret"].values)
    preds = model.predict(X_te)
    return te2, preds


def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    tr, va, te = temporal_split(df)
    print(f"  Train {len(tr):,} | Val {len(va):,} | Test {len(te):,}")

    results = {}

    # ── XGBoost ablation ──────────────────────────────────────────────────────
    print("\nRunning XGBoost preprocessing ablation...")
    for name, feats in FEAT_SETS.items():
        print(f"  {name.replace(chr(10),' ')}...", end=" ", flush=True)
        te2, preds = run_xgb(feats, tr, va, te)
        mean_ic, _, _ = monthly_rank_ic(te2, preds)
        net, sharpe, turn = backtest_ls_decile(te2, preds)
        results[name] = {"IC": mean_ic, "Net": net, "Sharpe": sharpe, "Turn": turn, "model": "XGBoost"}
        print(f"IC={mean_ic:.3f}  Net={net:+.1f}%  Sharpe={sharpe:.2f}  Turn={turn:.0f}%")

    # ── Ridge benchmark ───────────────────────────────────────────────────────
    print("\nRunning Ridge benchmark (full FinBERT features)...")
    full_feats = list(FEAT_SETS.values())[-1]
    te2, preds = run_ridge(full_feats, tr, va, te)
    mean_ic, _, _ = monthly_rank_ic(te2, preds)
    net, sharpe, turn = backtest_ls_decile(te2, preds)
    results["Ridge (FinBERT)"] = {"IC": mean_ic, "Net": net, "Sharpe": sharpe, "Turn": turn, "model": "Ridge"}
    print(f"  IC={mean_ic:.3f}  Net={net:+.1f}%  Sharpe={sharpe:.2f}  Turn={turn:.0f}%")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*75)
    print("PREPROCESSING ABLATION — FinBERT XGBoost (LS Decile, 2019–2023)")
    print("="*75)
    print(f"{'Feature set':45s} {'IC':>7} {'Net(%)':>8} {'Sharpe':>7} {'Turn':>6}")
    print("-"*75)
    for k, v in results.items():
        name = k.replace('\n', ' ')
        print(f"{name:45s} {v['IC']:>7.3f} {v['Net']:>8.2f} {v['Sharpe']:>7.2f} {v['Turn']:>5.0f}%")

    # ── LaTeX snippet ─────────────────────────────────────────────────────────
    print("\n\nLaTeX table rows:")
    for k, v in results.items():
        name = k.split('\n')[0]
        model = v['model']
        sign = '+' if v['Net'] > 0 else ''
        print(f"  {model:8s} & {name:42s} & ${v['IC']:.3f}$ & ${sign}{v['Net']:.2f}\\%$ & ${v['Sharpe']:.2f}$ & ${v['Turn']:.0f}\\%$ \\\\")


if __name__ == "__main__":
    main()
