"""
Rolling window robustness analysis — FinBERT_only signal.

Extends predictions to the full 2015-2023 post-training window using the
same model-selection protocol as the main pipeline: train on 2008-2014,
early stop on 2015-2018, then report the complete post-training IC path.

Outputs:
  1. Month-by-month Rank IC for each model
  2. Rolling 12-month annualized net return (LS_decile)
  3. % of months with positive Rank IC
  4. Walk-forward stability table

This directly addresses the concern that FinBERT performance is
period-specific (e.g. driven by COVID 2020 only).
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "outputs", "dataset_final_v2.parquet")
OUT_PATH  = os.path.join(ROOT, "outputs", "rolling_analysis.parquet")

TRAIN_END = "2014-12-31"
VAL_END   = "2018-12-31"
COST      = 0.002          # 0.2% one-way

FINBERT_COLS = [
    "p_pos", "p_neg", "p_neu",
    "net_sentiment", "sentiment_strength",
    "delta_net", "delta_pos", "delta_neg",
    "months_since_call",
    "net_sentiment_zscore", "delta_net_zscore", "delta_pos_zscore",
]

DELTA_COLS = ["delta_net", "delta_pos", "delta_neg",
              "net_sentiment_zscore", "delta_net_zscore", "delta_pos_zscore"]

_NON_JKP = {"PERMNO", "MthCalDt", "excess_ret", "SICCD"} | set(FINBERT_COLS)


def load_and_split(features: list[str]):
    df = pd.read_parquet(DATA_PATH)
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    df = df.dropna(subset=DELTA_COLS).copy()

    train = df[df["MthCalDt"] <= TRAIN_END]
    # val is used only for early stopping, mirroring 05_models.py.
    val   = df[(df["MthCalDt"] > TRAIN_END) & (df["MthCalDt"] <= VAL_END)]
    # Evaluated path includes validation + test months. We label it
    # "post-training" rather than pure test because 2015-2018 monitors
    # early stopping.
    oos   = df[df["MthCalDt"] > TRAIN_END]

    scaler  = StandardScaler().fit(train[features].values)
    return (scaler.transform(train[features].values), train["excess_ret"].values,
            scaler.transform(val[features].values),   val["excess_ret"].values,
            scaler.transform(oos[features].values),   oos[["PERMNO", "MthCalDt", "excess_ret"]])


def train_model(X_train, y_train, X_val, y_val, features: list[str],
                finbert_weight: float | None = None) -> xgb.Booster | XGBRegressor:
    if finbert_weight is None:
        m = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            early_stopping_rounds=30, eval_metric="rmse",
            tree_method="hist", device="cpu", random_state=42, verbosity=0,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return m
    weights = np.array([
        finbert_weight if f in set(FINBERT_COLS) else 1.0 for f in features
    ])
    params = dict(n_estimators=500, learning_rate=0.05, max_depth=4,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  eval_metric="rmse", tree_method="hist", device="cpu",
                  seed=42, verbosity=0)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_weights=weights,
                         feature_names=features)
    dval   = xgb.DMatrix(X_val, label=y_val, feature_weights=weights,
                         feature_names=features)
    return xgb.train(params, dtrain, num_boost_round=500,
                     evals=[(dval, "val")], early_stopping_rounds=30,
                     verbose_eval=False)


def predict(model, X, features):
    if isinstance(model, XGBRegressor):
        return model.predict(X)
    return model.predict(xgb.DMatrix(X, feature_names=features))


def monthly_rank_ic(preds_df: pd.DataFrame) -> pd.Series:
    def _ic(g):
        if len(g) < 5:
            return np.nan
        r, _ = spearmanr(g["excess_ret"], g["prediction"])
        return r
    return preds_df.groupby("MthCalDt").apply(_ic).rename("rank_ic")


def rolling_net_return(preds_df: pd.DataFrame, window: int = 12) -> pd.Series:
    """Rolling annualized net return for LS_decile strategy."""
    monthly = []
    prev_long: set = set()

    for date, g in preds_df.sort_values("MthCalDt").groupby("MthCalDt"):
        n = len(g)
        if n < 10:
            continue
        g = g.sort_values("prediction")
        n_long  = max(1, int(n * 0.10))
        n_short = max(1, int(n * 0.10))
        long_ret  = g["excess_ret"].iloc[-n_long:].mean()
        short_ret = g["excess_ret"].iloc[:n_short].mean()
        gross = long_ret - short_ret
        cur_long = set(g["PERMNO"].iloc[-n_long:])
        unchanged = len(prev_long & cur_long)
        turnover  = 1.0 - unchanged / len(cur_long) if cur_long else 0.0
        net = gross - turnover * COST * 2
        monthly.append({"MthCalDt": date, "net": net})
        prev_long = cur_long

    s = pd.DataFrame(monthly).set_index("MthCalDt")["net"]
    roll = s.rolling(window).apply(lambda x: (1 + x).prod() ** (12 / window) - 1)
    return roll


def print_stats(label: str, ic: pd.Series, roll: pd.Series):
    print(f"\n--- {label} ---")
    print(f"  Months evaluated     : {ic.notna().sum()}")
    print(f"  Mean monthly Rank IC : {ic.mean():.4f}")
    print(f"  Median Rank IC       : {ic.median():.4f}")
    print(f"  % months IC > 0      : {(ic > 0).mean()*100:.1f}%")
    print(f"  IC t-stat            : {ic.mean() / (ic.std() / ic.notna().sum()**0.5):.2f}")
    if roll.notna().sum() > 0:
        print(f"  Roll-12M net (mean)  : {roll.mean()*100:.2f}%")
        print(f"  Roll-12M net (min)   : {roll.min()*100:.2f}%")
        print(f"  Roll-12M net (max)   : {roll.max()*100:.2f}%")
        print(f"  % rolling windows>0  : {(roll > 0).mean()*100:.1f}%")


def yearly_ic(ic: pd.Series) -> pd.DataFrame:
    df = ic.reset_index()
    df["year"] = df["MthCalDt"].dt.year
    return df.groupby("year")["rank_ic"].agg(["mean", "median",
        lambda x: (x > 0).mean()]).rename(columns={
        "mean": "Mean IC", "median": "Median IC", "<lambda_0>": "% IC>0"
    }).round(4)


def main():
    df_all = pd.read_parquet(DATA_PATH)
    df_all["MthCalDt"] = pd.to_datetime(df_all["MthCalDt"])
    jkp_cols = [c for c in df_all.columns if c not in _NON_JKP]

    configs = [
        ("FinBERT_only",     FINBERT_COLS,              None),
        ("JKP_only",         jkp_cols,                  None),
        ("JKP+FinBERT",      jkp_cols + FINBERT_COLS,   None),
        ("JKP+FinBERT_w10x", jkp_cols + FINBERT_COLS,   10.0),
    ]

    all_ic   = {}
    all_roll = {}

    for name, features, fw in configs:
        print(f"\nTraining {name} ({len(features)} features)...")
        X_tr, y_tr, X_val, y_val, X_oos, meta = load_and_split(features)
        model = train_model(X_tr, y_tr, X_val, y_val, features, fw)
        preds = predict(model, X_oos, features)

        pred_df = meta.copy()
        pred_df["prediction"] = preds

        ic   = monthly_rank_ic(pred_df)
        roll = rolling_net_return(pred_df)

        all_ic[name]   = ic
        all_roll[name] = roll

        print_stats(name, ic, roll)
        print(f"\n  Year-by-year Rank IC:")
        print(yearly_ic(ic).to_string())

    # Summary comparison table
    print("\n\n=== SUMMARY — % months with positive Rank IC (2015-2023) ===")
    print(f"{'Model':<25} {'Mean IC':>9} {'% IC>0':>8} {'IC t-stat':>10}")
    print("-" * 58)
    for name, ic in all_ic.items():
        tstat = ic.mean() / (ic.std() / ic.notna().sum()**0.5)
        print(f"{name:<25} {ic.mean():>9.4f} {(ic>0).mean()*100:>7.1f}% {tstat:>10.2f}")

    # Save
    out = pd.concat([
        s.rename(name).to_frame() for name, s in all_ic.items()
    ], axis=1)
    out.to_parquet(OUT_PATH)
    print(f"\nRolling IC saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
