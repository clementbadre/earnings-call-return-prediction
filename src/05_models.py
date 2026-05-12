"""
Ablation study — 3 XGBoost models.

Input  : ../outputs/dataset_final.parquet
Output : ../outputs/model_results.parquet   (predictions + metrics)
         ../outputs/models/                 (saved model files)

Models:
  1. JKP only          — 153 features
  2. FinBERT only       — 3 features  (p_pos, p_neg, p_neu)
  3. JKP + FinBERT      — 156 features

Split (temporal, never random):
  Train      : 2008–2014
  Validation : 2015–2018  (hyperparameter tuning)
  Test       : 2019–2023  (out-of-sample evaluation)
"""

import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "outputs", "dataset_final.parquet")
OUT_PRED    = os.path.join(ROOT, "outputs", "model_results.parquet")
MODELS_DIR  = os.path.join(ROOT, "outputs", "models")

TRAIN_END = "2014-12-31"
VAL_END   = "2018-12-31"

FINBERT_COLS = ["p_pos", "p_neg", "p_neu"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    return df


def get_jkp_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"PERMNO", "MthCalDt", "excess_ret", "SICCD"} | set(FINBERT_COLS)
    return [c for c in df.columns if c not in exclude]


def temporal_split(df: pd.DataFrame):
    train = df[df["MthCalDt"] <= TRAIN_END]
    val   = df[(df["MthCalDt"] > TRAIN_END) & (df["MthCalDt"] <= VAL_END)]
    test  = df[df["MthCalDt"] > VAL_END]
    return train, val, test


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank IC — Spearman correlation between predictions and realized returns."""
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2":      round(r2_score(y_true, y_pred), 4),
        "rank_ic": round(information_coefficient(y_true, y_pred), 4),
    }


def fit_model(X_train, y_train, X_val, y_val) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric="rmse",
        tree_method="hist",
        device="cpu",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def run_model(
    name: str,
    features: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[XGBRegressor, pd.DataFrame, dict]:

    X_train, y_train = train[features].values, train["excess_ret"].values
    X_val,   y_val   = val[features].values,   val["excess_ret"].values
    X_test,  y_test  = test[features].values,  test["excess_ret"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\n  Training {name}  ({len(features)} features)...")
    model = fit_model(X_train, y_train, X_val, y_val)

    preds_test = model.predict(X_test)

    metrics = evaluate(y_test, preds_test)
    print(f"  R²       (test) : {metrics['r2']:.4f}")
    print(f"  Rank IC  (test) : {metrics['rank_ic']:.4f}")

    pred_df = test[["PERMNO", "MthCalDt", "excess_ret"]].copy()
    pred_df["prediction"] = preds_test
    pred_df["model"]      = name

    return model, pred_df, metrics


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_data()
    print(f"  Shape : {df.shape}")
    print(f"  Date range : {df['MthCalDt'].min().date()} → {df['MthCalDt'].max().date()}")

    jkp_cols = get_jkp_cols(df)
    print(f"  JKP features : {len(jkp_cols)}")

    train, val, test = temporal_split(df)
    print(f"\nSplit:")
    print(f"  Train      : {train['MthCalDt'].min().date()} → {train['MthCalDt'].max().date()}  ({len(train):,} rows)")
    print(f"  Validation : {val['MthCalDt'].min().date()} → {val['MthCalDt'].max().date()}  ({len(val):,} rows)")
    print(f"  Test       : {test['MthCalDt'].min().date()} → {test['MthCalDt'].max().date()}  ({len(test):,} rows)")

    all_preds   = []
    all_metrics = {}

    configs = [
        ("JKP_only",      jkp_cols),
        ("FinBERT_only",  FINBERT_COLS),
        ("JKP+FinBERT",   jkp_cols + FINBERT_COLS),
    ]

    for name, features in configs:
        model, pred_df, metrics = run_model(name, features, train, val, test)
        all_preds.append(pred_df)
        all_metrics[name] = metrics
        model.save_model(f"{MODELS_DIR}/{name}.json")

    print("\n=== ABLATION STUDY RESULTS ===")
    print(f"{'Model':<20} {'R²':>8} {'Rank IC':>10}")
    print("-" * 42)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['r2']:>8.4f} {m['rank_ic']:>10.4f}")

    results = pd.concat(all_preds, ignore_index=True)
    results.to_parquet(OUT_PRED, index=False)
    print(f"\nPredictions saved → {OUT_PRED}")
    print(f"Models saved      → {MODELS_DIR}/")


if __name__ == "__main__":
    main()
