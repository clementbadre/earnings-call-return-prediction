"""
Ablation study — 3 XGBoost models (v2: improved FinBERT features).

Input  : ../outputs/dataset_final_v2.parquet  (freshness-filtered, delta features, z-scores)
Output : ../outputs/model_results_v2.parquet  (predictions + metrics)
         ../outputs/models_v2/                (saved model files)

Models:
  1. JKP only          — 153 features
  2. FinBERT only       — 12 features (raw + delta + z-scores + freshness)
  3. JKP + FinBERT      — 165 features

Split (temporal, never random):
  Train      : 2008–2014
  Validation : 2015–2018  (hyperparameter tuning)
  Test       : 2019–2023  (out-of-sample evaluation)
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "outputs", "dataset_final_v2.parquet")
OUT_PRED    = os.path.join(ROOT, "outputs", "model_results_v2.parquet")
MODELS_DIR  = os.path.join(ROOT, "outputs", "models_v2")

TRAIN_END = "2014-12-31"
VAL_END   = "2018-12-31"

FINBERT_COLS = [
    "p_pos", "p_neg", "p_neu",
    "net_sentiment", "sentiment_strength",
    "delta_net", "delta_pos", "delta_neg",
    "months_since_call",
    "net_sentiment_zscore", "delta_net_zscore", "delta_pos_zscore",
]

# All columns that are not JKP factors
_NON_JKP = {"PERMNO", "MthCalDt", "excess_ret", "SICCD"} | set(FINBERT_COLS)


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple = (64, 32), dropout: float = 0.3):
        super().__init__()
        layers, dim = [], n_in
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int = 150, batch_size: int = 512,
    lr: float = 1e-3, patience: int = 15,
) -> tuple[MLP, float]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True,
    )
    Xva = torch.tensor(X_va, dtype=torch.float32).to(device)

    model = MLP(X_tr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.MSELoss()

    best_ic, best_state, no_improve = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(Xva).cpu().numpy()
        ic = information_coefficient(y_va, val_preds)

        if ic > best_ic:
            best_ic    = ic
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model.cpu(), best_ic


def run_mlp_model(
    name: str,
    features: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[MLP, pd.DataFrame, dict]:
    X_train = train[features].values.astype(np.float32)
    y_train = train["excess_ret"].values.astype(np.float32)
    X_val   = val[features].values.astype(np.float32)
    y_val   = val["excess_ret"].values.astype(np.float32)
    X_test  = test[features].values.astype(np.float32)
    y_test  = test["excess_ret"].values.astype(np.float32)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\n  Training {name}  ({len(features)} features, PyTorch MLP)...")
    model, best_val_ic = fit_mlp(X_train, y_train, X_val, y_val)
    print(f"  Best val Rank IC : {best_val_ic:.4f}")

    model.eval()
    with torch.no_grad():
        preds_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    metrics = evaluate(y_test, preds_test)
    print(f"  R²       (test) : {metrics['r2']:.4f}")
    print(f"  Rank IC  (test) : {metrics['rank_ic']:.4f}")

    pred_df = test[["PERMNO", "MthCalDt", "excess_ret"]].copy()
    pred_df["prediction"] = preds_test
    pred_df["model"]      = name
    return model, pred_df, metrics


# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    return df


def get_jkp_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_JKP]


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


def fit_weighted_model(
    X_train, y_train, X_val, y_val,
    features: list[str],
    finbert_weight: float = 10.0,
) -> xgb.Booster:
    """
    XGBoost with feature_weights: FinBERT features are finbert_weight× more
    likely to be selected at each column-sampling step than JKP features.
    Uses the low-level DMatrix API (sklearn wrapper doesn't expose feature_weights).
    """
    weights = np.array([
        finbert_weight if f in set(FINBERT_COLS) else 1.0
        for f in features
    ])
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="rmse",
        tree_method="hist",
        device="cpu",
        seed=42,
        verbosity=0,
    )
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_weights=weights,
                         feature_names=features)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_weights=weights,
                         feature_names=features)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return booster


def run_model(
    name: str,
    features: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    finbert_weight: float | None = None,
):
    X_train, y_train = train[features].values, train["excess_ret"].values
    X_val,   y_val   = val[features].values,   val["excess_ret"].values
    X_test,  y_test  = test[features].values,  test["excess_ret"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\n  Training {name}  ({len(features)} features)...")
    if finbert_weight is not None:
        model = fit_weighted_model(X_train, y_train, X_val, y_val,
                                   features, finbert_weight)
        dtest      = xgb.DMatrix(X_test, feature_names=features)
        preds_test = model.predict(dtest)
    else:
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

    # Drop rows missing delta features (first call per stock has no prior call)
    delta_cols = ["delta_net", "delta_pos", "delta_neg",
                  "net_sentiment_zscore", "delta_net_zscore", "delta_pos_zscore"]
    df_full  = df.copy()
    df_delta = df.dropna(subset=delta_cols).copy()
    print(f"\n  Full dataset  : {len(df_full):,} rows")
    print(f"  Delta-ready   : {len(df_delta):,} rows  (first call per stock removed)")

    all_preds   = []
    all_metrics = {}

    train_f,  val_f,  test_f  = temporal_split(df_full)
    train_d,  val_d,  test_d  = temporal_split(df_delta)

    all_feats = jkp_cols + FINBERT_COLS
    configs = [
        # (name, features, train, val, test, finbert_weight)
        ("JKP_only",          jkp_cols,   train_f, val_f, test_f, None),
        ("FinBERT_only",      FINBERT_COLS, train_d, val_d, test_d, None),
        ("JKP+FinBERT",       all_feats,  train_d, val_d, test_d, None),
        ("JKP+FinBERT_w10x",  all_feats,  train_d, val_d, test_d, 10.0),
    ]

    for name, features, train, val, test, fw in configs:
        model, pred_df, metrics = run_model(name, features, train, val, test, fw)
        all_preds.append(pred_df)
        all_metrics[name] = metrics
        model.save_model(f"{MODELS_DIR}/{name}.json")

    # MLP on FinBERT features (satisfies the deep-learning requirement)
    mlp_model, mlp_preds, mlp_metrics = run_mlp_model(
        "MLP_FinBERT", FINBERT_COLS, train_d, val_d, test_d
    )
    all_preds.append(mlp_preds)
    all_metrics["MLP_FinBERT"] = mlp_metrics
    torch.save(mlp_model.state_dict(), f"{MODELS_DIR}/MLP_FinBERT.pt")

    print("\n=== ABLATION STUDY RESULTS ===")
    print(f"{'Model':<24} {'R²':>8} {'Rank IC':>10}")
    print("-" * 46)
    for name, m in all_metrics.items():
        print(f"{name:<24} {m['r2']:>8.4f} {m['rank_ic']:>10.4f}")

    results = pd.concat(all_preds, ignore_index=True)
    results.to_parquet(OUT_PRED, index=False)
    print(f"\nPredictions saved → {OUT_PRED}")
    print(f"Models saved      → {MODELS_DIR}/")

    # ------------------------------------------------------------------
    # Freshness cutoff sensitivity (same JKP+FinBERT XGBoost, cutoffs 1–3)
    # ------------------------------------------------------------------
    print("\n=== FRESHNESS CUTOFF SENSITIVITY (JKP + FinBERT XGBoost) ===")
    print(f"{'Cutoff':>8} {'N_train':>10} {'N_test':>10} {'Rank IC':>10}")
    print("-" * 44)
    for cutoff in [1, 2, 3]:
        df_cut  = df_delta[df_delta["months_since_call"] <= cutoff].copy()
        tr_c, va_c, te_c = temporal_split(df_cut)
        feats        = jkp_cols + FINBERT_COLS
        scaler_fresh = StandardScaler()
        X_tr_c = scaler_fresh.fit_transform(tr_c[feats].values)
        X_va_c = scaler_fresh.transform(va_c[feats].values)
        X_te_c = scaler_fresh.transform(te_c[feats].values)
        m_cut  = fit_model(X_tr_c, tr_c["excess_ret"].values,
                           X_va_c, va_c["excess_ret"].values)
        preds_c = m_cut.predict(X_te_c)
        ic_c    = information_coefficient(te_c["excess_ret"].values, preds_c)
        print(f"{cutoff:>8} {len(tr_c):>10,} {len(te_c):>10,} {ic_c:>10.4f}")


if __name__ == "__main__":
    main()
