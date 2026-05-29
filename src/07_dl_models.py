"""
Deep learning ablation — 3 additional models to complement 06_models.py.

Models added to model_results_v2.parquet:
  MLP_JKP+FinBERT    — MLP on all 165 features (JKP + FinBERT), DL equivalent
                       of the XGBoost combined model
  LSTM_FinBERT       — 2-layer LSTM on sequences of the last 8 earnings-call
                       FinBERT scores per firm (temporal cross-sectional model)
  Attention_FinBERT  — Transformer encoder (2 layers, 4 heads) on the same
                       sequences, with positional encoding and padding mask

All models: same temporal split (2008–14 train, 2015–18 val, 2019–23 test),
same evaluation (R², Rank IC), early stopping on validation Rank IC.

Run after 06_models.py. Appends new rows to model_results_v2.parquet and
saves model weights to outputs/models_v2/.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "outputs", "dataset_final_v2.parquet")
SCORES_PATH = os.path.join(ROOT, "outputs", "finbert_scores.parquet")
OUT_PRED    = os.path.join(ROOT, "outputs", "model_results_v2.parquet")
MODELS_DIR  = os.path.join(ROOT, "outputs", "models_v2")

TRAIN_END = "2014-12-31"
VAL_END   = "2018-12-31"
SEQ_LEN   = 8   # last 8 quarterly calls ≈ 2 years of history

FINBERT_COLS = [
    "p_pos", "p_neg", "p_neu",
    "net_sentiment", "sentiment_strength",
    "delta_net", "delta_pos", "delta_neg",
    "months_since_call",
    "net_sentiment_zscore", "delta_net_zscore", "delta_pos_zscore",
]
_NON_JKP    = {"PERMNO", "MthCalDt", "excess_ret", "SICCD"} | set(FINBERT_COLS)
SEQ_FEATS   = ["p_pos", "p_neg", "p_neu"]   # raw 3-dim input for temporal models


# ── helpers ────────────────────────────────────────────────────────────────

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2":      round(r2_score(y_true, y_pred), 4),
        "rank_ic": round(information_coefficient(y_true, y_pred), 4),
    }

def temporal_split(df: pd.DataFrame):
    train = df[df["MthCalDt"] <= TRAIN_END]
    val   = df[(df["MthCalDt"] > TRAIN_END) & (df["MthCalDt"] <= VAL_END)]
    test  = df[df["MthCalDt"] > VAL_END]
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ── sequence creation ──────────────────────────────────────────────────────

def create_sequences(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    feat_cols: list = SEQ_FEATS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each row in df, build a [seq_len, n_feat] array of the most recent
    `seq_len` FinBERT scores strictly before MthCalDt. Left-pad with zeros.

    Returns
    -------
    seqs  : float32 array [N, seq_len, n_feat]
    masks : bool   array [N, seq_len]  True = padded position (ignore)
    """
    scores_df = scores_df.copy()
    scores_df["permno"] = scores_df["permno"].astype(int)

    seqs  = np.zeros((len(df), seq_len, len(feat_cols)), dtype=np.float32)
    masks = np.ones((len(df), seq_len), dtype=bool)   # True = padded

    by_permno = {p: g.sort_values("call_date")
                 for p, g in scores_df.groupby("permno")}

    for i, row in enumerate(df.itertuples(index=False)):
        permno = int(row.PERMNO)
        date   = row.MthCalDt
        if permno not in by_permno:
            continue
        grp   = by_permno[permno]
        prior = grp[grp["call_date"] < date][feat_cols].values
        if len(prior) == 0:
            continue
        take = min(len(prior), seq_len)
        seqs[i, -take:]  = prior[-take:]
        masks[i, -take:] = False      # these positions are real data

    return seqs, masks


# ── model architectures ────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple = (128, 64), dropout: float = 0.3):
        super().__init__()
        layers, dim = [], n_in
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LSTMModel(nn.Module):
    """2-layer LSTM; uses last hidden state for regression."""
    def __init__(self, input_size: int = 3, hidden: int = 32,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        _, (h, _) = self.lstm(x)       # h: [num_layers, B, hidden]
        return self.fc(self.drop(h[-1])).squeeze(-1)


class AttentionModel(nn.Module):
    """
    Pre-LN Transformer encoder (2 layers, 4 heads).
    Padding mask prevents attention to zero-padded timesteps.
    Aggregated via mean pooling over non-padded positions.
    """
    def __init__(self, input_size: int = 3, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj    = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=64,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T, input_size]   mask: [B, T] True = padded
        x = self.proj(x) + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x, src_key_padding_mask=mask)
        # Mean pool over non-padded positions
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)         # [B, T, 1]
            x = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.fc(x).squeeze(-1)


# ── training ───────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    mask_tr: np.ndarray = None, mask_va: np.ndarray = None,
    is_seq: bool = False,
    epochs: int = 150, batch_size: int = 512,
    lr: float = 1e-3, patience: int = 15,
) -> tuple[nn.Module, float]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = model.to(device)

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(X_va, dtype=torch.float32).to(device)
    mva = torch.tensor(mask_va, dtype=torch.bool).to(device) if mask_va is not None else None

    if is_seq and mask_tr is not None:
        ds = TensorDataset(Xtr, ytr, torch.tensor(mask_tr, dtype=torch.bool))
    else:
        ds = TensorDataset(Xtr, ytr)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit   = nn.MSELoss()

    best_ic, best_state, no_improve = -np.inf, None, 0

    for _ in range(epochs):
        model.train()
        for batch in loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            mb = batch[2].to(device) if len(batch) == 3 else None
            opt.zero_grad()
            pred = model(xb, mb) if is_seq else model(xb)
            crit(pred, yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xva, mva).cpu().numpy() if is_seq else model(Xva).cpu().numpy()
        ic = information_coefficient(y_va, pv)
        sched.step(-ic)

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


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    mask: np.ndarray = None,
    is_seq: bool = False,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
        mb = torch.tensor(mask[i:i+batch_size], dtype=torch.bool) if (mask is not None and is_seq) else None
        preds.append((model(xb, mb) if is_seq else model(xb)).numpy())
    return np.concatenate(preds)


# ── main ───────────────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    os.makedirs(MODELS_DIR, exist_ok=True)
    dev = "MPS" if torch.backends.mps.is_available() else "CPU"
    print(f"Device: {dev}\n")

    # --- Load ---
    df     = pd.read_parquet(DATA_PATH)
    scores = pd.read_parquet(SCORES_PATH)
    df["MthCalDt"]      = pd.to_datetime(df["MthCalDt"])
    scores["call_date"] = pd.to_datetime(scores["call_date"])

    jkp_cols  = [c for c in df.columns if c not in _NON_JKP]
    all_feats = jkp_cols + FINBERT_COLS

    delta_cols = ["delta_net", "delta_pos", "delta_neg",
                  "net_sentiment_zscore", "delta_net_zscore", "delta_pos_zscore"]
    df_delta = df.dropna(subset=delta_cols).copy()
    print(f"Delta-ready rows: {len(df_delta):,}")

    train_d, val_d, test_d = temporal_split(df_delta)
    print(f"  Train {len(train_d):,} | Val {len(val_d):,} | Test {len(test_d):,}")

    all_metrics = {}
    all_preds   = []

    # ── Model A: MLP_JKP+FinBERT (165 features) ───────────────────────────
    print(f"\n[1/3] MLP_JKP+FinBERT ({len(all_feats)} features)...")
    sc1  = StandardScaler()
    X_tr = sc1.fit_transform(train_d[all_feats].values.astype(np.float32))
    X_va = sc1.transform(val_d[all_feats].values.astype(np.float32))
    X_te = sc1.transform(test_d[all_feats].values.astype(np.float32))
    y_tr = train_d["excess_ret"].values.astype(np.float32)
    y_va = val_d["excess_ret"].values.astype(np.float32)
    y_te = test_d["excess_ret"].values.astype(np.float32)

    mlp_jkp, ic_va = train_model(MLP(len(all_feats)), X_tr, y_tr, X_va, y_va)
    pte = predict(mlp_jkp, X_te)
    m   = evaluate(y_te, pte)
    print(f"  val IC={ic_va:.4f} | test R²={m['r2']:.4f} | test IC={m['rank_ic']:.4f}")
    torch.save(mlp_jkp.state_dict(), f"{MODELS_DIR}/MLP_JKP+FinBERT.pt")

    df_p = test_d[["PERMNO", "MthCalDt", "excess_ret"]].copy()
    df_p["prediction"] = pte; df_p["model"] = "MLP_JKP+FinBERT"
    all_preds.append(df_p); all_metrics["MLP_JKP+FinBERT"] = m

    # ── Sequence data (shared for LSTM + Attention) ────────────────────────
    print("\nBuilding sequences (last 8 calls per firm)...")
    seqs_tr, mask_tr = create_sequences(train_d, scores)
    seqs_va, mask_va = create_sequences(val_d,   scores)
    seqs_te, mask_te = create_sequences(test_d,  scores)

    has_tr = (~mask_tr).any(axis=1)
    has_va = (~mask_va).any(axis=1)
    has_te = (~mask_te).any(axis=1)
    print(f"  Train rows with ≥1 prior call: {has_tr.sum():,} / {len(train_d):,}")
    print(f"  Test  rows with ≥1 prior call: {has_te.sum():,} / {len(test_d):,}")

    y_tr_s = train_d["excess_ret"].values.astype(np.float32)
    y_va_s = val_d["excess_ret"].values.astype(np.float32)
    y_te_s = test_d["excess_ret"].values.astype(np.float32)

    # ── Model B: LSTM_FinBERT ──────────────────────────────────────────────
    print(f"\n[2/3] LSTM_FinBERT (seq_len={SEQ_LEN}, hidden=32, 2 layers)...")
    lstm, ic_va = train_model(
        LSTMModel(),
        seqs_tr[has_tr], y_tr_s[has_tr],
        seqs_va[has_va], y_va_s[has_va],
        mask_tr=mask_tr[has_tr], mask_va=mask_va[has_va],
        is_seq=True,
    )
    pte_lstm = predict(lstm, seqs_te[has_te], mask_te[has_te], is_seq=True)
    m        = evaluate(y_te_s[has_te], pte_lstm)
    print(f"  val IC={ic_va:.4f} | test R²={m['r2']:.4f} | test IC={m['rank_ic']:.4f}")
    torch.save(lstm.state_dict(), f"{MODELS_DIR}/LSTM_FinBERT.pt")

    df_p = test_d[has_te][["PERMNO", "MthCalDt", "excess_ret"]].copy()
    df_p["prediction"] = pte_lstm; df_p["model"] = "LSTM_FinBERT"
    all_preds.append(df_p); all_metrics["LSTM_FinBERT"] = m

    # ── Model C: Attention_FinBERT ─────────────────────────────────────────
    print(f"\n[3/3] Attention_FinBERT (d=32, 4 heads, 2 layers)...")
    attn, ic_va = train_model(
        AttentionModel(),
        seqs_tr[has_tr], y_tr_s[has_tr],
        seqs_va[has_va], y_va_s[has_va],
        mask_tr=mask_tr[has_tr], mask_va=mask_va[has_va],
        is_seq=True,
    )
    pte_attn = predict(attn, seqs_te[has_te], mask_te[has_te], is_seq=True)
    m        = evaluate(y_te_s[has_te], pte_attn)
    print(f"  val IC={ic_va:.4f} | test R²={m['r2']:.4f} | test IC={m['rank_ic']:.4f}")
    torch.save(attn.state_dict(), f"{MODELS_DIR}/Attention_FinBERT.pt")

    df_p = test_d[has_te][["PERMNO", "MthCalDt", "excess_ret"]].copy()
    df_p["prediction"] = pte_attn; df_p["model"] = "Attention_FinBERT"
    all_preds.append(df_p); all_metrics["Attention_FinBERT"] = m

    # ── Append to existing parquet ─────────────────────────────────────────
    existing = pd.read_parquet(OUT_PRED)
    new_models = list(all_metrics.keys())
    existing = existing[~existing["model"].isin(new_models)]
    updated  = pd.concat([existing] + all_preds, ignore_index=True)
    updated.to_parquet(OUT_PRED, index=False)

    print("\n=== DL ABLATION RESULTS ===")
    print(f"{'Model':<24} {'R²':>8} {'Rank IC':>10}")
    print("-" * 46)
    for name, m in all_metrics.items():
        print(f"{name:<24} {m['r2']:>8.4f} {m['rank_ic']:>10.4f}")
    print(f"\nSaved → {OUT_PRED}")
    print(f"Weights → {MODELS_DIR}/")


if __name__ == "__main__":
    main()
