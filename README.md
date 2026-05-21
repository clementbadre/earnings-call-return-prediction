# ML in Finance — Earnings-Call Sentiment for Cross-Sectional Return Prediction

**EPFL ML in Finance course project**

Does fresh, firm-level earnings-call sentiment improve cross-sectional prediction of monthly excess stock returns? We show that freshness filtering, quarter-over-quarter delta features, and cross-sectional z-scoring convert a weak raw signal into a modest but economically exploitable predictor (FinBERT-only XGBoost: Sharpe 0.79, net +11.4% annualised, 2019–2023 out-of-sample).

---

## Repository structure

```
ML_Finance/
├── README.md
├── report/                   # LaTeX source
│   ├── main.tex
│   ├── 1-introduction.tex .. 6-conclusion.tex
│   ├── A-appendix.tex
│   └── references.bib
├── src/                      # Scripts — run in numbered order
│   ├── 01_preprocess_crsp.py
│   ├── 02_preprocess_jkp.py
│   ├── 03_finbert_inference.py        # ~3h on Apple M4 MPS
│   ├── 04_merge_datasets.py
│   ├── 04b_feature_engineering.py
│   ├── 05_models.py                   # XGBoost ablation + MLP_FinBERT
│   ├── 05b_dl_models.py               # LSTM, Attention, MLP_JKP+FinBERT
│   ├── 06_backtest.py
│   ├── 07_rolling_analysis.py
│   ├── 08_generate_figures.py         # generates all PDF figures
│   └── 09_preprocessing_ablation.py   # preprocessing ablation + Ridge benchmark
└── outputs/
    ├── figures/               # PDF figures (referenced by LaTeX)
    ├── models_v2/             # Saved weights (.pt, .json)
    ├── dataset_final_v2.parquet       # Freshness-filtered, feature-engineered dataset
    ├── model_results_v2.parquet       # Predictions for all 8 models
    └── backtest_results_v2.parquet    # Monthly portfolio returns
```

---

## Data requirements

| Dataset | Source | Notes |
|---|---|---|
| CRSP Monthly Stock File | WRDS | 2008–2023 |
| JKP Factors | openassetpricing.com | 153 monthly long-short factor returns |
| Earnings call transcripts | Provided separately | 34,643 calls matched by PERMNO |

Raw data files are not in the repository (size). The intermediate `outputs/*.parquet` files are committed and sufficient to reproduce all figures and tables without re-running FinBERT inference.

---

## Installation

```bash
pip install pandas numpy xgboost torch scikit-learn transformers scipy matplotlib seaborn pyarrow
```

---

## Reproducing results

If starting from the committed `outputs/*.parquet` files:

```bash
python src/05_models.py                   # ~5 min  — XGBoost models + MLP_FinBERT
python src/05b_dl_models.py              # ~20 min  — LSTM, Attention, MLP_JKP+FinBERT
python src/06_backtest.py                # ~2 min   — portfolio backtest
python src/07_rolling_analysis.py        # ~2 min   — 108-month rolling IC
python src/08_generate_figures.py        # ~1 min   — all PDF figures
python src/09_preprocessing_ablation.py # ~3 min   — preprocessing ablation + Ridge
```

If starting from raw data, first run scripts 01–04b in order (FinBERT inference takes ~3h).

All scripts use `SEED = 42`. Results may vary slightly on GPU due to non-deterministic operations but are qualitatively reproducible.

---

## Leakage controls

- **Temporal split only** — no random shuffling at any stage.
- **Scaler** — `StandardScaler` fitted on train set only, applied to val/test.
- **JKP features** — lagged by one month (month $t-1$ predicts return at $t$).
- **Delta features** — computed from prior-quarter calls only (strictly before prediction date).
- **Cross-sectional z-scores** — computed within each prediction month from available data only.
- **Validation** — used for early stopping only; test set never seen during training or tuning.
- **Backtest** — portfolio constructed only from ex-ante predictions.
