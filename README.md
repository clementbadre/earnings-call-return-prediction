# ML in Finance — Earnings-Call Sentiment for Return Prediction

EPFL ML in Finance course project.

We use FinBERT sentiment scores from earnings call transcripts to predict monthly excess stock returns in a cross-sectional long-short strategy. Models include XGBoost, MLP, LSTM, and a Transformer.

## Structure

```
├── src/                      # Run scripts in numbered order
│   ├── 01_preprocess_crsp.py
│   ├── 02_preprocess_jkp.py
│   ├── 03_finbert_inference.py     # ~3h on Apple M4 MPS
│   ├── 04_merge_datasets.py
│   ├── 05_feature_engineering.py
│   ├── 06_models.py                # XGBoost models
│   ├── 07_dl_models.py             # LSTM, Attention, MLP variants
│   ├── 08_backtest.py
│   ├── 09_rolling_analysis.py
│   ├── 10_generate_figures.py
│   └── 11_preprocessing_ablation.py
├── report/                   # LaTeX source
└── outputs/                  # Parquet files, figures, saved models
```

## Data

- CRSP Monthly Stock File (WRDS, 2008–2023)
- JKP factor returns (openassetpricing.com, 153 monthly factors)
- Earnings call transcripts (34,643 calls matched by PERMNO)

Raw data files are not in the repository. The committed `outputs/*.parquet` files are sufficient to reproduce figures and tables without re-running FinBERT inference.

## Reproducing results

```bash
pip install pandas numpy xgboost torch scikit-learn transformers scipy matplotlib seaborn pyarrow

# From committed parquet files:
python src/06_models.py
python src/07_dl_models.py
python src/08_backtest.py
python src/09_rolling_analysis.py
python src/10_generate_figures.py
python src/11_preprocessing_ablation.py
```

All scripts use `SEED = 42`.
