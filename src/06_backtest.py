"""
Investment strategy backtesting.

Input  : ../outputs/model_results.parquet  (predictions from 05_models.py)
         ../outputs/crsp_clean.parquet     (for sprtrn benchmark)
Output : ../outputs/backtest_results.parquet
         Printed performance table

Strategies (applied to best model predictions, per month):
  A. Long-short decile   — long top 10%, short bottom 10%, equal-weighted
  B. Long-short quintile — long top 20%, short bottom 20%, equal-weighted
  C. Long-only decile    — long top 10%, equal-weighted
  D. Long-short decile   — value-weighted (placeholder: equal-weighted here)

Transaction costs: 0.2% one-way (0.4% round-trip) applied on turnover.

Metrics per strategy:
  - Annualized return
  - Annualized Sharpe ratio
  - Maximum drawdown
  - Monthly alpha vs S&P500 (OLS)
  - Average monthly turnover
  - Net return after transaction costs
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_PATH = os.path.join(ROOT, "outputs", "model_results_v2.parquet")
CRSP_PATH = os.path.join(ROOT, "outputs", "crsp_clean.parquet")
OUT_PATH  = os.path.join(ROOT, "outputs", "backtest_results_v2.parquet")

COST_ONE_WAY = 0.002  # 0.2% per leg


def load_data():
    preds = pd.read_parquet(PRED_PATH)
    crsp  = pd.read_parquet(CRSP_PATH)[["MthCalDt", "sprtrn"]].drop_duplicates("MthCalDt")
    preds["MthCalDt"] = pd.to_datetime(preds["MthCalDt"])
    crsp["MthCalDt"]  = pd.to_datetime(crsp["MthCalDt"])
    return preds, crsp


def build_portfolio(
    group: pd.DataFrame,
    long_q: float,
    short_q: float | None,
) -> float:
    """Return equal-weighted long (minus short) portfolio excess return for one month."""
    n = len(group)
    if n < 10:
        return np.nan

    group = group.sort_values("prediction")
    n_long  = max(1, int(n * long_q))
    long_ret  = group["excess_ret"].iloc[-n_long:].mean()

    if short_q is None:
        return long_ret

    n_short = max(1, int(n * short_q))
    short_ret = group["excess_ret"].iloc[:n_short].mean()
    return long_ret - short_ret


def compute_turnover(positions_t: set, positions_t1: set) -> float:
    """Fraction of portfolio that changed between two months."""
    if not positions_t1:
        return 0.0
    unchanged = len(positions_t & positions_t1)
    return 1.0 - unchanged / len(positions_t1)


def run_strategy(
    df: pd.DataFrame,
    name: str,
    long_q: float,
    short_q: float | None,
) -> pd.DataFrame:
    monthly = []
    prev_long_permnos:  set = set()
    prev_short_permnos: set = set()

    for date, group in df.groupby("MthCalDt"):
        ret = build_portfolio(group, long_q, short_q)
        if np.isnan(ret):
            continue

        n = len(group)
        n_long  = max(1, int(n * long_q))
        cur_long  = set(group.sort_values("prediction")["PERMNO"].iloc[-n_long:])
        cur_short = set()
        turnover  = 0.0

        if short_q is not None:
            n_short   = max(1, int(n * short_q))
            cur_short = set(group.sort_values("prediction")["PERMNO"].iloc[:n_short])
            to_long   = compute_turnover(prev_long_permnos,  cur_long)
            to_short  = compute_turnover(prev_short_permnos, cur_short)
            turnover  = (to_long + to_short) / 2
        else:
            turnover = compute_turnover(prev_long_permnos, cur_long)

        cost    = turnover * COST_ONE_WAY * 2
        ret_net = ret - cost

        monthly.append({
            "MthCalDt": date,
            "ret_gross": ret,
            "ret_net":   ret_net,
            "turnover":  turnover,
            "strategy":  name,
        })
        prev_long_permnos  = cur_long
        prev_short_permnos = cur_short

    return pd.DataFrame(monthly)


def performance_metrics(rets: pd.Series, sprtrn: pd.Series) -> dict:
    ann_ret    = rets.mean() * 12
    ann_vol    = rets.std()  * np.sqrt(12)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum        = (1 + rets).cumprod()
    roll_max   = cum.cummax()
    drawdowns  = (cum - roll_max) / roll_max
    max_dd     = drawdowns.min()

    common     = rets.index.intersection(sprtrn.index)
    slope, intercept, *_ = stats.linregress(sprtrn.loc[common], rets.loc[common])
    alpha_ann  = intercept * 12
    beta       = slope

    return {
        "ann_return":  round(ann_ret * 100, 2),
        "ann_vol":     round(ann_vol * 100, 2),
        "sharpe":      round(sharpe, 3),
        "max_drawdown":round(max_dd * 100, 2),
        "alpha_ann":   round(alpha_ann * 100, 2),
        "beta":        round(beta, 3),
    }


def run_period(
    preds: pd.DataFrame,
    sprtrn: pd.Series,
    model_names: list[str],
    strategies: list[tuple],
    period_start: str,
    period_end: str,
) -> dict:
    """Run all model × strategy combos on a given date window."""
    mask  = (preds["MthCalDt"] >= period_start) & (preds["MthCalDt"] <= period_end)
    preds_p = preds[mask]
    summary = {}
    for model_name in model_names:
        df = preds_p[preds_p["model"] == model_name].copy()
        if len(df) == 0:
            continue
        for strat_name, lq, sq in strategies:
            key = f"{model_name} | {strat_name}"
            monthly  = run_strategy(df, strat_name, lq, sq)
            if len(monthly) == 0:
                continue
            rets_net = monthly.set_index("MthCalDt")["ret_net"]
            avg_turn = monthly["turnover"].mean()
            m_net    = performance_metrics(rets_net, sprtrn)
            summary[key] = {
                "Net (%)":      m_net["ann_return"],
                "Sharpe net":   m_net["sharpe"],
                "Alpha (%)":    m_net["alpha_ann"],
                "Max DD (%)":   m_net["max_drawdown"],
                "Turnover (%)": round(avg_turn * 100, 1),
            }
    return summary


def main():
    print("Loading predictions...")
    preds, crsp = load_data()
    print(f"  Models available: {list(preds['model'].unique())}")

    sprtrn = crsp.set_index("MthCalDt")["sprtrn"]

    model_names = ["JKP_only", "FinBERT_only", "JKP+FinBERT", "JKP+FinBERT_w10x", "MLP_FinBERT"]
    strategies  = [
        ("LS_decile",   0.10, 0.10),
        ("LS_quintile", 0.20, 0.20),
        ("LO_decile",   0.10, None),
    ]

    periods = [
        ("2019-2023 (full)",       "2019-01-01", "2023-12-31"),
        ("2019      (pre-COVID)",  "2019-01-01", "2019-12-31"),
        ("2020      (COVID)",      "2020-01-01", "2020-12-31"),
        ("2021-2023 (post-COVID)", "2021-01-01", "2023-12-31"),
    ]

    for period_label, start, end in periods:
        print(f"\n{'='*65}")
        print(f"  PERIOD: {period_label}")
        print(f"{'='*65}")
        summary = run_period(preds, sprtrn, model_names, strategies, start, end)
        df_sum  = pd.DataFrame(summary).T
        print(df_sum.to_string())

    # Save full-period monthly returns for the combined model
    all_monthly = []
    for model_name in model_names:
        df = preds[preds["model"] == model_name].copy()
        for strat_name, lq, sq in strategies:
            monthly = run_strategy(df, strat_name, lq, sq)
            monthly["model"] = model_name
            all_monthly.append(monthly)

    results = pd.concat(all_monthly, ignore_index=True)
    results.to_parquet(OUT_PATH, index=False)
    print(f"\nMonthly returns saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
