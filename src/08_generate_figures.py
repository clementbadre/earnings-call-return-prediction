"""
Generate all figures for the LaTeX report.
Saves to ../outputs/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy.stats import spearmanr

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE   = "#2C5F8A"
RED    = "#C0392B"
GREEN  = "#1A7A4A"
ORANGE = "#D4700A"
GRAY   = "#7F8C8D"
COLORS = [BLUE, RED, GREEN, ORANGE, GRAY]

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ── helpers ────────────────────────────────────────────────────────────────

def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {name}")


# ── 1. FinBERT score distributions ────────────────────────────────────────

def fig_finbert_dist():
    scores = pd.read_parquet(os.path.join(ROOT, "outputs", "finbert_scores.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    cols   = ["p_pos", "p_neg", "p_neu"]
    labels = ["Positive (p_pos)", "Negative (p_neg)", "Neutral (p_neu)"]
    clrs   = [GREEN, RED, GRAY]
    for ax, col, lbl, c in zip(axes, cols, labels, clrs):
        ax.hist(scores[col], bins=60, color=c, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(scores[col].mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {scores[col].mean():.3f}")
        ax.set_title(lbl)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if col == "p_pos" else "")
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    fig.suptitle("FinBERT Sentiment Score Distributions (34,643 earnings calls)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save("fig1_finbert_dist.pdf")


# ── 2. Signal freshness before and after filter ────────────────────────────

def fig_freshness():
    df = pd.read_parquet(os.path.join(ROOT, "outputs", "dataset_final_v2.parquet"))
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])

    # Recompute months_since_call from the already-filtered dataset
    # Load the finbert scores to get pre-filter distribution
    scores = pd.read_parquet(os.path.join(ROOT, "outputs", "finbert_scores.parquet"))
    crsp   = pd.read_parquet(os.path.join(ROOT, "outputs", "crsp_clean.parquet"))
    scores["call_date"] = pd.to_datetime(scores["call_date"])
    crsp["MthCalDt"]    = pd.to_datetime(crsp["MthCalDt"])

    crsp_s = crsp.sort_values("MthCalDt")
    scores_r = scores.rename(columns={"permno": "PERMNO"}).sort_values("call_date")

    merged = pd.merge_asof(
        crsp_s[["PERMNO", "MthCalDt"]],
        scores_r[["PERMNO", "call_date"]],
        left_on="MthCalDt", right_on="call_date",
        by="PERMNO", direction="backward", allow_exact_matches=False,
    )
    merged["months_since"] = (
        (merged["MthCalDt"] - merged["call_date"]) / pd.Timedelta(days=30.5)
    ).round()
    merged = merged.dropna(subset=["months_since"])
    merged = merged[merged["months_since"] >= 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Pre-filter
    vals = merged["months_since"].clip(0, 24)
    ax1.hist(vals, bins=25, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax1.axvline(vals.median(), color=RED, linestyle="--", linewidth=1.5,
                label=f"Median = {vals.median():.0f} months")
    ax1.axvline(3, color=ORANGE, linestyle="-", linewidth=2,
                label="Freshness cut-off (3 months)")
    ax1.set_title("Before Freshness Filter")
    ax1.set_xlabel("Months since last earnings call")
    ax1.set_ylabel("Count")
    ax1.legend(frameon=False)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Post-filter
    vals2 = df["months_since_call"].clip(0, 3)
    ax2.hist(vals2, bins=4, color=GREEN, alpha=0.8, edgecolor="white", linewidth=0.3,
             rwidth=0.7)
    ax2.set_title("After Freshness Filter (≤ 3 months)")
    ax2.set_xlabel("Months since last earnings call")
    ax2.set_ylabel("Count")
    ax2.set_xticks([0.5, 1.5, 2.5])
    ax2.set_xticklabels(["1", "2", "3"])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    fig.suptitle("Signal Freshness: Distribution of Months Since Last Earnings Call",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save("fig2_freshness.pdf")


# ── 3. Monthly Rank IC time series ────────────────────────────────────────

def fig_monthly_ic():
    ic_df = pd.read_parquet(os.path.join(ROOT, "outputs", "rolling_analysis.parquet"))
    ic_df.index = pd.to_datetime(ic_df.index)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: FinBERT monthly IC with 12M rolling mean
    ax = axes[0]
    ic = ic_df["FinBERT_only"].dropna()
    ax.bar(ic.index, ic.values, width=20,
           color=[GREEN if v > 0 else RED for v in ic.values], alpha=0.7)
    roll12 = ic.rolling(12).mean()
    ax.plot(roll12.index, roll12.values, color=BLUE, linewidth=2,
            label="12M rolling mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(ic.mean(), color=BLUE, linestyle="--", linewidth=1,
               label=f"Overall mean = {ic.mean():.4f}")
    ax.set_title(f"FinBERT_only: Monthly Rank IC (2015–2023)  |  t-stat = 2.67  |  "
                 f"{(ic>0).mean()*100:.0f}% positive months")
    ax.set_ylabel("Rank IC (Spearman)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(-0.25, 0.25)

    # Add COVID annotation
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
               alpha=0.12, color=ORANGE, label="COVID-19")

    # Bottom: JKP+FinBERT_w10x for comparison
    ax2 = axes[1]
    ic2 = ic_df["JKP+FinBERT_w10x"].dropna()
    ax2.bar(ic2.index, ic2.values, width=20,
            color=[GREEN if v > 0 else RED for v in ic2.values], alpha=0.7)
    roll12b = ic2.rolling(12).mean()
    ax2.plot(roll12b.index, roll12b.values, color=BLUE, linewidth=2,
             label="12M rolling mean")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(ic2.mean(), color=BLUE, linestyle="--", linewidth=1,
                label=f"Overall mean = {ic2.mean():.4f}")
    ax2.set_title(f"JKP+FinBERT w10×: Monthly Rank IC (2015–2023)  |  t-stat = 1.23  |  "
                  f"{(ic2>0).mean()*100:.0f}% positive months")
    ax2.set_ylabel("Rank IC (Spearman)")
    ax2.set_xlabel("Date")
    ax2.legend(frameon=False, loc="upper left")
    ax2.set_ylim(-0.25, 0.25)
    ax2.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
                alpha=0.12, color=ORANGE)

    plt.tight_layout()
    save("fig3_monthly_ic.pdf")


# ── 4. Year-by-year IC heatmap ────────────────────────────────────────────

def fig_ic_heatmap():
    ic_df = pd.read_parquet(os.path.join(ROOT, "outputs", "rolling_analysis.parquet"))
    ic_df.index = pd.to_datetime(ic_df.index)

    models = ["FinBERT_only", "JKP+FinBERT_w10x", "JKP+FinBERT"]
    labels = ["FinBERT only", "JKP+FinBERT ×10", "JKP+FinBERT"]

    rows = []
    for col in models:
        ic = ic_df[col].dropna()
        ic.index = pd.to_datetime(ic.index)
        by_year = ic.groupby(ic.index.year).mean()
        rows.append(by_year)

    heat = pd.DataFrame(rows, index=labels)

    fig, ax = plt.subplots(figsize=(11, 3))
    sns.heatmap(heat, annot=True, fmt=".3f", center=0,
                cmap="RdYlGn", ax=ax, linewidths=0.5,
                cbar_kws={"label": "Mean Monthly Rank IC"},
                annot_kws={"size": 10})
    ax.set_title("Mean Monthly Rank IC by Year and Model (2015–2023)", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    plt.tight_layout()
    save("fig4_ic_heatmap.pdf")


# ── 5. Cumulative returns ─────────────────────────────────────────────────

def fig_cumulative_returns():
    bt = pd.read_parquet(os.path.join(ROOT, "outputs", "backtest_results_v2.parquet"))
    bt["MthCalDt"] = pd.to_datetime(bt["MthCalDt"])
    crsp = pd.read_parquet(os.path.join(ROOT, "outputs", "crsp_clean.parquet"))
    crsp["MthCalDt"] = pd.to_datetime(crsp["MthCalDt"])
    sprtrn = crsp[["MthCalDt", "sprtrn"]].drop_duplicates("MthCalDt").set_index("MthCalDt")["sprtrn"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    PURPLE = "#7B2D8B"
    configs = [
        ("FinBERT_only", "LS_decile",  GREEN,  "FinBERT LS Decile (net)"),
        ("MLP_FinBERT",  "LS_decile",  PURPLE, "MLP FinBERT LS Decile (net)"),
        ("JKP+FinBERT",  "LS_decile",  BLUE,   "JKP+FinBERT LS Decile (net)"),
        ("JKP_only",     "LS_decile",  RED,    "JKP LS Decile (net)"),
    ]

    for model, strat, color, label in configs:
        sub = bt[(bt["model"] == model) & (bt["strategy"] == strat)].copy()
        sub = sub.sort_values("MthCalDt").set_index("MthCalDt")
        cum = (1 + sub["ret_net"]).cumprod()
        ax1.plot(cum.index, cum.values, color=color, linewidth=2, label=label)

    # S&P500 benchmark
    sp = sprtrn[sprtrn.index >= "2019-01-01"].sort_index()
    sp_cum = (1 + sp).cumprod()
    ax1.plot(sp_cum.index, sp_cum.values, color=GRAY, linewidth=1.5,
             linestyle="--", label="S&P500 (sprtrn)")
    ax1.axhline(1, color="black", linewidth=0.5)
    ax1.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
                alpha=0.12, color=ORANGE, label="COVID-19")
    ax1.set_title("Cumulative Net Returns (LS Decile Strategy, 2019–2023)")
    ax1.set_ylabel("Cumulative Return (base = 1)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_xlabel("Date")

    # Long-only comparison
    configs2 = [
        ("FinBERT_only", "LO_decile", GREEN,  "FinBERT LO Decile (net)"),
        ("JKP+FinBERT",  "LO_decile", BLUE,   "JKP+FinBERT LO Decile (net)"),
    ]
    for model, strat, color, label in configs2:
        sub = bt[(bt["model"] == model) & (bt["strategy"] == strat)].copy()
        sub = sub.sort_values("MthCalDt").set_index("MthCalDt")
        cum = (1 + sub["ret_net"]).cumprod()
        ax2.plot(cum.index, cum.values, color=color, linewidth=2, label=label)
    ax2.plot(sp_cum.index, sp_cum.values, color=GRAY, linewidth=1.5,
             linestyle="--", label="S&P500")
    ax2.axhline(1, color="black", linewidth=0.5)
    ax2.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
                alpha=0.12, color=ORANGE, label="COVID-19")
    ax2.set_title("Cumulative Net Returns (Long-Only Decile Strategy, 2019–2023)")
    ax2.set_ylabel("Cumulative Return (base = 1)")
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_xlabel("Date")

    plt.tight_layout()
    save("fig5_cumulative_returns.pdf")


# ── 6. Rolling 12-month net return ────────────────────────────────────────

def fig_rolling_12m():
    bt = pd.read_parquet(os.path.join(ROOT, "outputs", "backtest_results_v2.parquet"))
    bt["MthCalDt"] = pd.to_datetime(bt["MthCalDt"])

    # Extend to 2015-2023 using rolling_analysis predictions
    ic_df = pd.read_parquet(os.path.join(ROOT, "outputs", "rolling_analysis.parquet"))
    # Use the backtest monthly returns for FinBERT_only (test period only)
    fb = bt[(bt["model"] == "FinBERT_only") & (bt["strategy"] == "LS_decile")].copy()
    fb = fb.sort_values("MthCalDt").set_index("MthCalDt")

    roll = (fb["ret_net"].rolling(12)
            .apply(lambda x: (1+x).prod()**(12/12) - 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(roll.index, roll.values * 100, 0,
                    where=roll.values >= 0, alpha=0.4, color=GREEN, label="Positive")
    ax.fill_between(roll.index, roll.values * 100, 0,
                    where=roll.values < 0, alpha=0.4, color=RED, label="Negative")
    ax.plot(roll.index, roll.values * 100, color=BLUE, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(roll.mean() * 100, color=BLUE, linestyle="--", linewidth=1,
               label=f"Mean = {roll.mean()*100:.1f}%")
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
               alpha=0.12, color=ORANGE, label="COVID-19")
    pct_pos = (roll > 0).mean() * 100
    ax.set_title(f"FinBERT_only LS Decile: Rolling 12-Month Net Return  |  "
                 f"{pct_pos:.0f}% of windows positive")
    ax.set_ylabel("Annualized Net Return (%)")
    ax.set_xlabel("End of Rolling Window")
    ax.legend(frameon=False)
    plt.tight_layout()
    save("fig6_rolling_12m.pdf")


# ── 7. Ablation study bar chart ────────────────────────────────────────────

def fig_ablation():
    data = {
        "Model": ["JKP only", "FinBERT only", "JKP+FinBERT", "JKP+FinBERT ×10", "MLP (FinBERT)"],
        "Rank IC (test)": [0.0342, 0.0089, 0.0505, 0.0720, 0.0129],
        "Net Return % (LS decile)": [-4.50, 11.40, 0.49, -8.53, 7.87],
        "Sharpe (net)": [-0.361, 0.788, 0.039, -0.523, 0.576],
    }
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bar_colors = [RED, GREEN, BLUE, ORANGE, "#7B2D8B"]  # purple for MLP

    for ax, col, ylabel in zip(
        axes,
        ["Rank IC (test)", "Net Return % (LS decile)", "Sharpe (net)"],
        ["Rank IC (Spearman)", "Annualized Net Return (%)", "Sharpe Ratio (net)"]
    ):
        bars = ax.bar(df["Model"], df[col], color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5, width=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["Model"], rotation=22, ha="right", fontsize=9)
        for bar, val in zip(bars, df[col]):
            y = bar.get_height()
            offset = 0.003 if col == "Rank IC (test)" else 0.3
            va = "bottom" if val >= 0 else "top"
            y_txt = (y + offset) if val >= 0 else (y - offset)
            ax.text(bar.get_x() + bar.get_width()/2, y_txt,
                    f"{val:.3f}" if col == "Rank IC (test)" else f"{val:.2f}",
                    ha="center", va=va, fontsize=8.5, fontweight="bold")

    fig.suptitle("Ablation Study: Model Comparison (Out-of-Sample 2019–2023)\n"
                 "XGBoost (4 variants) + PyTorch MLP on FinBERT features", fontsize=12)
    plt.tight_layout()
    save("fig7_ablation.pdf")


# ── 8. Period robustness ──────────────────────────────────────────────────

def fig_period_robustness():
    periods   = ["2019\n(pre-COVID)", "2020\n(COVID)", "2021–2023\n(post-COVID)", "2019–2023\n(full)"]
    fb_ls     = [3.87,  14.99, 12.65, 11.40]
    jkp_ls    = [1.40,  11.37, -11.76, -4.50]
    comb_ls   = [7.50,  8.70,  -4.58,  0.49]

    x = np.arange(len(periods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w,   fb_ls,   w, label="FinBERT only",   color=GREEN,  alpha=0.85)
    ax.bar(x,       jkp_ls,  w, label="JKP only",       color=RED,    alpha=0.85)
    ax.bar(x + w,   comb_ls, w, label="JKP+FinBERT",    color=BLUE,   alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=10)
    ax.set_ylabel("Annualized Net Return, LS Decile (%)")
    ax.set_title("Period Robustness Check: Net Return by Sub-Period and Model")
    ax.legend(frameon=False)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    plt.tight_layout()
    save("fig8_period_robustness.pdf")


# ── 9. Pipeline diagram ───────────────────────────────────────────────────

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (1.0,  2.0, "CRSP\n(1.45M rows\n2008–2023)", BLUE),
        (3.5,  2.0, "JKP Factors\n(153 factors\nmonthly)", ORANGE),
        (6.0,  2.0, "Earnings Calls\n(34,643 calls\nFinBERT →\np_pos/p_neg/p_neu)", GREEN),
        (9.0,  2.0, "Feature\nEngineering\n(freshness, delta,\nz-scores)", BLUE),
        (11.8, 2.0, "Ablation\n(4 XGBoost\n+ MLP)", RED),
    ]
    for x, y, label, color in boxes:
        rect = plt.Rectangle((x - 0.9, y - 0.8), 1.8, 1.6,
                              facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
                multialignment="center")

    arrows = [(2.0, 2.0, 3.5 - 0.9), (4.5, 2.0, 6.0 - 0.9),
              (7.0, 2.0, 9.0 - 0.9), (10.0, 2.0, 11.8 - 0.9)]
    for x1, y, x2 in arrows:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    labels_below = [
        (1.0,  0.9, "Excess return\n(target)"),
        (3.5,  0.9, "1-month lag\n(look-ahead free)"),
        (6.0,  0.9, "MPS GPU\n(~3h inference)"),
        (9.0,  0.9, "≤3 months\nfreshness filter"),
        (11.8, 0.9, "Temporal split\n2008–14/15–18/19–23"),
    ]
    for x, y, lbl in labels_below:
        ax.text(x, y, lbl, ha="center", va="center", fontsize=7.5,
                color=GRAY, multialignment="center", style="italic")

    ax.set_title("Project Pipeline Overview", fontsize=13, pad=8)
    plt.tight_layout()
    save("fig9_pipeline.pdf")


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    fig_finbert_dist()
    fig_freshness()
    fig_monthly_ic()
    fig_ic_heatmap()
    fig_cumulative_returns()
    fig_rolling_12m()
    fig_ablation()
    fig_period_robustness()
    fig_pipeline()
    print(f"\nAll figures saved to: {FIG_DIR}")
