from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

def portfolio_returns(
    weights: pd.DataFrame,  # dates x tickers (weight at close t to hold next day)
    next_day_returns: pd.DataFrame,  # dates x tickers (label matrix)
    costs: pd.Series,       # daily costs (negative)
) -> pd.Series:
    """
    Daily portfolio return: sum_tickers( w_t * y_t ) + cost_t
    Assumes y_t is the return from t->t+1 (our label definition).
    """
    # Align indices/columns
    common_idx = weights.index.intersection(next_day_returns.index)
    common_cols = weights.columns.intersection(next_day_returns.columns)

    W = weights.loc[common_idx, common_cols]
    Y = next_day_returns.loc[common_idx, common_cols]
    C = costs.reindex(common_idx).fillna(0.0)

    gross = (W * Y).sum(axis=1)
    pnl = gross + C
    return pnl

def summarize_performance(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Vol": np.nan}

    ann_factor = 252.0
    mean = r.mean()
    vol = r.std(ddof=0)
    sharpe = (mean / vol) * np.sqrt(ann_factor) if vol > 0 else np.nan

    # CAGR with log compounding (approx)
    cum = (1.0 + r).prod()
    years = len(r) / ann_factor
    cagr = cum ** (1.0 / years) - 1.0 if years > 0 else np.nan

    # Max drawdown on equity curve
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    maxdd = dd.min()

    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(maxdd), "Vol": float(vol * np.sqrt(ann_factor))}


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative equity curve from daily returns."""
    returns = returns.fillna(0.0)
    return (1.0 + returns).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown series from an equity curve."""
    peak = equity.cummax()
    return equity / peak - 1.0


def yearly_metrics(r: pd.Series) -> pd.DataFrame:
    """Year-by-year Sharpe and CAGR (approx)."""
    r = r.dropna()
    if r.empty:
        return pd.DataFrame(columns=["Year", "Sharpe", "CAGR"])
    ann = 252.0
    by_year = []
    for yr, g in r.groupby(r.index.year):
        m = g.mean()
        v = g.std(ddof=0)
        sharpe = (m / v) * (ann ** 0.5) if v > 0 else float("nan")
        cagr = (1.0 + g).prod() ** (ann / len(g)) - 1.0
        by_year.append({"Year": int(yr), "Sharpe": float(sharpe), "CAGR": float(cagr)})
    return pd.DataFrame(by_year).set_index("Year").sort_index()


def save_equity_and_drawdown_plots(
    returns: pd.Series,
    out_equity_path: str,
    out_drawdown_path: str,
    title_prefix: str = "Walk-Forward ML Strategy",
) -> None:
    """Save equity curve and drawdown plots to files."""
    eq = equity_curve(returns)
    dd = drawdown_series(eq)

    # Equity
    plt.figure(figsize=(10, 4))
    eq.plot()
    plt.title(f"{title_prefix} - Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.tight_layout()
    plt.savefig(out_equity_path, dpi=150)
    plt.close()

    # Drawdown
    plt.figure(figsize=(10, 3))
    dd.plot()
    plt.title(f"{title_prefix} - Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_drawdown_path, dpi=150)
    plt.close()