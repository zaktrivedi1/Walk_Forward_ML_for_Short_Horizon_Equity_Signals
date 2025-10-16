# src/portfolio.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

def signal_to_weights(
    preds: pd.DataFrame,   # dates x tickers
    top_n: int = 20,
    long_short: bool = True,
    max_gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """
    Convert daily predictions into portfolio weights.
    - long_short=True: equal-weight top N long and bottom N short, gross capped to 1.0
    - long_short=False: equal-weight top N long only, sum(weights)=1
    """
    preds = preds.copy()

    w = pd.DataFrame(0.0, index=preds.index, columns=preds.columns)

    for dt, row in preds.iterrows():
        s = row.dropna()
        if s.empty:
            continue

        if long_short:
            longs = s.nlargest(top_n).index
            shorts = s.nsmallest(top_n).index

            n_long = len(longs)
            n_short = len(shorts)
            if n_long == 0 and n_short == 0:
                continue

            wl = (max_gross_leverage / 2.0) / max(1, n_long)
            ws = (max_gross_leverage / 2.0) / max(1, n_short)

            w.loc[dt, longs] = +wl
            w.loc[dt, shorts] = -ws
        else:
            longs = s.nlargest(top_n).index
            n_long = len(longs)
            if n_long == 0:
                continue
            w.loc[dt, longs] = 1.0 / n_long

    return w

def apply_transaction_costs(
    weights: pd.DataFrame,
    cost_bps_roundtrip: float = 10.0,
) -> pd.Series:
    """
    Simple linear costs on daily weight changes (turnover).
    - cost_bps_roundtrip: e.g., 10 bps for round-trip -> 5 bps per side.
    Returns a Series of daily cost (negative).
    """
    per_side = cost_bps_roundtrip / 2.0 / 1e4  # convert bps to decimal
    # Turnover = sum |w_t - w_{t-1}| across tickers
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = -per_side * dw
    return cost
