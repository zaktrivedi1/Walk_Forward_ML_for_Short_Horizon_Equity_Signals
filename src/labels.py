# src/labels.py
from __future__ import annotations

import pandas as pd
from typing import Literal

from .data import OHLCVData
from .features import _rolling_std


def make_next_day_return_label(data: OHLCVData) -> pd.DataFrame:
    """
    Returns a DataFrame with columns=tickers, index=dates,
    where each value is the next-day return based on adj_close.

    Alignment detail:
      - label[t] = (price[t+1] / price[t] - 1)
      - i.e., we SHIFT(-1) the price before pct_change(1), or equivalently shift the returns.
    """
    close = data.df.xs("adj_close", axis=1, level="field")
    # Compute 1d returns, then shift them UP so label at t is r_{t+1}
    r1 = close.pct_change(1, fill_method=None)
    y = r1.shift(-1)
    return y


def make_vol_scaled_label(data: OHLCVData, vol_window: int = 20) -> pd.DataFrame:
    """
    Next-day return divided by realized volatility (20d default).
    Helps stabilize the regression target.
    """
    y = make_next_day_return_label(data)
    close = data.df.xs("adj_close", axis=1, level="field")
    r1 = close.pct_change(1)
    vol = _rolling_std(r1, vol_window)
    y_scaled = y / vol
    return y_scaled
