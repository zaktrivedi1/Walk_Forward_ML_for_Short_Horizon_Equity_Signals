from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List

from .data import OHLCVData


def _pct_change(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return df.pct_change(periods=periods, fill_method=None)


def _rolling_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window, min_periods=window).mean()


def _rolling_std(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window, min_periods=window).std(ddof=0)


def _zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = _rolling_mean(df, window)
    std = _rolling_std(df, window)
    return (df - mean) / std


def _rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI classic calculation on price changes.
    Returns a DataFrame with same columns (tickers).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average True Range (ATR) using Wilder's classic definition.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    candidates = pd.concat([tr1, tr2, tr3], axis=1, keys=["tr1", "tr2", "tr3"])

    # Do groupby on the transposed view (no axis=1 deprecation), then transpose back.
    tr = candidates.T.groupby(level=1).max().T  # columns now back to tickers

    # Wilder's ATR often uses an EMA; here we keep a simple rolling mean
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr



def make_feature_panel(data: OHLCVData) -> pd.DataFrame:
    """
    Build a MultiIndex column DataFrame with columns=[ticker, feature_name].
    Features are computed from past data only (no leakage).
    """
    df = data.df

    # Extract field matrices (columns=tickers)
    close = df.xs("adj_close", axis=1, level="field")
    high = df.xs("high", axis=1, level="field")
    low = df.xs("low", axis=1, level="field")
    volume = df.xs("volume", axis=1, level="field")

    # === Basic returns ===
    ret_1d = _pct_change(close, 1)
    ret_5d = _pct_change(close, 5)
    ret_10d = _pct_change(close, 10)
    ret_20d = _pct_change(close, 20)

    # === Momentum z-scores (based on trailing returns) ===
    # Use rolling mean/std on past returns; returns already use past info.
    mom_5d_z = _zscore(ret_5d, 60)
    mom_10d_z = _zscore(ret_10d, 60)
    mom_20d_z = _zscore(ret_20d, 60)

    # === Mean-reversion style features ===
    # 1-day reversal: yesterday's return (already ret_1d)
    rev_1d = ret_1d

    # Deviation from 20d moving average (as %)
    ma_20 = _rolling_mean(close, 20)
    dev_from_ma20 = (close - ma_20) / ma_20

    # === Volatility / volume features ===
    vol_20d = _rolling_std(ret_1d, 20)  # realized vol of 1d returns
    vol_of_vol_20d = _rolling_std(vol_20d, 20)

    vol_z_20d = _zscore(vol_20d, 120)

    # ATR-based volatility
    atr_14 = _atr(high, low, close, 14)
    atr_pct_14 = atr_14 / close

    # Volume z-score (vs 60d history)
    vol_zscore_60 = _zscore(volume, 60)

    # === RSI ===
    rsi_14 = _rsi(close, 14)

    # Assemble as MultiIndex [ticker, feature]
    features: Dict[str, pd.DataFrame] = {
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "ret_20d": ret_20d,
        "mom_5d_z": mom_5d_z,
        "mom_10d_z": mom_10d_z,
        "mom_20d_z": mom_20d_z,
        "rev_1d": rev_1d,
        "dev_from_ma20": dev_from_ma20,
        "vol_20d": vol_20d,
        "vol_of_vol_20d": vol_of_vol_20d,
        "vol_z_20d": vol_z_20d,
        "atr_pct_14": atr_pct_14,
        "vol_zscore_60": vol_zscore_60,
        "rsi_14": rsi_14,
    }

    # Stack into MultiIndex columns
    parts = []
    for feat_name, mat in features.items():
        mat = mat.copy()
        mat.columns = pd.MultiIndex.from_product([mat.columns, [feat_name]], names=["ticker", "feature"])
        parts.append(mat)

    feat_panel = pd.concat(parts, axis=1).sort_index(axis=1)
    return feat_panel


def select_universe_features(feat_panel: pd.DataFrame, universe: List[str]) -> pd.DataFrame:
    """
    Filter feature panel to universe tickers only.
    """
    cols = feat_panel.columns
    mask = cols.get_level_values(0).isin(universe)
    return feat_panel.loc[:, mask]
