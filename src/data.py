from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("yfinance is required. Run `pip install yfinance`.") from e


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class OHLCVData:
    """
    Container for a panel-style OHLCV dataframe:
        - index: DatetimeIndex (UTC-naive, daily)
        - columns: MultiIndex [ticker, field]
        - fields: ['open','high','low','close','adj_close','volume']
    """
    df: pd.DataFrame

    @property
    def tickers(self) -> List[str]:
        return sorted(set(self.df.columns.get_level_values(0)))

    @property
    def fields(self) -> List[str]:
        return sorted(set(self.df.columns.get_level_values(1)))


# -----------------------------
# Download
# -----------------------------
def download_ohlcv(
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
    progress: bool = False,
) -> OHLCVData:
    """
    Download OHLCV with yfinance and return a clean, column-harmonized MultiIndex DataFrame.

    Notes on shape:
    - yfinance.download returns a wide DataFrame with column level 0 = fields and level 1 = tickers.
      We transpose/reorder into columns level 0 = tickers and level 1 = fields.
    - Includes both Close and Adj Close. Adj Close used for returns; raw Close is handy for sanity checks.

    Parameters
    ----------
    tickers : Iterable[str]
        Symbols (e.g., ["AAPL","MSFT","SPY"])
    start : str
        e.g., "2010-01-01"
    end : Optional[str]
        e.g., "2024-12-31"
    interval : str
        "1d" (daily) for this project
    auto_adjust : bool
        If True, yfinance returns adjusted prices but drops Adj Close. False kept and Adj Close requested explicitly.
    progress : bool
        Show yfinance download progress.
    """
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        raise ValueError("No valid tickers provided.")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",   
        progress=progress,
        threads=True,
    )

    # --- Make columns consistently [ticker, field] ---
    if isinstance(raw.columns, pd.MultiIndex):
        # Detect which level holds fields
        lvl0 = set(map(str, raw.columns.get_level_values(0)))
        lvl1 = set(map(str, raw.columns.get_level_values(1)))
        field_candidates = {
            "Open","High","Low","Close","Adj Close","Volume",
            "open","high","low","close","adj_close","volume"
        }

        # If level 0 looks like fields, swap; else keep as-is
        if len(lvl0 & field_candidates) > len(lvl1 & field_candidates):
            raw = raw.swaplevel(0, 1, axis=1)

        # Ensure column level names
        raw.columns = pd.MultiIndex.from_arrays(
            [raw.columns.get_level_values(0).astype(str), raw.columns.get_level_values(1).astype(str)],
            names=["ticker", "field"]
        )
    else:
        # Single-ticker DataFrame: wrap to MultiIndex [ticker, field]
        single_ticker = tickers[0]
        raw = pd.concat({single_ticker: raw}, axis=1)
        raw.columns = pd.MultiIndex.from_arrays(
            [raw.columns.get_level_values(0).astype(str), raw.columns.get_level_values(1).astype(str)],
            names=["ticker", "field"]
        )

    # --- Normalize field names ---
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        # to be robust if already lowercase or mixed
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "adj_close": "adj_close",
        "volume": "volume",
    }

    lvl0 = raw.columns.get_level_values(0)  # tickers
    lvl1 = raw.columns.get_level_values(1)  # fields
    lvl1_norm = [rename_map.get(str(f), str(f).lower()) for f in lvl1]
    raw.columns = pd.MultiIndex.from_arrays([lvl0, lvl1_norm], names=["ticker", "field"])

    # --- Ensure all fields exist for each ticker in a stable order ---
    wanted_fields = ["open", "high", "low", "close", "adj_close", "volume"]
    raw = _ensure_all_fields(raw, wanted_fields)

    # --- Clean index to tz-naive daily dates ---
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()

    return OHLCVData(df=raw)


def _ensure_all_fields(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    tickers = sorted(set(df.columns.get_level_values(0)))
    existing = set(df.columns.get_level_values(1))
    missing = [f for f in fields if f not in existing]

    if missing:
        for t in tickers:
            for f in missing:
                df[(t, f)] = np.nan
        df = df.sort_index(axis=1)

    # Reorder strictly to fields order for each ticker
    new_cols = []
    for t in tickers:
        for f in fields:
            new_cols.append((t, f))
    return df.reindex(columns=pd.MultiIndex.from_tuples(new_cols, names=["ticker", "field"]))


# -----------------------------
# Cleaning & alignment
# -----------------------------
def align_calendar_daily(data: OHLCVData, drop_weekends: bool = True) -> OHLCVData:
    """
    Aligns index to a common daily calendar across all tickers.
    """
    df = data.df.copy()

    start = df.index.min()
    end = df.index.max()
    if drop_weekends:
        idx = pd.bdate_range(start=start, end=end)
    else:
        idx = pd.date_range(start=start, end=end, freq="D")

    df = df.reindex(idx)

    # Forward-fill prices only; leave volumes as NaN (then fill zeros) to avoid synthetic turnover on holidays
    price_fields = ["open", "high", "low", "close", "adj_close"]
    vol_field = "volume"

    # Forward-fill per ticker/field for prices
    for f in price_fields:
        cols = df.xs(f, axis=1, level="field", drop_level=False)
        # group by ticker on the transposed view, then ffill, then transpose back
        filled = cols.T.groupby(level=0).ffill().T
        df.loc[:, cols.columns] = filled

    # Volume: set missing to 0 (no trading)
    vol_cols = df.xs(vol_field, axis=1, level="field", drop_level=False).columns
    df.loc[:, vol_cols] = df.loc[:, vol_cols].fillna(0.0)

    return OHLCVData(df=df)


def drop_sparse_tickers(
    data: OHLCVData,
    min_start_date: Optional[str] = None,
    max_na_frac: float = 0.2,
) -> OHLCVData:
    """
    Keep tickers that:
      - (optionally) have any valid price on/before `min_start_date`
      - have acceptable missingness AFTER their first valid price (ignoring pre-IPO NaNs)
    A day counts as 'valid' if ANY of ['adj_close','close','open','high','low'] is non-NaN.
    """
    df = data.df.copy()
    tickers = sorted(set(df.columns.get_level_values(0)))

    keep = []
    min_start_dt = pd.Timestamp(min_start_date) if min_start_date is not None else None

    price_fields = ["adj_close", "close", "open", "high", "low"]

    for t in tickers:
        sub = df[t]

        # Row-wise "any price present"
        available_any = sub[price_fields].notna().any(axis=1)
        if not available_any.any():
            # No price at all
            continue

        # First day with any valid price (IPO or first coverage)
        first_valid = available_any.idxmax()  # returns first True index

        if min_start_dt is not None and (first_valid is None or first_valid > min_start_dt):
            # Ticker starts after required start date
            continue

        # Post-IPO period only
        post_mask = available_any.loc[first_valid:]
        na_frac_post = 1.0 - float(post_mask.mean())  # fraction of days with NO price

        if na_frac_post <= max_na_frac:
            keep.append(t)

    if not keep:
        raise ValueError(
            "All tickers filtered out by sparsity. "
            "Try increasing `max_na_frac` (e.g., 0.4), use `min_start_date=None`, "
            "or start the download earlier."
        )

    filtered = df.loc[:, df.columns.get_level_values(0).isin(keep)]
    return OHLCVData(df=filtered)

# -----------------------------
# Liquidity / universe building
# -----------------------------
def compute_dollar_volume(data: OHLCVData, window: int = 20) -> pd.DataFrame:
    """
    Rolling average dollar volume (Adj Close * Volume), per ticker.
    Returns a DataFrame with columns = tickers.
    """
    prices = data.df.xs("adj_close", axis=1, level="field")
    vols = data.df.xs("volume", axis=1, level="field")
    dollar_vol = prices * vols
    return dollar_vol.rolling(window=window, min_periods=1).mean()


def build_liquid_universe(
    data: OHLCVData,
    top_n: int = 100,
    lookback_days: int = 60,
    min_history_days: int = 252,
) -> List[str]:
    """
    Select top-N liquid tickers by average dollar volume over the last `lookback_days`,
    and require at least `min_history_days` of non-NaN adj_close history.

    Returns the list of selected tickers.
    """
    df = data.df
    tickers = sorted(set(df.columns.get_level_values(0)))

    # Require minimum history
    long_enough = []
    for t in tickers:
        good = df[t]["adj_close"].notna().sum()
        if good >= min_history_days:
            long_enough.append(t)

    if not long_enough:
        raise ValueError("No tickers satisfy minimum history requirement.")

    dv = compute_dollar_volume(data, window=lookback_days)
    recent = dv.tail(lookback_days)
    avg_dv = recent.mean(axis=0).dropna()

    # Intersect with long_enough
    avg_dv = avg_dv[avg_dv.index.isin(long_enough)]
    if avg_dv.empty:
        raise ValueError("No tickers with both sufficient history and dollar volume data.")

    top = avg_dv.sort_values(ascending=False).head(top_n).index.tolist()
    return top


# -----------------------------
# Caching helpers
# -----------------------------
def save_parquet(data: OHLCVData, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.df.to_parquet(path)


def load_parquet(path: str) -> OHLCVData:
    df = pd.read_parquet(path)
    # Ensure columns are MultiIndex after reload
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns [ticker, field] in cached parquet.")
    return OHLCVData(df=df)


# -----------------------------
# Convenience: end-to-end data prep
# -----------------------------
def prepare_data_pipeline(
    tickers: Iterable[str],
    start: str,
    end: Optional[str],
    cache_path: Optional[str] = None,
    min_start_date: Optional[str] = None,
    universe_top_n: int = 100,
) -> Tuple[OHLCVData, List[str]]:
    """
    One-shot convenience to:
        1) download
        2) align calendar
        3) drop sparse tickers
        4) compute top-N liquid universe (based on recent dollar volume)
        5) optional cache

    Returns:
        cleaned_data, universe_tickers
    """
    data = download_ohlcv(tickers, start=start, end=end)
    data = align_calendar_daily(data)
    data = drop_sparse_tickers(data, min_start_date=min_start_date)
    universe = build_liquid_universe(data, top_n=universe_top_n)

    if cache_path:
        save_parquet(data, cache_path)

    return data, universe
