# src/model.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor
import time

def _to_long_features(feat_panel: pd.DataFrame) -> pd.DataFrame:
    # columns=[ticker, feature] -> index=(date,ticker), columns=features
    X_long = feat_panel.stack(level=0, future_stack=True)
    X_long.index.set_names(["date", "ticker"], inplace=True)
    return X_long

def _to_long_labels(y_wide: pd.DataFrame) -> pd.Series:
    # wide labels (columns=tickers) -> Series indexed by (date,ticker)
    y_long = y_wide.stack(future_stack=True)
    y_long.index.set_names(["date", "ticker"], inplace=True)
    return y_long

def default_model_factory() -> GradientBoostingRegressor:
    # Simple, fast, decent baseline; tune later if you like
    return GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        random_state=42,
    )

def walkforward_predict(
    feat_panel_universe: pd.DataFrame,  # MultiIndex [ticker, feature]
    y_universe: pd.DataFrame,           # wide: columns=tickers
    train_start: Optional[str] = None,  # e.g., "2013-01-01"
    retrain_freq: str = "M",            # "M" (monthly), "W" (weekly), "D" (daily)
    min_train_days: int = 126,          # start with ~6 months; can raise to 252 later
    model_factory: Callable[[], object] = default_model_factory,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward:
      - At each retrain point (monthly/weekly/daily), fit on all prior data (>= train_start).
      - Predict for the period up to the next retrain point.
    Returns a wide DataFrame of predictions with columns=tickers.
    """
    # 1) Reshape to long and align
    X_long = _to_long_features(feat_panel_universe)      # (date,ticker) x features
    y_long = _to_long_labels(y_universe)                 # (date,ticker) series

    # Join but ONLY drop rows where label is NaN (keep feature NaNs for imputation)
    df = X_long.join(y_long.to_frame("y"), how="inner")
    df = df.dropna(subset=["y"])

    # Sort by (date, ticker) and extract the date level for splitting
    df = df.sort_index()  # by (date, ticker)
    dates = df.index.get_level_values("date")

    # Ensure we have a proper DatetimeIndex of unique trading dates
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    # Do NOT filter cut points by train_start; enforce via train_mask instead

    # 2) Build retrain cut points based on retrain_freq
    if retrain_freq == "D":
        cut_points = list(unique_dates)  # retrain daily
    elif retrain_freq == "W":
        week_keys = unique_dates.to_period("W")
        cut_points = [unique_dates[week_keys == w][0] for w in week_keys.unique()]
    else:
        # default monthly retrain: first trading day in each calendar month
        month_keys = unique_dates.to_period("M")
        cut_points = [unique_dates[month_keys == m][0] for m in month_keys.unique()]

    preds_records: list[pd.DataFrame] = []

    def _block_end(i: int) -> pd.Timestamp:
        if i < len(cut_points) - 1:
            return cut_points[i + 1] - pd.Timedelta(days=1)
        else:
            return unique_dates[-1]

    for i, cut in enumerate(cut_points):
        t0 = time.perf_counter()
        block_start = cut
        block_end = _block_end(i)

        # Masks over the full long df (indexed by (date,ticker))
        pred_mask = (dates >= block_start) & (dates <= block_end)

        # Training uses all dates strictly before this cut
        train_mask = (dates < block_start)
        if train_start is not None:
            train_mask &= (dates >= pd.Timestamp(train_start))

        # Require enough unique training days
        train_dates = dates[train_mask]
        if train_dates.unique().size < min_train_days:
            continue

        # Split into X/y
        X_train = df.loc[train_mask].drop(columns=["y"])
        y_train = df.loc[train_mask, "y"]

        # ---- Robust, leakage-safe imputation ----
        # Compute per-feature medians on the training window (past-only)
        feat_medians = X_train.median(axis=0, skipna=True)

        # Fill NaNs in train with train medians
        X_train = X_train.fillna(feat_medians)

        # If any columns are still all-NaN (can happen rarely), drop them consistently
        all_nan_cols = X_train.columns[X_train.isna().all(0)]
        if len(all_nan_cols) > 0:
            X_train = X_train.drop(columns=all_nan_cols)

        # Drop any remaining rows with NaNs (should be few after fill)
        valid_train_mask = X_train.notna().all(axis=1)
        X_train = X_train.loc[valid_train_mask]
        y_train = y_train.loc[valid_train_mask]

        if X_train.empty:
            continue

        print(f"[WF] Block {i+1}/{len(cut_points)} | Train days={train_dates.unique().size} | "
          f"{block_start.date()} → {block_end.date()} | rows={X_train.shape[0]} | features={X_train.shape[1]}")
        # Fit model
        model = model_factory()
        model.fit(X_train.values, y_train.values)

        # Predict this block; fill using the SAME train medians (no leakage)
        X_pred = df.loc[pred_mask].drop(columns=["y"])
        # Keep only the columns used for training (in case we dropped any)
        X_pred = X_pred.reindex(columns=X_train.columns)
        X_pred = X_pred.fillna(feat_medians.reindex(X_train.columns))

        # Drop any remaining NaN rows (should be few)
        valid_pred_mask = X_pred.notna().all(axis=1)
        X_pred = X_pred.loc[valid_pred_mask]

        if X_pred.empty:
            continue

        y_hat = model.predict(X_pred.values)

        preds_records.append(pd.DataFrame({"pred": y_hat}, index=X_pred.index))

    if not preds_records:
        raise ValueError("No predictions produced. Consider reducing min_train_days or checking feature NaNs.")

    preds_long = pd.concat(preds_records).sort_index()
    preds_wide = preds_long["pred"].unstack("ticker").sort_index()
    dt_elapsed = time.perf_counter() - t0
    print(f"[WF] Done block {i+1}/{len(cut_points)} in {dt_elapsed:.2f}s | pred rows={X_pred.shape[0]}")
    return preds_wide
