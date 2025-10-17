from __future__ import annotations

import os
import pandas as pd
import time

from src.data import prepare_data_pipeline
from src.features import make_feature_panel, select_universe_features
from src.labels import make_next_day_return_label
from src.model import walkforward_predict, default_model_factory  # or your fast_model_factory
from src.portfolio import signal_to_weights, apply_transaction_costs
from src.backtest import portfolio_returns, summarize_performance, yearly_metrics, save_equity_and_drawdown_plots

def main():
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ----- Config -----
    tickers = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","SPY"]
    start = "2012-01-01"
    end = None
    universe_top_n = 5
    min_start_date = None

    # Model / walk-forward settings
    retrain_freq = "M"        # "M" monthly, "W" weekly (faster), "D" daily (slow)
    min_train_days = 126      
    model_factory = default_model_factory  

    # Portfolio settings
    top_n = 2
    long_short = True
    max_gross_leverage = 1.0
    cost_bps_roundtrip = 10.0

    # ----- Data -----
    data, universe = prepare_data_pipeline(
        tickers=tickers,
        start=start,
        end=end,
        cache_path="data/prices.parquet",
        min_start_date=min_start_date,
        universe_top_n=universe_top_n,
    )

    # ----- Features & Labels -----
    feat_panel = make_feature_panel(data)
    feat_univ = select_universe_features(feat_panel, universe)

    y = make_next_day_return_label(data)
    y_univ = y[universe].loc[feat_univ.index]

    # Align (defensive)
    common_idx = feat_univ.index.intersection(y_univ.index)
    feat_univ = feat_univ.loc[common_idx]
    y_univ = y_univ.loc[common_idx]

    # ----- Walk-forward predictions -----
    preds = walkforward_predict(
        feat_panel_universe=feat_univ,
        y_universe=y_univ,
        train_start=None,
        retrain_freq=retrain_freq,
        min_train_days=min_train_days,
        model_factory=model_factory,
    )

    # ----- Portfolio -----
    weights = signal_to_weights(
        preds,
        top_n=top_n,
        long_short=long_short,
        max_gross_leverage=max_gross_leverage,
    )
    costs = apply_transaction_costs(weights, cost_bps_roundtrip=cost_bps_roundtrip)
    rets = portfolio_returns(weights, y_univ, costs)

    # ----- Reports -----
    summary = summarize_performance(rets)
    yr = yearly_metrics(rets)

    # Save tables
    pd.Series(summary).to_csv("reports/tables/summary.csv")
    yr.to_csv("reports/tables/yearly_metrics.csv")

    # Save plots
    save_equity_and_drawdown_plots(
        rets,
        out_equity_path="reports/figures/equity_curve.png",
        out_drawdown_path="reports/figures/drawdown.png",
        title_prefix="Walk-Forward ML (Top2 L/S, 10bps)"
    )

    print("Universe:", universe)
    print("Preds shape:", preds.shape)
    print("Performance summary:", summary)
    print("Yearly metrics head:\n", yr.head())
    print("Saved: reports/figures/equity_curve.png, drawdown.png")
    print("Saved: reports/tables/summary.csv, yearly_metrics.csv")

if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f"\n Total runtime: {t1 - t0:.2f} seconds")
