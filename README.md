# Walk-Forward ML for Short-Horizon Equity Signals

**Goal:** Demonstrate a realistic, leakage-safe ML pipeline for short-horizon equity signals, including walk-forward training, portfolio construction with costs, and performance reporting.

## Highlights
- **Leakage control:** Strict time-ordered splits, labels shifted to predict **t→t+1**.
- **Walk-forward retraining:** Expanding window, monthly by default.
- **Features:** Momentum, mean reversion, volatility (ATR/realized), RSI, volume z-scores.
- **Portfolio:** Top-N long / bottom-N short, caps gross leverage, transaction costs.
- **Reporting:** Equity curve, drawdown, yearly Sharpe/CAGR, turnover.

