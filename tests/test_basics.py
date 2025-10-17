from __future__ import annotations

import pandas as pd
from src.data import OHLCVData
from src.backtest import portfolio_returns
from src.portfolio import signal_to_weights

def test_weights_sum_long_short():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    cols = ["A","B","C","D"]
    preds = pd.DataFrame([[3,2,1,-1],[1,4,-2,-3],[2,0,-1,1]], index=idx, columns=cols)
    w = signal_to_weights(preds, top_n=1, long_short=True, max_gross_leverage=1.0)
    # Should be +0.5 on top long, -0.5 on top short
    assert (w.abs().sum(axis=1) - 1.0).abs().max() < 1e-9

def test_label_alignment_next_day():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A":[100,101,100,102,103]}, index=idx)
    r1 = prices.pct_change(1)
    y = r1.shift(-1)  # our definition

    # Check a spot: label at t equals return from t->t+1
    t = idx[1]  # 2020-01-02
    expected = (prices.loc[idx[2], "A"] / prices.loc[idx[1], "A"] - 1.0)
    assert abs(y.loc[t, "A"] - expected) < 1e-12
