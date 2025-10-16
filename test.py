from src.data import prepare_data_pipeline, OHLCVData
from src.features import make_feature_panel, select_universe_features
from src.labels import make_next_day_return_label
from src.features import make_feature_panel, select_universe_features
from src.labels import make_next_day_return_label
from src.model import walkforward_predict
from src.portfolio import signal_to_weights, apply_transaction_costs
from src.backtest import portfolio_returns, summarize_performance


# A small, liquid set to start — we’ll expand to ~100 later.
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "SPY"]

data, universe = prepare_data_pipeline(
    tickers=tickers,
    start="2012-01-01",
    end=None,  # up to latest
    cache_path="data/prices.parquet",
    min_start_date=None,
    universe_top_n=5,
)

print("Columns (levels):", data.df.columns.names)
print("Tickers in data:", len(data.tickers), data.tickers[:5], "...")
print("Fields:", data.fields)
print("Data idx range:", data.df.index.min(), "→", data.df.index.max())
print("Universe (top 5 by recent dollar volume):", universe)

# Build features for all tickers then filter to universe
feat_panel = make_feature_panel(data)
feat_univ = select_universe_features(feat_panel, universe)

print("Feature panel shape (all):", feat_panel.shape)
print("Feature panel shape (universe):", feat_univ.shape)

# Build labels (next-day returns)
y = make_next_day_return_label(data)

# Align to universe
y_univ = y[universe]

# Check alignment — drop rows where everything is NaN initially
common_index = feat_univ.index.intersection(y_univ.index)
feat_univ = feat_univ.loc[common_index]
y_univ = y_univ.loc[common_index]

print("Label shape (universe):", y_univ.shape)

# Peek at a couple columns to ensure shifts look right
t0 = feat_univ.index.min()
print("First feature date:", t0)
print("Example feature columns for", universe[0], ":\n", feat_univ.xs(universe[0], axis=1, level="ticker").head(3))
print("Example labels head:\n", y_univ[universe[0]].head(3))

# 2) Walk-forward predictions (expanding window, monthly)
preds = walkforward_predict(
    feat_panel_universe=feat_univ,
    y_universe=y_univ,
    train_start="2013-01-01",
    retrain_freq="M",
    min_train_days=126,
)

print("Preds shape:", preds.shape)
print(preds.tail())

# 3) Portfolio construction
weights = signal_to_weights(preds, top_n=2, long_short=True, max_gross_leverage=1.0)

# 4) Costs and backtest
costs = apply_transaction_costs(weights, cost_bps_roundtrip=10.0)
rets = portfolio_returns(weights, y_univ, costs)

summary = summarize_performance(rets)
print("Performance summary:", summary)
print("Start/End:", rets.index.min(), "→", rets.index.max())
print("Turnover (median daily):", float(weights.diff().abs().sum(axis=1).median()))