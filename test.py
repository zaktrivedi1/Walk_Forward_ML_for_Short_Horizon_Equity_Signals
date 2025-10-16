# quick_test_data.py
from src.data import prepare_data_pipeline, OHLCVData

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

print("hello world")
print("Columns (levels):", data.df.columns.names)
print("Tickers in data:", len(data.tickers), data.tickers[:5], "...")
print("Fields:", data.fields)
print("Data idx range:", data.df.index.min(), "→", data.df.index.max())
print("Universe (top 5 by recent dollar volume):", universe)
