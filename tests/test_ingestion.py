import pytest
import pandas as pd
import os
from src.ingestion import fetch_stock_data, fetch_news


def test_fetch_stock_data():
    df = fetch_stock_data("AAPL", start="2023-01-01")
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert os.path.exists("data/AAPL_prices.csv")


def test_fetch_stock_data_invalid():
    df = fetch_stock_data("INVALID_TICKER_XYZ", start="2023-01-01")
    assert df is None or (isinstance(df, pd.DataFrame) and df.empty)


def test_fetch_news():
    news = fetch_news("NVDA")
    assert news is not None
    assert isinstance(news, pd.DataFrame)
    if not news.empty:
        assert "title" in news.columns
        assert os.path.exists("data/NVDA_news.csv")
