import pytest
import pandas as pd
import os
from src.ingestion import fetch_stock_data, fetch_news

def test_fetch_stock_data():
    # Test avec un ticker valide
    df = fetch_stock_data("AAPL", start="2023-01-01")
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert os.path.exists("data/AAPL_prices.csv")

def test_fetch_stock_data_invalid():
    # Test avec un ticker invalide
    df = fetch_stock_data("INVALID", start="2023-01-01")
    assert df is None or df.empty

def test_fetch_news():
    # Test avec un ticker valide
    news = fetch_news("NVDA")
    assert news is not None
    assert isinstance(news, list)
    if news:
        assert "title" in news[0]
        assert os.path.exists("data/NVDA_news.csv")