import yfinance as yf
import pandas as pd
import feedparser
import os
from datetime import datetime, timedelta, timezone


def _save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _is_cache_valid(path: str, max_age_hours: int = 24) -> bool:
    if not os.path.exists(path):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - file_time < timedelta(hours=max_age_hours)


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker, start="2023-01-01", force_refresh=False):
    path = f"data/{ticker}_prices.csv"

    if not force_refresh and _is_cache_valid(path):
        print(f"📂 Chargement des données de prix depuis le cache pour {ticker}...")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    print(f"📥 Téléchargement des données de prix pour {ticker}...")
    df = yf.download(ticker, start=start, progress=False)

    if df.empty:
        print(f"❌ Erreur : Aucune donnée de prix trouvée pour {ticker}.")
        return None

    # Flatten MultiIndex columns (yfinance returns MultiIndex for single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    _save_dataframe(df, path)
    print(f"✅ Prix sauvegardés : {path}")
    return df


# ── News fetchers ─────────────────────────────────────────────────────────────

def _parse_rss_entry(entry, default_publisher="Unknown"):
    title = entry.get("title", "").strip()
    if not title:
        return None

    link = entry.get("link", "")

    # Publisher: try source.title, then feed author, then default
    src = entry.get("source")
    if isinstance(src, dict):
        publisher = src.get("title", default_publisher)
    else:
        publisher = entry.get("author", default_publisher)

    # Timestamp: published_parsed (struct_time) → ISO string
    published_at = None
    if entry.get("published_parsed"):
        try:
            published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
    if not published_at and entry.get("updated_parsed"):
        try:
            published_at = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass

    return {"title": title, "publisher": publisher, "link": link, "publishedAt": published_at}


def _fetch_yahoo_rss(ticker: str) -> list[dict]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        articles = [_parse_rss_entry(e, "Yahoo Finance") for e in feed.entries]
        return [a for a in articles if a]
    except Exception as e:
        print(f"⚠️ Yahoo RSS failed: {e}")
        return []


def _fetch_google_news_rss(ticker: str) -> list[dict]:
    query = f"{ticker}+stock"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        articles = [_parse_rss_entry(e, "Google News") for e in feed.entries]
        return [a for a in articles if a]
    except Exception as e:
        print(f"⚠️ Google News RSS failed: {e}")
        return []


def _fetch_yfinance_news(ticker: str) -> list[dict]:
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"⚠️ yfinance news failed: {e}")
        return []

    articles = []
    for n in news:
        title = n.get("title") or n.get("headline")
        if not title:
            continue
        published_at = n.get("providerPublishTime") or n.get("pubDate") or n.get("date")
        if isinstance(published_at, (int, float)):
            published_at = datetime.fromtimestamp(int(published_at), tz=timezone.utc).isoformat()
        elif isinstance(published_at, datetime):
            published_at = published_at.isoformat()
        articles.append({
            "title":       title,
            "publisher":   n.get("publisher") or n.get("provider", "Unknown"),
            "link":        n.get("link", ""),
            "publishedAt": published_at,
        })
    return articles


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_news(ticker, force_refresh=False):
    """
    Fetch news from RSS feeds (Yahoo Finance + Google News) with yfinance fallback.
    Accumulates articles over time: new articles are merged with existing CSV so
    the rolling correlation has enough data after a few refreshes.
    force_refresh=True replaces the CSV entirely instead of appending.
    """
    path_news = f"data/{ticker}_news.csv"

    # Use cache when fresh and not forcing
    if not force_refresh and _is_cache_valid(path_news, max_age_hours=6):
        print(f"📂 Chargement des news depuis le cache pour {ticker}...")
        return pd.read_csv(path_news)

    print(f"📰 Récupération des news pour {ticker} (RSS + yfinance)...")

    # Collect from all sources; deduplicate by title
    all_articles: list[dict] = []
    all_articles += _fetch_yahoo_rss(ticker)
    all_articles += _fetch_google_news_rss(ticker)
    all_articles += _fetch_yfinance_news(ticker)

    if not all_articles:
        print(f"⚠️ Aucune news trouvée pour {ticker}.")
        if os.path.exists(path_news):
            return pd.read_csv(path_news)
        return None

    df_new = pd.DataFrame(all_articles).drop_duplicates(subset=["title"]).reset_index(drop=True)

    # Accumulate: merge with historical CSV unless force_refresh
    if not force_refresh and os.path.exists(path_news):
        df_existing = pd.read_csv(path_news)
        df_combined = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .drop_duplicates(subset=["title"])
            .reset_index(drop=True)
        )
    else:
        df_combined = df_new

    _save_dataframe(df_combined, path_news)
    print(f"✅ {len(df_new)} nouvelles news → {len(df_combined)} total sauvegardé dans {path_news}")
    return df_combined


if __name__ == "__main__":
    fetch_stock_data("NVDA")
    fetch_news("NVDA")
