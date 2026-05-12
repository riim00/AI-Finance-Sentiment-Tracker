import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


def _save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _is_cache_valid(path: str, max_age_hours: int = 24) -> bool:
    """Vérifie si le fichier cache est récent."""
    if not os.path.exists(path):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - file_time < timedelta(hours=max_age_hours)


def fetch_stock_data(ticker, start="2023-01-01", force_refresh=False):
    """Télécharge les prix historiques et les sauvegarde dans data/, avec cache local."""
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

def fetch_news(ticker, force_refresh=False):
    """Récupère les dernières news et gère les erreurs de clés manquantes, avec cache local."""
    path_news = f"data/{ticker}_news.csv"
    
    if not force_refresh and _is_cache_valid(path_news, max_age_hours=6):
        print(f"📂 Chargement des news depuis le cache pour {ticker}...")
        df_news = pd.read_csv(path_news)
        return df_news
    
    print(f"📰 Récupération des news pour {ticker}...")
    t = yf.Ticker(ticker)
    
    try:
        news = t.news
    except Exception as e:
        print(f"❌ Erreur lors de l'accès aux news : {e}")
        return None
    
    if not news:
        print(f"⚠️ Aucune news trouvée pour {ticker}.")
        return None
    
    news_list = []
    for n in news:
        title = n.get('title') or n.get('headline')
        publisher = n.get('publisher') or n.get('provider', 'Unknown')
        link = n.get('link', '')
        published_at = n.get('providerPublishTime') or n.get('pubDate') or n.get('date')

        if isinstance(published_at, (int, float)):
            published_at = datetime.fromtimestamp(int(published_at)).isoformat()
        elif isinstance(published_at, datetime):
            published_at = published_at.isoformat()

        if title:
            news_list.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "publishedAt": published_at,
            })
    
    if not news_list:
        print("⚠️ Liste de news vide après filtrage.")
        return None

    df_news = pd.DataFrame(news_list)
    _save_dataframe(df_news, path_news)
    print(f"✅ {len(df_news)} news sauvegardées dans {path_news}")
    return df_news

if __name__ == "__main__":
    ticker_symbol = "NVDA"
    fetch_stock_data(ticker_symbol)
    fetch_news(ticker_symbol)