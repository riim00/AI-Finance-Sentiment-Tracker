import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start="2023-01-01"):
    """Télécharge les prix historiques et les sauvegarde dans data/"""
    print(f"📥 Téléchargement des données de prix pour {ticker}...")
    
    df = yf.download(ticker, start=start)
    
    if df.empty:
        print(f"❌ Erreur : Aucune donnée de prix trouvée pour {ticker}.")
        return None

    os.makedirs('data', exist_ok=True)
    path = f"data/{ticker}_prices.csv"
    df.to_csv(path)
    print(f"✅ Prix sauvegardés : {path}")
    return df

def fetch_news(ticker):
    """Récupère les dernières news et gère les erreurs de clés manquantes"""
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
        # On utilise .get() pour éviter le crash si une information est absente
        # Certains titres sont parfois dans la clé 'title', d'autres dans 'headline'
        title = n.get('title') or n.get('headline')
        publisher = n.get('publisher', 'Unknown')
        link = n.get('link', '')
        
        if title:
            news_list.append({
                "title": title,
                "publisher": publisher,
                "link": link
            })
    
    if not news_list:
        print("⚠️ Liste de news vide après filtrage.")
        return None

    df_news = pd.DataFrame(news_list)
    
    os.makedirs('data', exist_ok=True)
    path_news = f"data/{ticker}_news.csv"
    df_news.to_csv(path_news, index=False)
    print(f"✅ {len(df_news)} news sauvegardées dans {path_news}")
    return df_news

if __name__ == "__main__":
    ticker_symbol = "NVDA"
    # Étape 1 : Prix
    fetch_stock_data(ticker_symbol)
    # Étape 2 : News
    fetch_news(ticker_symbol)