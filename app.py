import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Import de tes fonctions personnalisées
from src.ingestion import fetch_stock_data, fetch_news
from src.processing import analyze_sentiment

st.set_page_config(page_title="AI Finance Sentiment Tracker", layout="wide")

@st.cache_data
def load_price_data(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

@st.cache_data
def load_sentiment_data(path):
    return pd.read_csv(path)

# --- BARRE LATÉRALE (Configuration) ---
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Symbole Boursier (Ticker)", value="NVDA").upper()
start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2024-01-01"))

sentiment_filter = st.sidebar.selectbox(
    "Filtrer le sentiment",
    options=["Tous", "positive", "negative", "neutral"],
    help="Affiche uniquement les actualités du sentiment sélectionné."
)

run_button = st.sidebar.button("Lancer l'analyse IA")

# --- TITRE PRINCIPAL ---
st.title(f"📈 AI Finance Sentiment Tracker : {ticker}")
st.write(f"Analyse en temps réel pour **{ticker}** utilisant **FinBERT** (NLP).")

# --- LOGIQUE D'ANALYSE ---
if run_button:
    with st.spinner(f"Récupération des données et analyse de sentiment pour {ticker}..."):
        # 1. Ingestion
        fetch_stock_data(ticker, start=start_date)
        fetch_news(ticker)
        
        # 2. IA Processing
        analyze_sentiment(ticker)
        st.sidebar.success("Analyse terminée !")

# --- AFFICHAGE DES RÉSULTATS ---
try:
    # Chemins des fichiers
    price_path = f"data/{ticker}_prices.csv"
    sent_path = f"data/{ticker}_sentiment.csv"

    if os.path.exists(price_path) and os.path.exists(sent_path):
        # Chargement
        prices = load_price_data(price_path)
        raw_sentiment = load_sentiment_data(sent_path)
        sentiment = raw_sentiment

        if sentiment_filter != "Tous":
            sentiment = raw_sentiment[raw_sentiment['sentiment'] == sentiment_filter]

        # --- MÉTRIQUES DE SYNTHÈSE ---
        counts = raw_sentiment['sentiment'].value_counts().to_dict()
        total_news = len(raw_sentiment)
        st.subheader("Synthèse du sentiment")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total news", total_news)
        m2.metric("Positif", counts.get('positive', 0))
        m3.metric("Neutre", counts.get('neutral', 0))
        m4.metric("Négatif", counts.get('negative', 0))

        # --- GRAPHIQUE DES PRIX ---
        st.subheader(f"Évolution du cours : {ticker}")
        fig_price = px.line(prices, y="Close", template="plotly_dark")
        st.plotly_chart(fig_price, use_container_width=True)

        if sentiment.empty:
            st.warning("Aucun article ne correspond au filtre sélectionné. Changez le filtre ou réexécutez l'analyse.")
        else:
            # --- ANALYSE DE SENTIMENT ---
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Humeur du Marché")
                fig_pie = px.pie(sentiment, names='sentiment', color='sentiment',
                             color_discrete_map={'positive':'#00CC96', 'neutral':'#636EFA', 'negative':'#EF553B'})
                st.plotly_chart(fig_pie)

            with col2:
                st.subheader("Détails des dernières actualités")
                st.dataframe(sentiment[['title', 'sentiment', 'positive', 'negative']], use_container_width=True)

            # --- CORRÉLATION SENTIMENT-PRIX ---
            st.subheader("Corrélation Sentiment & Prix")
            sentiment_score = sentiment['positive'].mean() - sentiment['negative'].mean()
            avg_price = prices['Close'].mean()

            fig_corr = px.scatter(
                x=[sentiment_score],
                y=[avg_price],
                labels={'x': 'Score Sentiment Moyen', 'y': 'Prix Moyen de Clôture'},
                title=f"Corrélation pour {ticker}"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.write(f"**Score Sentiment Moyen** : {sentiment_score:.2f} (positif - négatif)")
            st.write(f"**Prix Moyen** : {avg_price:.2f}")
            
    else:
        st.info("👋 Entrez un ticker dans la barre latérale et cliquez sur 'Lancer l'analyse' pour voir les résultats.")

except Exception as e:
    st.error(f"Erreur lors de l'affichage : {e}")

st.divider()
st.caption("Projet Data & IA - Portfolio | Modèle : ProsusAI/FinBERT | Source : Yahoo Finance")