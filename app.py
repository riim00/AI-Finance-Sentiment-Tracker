import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Finance Sentiment Tracker", layout="wide")

st.title("📈 AI Finance Sentiment Tracker : NVIDIA")
st.write("Analyse des prix et du sentiment des news via FinBERT")

# Chargement des données
prices = pd.read_csv("data/NVDA_prices.csv", index_col=0, parse_dates=True)
sentiment = pd.read_csv("data/NVDA_sentiment.csv")

# --- PARTIE 1 : Graphique des prix ---
st.subheader("Évolution du cours de l'action")
fig_price = px.line(prices, y="Close", title="Prix de clôture (NVDA)")
st.plotly_chart(fig_price, use_container_width=True)

# --- PARTIE 2 : Analyse de sentiment ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution du Sentiment")
    fig_pie = px.pie(sentiment, names='sentiment', color='sentiment',
                 color_discrete_map={'positive':'#00CC96', 'neutral':'#636EFA', 'negative':'#EF553B'})
    st.plotly_chart(fig_pie)

with col2:
    st.subheader("Détails des News")
    st.dataframe(sentiment[['title', 'sentiment', 'positive', 'negative']])

st.success("Données analysées avec succès par le modèle FinBERT !")