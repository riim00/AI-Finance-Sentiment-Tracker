# 📈 AI Finance Sentiment Tracker

Ce projet utilise l'Intelligence Artificielle (NLP) pour analyser le sentiment des actualités financières et prédire l'impact sur le cours des actions (ex: NVIDIA).

## 🚀 Fonctionnalités
- Collecte de données boursières en temps réel via `yfinance`.
- Analyse de sentiment avec le modèle spécialisé **FinBERT**.
- Dashboard interactif avec **Streamlit**.

## 📁 Structure du Projet
- `src/` : Code source (ingestion, processing, modèles).
- `data/` : Stockage des fichiers CSV (ignorés par Git).
- `notebooks/` : Expérimentations et analyses exploratoires.

## 🛠 Installation
1. `pip install -r requirements.txt`
2. `python src/ingestion.py`