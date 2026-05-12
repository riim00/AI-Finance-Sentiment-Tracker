# 📈 AI Finance Sentiment Tracker

Ce projet utilise l'Intelligence Artificielle (NLP) pour analyser le sentiment des actualités financières et corréler avec l'évolution des cours boursiers (ex: NVIDIA, AAPL).

## 🚀 Fonctionnalités
- **Collecte de Données** : Récupération des prix boursiers et actualités en temps réel via `yfinance`.
- **Analyse de Sentiment** : Utilisation du modèle spécialisé **FinBERT** (ProsusAI) pour classifier les sentiments (positif, négatif, neutre).
- **Dashboard Interactif** : Interface Streamlit pour visualiser les prix, sentiments et corrélations.
- **Modularité** : Code organisé pour faciliter l'extension (ajout de sources de données, modèles IA).

## 📁 Structure du Projet
```
.
├── app.py                 # Application Streamlit (dashboard)
├── src/
│   ├── __init__.py
│   ├── ingestion.py       # Collecte des données (prix et news)
│   └── processing.py      # Analyse IA avec FinBERT
├── data/                  # Fichiers CSV générés (ignorés par Git)
├── tests/                 # Tests unitaires (pytest)
├── notebooks/             # Analyses exploratoires (Jupyter)
├── requirements.txt       # Dépendances Python
└── README.md              # Ce fichier
```

## 🛠 Installation et Utilisation
### Prérequis
- Python 3.8+
- Compte GitHub (pour cloner)

### Installation
1. Clonez le repo :
   ```bash
   git clone https://github.com/votre-username/AI-Finance-Sentiment-Tracker.git
   cd AI-Finance-Sentiment-Tracker
   ```

2. Créez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

### Lancement
- **Dashboard** : `streamlit run app.py`
- **Tests** : `pytest tests/`
- **Ingestion manuelle** : `python src/ingestion.py` (pour un ticker spécifique)

### Exemple d'Usage
1. Lancez le dashboard.
2. Entrez un ticker (ex: NVDA) et une date de début.
3. Cliquez sur "Lancer l'analyse IA".
4. Visualisez les graphiques de prix, le camembert de sentiment et le tableau des news.

## 🧪 Tests
Les tests couvrent les fonctions clés :
- `pytest tests/test_ingestion.py` : Test de la collecte de données.
- `pytest tests/test_processing.py` : Test de l'analyse de sentiment (avec mocks pour le modèle IA).

## 🤖 Modèle IA
- **FinBERT** : Modèle BERT fine-tuné sur des données financières.
- **Précision** : Bonne pour les textes financiers, mais peut être améliorée avec fine-tuning.
- **Limites** : Chargement lent ; optimisations possibles (cache, GPU).

## 📊 Visualisations
- Graphique des prix (Plotly).
- Camembert des sentiments.
- Tableau des actualités avec scores de probabilité.

## 🔧 Améliorations Futures
- Prédiction de prix basée sur sentiment.
- Intégration d'APIs externes (NewsAPI).
- Déploiement cloud (Streamlit Cloud).
- Tests d'intégration et CI/CD.

## 📄 Licence
MIT License - Libre d'utilisation et modification.

## 👤 Auteur
[Votre Nom] - Portfolio Data Science & IA.