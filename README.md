# AI Finance Sentiment Tracker

Système d'analyse financière end-to-end combinant ingestion de données boursières en temps réel et NLP (FinBERT) pour évaluer l'humeur du marché via un dashboard interactif.

---

## Aperçu

```
News RSS feeds  ──►  FinBERT (NLP)  ──►  Score de sentiment
yFinance API    ──►  Indicateurs techniques              ──►  Dashboard Streamlit
                         Cache local (CSV)
```

Le pipeline récupère les cours boursiers et les actualités financières, les analyse avec [FinBERT](https://huggingface.co/ProsusAI/finbert) (transformer pré-entraîné sur le corpus financier), puis affiche les résultats sur un dashboard avec des indicateurs techniques avancés.

---

## Fonctionnalités

| Catégorie | Détail |
|---|---|
| **Analyse IA** | FinBERT — classification positif / négatif / neutre sur chaque article |
| **Données** | yFinance + flux RSS ; cache CSV 24 h pour limiter les appels API |
| **Indicateurs techniques** | SMA 20/50, RSI 14, Bandes de Bollinger 20-2, MACD |
| **Dashboard** | Filtres de sentiment, favoris persistants, plage de dates, corrélation glissante sentiment/cours |
| **Qualité** | Tests unitaires `pytest` pour l'ingestion et le traitement |

---

## Stack technique

- **Python 3.x**
- **IA/ML** — PyTorch · Hugging Face Transformers (FinBERT)
- **Data** — Pandas · yFinance · feedparser
- **UI** — Streamlit · Plotly

---

## Structure du projet

```
AI-Finance-Sentiment-Tracker/
├── app.py                  # Point d'entrée Streamlit
├── src/
│   ├── ingestion.py        # Récupération prix et news (yfinance + RSS)
│   └── processing.py       # Analyse de sentiment FinBERT
├── tests/
│   ├── test_ingestion.py
│   └── test_processing.py
├── data/                   # Cache CSV généré automatiquement
├── scripts/                # Utilitaires (corrélation, etc.)
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/votre-username/AI-Finance-Sentiment-Tracker.git
cd AI-Finance-Sentiment-Tracker

python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

> Le modèle FinBERT (~400 Mo) est téléchargé automatiquement depuis Hugging Face au premier lancement.

---

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

---

## Utilisation

1. **Sélectionner un ticker** — recherche parmi les symboles populaires ou saisie libre avec validation en temps réel.
2. **Choisir la plage de dates** — début et fin de la période à analyser.
3. **Lancer l'analyse IA** — récupère les données, infère le sentiment sur chaque article.
4. **Explorer les résultats** :
   - Graphique de prix avec indicateurs techniques (SMA, RSI, Bollinger, MACD)
   - Répartition des sentiments et actualités filtrables
   - Évolution temporelle du score moyen de sentiment
   - Corrélation glissante sentiment / cours
5. **Gérer les favoris** — sauvegardez vos tickers pour un accès rapide.
6. **Forcer le rafraîchissement** — ignore le cache local pour obtenir les données les plus récentes.

---

## Tests

```bash
pytest tests/ -v
```

---

## Licence

MIT
