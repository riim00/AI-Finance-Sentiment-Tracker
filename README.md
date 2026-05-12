# 📈 AI Finance Sentiment Tracker

Un système d'analyse financière de bout en bout qui combine l'ingestion de données boursières en temps réel et le Deep Learning (NLP) pour évaluer l'humeur du marché.

## ✨ Points Forts du Projet
- **Architecture NLP** : Utilisation de **FinBERT** (ProsusAI), un modèle transformer optimisé pour le langage financier.
- **Performance** : Implémentation de caches (`st.cache_data`) et de traitement par batch pour une inférence rapide, plus cache local des données API pour éviter les téléchargements répétés.
- **Visualisation Dynamique** : Dashboard interactif avec Streamlit et Plotly, filtres de sentiment, sélection des actualités clés, évolution temporelle du score de sentiment, recherche de tickers avec plage de dates personnalisable, gestion des favoris persistants, validation en temps réel des tickers, et indicateurs techniques avancés (moyennes mobiles SMA 20/50, RSI 14, bandes de Bollinger 20-2).
- **Qualité de Code** : Tests unitaires automatisés avec `pytest` et gestion rigoureuse des erreurs d'API.

## 🛠️ Stack Technique
- **Langage** : Python 3.x
- **IA/ML** : PyTorch, Hugging Face (Transformers)
- **Data** : Pandas, YFinance
- **UI** : Streamlit, Plotly

## 🚀 Installation & Utilisation
1. Clonez le repo : `git clone https://github.com/votre-username/votre-repo.git`
2. Installez les dépendances : `pip install -r requirements.txt`
3. Lancez l'application : `streamlit run app.py`

## 📖 Utilisation
1. **Sélection du stock** : Utilisez la barre de recherche pour filtrer les tickers populaires ou saisissez un symbole personnalisé (validation en temps réel avec suggestions).
2. **Gestion des favoris** : Cliquez sur "⭐ Ajouter aux favoris" pour sauvegarder un ticker. Les favoris sont persistants pendant la session.
3. **Plage de dates** : Choisissez les dates de début et de fin pour analyser une période spécifique.
4. **Filtre de sentiment** : Filtrez les actualités par sentiment (positif, négatif, neutre ou tous).
5. **Indicateurs techniques** : Cochez "Afficher les indicateurs techniques" pour ajouter les moyennes mobiles (SMA 20 et SMA 50) au graphique des prix, "Afficher le RSI" pour voir l'Indice de Force Relative avec niveaux de surachat/survente, et "Afficher les bandes de Bollinger" pour identifier les zones de volatilité.
6. **Rafraîchissement** : Cochez "Forcer le rafraîchissement des données" pour ignorer le cache local (utile pour les données très récentes).
7. **Lancer l'analyse** : Cliquez sur "Lancer l'analyse IA" pour récupérer les données et analyser le sentiment.
8. **Explorer les résultats** : Visualisez les graphiques de prix, le sentiment, les actualités clés et l'évolution temporelle.