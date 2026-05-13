"""
Script de test : génère des données de sentiment synthétiques sur 60 jours
pour tester le graphique de corrélation glissante.

Usage:
    python scripts/generate_test_correlation.py
Puis dans l'app, sélectionner NVDA et lancer l'analyse sur une large plage.
"""

import pandas as pd
import numpy as np
import os

TICKER = "NVDA"
N_DAYS = 60
N_ARTICLES_PER_DAY = 3
SEED = 42

np.random.seed(SEED)

os.makedirs("data", exist_ok=True)

# ── Sentiment synthétique ────────────────────────────────────────────────────
dates = pd.bdate_range(end=pd.Timestamp.today(), periods=N_DAYS)

rows = []
for date in dates:
    # Sentiment qui dérive doucement (trend + bruit)
    trend = 0.3 * np.sin(np.linspace(0, 4 * np.pi, N_DAYS)[list(dates).index(date)])
    for _ in range(N_ARTICLES_PER_DAY):
        pos = np.clip(0.4 + trend + np.random.normal(0, 0.15), 0.01, 0.98)
        neg = np.clip(0.3 - trend + np.random.normal(0, 0.12), 0.01, 0.98)
        total = pos + neg + 0.05
        pos, neg = pos / total, neg / total
        neu = max(0, 1 - pos - neg)
        sentiment = "positive" if pos > neg and pos > neu else ("negative" if neg > pos and neg > neu else "neutral")
        rows.append({
            "title":       f"Test article — {date.date()} #{_+1}",
            "publisher":   np.random.choice(["Reuters", "Bloomberg", "FT", "WSJ"]),
            "link":        "",
            "positive":    round(pos, 6),
            "negative":    round(neg, 6),
            "neutral":     round(neu, 6),
            "sentiment":   sentiment,
            "publishedAt": date.isoformat(),
        })

df_sent = pd.DataFrame(rows)
sent_path = f"data/{TICKER}_sentiment.csv"
df_sent.to_csv(sent_path, index=False)
print(f"✅ {len(df_sent)} articles synthétiques → {sent_path}")
print(f"   Dates : {dates[0].date()} → {dates[-1].date()}")
print(f"   Distribution : {df_sent['sentiment'].value_counts().to_dict()}")
print()
print("Ouvre l'app, sélectionne NVDA et clique sur 'Lancer l'analyse IA'.")
print("La corrélation glissante 14j devrait maintenant s'afficher.")
