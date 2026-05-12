import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from functools import lru_cache

MODEL_NAME = "ProsusAI/finbert"

@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    print(f"🧠 Chargement du modèle IA ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def analyze_sentiment(ticker, batch_size=16):
    news_path = f"data/{ticker}_news.csv"
    if not os.path.exists(news_path):
        print("❌ Fichier news introuvable. Lance ingestion.py d'abord.")
        return

    df = pd.read_csv(news_path)
    if 'title' not in df.columns:
        print("❌ Colonne 'title' manquante dans le fichier de news.")
        return

    titles = df['title'].dropna().astype(str).tolist()
    if not titles:
        print("⚠️ Aucun titre valide à analyser.")
        return

    tokenizer, model = load_model_and_tokenizer()
    print(f"📊 Analyse de {len(titles)} titres...")

    predictions = []
    for start in range(0, len(titles), batch_size):
        batch_titles = titles[start:start + batch_size]
        inputs = tokenizer(batch_titles, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions.append(batch_preds.cpu())

    if not predictions:
        print("❌ Aucune prédiction générée.")
        return

    predictions = torch.vstack(predictions)
    df['positive'] = predictions[:, 0].numpy()
    df['negative'] = predictions[:, 1].numpy()
    df['neutral'] = predictions[:, 2].numpy()

    labels = ['positive', 'negative', 'neutral']
    df['sentiment'] = [labels[i] for i in torch.argmax(predictions, dim=1).tolist()]

    os.makedirs('data', exist_ok=True)
    output_path = f"data/{ticker}_sentiment.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Analyse terminée. Résultats sauvegardés dans {output_path}")
    return df

if __name__ == "__main__":
    analyze_sentiment("NVDA")