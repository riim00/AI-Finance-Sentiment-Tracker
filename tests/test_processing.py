import pytest
import pandas as pd
import os
import torch
from unittest.mock import patch, MagicMock
from src.processing import analyze_sentiment

def test_analyze_sentiment():
    # Mock le modèle pour éviter le chargement lourd
    with patch('src.processing.AutoTokenizer') as mock_tokenizer, \
         patch('src.processing.AutoModelForSequenceClassification') as mock_model, \
         patch('torch.no_grad'), \
         patch('torch.nn.functional.softmax') as mock_softmax:

        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_softmax.return_value = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6]
        ])  # Simuler les probabilités de sentiment

        # Créer un fichier news fictif
        news_data = pd.DataFrame({
            'title': ['Good news', 'Bad news', 'Neutral news']
        })
        os.makedirs('data', exist_ok=True)
        news_data.to_csv("data/TEST_news.csv", index=False)  # Pour le test, on utilise TEST

        # Appeler la fonction
        result = analyze_sentiment("TEST")

        # Assertions
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'sentiment' in result.columns
        assert os.path.exists("data/TEST_sentiment.csv")

        # Nettoyer
        os.remove("data/TEST_news.csv")
        os.remove("data/TEST_sentiment.csv")