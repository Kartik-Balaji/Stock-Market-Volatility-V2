"""
CausalFolio - FinBERT Sentiment Analysis
==========================================
Uses FinBERT (pre-trained on financial text) to extract sentiment from news.

Key concepts:
- FinBERT: BERT fine-tuned on financial news
- Outputs: positive, negative, neutral probabilities
- Sentiment score: positive - negative (range: -1 to +1)

Usage:
    from models.sentiment import FinBERTSentiment
    
    sentiment = FinBERTSentiment()
    scores = sentiment.analyze(["Market rallies on strong earnings"])
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
import numpy as np

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers not installed. Run: pip install transformers")
    HAS_TRANSFORMERS = False


class FinBERTSentiment(nn.Module):
    """
    FinBERT-based sentiment analyzer for financial text.
    
    Uses ProsusAI/finbert which is pre-trained on:
    - Financial news articles
    - Analyst reports
    - Earnings calls
    
    Outputs sentiment scores in range [-1, +1]:
    - +1 = Very positive (bullish)
    - 0 = Neutral
    - -1 = Very negative (bearish)
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        max_length: int = 128
    ):
        """
        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
            device: 'cuda', 'cpu', or None (auto-detect)
            max_length: Maximum token length for input text
        """
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required. Install with: pip install transformers")
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load tokenizer and model
        print(f"Loading FinBERT from '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        self.max_length = max_length
        
        # FinBERT label mapping
        self.labels = ['positive', 'negative', 'neutral']
        
        print(f"✓ FinBERT loaded on {device}")
    
    @torch.no_grad()
    def analyze(self, texts: Union[str, List[str]]) -> Dict:
        """
        Analyze sentiment of text(s).
        
        Args:
            texts: Single string or list of strings
        
        Returns:
            Dict with:
            - 'scores': sentiment scores [-1 to +1]
            - 'labels': predicted labels ('positive', 'negative', 'neutral')
            - 'probabilities': raw probabilities for each class
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # Get predictions
        pred_indices = probs.argmax(dim=-1)
        pred_labels = [self.labels[i] for i in pred_indices.tolist()]
        
        # Calculate sentiment score: positive - negative
        pos_probs = probs[:, 0].cpu().numpy()  # positive
        neg_probs = probs[:, 1].cpu().numpy()  # negative
        scores = pos_probs - neg_probs  # Range: -1 to +1
        
        return {
            'scores': scores.tolist(),
            'labels': pred_labels,
            'probabilities': probs.cpu().numpy().tolist()
        }
    
    def batch_analyze(
        self, 
        texts: List[str], 
        batch_size: int = 16
    ) -> Dict:
        """
        Analyze large list of texts in batches.
        
        Args:
            texts: List of strings
            batch_size: Batch size for processing
        
        Returns:
            Same format as analyze()
        """
        all_scores = []
        all_labels = []
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.analyze(batch)
            all_scores.extend(result['scores'])
            all_labels.extend(result['labels'])
            all_probs.extend(result['probabilities'])
        
        return {
            'scores': all_scores,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def analyze_stock_news(
        self, 
        news_dict: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Analyze news for multiple stocks.
        
        Args:
            news_dict: {ticker: [headline1, headline2, ...]}
        
        Returns:
            {ticker: average_sentiment_score}
        """
        stock_sentiments = {}
        
        for ticker, headlines in news_dict.items():
            if not headlines:
                stock_sentiments[ticker] = 0.0  # Neutral if no news
                continue
            
            result = self.analyze(headlines)
            avg_score = np.mean(result['scores'])
            stock_sentiments[ticker] = float(avg_score)
        
        return stock_sentiments


class SimpleSentiment:
    """
    Simple rule-based sentiment (fallback without FinBERT).
    
    Uses keyword matching for basic sentiment analysis.
    Not as accurate as FinBERT but works without GPU/transformers.
    """
    
    POSITIVE_WORDS = {
        'rally', 'surge', 'gain', 'rise', 'jump', 'soar', 'climb',
        'bullish', 'growth', 'profit', 'beat', 'strong', 'positive',
        'upgrade', 'buy', 'outperform', 'record', 'high', 'boost'
    }
    
    NEGATIVE_WORDS = {
        'fall', 'drop', 'decline', 'crash', 'plunge', 'sink', 'tumble',
        'bearish', 'loss', 'miss', 'weak', 'negative', 'downgrade',
        'sell', 'underperform', 'low', 'cut', 'warning', 'fear'
    }
    
    def analyze(self, texts: Union[str, List[str]]) -> Dict:
        """
        Analyze sentiment using keyword matching.
        
        Args:
            texts: Single string or list of strings
        
        Returns:
            Dict with 'scores' and 'labels'
        """
        if isinstance(texts, str):
            texts = [texts]
        
        scores = []
        labels = []
        
        for text in texts:
            words = set(text.lower().split())
            
            pos_count = len(words & self.POSITIVE_WORDS)
            neg_count = len(words & self.NEGATIVE_WORDS)
            
            # Calculate score
            total = pos_count + neg_count
            if total == 0:
                score = 0.0
                label = 'neutral'
            else:
                score = (pos_count - neg_count) / total
                if score > 0.2:
                    label = 'positive'
                elif score < -0.2:
                    label = 'negative'
                else:
                    label = 'neutral'
            
            scores.append(score)
            labels.append(label)
        
        return {
            'scores': scores,
            'labels': labels
        }
    
    def analyze_stock_news(
        self, 
        news_dict: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Analyze news for multiple stocks."""
        stock_sentiments = {}
        
        for ticker, headlines in news_dict.items():
            if not headlines:
                stock_sentiments[ticker] = 0.0
                continue
            
            result = self.analyze(headlines)
            avg_score = np.mean(result['scores'])
            stock_sentiments[ticker] = float(avg_score)
        
        return stock_sentiments


# ===================
# Quick Test
# ===================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sentiment Analysis")
    print("=" * 60)
    
    # Test headlines
    test_headlines = [
        "TCS reports record quarterly profit, stock surges 5%",
        "HDFC Bank faces regulatory concerns, shares drop",
        "Market trades flat amid global uncertainty",
        "Reliance announces major acquisition, investors optimistic",
        "IT sector witnesses massive sell-off on US recession fears"
    ]
    
    print("\nTest headlines:")
    for i, h in enumerate(test_headlines):
        print(f"  {i+1}. {h}")
    
    # Test SimpleSentiment (always works)
    print("\n" + "-" * 40)
    print("Testing SimpleSentiment (rule-based):")
    simple = SimpleSentiment()
    result = simple.analyze(test_headlines)
    
    for h, score, label in zip(test_headlines, result['scores'], result['labels']):
        print(f"  [{label:^8}] {score:+.2f} | {h[:50]}...")
    
    # Test FinBERT (if available)
    if HAS_TRANSFORMERS:
        print("\n" + "-" * 40)
        print("Testing FinBERT:")
        try:
            finbert = FinBERTSentiment()
            result = finbert.analyze(test_headlines)
            
            for h, score, label in zip(test_headlines, result['scores'], result['labels']):
                print(f"  [{label:^8}] {score:+.2f} | {h[:50]}...")
        except Exception as e:
            print(f"  FinBERT error: {e}")
    else:
        print("\nSkipping FinBERT (transformers not installed)")
    
    print("\n" + "=" * 60)
    print("Sentiment tests complete!")
    print("=" * 60)
