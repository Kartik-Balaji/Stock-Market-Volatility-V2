"""
CausalFolio - Daily Update Script
==================================
Fine-tunes the model with new data and generates predictions.

Run this DAILY after market close (after 3:30 PM IST).

Usage:
    python scripts/daily_update.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
from pathlib import Path

def get_base_dir():
    """Robustly get the base project directory, handling Colab exec()."""
    try:
        return Path(__file__).parent.parent
    except NameError:
        colab_path = Path('/content/drive/MyDrive/BSEpredictionNew')
        if colab_path.exists():
            return colab_path
        return Path.cwd()

# Setup paths automatically 
BASE_DIR = get_base_dir()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ============================================================
# Configuration
# ============================================================

from scripts.train_initial import DIRECTION_DOWN, DIRECTION_SIDEWAYS, DIRECTION_UP, DIRECTION_THRESHOLD

CONFIG = {
    'config_path': str(BASE_DIR / 'config' / 'config.yaml'),
    'checkpoint_dir': str(BASE_DIR / 'checkpoints'),
    'model_name': 'causalfolio_v3.pt',
    'finetune_epochs': 5,
    'finetune_lr': 0.0001,
    'lookback_days': 250,
}

def setup():
    print("=" * 60)
    print("CausalFolio - Daily Update")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_stocks():
    from data.universe_loader import get_universe_tickers, get_sector_mapping
    with open(CONFIG['config_path'], 'r') as f:
        config = yaml.safe_load(f)
    tickers = get_universe_tickers(config)
    sectors = get_sector_mapping(config)
    return tickers, sectors

def fetch_recent_data(tickers, days=250):
    from data.bse_loader import get_prices
    print("\n" + "=" * 60)
    print("Step 1: Fetching Recent Data")
    print("=" * 60)
    
    end = datetime.now()
    start = end - timedelta(days=days + 10)
    
    data = get_prices(
        tickers,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        progress=True
    )
    print(f"✓ Downloaded shape: {data.shape}")
    return data

def compute_features(data, tickers, sectors):
    from features.classical import build_multi_stock_features, features_to_tensor
    from features.graph_builder import build_graph
    print("\n" + "=" * 60)
    print("Step 2: Computing Features")
    print("=" * 60)
    
    features_dict = build_multi_stock_features(data, tickers)
    tensor, feature_names, dates = features_to_tensor(features_dict, tickers)
    print(f"✓ Feature tensor: {tensor.shape}")
    
    edge_index, _ = build_graph(features_dict, tickers, sectors, corr_threshold=0.3, max_edges_per_node=5)
    print(f"✓ Graph edges: {edge_index.shape[1]}")
    
    return tensor, edge_index, features_dict, dates, feature_names

NEWSAPI_KEY = '259fbcbeb1ce4587b0faf50ba7286356'

def get_sentiment(tickers):
    from data import news_scraper as news_module
    from models import sentiment as sentiment_module
    
    print("\n" + "=" * 60)
    print("Step 3: Live Sentiment Analysis")
    print("=" * 60)
    
    sentiment_scores = {}
    try:
        try:
            analyzer = sentiment_module.FinBERTSentiment()
        except:
            analyzer = sentiment_module.SimpleSentiment()
            
        for ticker in tickers:
            try:
                headlines = news_module.get_news_headlines(ticker, max_per_source=3, newsapi_key=NEWSAPI_KEY)
                if headlines:
                    texts = [h['headline'] for h in headlines]
                    result = analyzer.analyze(texts)
                    scores = result.get('scores', [0.0])
                    score = sum(scores) / len(scores) if scores else 0.0
                    sentiment_scores[ticker] = score
                    emoji = "📈" if score > 0.1 else "📉" if score < -0.1 else "➡️"
                    print(f"  {emoji} {ticker}: {score:+.3f} ({len(headlines)} headlines)")
                else:
                    sentiment_scores[ticker] = 0.0
            except Exception as e:
                sentiment_scores[ticker] = 0.0
    except Exception as e:
        print(f"⚠ Sentiment failed: {e}")
        
    return torch.tensor([sentiment_scores.get(t, 0.0) for t in tickers], dtype=torch.float32)

def load_model(num_stocks):
    import importlib
    import models.gnn, models.tcn, models.model_minimal
    importlib.reload(models.gnn)
    importlib.reload(models.tcn)
    importlib.reload(models.model_minimal)
    from models.model_minimal import CausalFolioMinimal, TrainingModule
    
    print("\n" + "=" * 60)
    print("Step 4: Loading Model")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['model_name'])
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model {checkpoint_path} not found. Did you run train_initial.py?")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('config', {})
    
    model = CausalFolioMinimal(
        num_features=model_config.get('num_features', 10),
        num_stocks=num_stocks,
        gnn_hidden=model_config.get('gnn_hidden', 32),
        gnn_output=model_config.get('gnn_output', 16),
        tcn_hidden=model_config.get('tcn_hidden', 32),
        tcn_layers=model_config.get('tcn_layers', 4),
        dropout=model_config.get('dropout', 0.2),
        use_sentiment=True
    )
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    
    trainer = TrainingModule(model, learning_rate=CONFIG['finetune_lr'], device=device)
    print(f"✓ Model loaded: {checkpoint_path}")
    return model, trainer, device

def compute_targets(features_dict, dates, tickers):
    print("\n" + "=" * 60)
    print("Step 5: Computing Finetune Targets")
    print("=" * 60)
    
    vol_targets_list = []
    ret_values_list = []
    
    for ticker in tickers:
        df = features_dict.get(ticker, pd.DataFrame())
        if 'vol_5d' in df.columns:
            vol_targets_list.append(df['vol_5d'].values)
        else:
            vol_targets_list.append(np.zeros(len(df)))
            
        if 'forward_return_5d' in df.columns:
            ret_values_list.append(df['forward_return_5d'].values)
        elif 'return_1d' in df.columns:
            ret = df['return_1d'].shift(-5).values
            ret_values_list.append(np.nan_to_num(ret, 0))
        else:
            ret_values_list.append(np.zeros(len(df)))
            
    vol_targets = np.stack(vol_targets_list, axis=1)  
    ret_values = np.stack(ret_values_list, axis=1)    
    
    direction_labels = np.ones_like(ret_values, dtype=np.int64) * DIRECTION_SIDEWAYS
    direction_labels[ret_values < -DIRECTION_THRESHOLD] = DIRECTION_DOWN
    direction_labels[ret_values > DIRECTION_THRESHOLD] = DIRECTION_UP
    
    vol_targets_tensor = torch.nan_to_num(torch.tensor(vol_targets, dtype=torch.float32).unsqueeze(-1), 0.0)
    direction_labels_tensor = torch.tensor(direction_labels, dtype=torch.long)
    return vol_targets_tensor, direction_labels_tensor

def predict(model, features, edge_index, sentiment, tickers, device):
    print("\n" + "=" * 60)
    print("Generating Predictions")
    print("=" * 60)
    model.eval()
    
    with torch.no_grad():
        outputs = model(features.to(device), edge_index.to(device), sentiment.to(device))
        volatility_preds = outputs['volatility'][-1].cpu().numpy().flatten()
        direction_logits = outputs['direction'][-1].cpu()
        direction_preds = torch.argmax(direction_logits, dim=-1).numpy()
    
    results = []
    for i, ticker in enumerate(tickers):
        vol = volatility_preds[i]
        dir_class = direction_preds[i]
        direction = ['DOWN', 'SIDEWAYS', 'UP'][dir_class]
        results.append({'Ticker': ticker, 'Volatility': vol, 'Direction': direction})
        
    df = pd.DataFrame(results).sort_values('Volatility', ascending=False)
    
    print("\n🔮 PREDICTIONS (Next 5-Day):")
    for _, row in df.iterrows():
        emoji = '🔴' if row['Direction'] == 'DOWN' else '🟢' if row['Direction'] == 'UP' else '🟡'
        print(f"  {emoji} {row['Ticker']:<15} Dir: {row['Direction']:<10} Vol: {row['Volatility']:.2%}")
    return df

def run_daily_update():
    setup()
    tickers, sectors = load_stocks()
    data = fetch_recent_data(tickers, days=CONFIG['lookback_days'])
    sentiment = get_sentiment(tickers)
    features, edge_index, features_dict, dates, _ = compute_features(data, tickers, sectors)
    
    model, trainer, device = load_model(len(tickers))
    vol_targets, dir_targets = compute_targets(features_dict, dates, tickers)
    
    min_len = min(features.shape[0], vol_targets.shape[0], dir_targets.shape[0])
    
    print("\n" + "=" * 60)
    print("Step 6: Fine-tuning")
    print("=" * 60)
    trainer.train(
        features=features[:min_len].to(device),
        edge_index=edge_index.to(device),
        vol_targets=vol_targets[:min_len].to(device),
        direction_labels=dir_targets[:min_len].to(device),
        sentiment=sentiment.to(device),
        epochs=CONFIG['finetune_epochs'],
        val_split=0.2,
        early_stopping=100,
        verbose=True
    )
    
    trainer.save_checkpoint(os.path.join(CONFIG['checkpoint_dir'], CONFIG['model_name']))
    
    return predict(model, features, edge_index, sentiment, tickers, device)
