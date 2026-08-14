"""
CausalFolio - Price Prediction Script
======================================
Predicts the 5-day return direction and bounds using the v3 classification model.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

def get_base_dir():
    """Robustly get the base project directory, handling Colab exec()."""
    try:
        return Path(__file__).parent.parent
    except NameError:
        colab_path = Path('/content/drive/MyDrive/BSEpredictionNew')
        if colab_path.exists():
            return colab_path
        return Path.cwd()

BASE_DIR = get_base_dir()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CONFIG = {
    'config_path': str(BASE_DIR / 'config' / 'config.yaml'),
    'checkpoint_dir': str(BASE_DIR / 'checkpoints'),
    'model_name': 'causalfolio_v3.pt',
    'lookback_days': 250,
}
NEWSAPI_KEY = '259fbcbeb1ce4587b0faf50ba7286356'
MIN_DATA_FRACTION = 0.9
MIN_TICKERS_BY_INDEX = {
    'NIFTY_50': 40,
    'NIFTY_100': 80,
    'NIFTY_500': 400,
}

def setup():
    print("=" * 60)
    print("CausalFolio - Price Prediction")
    print("=" * 60)

def load_config() -> dict:
    with open(CONFIG['config_path'], 'r') as f:
        return yaml.safe_load(f)

def validate_universe_size(tickers: List[str], config: dict) -> None:
    index_name = str(config.get('universe', {}).get('index', '')).upper()
    min_expected = MIN_TICKERS_BY_INDEX.get(index_name)
    if min_expected and len(tickers) < min_expected:
        raise ValueError(
            f"Universe size {len(tickers)} is below expected minimum {min_expected} for {index_name}. "
            "Check the local NSE CSV files under data/."
        )

def get_close_series(data: pd.DataFrame, ticker: str) -> pd.Series | None:
    if isinstance(data.columns, pd.MultiIndex):
        level_0 = data.columns.get_level_values(0).unique()
        level_1 = data.columns.get_level_values(1).unique()
        if ticker in level_0:
            return data[ticker]['Close'] if 'Close' in data[ticker] else None
        if ticker in level_1:
            cols = data.xs(ticker, level=1, axis=1)
            return cols['Close'] if 'Close' in cols else None
        return None
    if 'Close' in data.columns:
        if ticker in data.columns:
            return data['Close'][ticker]
        return data['Close']
    return None

def filter_tickers_by_completeness(
    data: pd.DataFrame,
    tickers: List[str],
    min_fraction: float = MIN_DATA_FRACTION,
    verbose: bool = True
) -> List[str]:
    expected_rows = len(data.index)
    if expected_rows == 0:
        return []

    min_rows = int(np.ceil(expected_rows * min_fraction))
    valid = []
    dropped = []

    for ticker in tickers:
        close = get_close_series(data, ticker)
        if close is None:
            dropped.append(ticker)
            continue
        valid_rows = int(close.dropna().shape[0])
        if valid_rows >= min_rows:
            valid.append(ticker)
        else:
            dropped.append(ticker)

    if verbose:
        print(f"→ Tickers kept: {len(valid)} / {len(tickers)} (>= {min_fraction:.0%} data)")
        if dropped:
            print(f"  Dropped {len(dropped)} tickers due to incomplete data")

    return valid

def build_features_tensor_strict(
    features_dict: Dict[str, pd.DataFrame],
    tickers: List[str]
) -> Tuple[torch.Tensor, List[str], pd.DatetimeIndex, List[str], Dict[str, pd.DataFrame]]:
    cleaned = {}
    for ticker in tickers:
        df = features_dict.get(ticker)
        if df is None:
            continue
        df = df.dropna()
        if not df.empty:
            cleaned[ticker] = df

    if not cleaned:
        raise ValueError("No tickers with complete feature rows")

    common_dates = set.intersection(*[set(df.index) for df in cleaned.values()])
    if not common_dates:
        raise ValueError("No common dates across tickers after cleaning")

    common_dates = sorted(common_dates)
    feature_names = list(next(iter(cleaned.values())).columns)

    tensor_list = []
    valid_tickers = []
    aligned_features = {}

    for ticker in tickers:
        if ticker not in cleaned:
            continue
        aligned = cleaned[ticker].loc[common_dates, feature_names]
        if aligned.isna().any().any():
            continue
        tensor_list.append(torch.tensor(aligned.values, dtype=torch.float32))
        valid_tickers.append(ticker)
        aligned_features[ticker] = aligned

    if not valid_tickers:
        raise ValueError("No tickers with fully aligned feature data")

    tensor = torch.stack(tensor_list, dim=1)
    print(f"  ✓ Tensor: {tensor.shape} [T={tensor.shape[0]} days, N={tensor.shape[1]} stocks, F={tensor.shape[2]} features]")

    return tensor, feature_names, pd.DatetimeIndex(common_dates), valid_tickers, aligned_features
    
def load_stocks():
    from data.universe_loader import get_universe_tickers, get_sector_mapping
    config = load_config()
    tickers = get_universe_tickers(config)
    sectors = get_sector_mapping(config)
    validate_universe_size(tickers, config)
    return tickers, sectors

def fetch_data(tickers, days=250):
    from data.bse_loader import get_prices
    end = datetime.now()
    start = end - timedelta(days=days + 10)
    return get_prices(tickers, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=True)

def compute_price_indicators(data, tickers):
    indicators = {}
    for ticker in tickers:
        close = get_close_series(data, ticker)
        if close is None:
            continue
        close = close.dropna()

        if len(close) < 50:
            continue

        current_price = close.iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        momentum_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        trend_signal = "BULLISH" if current_price > sma_20 > sma_50 else "BEARISH" if current_price < sma_20 < sma_50 else "NEUTRAL"

        indicators[ticker] = {
            'current_price': current_price, 'sma_20': sma_20, 'sma_50': sma_50,
            'rsi': rsi, 'momentum_5d': momentum_5d, 'trend_signal': trend_signal
        }
    return indicators

def get_predictions(tickers, sectors, data):
    from features.classical import build_multi_stock_features
    from features.graph_builder import build_graph
    from data import news_scraper as news_module
    from models import sentiment as sentiment_module
    import importlib
    import models.gnn, models.tcn, models.model_minimal
    importlib.reload(models.gnn)
    importlib.reload(models.tcn)
    importlib.reload(models.model_minimal)
    from models.model_minimal import CausalFolioMinimal
    
    features_dict = build_multi_stock_features(data, tickers)
    tensor, _, _, valid_tickers, aligned_features = build_features_tensor_strict(features_dict, tickers)
    sector_map = {t: sectors.get(t, 'Unknown') for t in valid_tickers}
    edge_index, _ = build_graph(aligned_features, valid_tickers, sector_map, corr_threshold=0.3, max_edges_per_node=5)
    
    sentiment_scores = {t: 0.0 for t in valid_tickers}
    try:
        analyzer = sentiment_module.FinBERTSentiment()
        for t in valid_tickers:
            h = news_module.get_news_headlines(t, max_per_source=3, newsapi_key=NEWSAPI_KEY)
            if h:
                res = analyzer.analyze([x['headline'] for x in h])
                scores = res.get('scores', [0.0])
                sentiment_scores[t] = sum(scores) / len(scores) if scores else 0.0
    except:
        pass
        
    sentiment = torch.tensor([sentiment_scores[t] for t in valid_tickers], dtype=torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['model_name'])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = CausalFolioMinimal(
        num_features=checkpoint.get('config', {}).get('num_features', 10),
        num_stocks=len(valid_tickers),
        gnn_hidden=checkpoint.get('config', {}).get('gnn_hidden', 32),
        gnn_output=checkpoint.get('config', {}).get('gnn_output', 16),
        tcn_hidden=checkpoint.get('config', {}).get('tcn_hidden', 32),
        tcn_layers=checkpoint.get('config', {}).get('tcn_layers', 4),
        dropout=checkpoint.get('config', {}).get('dropout', 0.2),
        use_sentiment=True
    )
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device).eval()
    
    with torch.no_grad():
        outputs = model(tensor.to(device), edge_index.to(device), sentiment.to(device))
        vol = outputs['volatility'][-1].cpu().numpy().flatten()
        direction_logits = outputs['direction'][-1].cpu()
        direction_classes = torch.argmax(direction_logits, dim=-1).numpy()

    return valid_tickers, vol, direction_classes, sentiment_scores

def predict_prices(tickers, indicators, vol, dir_classes, sentiment, target_ticker=None):
    print("\n" + "=" * 60)
    if target_ticker:
        print(f"Step 4: Price Predictions (Calibrated with v3 Model) for {target_ticker}")
    else:
        print("Step 4: Price Predictions (Calibrated with v3 Model)")
    print("=" * 60)
    
    predictions = []
    for i, ticker in enumerate(tickers):
        if target_ticker and ticker != target_ticker:
            continue
            
        if ticker not in indicators: continue
        
        ind = indicators[ticker]
        v = vol[i]
        s = sentiment.get(ticker, 0.0)
        c = dir_classes[i]
        
        current = ind['current_price']
        
        # DOWN=0, SIDEWAYS=1, UP=2 (calibrated threshold is 2%)
        base_expected = {0: -2.5, 1: 0.0, 2: 2.5}[c]
        
        # Combine with momentum & sentiment
        expected_move_pct = base_expected + (s * 5) + (ind['momentum_5d'] * 0.1)
        expected_move_pct = max(-15, min(15, expected_move_pct))
        
        target = current * (1 + expected_move_pct / 100)
        
        dir_str = "🟢 UP" if expected_move_pct > 1.0 else "🔴 DOWN" if expected_move_pct < -1.0 else "🟡 SIDEWAYS"
        
        confidence = 40 + (abs(expected_move_pct) * 3) + ((1 - min(v, 0.4)) * 20)
        confidence = min(confidence, 90)
        if base_expected > 0 and ind['trend_signal'] == "BULLISH": confidence += 5
        
        five_day_vol = v / 7.0
        
        predictions.append({
            'Ticker': ticker, 'Current_Price': current, 'Direction': dir_str,
            'Target_Price': target, 'Expected_Move': expected_move_pct,
            'Support': current * (1 - five_day_vol), 'Resistance': current * (1 + five_day_vol),
            'Volatility': v, 'Confidence': confidence
        })
        
    predictions.sort(key=lambda x: -x['Confidence'])
    
    print("\n🎯 PRICE PREDICTIONS (Next 5 Days):")
    print("=" * 90)
    print(f"{'Ticker':<12} {'Price':>10} {'Direction':>12} {'Target':>10} {'Move':>8} {'Support':>10} {'Resist':>10} {'Conf':>6}")
    print("-" * 90)
    
    for p in predictions:
        print(f"{p['Ticker']:<12} ₹{p['Current_Price']:>8.2f} {p['Direction']:>12} "
              f"₹{p['Target_Price']:>8.2f} {p['Expected_Move']:+.1f}%    "
              f"₹{p['Support']:>8.2f} ₹{p['Resistance']:>8.2f} {p['Confidence']:>5.1f}%")
              
    return pd.DataFrame(predictions)

def predict_prices_main(target_ticker=None):
    setup()
    tickers, sectors = load_stocks()
    data = fetch_data(tickers)
    valid_tickers = filter_tickers_by_completeness(data, tickers, MIN_DATA_FRACTION, verbose=True)
    if not valid_tickers:
        raise ValueError("No tickers meet the data completeness threshold")
    indicators = compute_price_indicators(data, valid_tickers)
    used_tickers, vol, dir_classes, sentiment = get_predictions(valid_tickers, sectors, data)
    return predict_prices(used_tickers, indicators, vol, dir_classes, sentiment, target_ticker)
