"""
CausalFolio - Walk-Forward Backtest
====================================
Tests model predictions against historical data using walk-forward methodology.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass
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
    'universe': 'NIFTY_100',
    'max_tickers': None,
    'years': 3,
    'initial_train_days': 252,
    'test_days': 5,
    'step_days': 5,
    'total_test_periods': 20,
    'threshold': 2.0,
    'retrain': False,
    'retrain_epochs': 25,
}
MIN_DATA_FRACTION = 0.9
MIN_TICKERS_BY_INDEX = {
    'NIFTY_50': 40,
    'NIFTY_100': 80,
    'NIFTY_500': 400,
}

@dataclass
class PredictionResult:
    date: str
    ticker: str
    predicted_direction: str
    predicted_move_pct: float
    actual_move_pct: float
    direction_correct: bool
    volatility_pred: float
    sentiment: float

def setup():
    print("=" * 60)
    print("CausalFolio - Walk-Forward Backtest")
    print("=" * 60)

def load_config() -> dict:
    with open(CONFIG['config_path'], 'r') as f:
        return yaml.safe_load(f)

def validate_universe_size(tickers: List[str], config: dict) -> None:
    if CONFIG.get('max_tickers'):
        # Deliberately running a capped/small universe (CPU); skip the strict check.
        return
    index_name = str(CONFIG['universe'] if CONFIG.get('universe') else config.get('universe', {}).get('index', '')).upper()
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
    min_fraction: float = MIN_DATA_FRACTION
) -> List[str]:
    expected_rows = len(data.index)
    if expected_rows == 0:
        return []

    min_rows = int(np.ceil(expected_rows * min_fraction))
    valid = []

    for ticker in tickers:
        close = get_close_series(data, ticker)
        if close is None:
            continue
        valid_rows = int(close.dropna().shape[0])
        if valid_rows >= min_rows:
            valid.append(ticker)

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
    return tensor, feature_names, pd.DatetimeIndex(common_dates), valid_tickers, aligned_features

def load_stocks():
    from data.universe_loader import get_universe_tickers, get_sector_mapping
    config = load_config()
    tickers = get_universe_tickers(config)
    sectors = get_sector_mapping(config)
    
    # Restrict to a smaller universe for local CPU runs if requested
    index_name = str(config.get('universe', {}).get('index', '')).upper()
    if CONFIG['universe'] and CONFIG['universe'].upper() != index_name:
        universe_sizes = {'NIFTY_50': 50, 'NIFTY_100': 100, 'NIFTY_500': 500}
        limit = universe_sizes.get(CONFIG['universe'].upper())
        if limit:
            tickers = tickers[:limit]
    
    if CONFIG['max_tickers'] and len(tickers) > CONFIG['max_tickers']:
        tickers = tickers[:CONFIG['max_tickers']]
    
    validate_universe_size(tickers, config)
    return tickers, sectors

def fetch_historical_data(tickers, years: int = 3):
    from data.bse_loader import get_prices
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    return get_prices(tickers, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=True)

def get_price_at_date(data, ticker, date_idx):
    try:
        cols = data.columns
        if isinstance(cols, pd.MultiIndex):
            level_0 = cols.get_level_values(0)
            level_1 = cols.get_level_values(1)
            if ('Close', ticker) in cols:
                return float(data[('Close', ticker)].iloc[date_idx])
            if (ticker, 'Close') in cols:
                return float(data[(ticker, 'Close')].iloc[date_idx])
            if ticker in level_0 and 'Close' in level_1:
                return float(data.xs(ticker, level=0, axis=1)['Close'].iloc[date_idx])
            if ticker in level_1 and 'Close' in level_0:
                return float(data.xs(ticker, level=1, axis=1)['Close'].iloc[date_idx])
        elif 'Close' in cols and ticker in cols:
            return float(data['Close'][ticker].iloc[date_idx])
        return float(data['Close'].iloc[date_idx])
    except Exception:
        return None

def calculate_actual_return(data, ticker, start_idx, days=5):
    start_price = get_price_at_date(data, ticker, start_idx)
    end_price = get_price_at_date(data, ticker, start_idx + days)
    if start_price and end_price:
        return ((end_price - start_price) / start_price) * 100
    return None

def run_single_prediction(data, tickers, sectors, train_end_idx, device='cpu', retrain=False):
    from features.classical import build_multi_stock_features
    from features.graph_builder import build_graph
    import importlib
    import models.gnn, models.tcn, models.model_minimal
    importlib.reload(models.gnn)
    importlib.reload(models.tcn)
    importlib.reload(models.model_minimal)
    from models.model_minimal import CausalFolioMinimal

    train_data = data.iloc[:train_end_idx + 1]
    valid_tickers = filter_tickers_by_completeness(train_data, tickers, MIN_DATA_FRACTION)
    if not valid_tickers:
        return {}

    try:
        # Build feature frames INCLUDING true 5-day forward returns so we can
        # (a) train on clean rows and (b) derive honest class labels.
        features_dict = build_multi_stock_features(train_data, valid_tickers, include_forward_returns=True)

        cleaned = {}
        for t in valid_tickers:
            df = features_dict.get(t)
            if df is None:
                continue
            df = df.dropna()
            if not df.empty:
                cleaned[t] = df
        if not cleaned:
            return {}

        common_dates = set.intersection(*[set(df.index) for df in cleaned.values()])
        if not common_dates:
            return {}
        common_dates = sorted(common_dates)

        feature_names_all = list(next(iter(cleaned.values())).columns)
        model_cols = [c for c in feature_names_all if not c.startswith('forward_return')]

        tensor_list, vol_list, ret_list = [], [], []
        aligned_features = {}
        aligned_tickers = []
        sector_map = {}

        for t in valid_tickers:
            if t not in cleaned:
                continue
            alg = cleaned[t].loc[common_dates]
            if alg.isna().any().any():
                continue
            tensor_list.append(torch.tensor(alg[model_cols].values, dtype=torch.float32))
            vol_list.append(alg['vol_5d'].values)
            ret_list.append(alg['forward_return_5d'].values)
            aligned_tickers.append(t)
            aligned_features[t] = alg
            sector_map[t] = sectors.get(t, 'Unknown')

        if not aligned_tickers:
            return {}

        tensor = torch.stack(tensor_list, dim=1)
        edge_index, _ = build_graph(aligned_features, aligned_tickers, sector_map, corr_threshold=0.3, max_edges_per_node=5)
    except Exception:
        return {}

    # ---- Retrain a fresh model on the training window (walk-forward) ----
    if retrain:
        from features.classical import market_neutralize, normalize_features
        from models.model_minimal import TrainingModule

        T, N = tensor.shape[0], len(aligned_tickers)
        vol_targets = torch.tensor(np.stack(vol_list, axis=1), dtype=torch.float32).unsqueeze(-1)  # [T,N,1]
        ret_matrix = np.stack(ret_list, axis=1)  # [T,N]
        thr = CONFIG.get('threshold', 2.0) / 100.0
        labels = np.ones_like(ret_matrix, dtype=np.int64)
        labels[ret_matrix < -thr] = 0
        labels[ret_matrix > thr] = 2
        lab_tensor = torch.tensor(labels, dtype=torch.long)

        counts = np.bincount(labels.flatten(), minlength=3).astype(float)
        total = counts.sum()
        w = np.where(counts > 0, total / (3.0 * counts), 1.0)
        w = w / w.sum() * 3.0
        class_weights = torch.tensor(w, dtype=torch.float32)

        sentiment = torch.zeros(N)
        tensor = market_neutralize(tensor, model_cols)
        tensor, stats = normalize_features(tensor)
        stats_mean = stats['mean'].tolist()
        stats_std = stats['std'].tolist()

        model = CausalFolioMinimal(
            num_features=len(model_cols),
            num_stocks=N,
            gnn_hidden=32,
            gnn_output=16,
            tcn_hidden=64,
            tcn_layers=3,
            dropout=0.3,
            use_sentiment=True
        )
        trainer = TrainingModule(model, learning_rate=1e-3, weight_decay=1e-4, device=device)
        trainer.train(
            tensor, edge_index, vol_targets, lab_tensor, class_weights, sentiment,
            epochs=CONFIG.get('retrain_epochs', 25),
            val_split=0.15,
            early_stopping=10,
            verbose=False
        )
        model = model.to(device).eval()
        inference_tensor = tensor
    # ---- Use the pre-trained checkpoint (with persisted preprocessing) ----
    else:
        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['model_name'])
        if not os.path.exists(checkpoint_path):
            return {}

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint.get('config', {})

        model = CausalFolioMinimal(
            num_features=model_config.get('num_features', 10),
            num_stocks=len(aligned_tickers),
            gnn_hidden=model_config.get('gnn_hidden', 32),
            gnn_output=model_config.get('gnn_output', 16),
            tcn_hidden=model_config.get('tcn_hidden', 32),
            tcn_layers=model_config.get('tcn_layers', 4),
            dropout=model_config.get('dropout', 0.2),
            use_sentiment=True
        )
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device).eval()

        feature_mean = model_config.get('feature_mean')
        feature_std = model_config.get('feature_std')
        feature_names_config = model_config.get('feature_names')
        from features.classical import apply_preprocessing
        inference_tensor = apply_preprocessing(
            tensor.float(),
            feature_names_config if feature_names_config is not None else model_cols,
            feature_mean,
            feature_std,
            market_neutralized=model_config.get('market_neutralized', False)
        )
        sentiment = torch.zeros(len(aligned_tickers))

    with torch.no_grad():
        outputs = model(inference_tensor.to(device), edge_index.to(device), sentiment.to(device))
        volatility_preds = outputs['volatility'][-1].cpu().numpy().flatten()
        direction_logits = outputs['direction'][-1].cpu()
        direction_preds = torch.argmax(direction_logits, dim=-1).numpy()

    results = {}
    for i, ticker in enumerate(aligned_tickers):
        vol_pred = volatility_preds[i]
        pred_class = direction_preds[i]
        expected_move = {0: -2.5, 1: 0.0, 2: 2.5}[pred_class]
        direction = ['DOWN', 'SIDEWAYS', 'UP'][pred_class]

        results[ticker] = {
            'direction': direction,
            'expected_move': expected_move,
            'volatility': vol_pred,
            'sentiment': 0.0
        }
    return results

def calculate_metrics(results: List[PredictionResult]) -> Dict:
    if not results: return {}
    direction_correct = sum(1 for r in results if r.direction_correct)
    up_preds = [r for r in results if r.predicted_direction == 'UP']
    down_preds = [r for r in results if r.predicted_direction == 'DOWN']
    thr = CONFIG.get('threshold', 2.0)
    
    up_acc = sum(1 for r in up_preds if r.direction_correct) / len(up_preds) * 100 if up_preds else 0
    down_acc = sum(1 for r in down_preds if r.direction_correct) / len(down_preds) * 100 if down_preds else 0
    
    # Baseline: always predict the majority class (SIDEWAYS) -> accuracy = fraction |move|<=threshold%
    side_ways_correct = sum(1 for r in results if abs(r.actual_move_pct) <= thr)
    n = len(results)
    
    return {
        'total_predictions': n,
        'direction_accuracy': direction_correct / n * 100,
        'up_accuracy': up_acc,
        'down_accuracy': down_acc,
        'mae': np.mean([abs(r.predicted_move_pct - r.actual_move_pct) for r in results]),
        'sideways_baseline': side_ways_correct / n * 100,
        'lift_over_baseline': (direction_correct / n * 100) - (side_ways_correct / n * 100),
    }

def run_walkforward_backtest():
    setup()
    tickers, sectors = load_stocks()
    data = fetch_historical_data(tickers, years=CONFIG['years'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_days = len(data)
    initial_train, test_days, step, n_periods = CONFIG['initial_train_days'], CONFIG['test_days'], CONFIG['step_days'], CONFIG['total_test_periods']
    
    all_results = []
    
    print("\n" + "=" * 60)
    print("Step 2: Running Walk-Forward Test")
    print("=" * 60)
    
    for period in range(n_periods):
        train_end = initial_train + period * step
        test_end = train_end + test_days
        
        if test_end >= n_days: break
        
        test_date = data.index[train_end].strftime('%Y-%m-%d')
        print(f"  Period {period + 1}/{n_periods}: Train→{test_date}, Test next {test_days} days")
        
        predictions = run_single_prediction(data, tickers, sectors, train_end, device, retrain=CONFIG['retrain'])
        
        for ticker, pred in predictions.items():
            actual_return = calculate_actual_return(data, ticker, train_end, test_days)
            if actual_return is None: continue
            
            thr = CONFIG.get('threshold', 2.0)
            pred_dir = pred['direction']
            if pred_dir == 'UP': correct = actual_return > thr
            elif pred_dir == 'DOWN': correct = actual_return < -thr
            else: correct = abs(actual_return) <= thr
            
            all_results.append(PredictionResult(
                date=test_date, ticker=ticker, predicted_direction=pred_dir,
                predicted_move_pct=pred['expected_move'], actual_move_pct=actual_return,
                direction_correct=correct, volatility_pred=pred['volatility'], sentiment=pred['sentiment']
            ))
            
    metrics = calculate_metrics(all_results)
    print(f"\n📊 BACKTEST RESULTS:")
    print(f"  Total predictions: {metrics.get('total_predictions', 0)}")
    print(f"  Direction accuracy: {metrics.get('direction_accuracy', 0):.1f}%")
    print(f"    ↑ UP accuracy (>+{CONFIG['threshold']}%): {metrics.get('up_accuracy', 0):.1f}%")
    print(f"    ↓ DOWN accuracy (<-{CONFIG['threshold']}%): {metrics.get('down_accuracy', 0):.1f}%")
    print(f"  Baseline (always SIDEWAYS): {metrics.get('sideways_baseline', 0):.1f}%")
    print(f"  Lift over baseline: {metrics.get('lift_over_baseline', 0):+.1f}%")
    print(f"  MAE (predicted vs actual move %): {metrics.get('mae', 0):.2f}")
    
    return pd.DataFrame([vars(r) for r in all_results]), metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CausalFolio Walk-Forward Backtest')
    parser.add_argument('--universe', default=CONFIG['universe'], choices=['NIFTY_50', 'NIFTY_100', 'NIFTY_500'])
    parser.add_argument('--max-tickers', type=int, default=CONFIG['max_tickers'])
    parser.add_argument('--model-name', default=CONFIG['model_name'])
    parser.add_argument('--years', type=int, default=CONFIG['years'])
    parser.add_argument('--periods', type=int, default=CONFIG['total_test_periods'])
    parser.add_argument('--threshold', type=float, default=CONFIG['threshold'], help='Direction threshold %% for UP/DOWN (default 2.0)')
    parser.add_argument('--retrain', action='store_true', help='Walk-forward retrain a fresh model per period')
    parser.add_argument('--retrain-epochs', type=int, default=CONFIG['retrain_epochs'])
    args = parser.parse_args()
    
    CONFIG['universe'] = args.universe
    CONFIG['max_tickers'] = args.max_tickers
    CONFIG['model_name'] = args.model_name
    CONFIG['years'] = args.years
    CONFIG['total_test_periods'] = args.periods
    CONFIG['threshold'] = args.threshold
    CONFIG['retrain'] = args.retrain
    CONFIG['retrain_epochs'] = args.retrain_epochs
    
    results, metrics = run_walkforward_backtest()
