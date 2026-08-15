"""
CausalFolio - Initial Training Script
======================================
Trains the model on multi-year historical BSE/NSE data using continuous
return regression and realized volatility prediction.

Usage:
    python scripts/train_initial.py --universe NIFTY_100 --start-date 2018-01-01
"""

import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
from sklearn.metrics import classification_report, mean_absolute_error

def get_base_dir():
    """Robustly get the base project directory."""
    try:
        return Path(__file__).parent.parent
    except NameError:
        return Path.cwd()

_base_dir = get_base_dir()
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = get_base_dir() / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Training-specific defaults
TRAINING_CONFIG = {
    'gnn_hidden': 128,
    'gnn_output': 64,
    'tcn_hidden': 256,
    'tcn_layers': 5,
    'dropout': 0.3,
    
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'val_split': 0.15,
    'early_stopping': 30,
    
    'lr_patience': 10,
    'lr_factor': 0.5,
}

def load_stocks_from_config(config_path=None):
    """Load stocks and sectors dynamically from universe_loader."""
    print("\n" + "=" * 60)
    print("Step 1: Loading Universe Constituents")
    print("=" * 60)
    
    from data.universe_loader import get_universe_tickers, get_sector_mapping
    
    config = load_config(config_path)
    tickers = get_universe_tickers(config)
    sectors = get_sector_mapping(config)
    
    print(f"✓ Loaded {len(tickers)} stocks for index: {config.get('universe', {}).get('index', 'Unknown')}")
    return tickers, sectors

def setup_paths(checkpoint_dir):
    """Ensure all required paths exist."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✓ Checkpoint directory ready: {checkpoint_dir}")

def load_data(tickers, start_date, end_date):
    """Load historical OHLCV data using batched bse_loader."""
    from data.bse_loader import get_prices
    
    print("\n" + "=" * 60)
    print("Step 2: Loading Historical Price Data")
    print("=" * 60)
    print(f"Tickers to fetch: {len(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    
    data = get_prices(
        tickers,
        start=start_date,
        end=end_date,
        progress=True
    )
    print(f"✓ Downloaded shape: {data.shape}")
    return data

def compute_features(data, tickers, sectors):
    """Compute classical technical features and build graph structure."""
    from features.classical import build_multi_stock_features, features_to_tensor
    from features.graph_builder import build_graph
    
    print("\n" + "=" * 60)
    print("Step 3: Feature Engineering & Graph Construction")
    print("=" * 60)
    
    # Compute features with forward returns
    features_dict = build_multi_stock_features(data, tickers, include_forward_returns=True)
    valid_tickers = [t for t in tickers if t in features_dict and len(features_dict[t]) > 0]
    
    # Convert clean input features to tensor
    tensor, feature_names, dates, valid_tickers = features_to_tensor(features_dict, valid_tickers)
    
    # Build graph with sector + correlation edges
    edge_index, _ = build_graph(
        features_dict, 
        valid_tickers, 
        sectors,
        corr_threshold=0.3,
        max_edges_per_node=5
    )
    
    print(f"✓ Feature tensor: {tensor.shape} (T={tensor.shape[0]} days, N={tensor.shape[1]} stocks, F={tensor.shape[2]} features)")
    print(f"✓ Graph edges: {edge_index.shape[1]}")
    
    return tensor, feature_names, dates, edge_index, features_dict, valid_tickers

def compute_targets(features_dict, valid_tickers, dates, horizon=5):
    """
    Compute continuous targets:
    1. vol_5d: 5-day realized volatility [T, N, 1]
    2. forward_return_5d: 5-day cumulative forward return [T, N, 1]
    
    Drops the trailing `horizon` rows where forward returns cannot be observed.
    """
    print("\n" + "=" * 60)
    print("Step 4: Computing Continuous Targets (Volatility & Returns)")
    print("=" * 60)
    
    vol_targets_list = []
    ret_targets_list = []
    
    dates = pd.DatetimeIndex(dates)
    
    for ticker in valid_tickers:
        df = features_dict[ticker].reindex(dates)
        
        # Volatility target
        if 'vol_5d' in df.columns:
            vol_targets_list.append(df['vol_5d'].values)
        else:
            vol_targets_list.append(np.zeros(len(df)))
            
        # Continuous Forward Return target
        if 'forward_return_5d' in df.columns:
            ret_targets_list.append(df['forward_return_5d'].values)
        else:
            # Fallback compute from Close
            if 'Close' in df.columns:
                f_ret = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
                ret_targets_list.append(f_ret.values)
            else:
                ret_targets_list.append(np.zeros(len(df)))
    
    vol_targets = np.stack(vol_targets_list, axis=1)  # [T, N]
    ret_targets = np.stack(ret_targets_list, axis=1)  # [T, N]
    
    # Trim the trailing unclosed forward window (last 5 rows)
    if len(dates) > horizon:
        vol_targets = vol_targets[:-horizon]
        ret_targets = ret_targets[:-horizon]
        valid_dates = dates[:-horizon]
    else:
        valid_dates = dates
    
    vol_targets_tensor = torch.tensor(np.nan_to_num(vol_targets, 0.0), dtype=torch.float32).unsqueeze(-1)
    ret_targets_tensor = torch.tensor(np.nan_to_num(ret_targets, 0.0), dtype=torch.float32).unsqueeze(-1)
    
    # Diagnostic stats on return target distribution
    ret_flat = ret_targets.flatten()
    up_count = (ret_flat > 0.005).sum()
    down_count = (ret_flat < -0.005).sum()
    flat_count = len(ret_flat) - up_count - down_count
    
    print(f"✓ Volatility targets: {vol_targets_tensor.shape}, mean={vol_targets_tensor.mean():.4f}")
    print(f"✓ Return targets: {ret_targets_tensor.shape}, mean={ret_targets_tensor.mean():.4f}, std={ret_targets_tensor.std():.4f}")
    print(f"  Distribution: UP(>+0.5%)={up_count} ({up_count/len(ret_flat)*100:.1f}%), "
          f"DOWN(<-0.5%)={down_count} ({down_count/len(ret_flat)*100:.1f}%), "
          f"FLAT={flat_count} ({flat_count/len(ret_flat)*100:.1f}%)")
    
    return vol_targets_tensor, ret_targets_tensor, valid_dates

def create_model(num_features, num_stocks, config):
    """Instantiate CausalFolioMinimal and TrainingModule."""
    from models.model_minimal import CausalFolioMinimal, TrainingModule
    
    model = CausalFolioMinimal(
        num_features=num_features,
        num_stocks=num_stocks,
        gnn_hidden=config['gnn_hidden'],
        gnn_output=config['gnn_output'],
        tcn_hidden=config['tcn_hidden'],
        tcn_layers=config['tcn_layers'],
        dropout=config['dropout'],
        use_sentiment=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n✓ Model initialized on {device.upper()}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  TCN Receptive field: {model.get_receptive_field()} trading days")
    
    trainer = TrainingModule(
        model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device
    )
    
    return model, trainer, device

def evaluate_predictions(model, features, edge_index, sentiment, vol_targets, ret_targets):
    """Comprehensive post-training evaluation across multiple directional thresholds."""
    print("\n" + "=" * 60)
    print("Step 6: Evaluation & Directional Accuracy Breakdown")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        outputs = model(features, edge_index, sentiment)
        pred_vol = outputs['volatility'].cpu().numpy().squeeze(-1)
        pred_ret = outputs['returns'].cpu().numpy().squeeze(-1)
        
    true_vol = vol_targets.cpu().numpy().squeeze(-1)
    true_ret = ret_targets.cpu().numpy().squeeze(-1)
    
    # 1. Sign Accuracy (Sign of Return: Directional accuracy)
    sign_correct = ((pred_ret > 0) == (true_ret > 0)).mean() * 100.0
    
    # 2. Confident moves (filter out near-zero noise)
    for thr in [0.005, 0.01, 0.015, 0.02]:
        mask = np.abs(pred_ret) > thr
        if mask.sum() > 0:
            conf_acc = ((pred_ret[mask] > 0) == (true_ret[mask] > 0)).mean() * 100.0
            print(f"  Directional Accuracy (Conviction |pred| > {thr*100:.1f}%): {conf_acc:.2f}% (on {mask.sum()}/{mask.size} trades)")
        else:
            print(f"  Directional Accuracy (Conviction |pred| > {thr*100:.1f}%): N/A (no trades)")
    
    # 3. Overall Metrics
    vol_mae = mean_absolute_error(true_vol.flatten(), pred_vol.flatten())
    ret_mae = mean_absolute_error(true_ret.flatten(), pred_ret.flatten())
    
    print(f"\n  Overall Directional Sign Accuracy (All samples): {sign_correct:.2f}%")
    print(f"  Volatility MAE: {vol_mae:.4f}")
    print(f"  Return MAE: {ret_mae*100:.2f}%")
    
    # 4. 3-Class Binned Representation for comparison
    thr_3c = 0.02
    pred_3c = np.where(pred_ret > thr_3c, 2, np.where(pred_ret < -thr_3c, 0, 1))
    true_3c = np.where(true_ret > thr_3c, 2, np.where(true_ret < -thr_3c, 0, 1))
    acc_3c = (pred_3c == true_3c).mean() * 100.0
    baseline_3c = (true_3c == 1).mean() * 100.0
    print(f"  3-Class Exact Bin Match (±{thr_3c*100:.0f}%): {acc_3c:.1f}% vs Sideways Baseline {baseline_3c:.1f}%")

def save_model(trainer, checkpoint_path, config, feature_names, norm_stats, valid_tickers):
    """Save model checkpoint with full normalization metadata."""
    print("\n" + "=" * 60)
    print("Step 7: Saving Model Checkpoint")
    print("=" * 60)
    
    extra_config = {
        'feature_mean': norm_stats['mean'].tolist(),
        'feature_std': norm_stats['std'].tolist(),
        'feature_names': feature_names,
        'market_neutralized': True,
        'num_stocks_train': len(valid_tickers),
        'tickers': valid_tickers,
    }
    trainer.save_checkpoint(checkpoint_path, extra_config=extra_config)
    print(f"✓ Saved to {checkpoint_path}")

def main(
    config_path=None,
    start_date='2018-01-01',
    end_date=None,
    universe='NIFTY_100',
    max_tickers=None,
    epochs=None,
    model_name='causalfolio_v3.pt'
):
    """
    Main training execution pipeline.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    print("=" * 60)
    print("CausalFolio - Scaled Model Training Pipeline")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe: {universe}, Period: {start_date} to {end_date}")
    
    base_dir = get_base_dir()
    checkpoint_dir = str(base_dir / 'checkpoints')
    setup_paths(checkpoint_dir)
    
    yaml_config = load_config(config_path)
    model_cfg = yaml_config.get('model', {})
    
    config = {
        'start_date': start_date,
        'end_date': end_date,
        'num_features': 10,
        'checkpoint_dir': checkpoint_dir,
        'model_name': model_name,
        'gnn_hidden': model_cfg.get('gnn_hidden_dim', TRAINING_CONFIG['gnn_hidden']),
        'gnn_output': model_cfg.get('gnn_hidden_dim', TRAINING_CONFIG['gnn_hidden']) // 2,
        'tcn_hidden': model_cfg.get('tcn_hidden_dim', TRAINING_CONFIG['tcn_hidden']),
        'tcn_layers': model_cfg.get('tcn_num_layers', TRAINING_CONFIG['tcn_layers']),
        'dropout': model_cfg.get('gnn_dropout', TRAINING_CONFIG['dropout']),
        **{k: v for k, v in TRAINING_CONFIG.items() if k not in ['gnn_hidden', 'gnn_output', 'tcn_hidden', 'tcn_layers', 'dropout']}
    }
    if epochs is not None:
        config['epochs'] = epochs
    
    # 1. Load universe tickers
    tickers, sectors = load_stocks_from_config(config_path)
    if universe == 'NIFTY_50':
        tickers = tickers[:50]
    elif universe == 'NIFTY_100':
        tickers = tickers[:100]
    if max_tickers is not None and len(tickers) > max_tickers:
        tickers = tickers[:max_tickers]
    sectors = {t: sectors.get(t, 'Unknown') for t in tickers}
    print(f"  Training on {len(tickers)} stocks")
    
    # 2. Download historical data
    data = load_data(tickers, start_date, end_date)
    
    # 3. Compute features & graph
    features, feature_names, dates, edge_index, features_dict, valid_tickers = compute_features(data, tickers, sectors)
    
    # 4. Market-neutralize + Z-Score normalize features
    from features.classical import market_neutralize, normalize_features
    features = market_neutralize(features, feature_names)
    features, norm_stats = normalize_features(features)
    print(f"\n✓ Features market-neutralized & normalized: mean={norm_stats['mean'].shape}, std={norm_stats['std'].shape}")
    
    # 5. Compute continuous targets (volatility + 5d forward returns)
    vol_targets, ret_targets, valid_dates = compute_targets(features_dict, valid_tickers, dates, horizon=5)
    
    # Align features with valid targets (dropping trailing unobservable horizon)
    min_T = min(features.shape[0], vol_targets.shape[0], ret_targets.shape[0])
    features = features[:min_T]
    vol_targets = vol_targets[:min_T]
    ret_targets = ret_targets[:min_T]
    
    sentiment = torch.zeros(len(valid_tickers))
    
    # 6. Instantiate model & trainer
    model, trainer, device = create_model(len(feature_names), len(valid_tickers), config)
    
    # 7. Train model
    print("\n" + "=" * 60)
    print("Step 5: Training Dual Continuous Backbone")
    print("=" * 60)
    
    features = features.to(device)
    edge_index = edge_index.to(device)
    vol_targets = vol_targets.to(device)
    ret_targets = ret_targets.to(device)
    sentiment = sentiment.to(device)
    
    history = trainer.train(
        features=features,
        edge_index=edge_index,
        vol_targets=vol_targets,
        ret_targets=ret_targets,
        sentiment=sentiment,
        epochs=config['epochs'],
        val_split=config['val_split'],
        early_stopping=config['early_stopping'],
        verbose=True
    )
    
    # 8. Detailed Evaluation
    evaluate_predictions(model, features, edge_index, sentiment, vol_targets, ret_targets)
    
    # 9. Save Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, model_name)
    save_model(trainer, checkpoint_path, config, feature_names, norm_stats, valid_tickers)
    
    print("\n" + "=" * 60)
    print("ALL TRAINING & EVALUATION TASKS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return model, trainer, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CausalFolio Scaled Training")
    parser.add_argument('--config', default=None)
    parser.add_argument('--start-date', default='2018-01-01')
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--universe', default='NIFTY_100', choices=['NIFTY_50', 'NIFTY_100', 'NIFTY_500'])
    parser.add_argument('--max-tickers', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--model-name', default='causalfolio_v3.pt')
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        universe=args.universe,
        max_tickers=args.max_tickers,
        epochs=args.epochs,
        model_name=args.model_name
    )
