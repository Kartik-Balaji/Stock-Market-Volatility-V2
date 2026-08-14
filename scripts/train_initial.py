"""
CausalFolio - Initial Training Script
======================================
Trains the model on 5 years of historical BSE data.

Run this ONCE to create the base model, then use daily_update.py
for ongoing fine-tuning.

Usage (in Colab):
    !python train_initial.py

Or run cells from this file in a notebook.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ============================================================
# Configuration
# ============================================================

import yaml

# ============================================================
# Configuration - Load from config.yaml
# ============================================================

from pathlib import Path

def get_base_dir():
    """Robustly get the base project directory, handling Colab exec()."""
    try:
        return Path(__file__).parent.parent
    except NameError:
        colab_path = Path('/content/drive/MyDrive/BSEpredictionNew')
        if colab_path.exists():
            return colab_path
        # Fallback when running via exec() in Colab where __file__ is not defined.
        # This assumes you have changed your current working directory to the project root.
        return Path.cwd()

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = get_base_dir() / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Ensure base project dir is importable (data/, features/, models/, config/)
_base_dir = get_base_dir()
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

# Training-specific settings (can override config.yaml)
TRAINING_CONFIG = {
    # Model settings (INTENSIVE)
    'gnn_hidden': 64,
    'gnn_output': 32,
    'tcn_hidden': 128,
    'tcn_layers': 7,       # Receptive field: 63 days
    'dropout': 0.2,
    
    # Training settings (INTENSIVE)
    'epochs': 300,
    'batch_size': 16, # Reduced from 64 to prevent OOM with large receptive fields & 500 stocks
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'val_split': 0.15,
    'early_stopping': 50,
    
    # Learning rate scheduling
    'lr_patience': 15,
    'lr_factor': 0.5,
}

# ============================================================
# Stock Loading from config.yaml
# ============================================================

def load_stocks_from_config(config_path=None):
    """
    Load stocks and sectors dynamically from universe_loader.
    """
    print("\n" + "=" * 60)
    print("Step 1: Loading Stocks from universe_loader")
    print("=" * 60)
    
    from data.universe_loader import get_universe_tickers, get_sector_mapping
    
    config = load_config(config_path)
    tickers = get_universe_tickers(config)
    sectors = get_sector_mapping(config)
    
    print(f"✓ Loaded {len(tickers)} stocks for universe: {config.get('universe', {}).get('index', 'Unknown')}")
    
    return tickers, sectors


# ============================================================
# Setup
# ============================================================

def setup_paths(checkpoint_dir):
    """Ensure all required paths exist."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add ml folder to path
    ml_path = str(get_base_dir().absolute())
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)
    
    print(f"✓ Checkpoint directory: {checkpoint_dir}")

# ============================================================
# Data Loading
# ============================================================

def load_data(tickers, start_date, end_date):
    """Load historical data for given tickers using batched bse_loader."""
    from data.bse_loader import get_prices
    
    print("\n" + "=" * 60)
    print("Step 2: Loading Historical Data")
    print("=" * 60)
    
    print(f"Tickers to fetch: {len(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Download data using our batched loader to prevent Rate Limits
    data = get_prices(
        tickers,
        start=start_date,
        end=end_date,
        progress=True
    )
    
    print(f"✓ Downloaded shape: {data.shape}")
    return data

# ============================================================
# Feature Engineering
# ============================================================

def compute_features(data):
    """Compute classical features for all stocks."""
    import importlib
    import features.classical
    importlib.reload(features.classical)
    import features.graph_builder
    importlib.reload(features.graph_builder)
    from features.classical import build_multi_stock_features, features_to_tensor
    from features.graph_builder import build_graph
    
    print("\n" + "=" * 60)
    print("Step 2: Computing Features")
    print("=" * 60)
    
    # Build features WITH forward returns so we can compute correct target labels.
    # The model input tensor uses only the non-forward columns (10 features).
    features_dict = build_multi_stock_features(data, CONFIG['tickers'], include_forward_returns=True)
    
    for ticker in CONFIG['tickers']:
        if ticker in features_dict:
            print(f"  ✓ {ticker}: {len(features_dict[ticker])} days, "
                  f"{len(features_dict[ticker].columns)} features")
    
    # Strip forward-return columns before building the model input tensor
    # (forward returns are TARGETS, not features).
    first_ticker = next(iter(features_dict))
    model_feature_cols = [c for c in features_dict[first_ticker].columns
                          if not c.startswith('forward_return')]
    model_features = {t: df[model_feature_cols] for t, df in features_dict.items()}
    
    # Convert to tensor
    tensor, feature_names, dates = features_to_tensor(model_features, CONFIG['tickers'])
    print(f"\n✓ Feature tensor: {tensor.shape}")
    print(f"  Features: {feature_names}")
    
    # Build graph with max edges limit to avoid OOM for 500 stocks
    edge_index, edge_info = build_graph(
        model_features, 
        CONFIG['tickers'], 
        CONFIG['sectors'],
        corr_threshold=0.3,
        max_edges_per_node=10
    )
    print(f"✓ Graph edges: {edge_index.shape[1]}")
    
    return tensor, feature_names, dates, edge_index, features_dict

# ============================================================
# Sentiment (Optional - for historical data, we use neutral)
# ============================================================

def get_sentiment_scores():
    """
    Get sentiment scores for each stock.
    
    For historical training, we use neutral (0.0) since we don't have
    historical news. Daily updates will include real sentiment.
    """
    print("\n" + "=" * 60)
    print("Step 3: Sentiment Scores")
    print("=" * 60)
    
    # For initial training, use neutral sentiment
    # Daily updates will incorporate real FinBERT scores
    sentiment = torch.zeros(len(CONFIG['tickers']))
    
    print("  ℹ Using neutral sentiment for historical training")
    print("  (Real sentiment will be used in daily updates)")
    
    return sentiment

# ============================================================
# Model Creation
# ============================================================

def create_model():
    """Create the CausalFolioMinimal model."""
    # Import model internally referring to relative structure
    import importlib
    import models.model_minimal
    importlib.reload(models.model_minimal)
    from models.model_minimal import CausalFolioMinimal, TrainingModule
    
    # Create model
    model = CausalFolioMinimal(
        num_features=CONFIG['num_features'],
        num_stocks=len(CONFIG['tickers']),
        gnn_hidden=CONFIG['gnn_hidden'],
        gnn_output=CONFIG['gnn_output'],
        tcn_hidden=CONFIG['tcn_hidden'],
        tcn_layers=CONFIG['tcn_layers'],
        dropout=CONFIG['dropout'],
        use_sentiment=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Receptive field: {model.get_receptive_field()} days")
    
    # Create trainer
    trainer = TrainingModule(
        model,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        device=device
    )
    
    return model, trainer, device

# ============================================================
# Target Computation
# ============================================================

# Direction class labels
DIRECTION_DOWN = 0
DIRECTION_SIDEWAYS = 1
DIRECTION_UP = 2
DIRECTION_THRESHOLD = 0.02  # ±2% threshold

def compute_targets(features_dict, dates):
    """
    Compute DUAL training targets (volatility AND direction CLASS LABELS).
    
    Targets:
    1. vol_5d: 5-day realized volatility [T, N, 1]
    2. direction_labels: 0=DOWN, 1=SIDEWAYS, 2=UP [T, N]
    
    Direction thresholds (applied to the TRUE forward return):
    - DOWN: return < -2%
    - SIDEWAYS: -2% <= return <= +2%
    - UP: return > +2%
    
    The forward return used is the actual 5-day cumulative return
    (close[t+5]/close[t] - 1), extracted from the forward_return_5d column
    that features_dict now contains (see compute_features).
    """
    print("\n" + "=" * 60)
    print("Step 5: Computing Targets (Classification)")
    print("=" * 60)
    
    vol_targets_list = []
    ret_values_list = []
    
    # Align every target to the same date axis used for the feature tensor
    if dates is not None:
        dates = pd.DatetimeIndex(dates)
    
    for ticker in CONFIG['tickers']:
        if ticker in features_dict:
            df = features_dict[ticker]
            
            if dates is not None and len(df.index) != len(dates):
                # Intersect with the tensor's common dates (keep same order)
                df = df.reindex(dates)
            
            # Volatility target
            if 'vol_5d' in df.columns:
                vol_targets_list.append(df['vol_5d'].values)
            else:
                print(f"  ⚠ {ticker}: Missing vol_5d, using zeros")
                vol_targets_list.append(np.zeros(len(df)))
            
            # True forward return (5-day cumulative) - present in features_dict
            if 'forward_return_5d' in df.columns:
                ret_values_list.append(df['forward_return_5d'].values)
            else:
                # Fallback: attempt to approximate from 1-day returns is WRONG;
                # compute the real 5-day cumulative forward return when possible.
                if 'return_1d' in df.columns:
                    # Approximate cumulative forward return from 1d log returns
                    # sum of the next 5 log returns ~ ln(close[t+5]/close[t])
                    r1 = df['return_1d']
                    approx = np.zeros(len(df))
                    for i in range(1, 6):
                        si = r1.shift(-i).values
                        approx += np.nan_to_num(si)
                    ret_values_list.append(approx)
                else:
                    print(f"  ⚠ {ticker}: Missing forward_return_5d, using zeros")
                    ret_values_list.append(np.zeros(len(df)))
    
    # Stack arrays
    vol_targets = np.stack(vol_targets_list, axis=1)  # [T, N]
    ret_values = np.stack(ret_values_list, axis=1)    # [T, N]
    
    # Convert returns to class labels
    # DOWN=0: return < -THRESHOLD
    # SIDEWAYS=1: -THRESHOLD <= return <= +THRESHOLD
    # UP=2: return > +THRESHOLD
    direction_labels = np.ones_like(ret_values, dtype=np.int64) * DIRECTION_SIDEWAYS  # Default SIDEWAYS
    direction_labels[ret_values < -DIRECTION_THRESHOLD] = DIRECTION_DOWN
    direction_labels[ret_values > DIRECTION_THRESHOLD] = DIRECTION_UP
    
    # Create tensors
    vol_targets_tensor = torch.tensor(vol_targets, dtype=torch.float32).unsqueeze(-1)  # [T, N, 1]
    direction_labels_tensor = torch.tensor(direction_labels, dtype=torch.long)  # [T, N]
    
    # Handle any NaN values
    vol_targets_tensor = torch.nan_to_num(vol_targets_tensor, 0.0)
    
    # Count class distribution
    num_down = (direction_labels == DIRECTION_DOWN).sum()
    num_sideways = (direction_labels == DIRECTION_SIDEWAYS).sum()
    num_up = (direction_labels == DIRECTION_UP).sum()
    total = direction_labels.size
    
    print(f"✓ Volatility targets shape: {vol_targets_tensor.shape}")
    print(f"✓ Direction labels shape: {direction_labels_tensor.shape}")
    print(f"  Class distribution (using ±{DIRECTION_THRESHOLD*100:.0f}% threshold):")
    print(f"    DOWN:     {num_down:,} ({100*num_down/total:.1f}%)")
    print(f"    SIDEWAYS: {num_sideways:,} ({100*num_sideways/total:.1f}%)")
    print(f"    UP:       {num_up:,} ({100*num_up/total:.1f}%)")
    
    # Calculate Class Weights
    weights = np.zeros(3)
    if num_down > 0: weights[DIRECTION_DOWN] = total / (3 * num_down)
    if num_sideways > 0: weights[DIRECTION_SIDEWAYS] = total / (3 * num_sideways)
    if num_up > 0: weights[DIRECTION_UP] = total / (3 * num_up)
    
    # Normalize weights so they sum to 3 (num_classes)
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights * (3.0 / weight_sum)
        
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    print(f"\n  Computed Class Weights:")
    print(f"    DOWN:     {weights[DIRECTION_DOWN]:.4f}")
    print(f"    SIDEWAYS: {weights[DIRECTION_SIDEWAYS]:.4f}")
    print(f"    UP:       {weights[DIRECTION_UP]:.4f}")
    
    return vol_targets_tensor, direction_labels_tensor, class_weights_tensor

# ============================================================
# Training
# ============================================================

def train_model(model, trainer, features, edge_index, vol_targets, direction_labels, sentiment, class_weights, device):
    """Train the model with DUAL TARGETS (Classification)."""
    print("\n" + "=" * 60)
    print("Step 6: Training (Direction Classification)")
    print("=" * 60)
    
    features = features.to(device)
    edge_index = edge_index.to(device)
    vol_targets = vol_targets.to(device)
    direction_labels = direction_labels.to(device)
    sentiment = sentiment.to(device)
    
    print(f"Training configuration:")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Validation split: {CONFIG['val_split']}")
    print(f"  Early stopping patience: {CONFIG['early_stopping']}")
    print(f"  Targets: Volatility (MSE) + Direction (CrossEntropy)")
    print()
    
    # Train with dual targets (classification)
    history = trainer.train(
        features=features,
        edge_index=edge_index,
        vol_targets=vol_targets,
        direction_labels=direction_labels,
        class_weights=class_weights,
        sentiment=sentiment,
        epochs=CONFIG['epochs'],
        val_split=CONFIG['val_split'],
        early_stopping=CONFIG['early_stopping'],
        verbose=True
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Best epoch: {history['best_epoch'] + 1}")
    print(f"  Best val loss: {history['best_val_loss']:.6f}")
    
    print("\n" + "=" * 60)
    print("Step 6a: Final Evaluation (Classification Report)")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        outputs = model(features, edge_index, sentiment)
        dir_logits = outputs['direction'].cpu() # [T, N, 3]
        
        # Predict classes
        pred_classes = torch.argmax(dir_logits, dim=-1).flatten().numpy()
        true_classes = direction_labels.cpu().flatten().numpy()
        
        target_names = ['DOWN', 'SIDEWAYS', 'UP']
        print(classification_report(true_classes, pred_classes, target_names=target_names, zero_division=0))
    
    return history

# ============================================================
# Save Model
# ============================================================

def save_model(trainer):
    """Save the trained model."""
    print("\n" + "=" * 60)
    print("Step 7: Saving Model")
    print("=" * 60)
    
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['model_name'])
    # Embed feature normalization stats + feature names so inference can
    # apply the exact same preprocessing.
    extra_config = {
        'feature_mean': CONFIG.get('feature_mean'),
        'feature_std': CONFIG.get('feature_std'),
        'feature_names': CONFIG.get('feature_names'),
        'market_neutralized': CONFIG.get('market_neutralized', False),
        'num_stocks_train': len(CONFIG.get('tickers', [])),
    }
    trainer.save_checkpoint(checkpoint_path, extra_config=extra_config)
    
    print(f"✓ Model saved to: {checkpoint_path}")
    
    return checkpoint_path

def main(config_path=None, start_date='2019-01-01', end_date=None,
         universe='NIFTY_100', max_tickers=None, epochs=None,
         tcn_layers=None, model_name='causalfolio_v3.pt'):
    """
    Full training pipeline using dynamically fetched stocks.
    
    Args:
        config_path: Path to config.yaml
        start_date: Training start date (default: 2019-01-01)
        end_date: Training end date (default: today)
        universe: NIFTY_50, NIFTY_100, or NIFTY_500
        max_tickers: Cap number of tickers (for fast local CPU runs)
        epochs: Override epoch count
        tcn_layers: Override TCN layers
        model_name: Checkpoint file name
    """
    global CONFIG
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Colab Ghost Memory Fix: Purge previous traceback tensors from VRAM
    import gc
    import sys
    sys.last_traceback = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("=" * 60)
    print("CausalFolio - Initial Training (INTENSIVE)")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load universe
    tickers, sectors = load_stocks_from_config(config_path)
    
    # Optionally restrict universe for fast local CPU runs
    if universe is not None and universe != 'NIFTY_500':
        # Use the first N tickers from the loaded list (already ordered by index)
        _univ_map = {'NIFTY_50': 50, 'NIFTY_100': 100}
        n = _univ_map.get(universe, len(tickers))
        tickers = tickers[:n]
        sectors = {t: sectors[t] for t in tickers if t in sectors}
    if max_tickers is not None and len(tickers) > max_tickers:
        tickers = tickers[:max_tickers]
        sectors = {t: sectors[t] for t in tickers if t in sectors}
    print(f"  Universe restricted to: {len(tickers)} stocks")
    
    # Setup paths
    base_dir = get_base_dir()
    checkpoint_dir = str(base_dir / 'checkpoints')
    
    # Overwrite config values from yaml
    yaml_config = load_config(config_path)
    model_cfg = yaml_config.get('model', {})
    
    # Build CONFIG
    CONFIG = {
        'tickers': tickers,
        'sectors': sectors,
        'start_date': start_date,
        'end_date': end_date,
        'num_features': 10,
        'checkpoint_dir': checkpoint_dir,
        'model_name': model_name,
        'gnn_hidden': model_cfg.get('gnn_hidden_dim', TRAINING_CONFIG['gnn_hidden']),
        'gnn_output': model_cfg.get('gnn_hidden_dim', TRAINING_CONFIG['gnn_hidden']) // 2,
        'tcn_hidden': model_cfg.get('tcn_hidden_dim', TRAINING_CONFIG['tcn_hidden']),
        'tcn_layers': tcn_layers if tcn_layers is not None else model_cfg.get('tcn_num_layers', TRAINING_CONFIG['tcn_layers']),
        'dropout': model_cfg.get('gnn_dropout', TRAINING_CONFIG['dropout']),
        **{k: v for k, v in TRAINING_CONFIG.items() if k not in ['gnn_hidden', 'gnn_output', 'tcn_hidden', 'tcn_layers', 'dropout']}
    }
    if epochs is not None:
        CONFIG['epochs'] = epochs
    
    print(f"\n→ Configuration:")
    print(f"  Stocks: {len(CONFIG['tickers'])}")
    print(f"  Date range: {CONFIG['start_date']} to {CONFIG['end_date']}")
    print(f"  Intensive training: {CONFIG['epochs']} epochs")
    
    # Setup
    setup_paths(checkpoint_dir)
    
    # Step 2: Load data
    data = load_data(tickers, start_date, end_date)
    
    # Step 3: Compute features
    features, feature_names, dates, edge_index, features_dict = compute_features(data)
    CONFIG['feature_names'] = feature_names
    
    # Step 3b: Market-neutralize cross-sectional return features, then z-score
    from features.classical import normalize_features, market_neutralize
    features = market_neutralize(features, feature_names)
    features, norm_stats = normalize_features(features)
    CONFIG['feature_mean'] = norm_stats['mean'].tolist()
    CONFIG['feature_std'] = norm_stats['std'].tolist()
    CONFIG['market_neutralized'] = True
    print(f"\n✓ Features market-neutralized + z-scored, mean={norm_stats['mean'].shape}, std={norm_stats['std'].shape}")
    
    # Step 4: Get sentiment (neutral for historical)
    sentiment = get_sentiment_scores()
    
    # Step 5: Compute DUAL targets
    vol_targets, ret_targets, class_weights = compute_targets(features_dict, dates)
    
    # Ensure shapes match
    min_len = min(features.shape[0], vol_targets.shape[0], ret_targets.shape[0])
    features = features[:min_len]
    vol_targets = vol_targets[:min_len]
    ret_targets = ret_targets[:min_len]
    
    # Step 6: Create model
    model, trainer, device = create_model()
    
    # Step 7: Train with dual targets
    history = train_model(model, trainer, features, edge_index, vol_targets, ret_targets, sentiment, class_weights, device)
    
    # Step 8: Save (with normalization stats embedded in checkpoint)
    checkpoint_path = save_model(trainer)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"\nTrained on: {tickers}")
    print("\nNext steps:")
    print("  1. Run daily_update.py to fine-tune with new data")
    print("  2. Use model for predictions")
    
    return model, trainer, history, CONFIG


# Global CONFIG (will be populated by main())
CONFIG = {}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CausalFolio initial training")
    parser.add_argument('--config', default=None, help='Path to config.yaml')
    parser.add_argument('--start-date', default='2019-01-01', help='Training start date')
    parser.add_argument('--end-date', default=None, help='Training end date (default: today)')
    parser.add_argument('--universe', default='NIFTY_100', choices=['NIFTY_50', 'NIFTY_100', 'NIFTY_500'],
                        help='Universe to use (default NIFTY_100 for local runs)')
    parser.add_argument('--max-tickers', type=int, default=None, help='Cap number of tickers')
    parser.add_argument('--epochs', type=int, default=None, help='Override epoch count')
    parser.add_argument('--tcn-layers', type=int, default=None, help='Override TCN layers')
    parser.add_argument('--model-name', default='causalfolio_v3.pt', help='Checkpoint file name')
    args = parser.parse_args()
    
    model, trainer, history, config = main(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        universe=args.universe,
        max_tickers=args.max_tickers,
        epochs=args.epochs,
        tcn_layers=args.tcn_layers,
        model_name=args.model_name
    )
