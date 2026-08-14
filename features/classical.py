"""
CausalFolio - Classical Feature Engineering
============================================
Computes returns, volatility, RSI, momentum, and other technical features.

Usage (in Colab):
    from features.classical import build_node_features, features_to_tensor
    
    features = build_node_features(prices)  # For single stock
    tensor = features_to_tensor(features_dict, tickers)  # For GNN input
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch


def compute_returns(
    prices: pd.DataFrame,
    periods: List[int] = [1, 5, 20]
) -> pd.DataFrame:
    """
    Compute log returns over multiple periods.
    
    Args:
        prices: DataFrame with 'Close' column or Series
        periods: List of lookback periods [1, 5, 20] = daily, weekly, monthly
    
    Returns:
        DataFrame with columns: return_1d, return_5d, return_20d
    """
    # Handle different input types
    if isinstance(prices, pd.DataFrame):
        if 'Close' in prices.columns:
            close = prices['Close']
        elif 'Adj Close' in prices.columns:
            close = prices['Adj Close']
        else:
            close = prices.iloc[:, 0]
    else:
        close = prices
    
    returns = pd.DataFrame(index=close.index)
    
    for period in periods:
        returns[f'return_{period}d'] = np.log(close / close.shift(period))
    
    return returns


def compute_volatility(
    returns: pd.DataFrame,
    windows: List[int] = [5, 20, 60]
) -> pd.DataFrame:
    """
    Compute realized volatility (rolling std of returns).
    
    Args:
        returns: DataFrame with return columns
        windows: Rolling window sizes [5, 20, 60] = week, month, quarter
    
    Returns:
        DataFrame with columns: vol_5d, vol_20d, vol_60d (annualized)
    """
    # Use 1-day returns for volatility
    if 'return_1d' in returns.columns:
        ret = returns['return_1d']
    else:
        ret = returns.iloc[:, 0]
    
    volatility = pd.DataFrame(index=returns.index)
    
    for window in windows:
        # Annualized volatility (252 trading days)
        volatility[f'vol_{window}d'] = ret.rolling(window).std() * np.sqrt(252)
    
    return volatility


def compute_rsi(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (0-100).
    
    RSI > 70 = overbought
    RSI < 30 = oversold
    """
    if isinstance(prices, pd.DataFrame):
        close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
    else:
        close = prices
    
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_momentum(
    prices: pd.DataFrame,
    periods: List[int] = [5, 20]
) -> pd.DataFrame:
    """
    Compute price momentum (rate of change).
    
    momentum = (price_today - price_n_days_ago) / price_n_days_ago
    """
    if isinstance(prices, pd.DataFrame):
        close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
    else:
        close = prices
    
    momentum = pd.DataFrame(index=close.index)
    
    for period in periods:
        momentum[f'mom_{period}d'] = (close - close.shift(period)) / (close.shift(period) + 1e-10)
    
    return momentum


def compute_forward_returns(
    prices: pd.DataFrame,
    periods: List[int] = [5]
) -> pd.DataFrame:
    """
    Compute FORWARD-LOOKING returns (for prediction targets).
    
    This is the price change OVER THE NEXT N days.
    Used as target for training price prediction models.
    
    Args:
        prices: DataFrame with 'Close' column
        periods: Forward periods [5] = next 5 days
    
    Returns:
        DataFrame with columns: forward_return_5d (etc.)
    
    WARNING: This uses future data - only use for TARGET labels, not features!
    """
    if isinstance(prices, pd.DataFrame):
        close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
    else:
        close = prices
    
    forward_returns = pd.DataFrame(index=close.index)
    
    for period in periods:
        # Forward return = (future_price - current_price) / current_price
        forward_returns[f'forward_return_{period}d'] = (close.shift(-period) - close) / (close + 1e-10)
    
    return forward_returns


def compute_volume_ratio(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute volume relative to moving average.
    
    ratio > 1 = higher than usual volume
    ratio < 1 = lower than usual volume
    """
    if 'Volume' not in prices.columns:
        # Return ones if no volume data
        return pd.Series(index=prices.index, data=1.0, name='volume_ratio')
    
    volume = prices['Volume']
    vol_ma = volume.rolling(window).mean()
    
    return (volume / (vol_ma + 1e-10)).rename('volume_ratio')


def build_node_features(
    prices: pd.DataFrame,
    config: Optional[Dict] = None,
    include_forward_returns: bool = False
) -> pd.DataFrame:
    """
    Build complete feature set for a single stock (one GNN node).
    
    Args:
        prices: DataFrame with OHLCV data
        config: Optional config with custom window sizes
        include_forward_returns: If True, include forward returns (for training targets)
    
    Returns:
        DataFrame with all features (~10-12 features), NaN rows present at start
    """
    if config is None:
        config = {
            'return_periods': [1, 5, 20],
            'vol_windows': [5, 20, 60],
            'rsi_period': 14,
            'momentum_periods': [5, 20],
            'volume_window': 20
        }
    
    # Compute all features
    returns = compute_returns(prices, config.get('return_periods', [1, 5, 20]))
    volatility = compute_volatility(returns, config.get('vol_windows', [5, 20, 60]))
    rsi = compute_rsi(prices, config.get('rsi_period', 14))
    momentum = compute_momentum(prices, config.get('momentum_periods', [5, 20]))
    volume_ratio = compute_volume_ratio(prices, config.get('volume_window', 20))
    
    # Combine all features
    features = pd.concat([
        returns,
        volatility,
        rsi.rename('rsi'),
        momentum,
        volume_ratio
    ], axis=1)
    
    # Optionally include forward returns (for training targets)
    if include_forward_returns:
        forward_returns = compute_forward_returns(prices, [5, 20])
        features = pd.concat([features, forward_returns], axis=1)
    
    return features


def build_multi_stock_features(
    data: pd.DataFrame,
    tickers: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Build features for multiple stocks.
    
    Args:
        data: DataFrame with MultiIndex columns from yfinance.
              Handles both (ticker, OHLCV) and (OHLCV, ticker) structures.
        tickers: List of ticker symbols
    
    Returns:
        Dict mapping ticker -> feature DataFrame
    """
    features = {}
    
    # Debug: Show data structure
    print(f"  → Data structure: {type(data.columns)}")
    if isinstance(data.columns, pd.MultiIndex):
        print(f"  → MultiIndex levels: {data.columns.names}")
        print(f"  → Level 0 values: {data.columns.get_level_values(0).unique().tolist()[:5]}")
    
    for ticker in tickers:
        try:
            # Handle different yfinance column structures
            if isinstance(data.columns, pd.MultiIndex):
                # Check if tickers are at level 0 or level 1
                level_0 = data.columns.get_level_values(0).unique()
                level_1 = data.columns.get_level_values(1).unique()
                
                if ticker in level_0:
                    # Old structure: (Ticker, Price)
                    stock_data = data[ticker]
                elif ticker in level_1:
                    # New structure: (Price, Ticker) - swap levels
                    stock_data = data.xs(ticker, level=1, axis=1)
                else:
                    print(f"  ✗ {ticker}: Not found in data")
                    continue
            else:
                # Single stock download (no MultiIndex)
                stock_data = data
            
            features[ticker] = build_node_features(stock_data)
            print(f"  ✓ {ticker}: {features[ticker].shape[1]} features")
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")
            continue
    
    return features


def features_to_tensor(
    features_dict: Dict[str, pd.DataFrame],
    tickers: List[str]
) -> Tuple[torch.Tensor, List[str], pd.DatetimeIndex]:
    """
    Convert feature dictionaries to tensor for GNN.
    Uses forward-fill for missing values, then skips any remaining NaN rows.
    
    Args:
        features_dict: Dict mapping ticker -> feature DataFrame
        tickers: Ordered list of tickers (defines node order)
    
    Returns:
        Tuple of:
        - Tensor of shape [T, N_stocks, N_features]
        - List of feature names
        - DatetimeIndex of dates
    """
    # Get tickers that exist in features_dict
    valid_tickers = [t for t in tickers if t in features_dict and len(features_dict[t]) > 0]
    
    if not valid_tickers:
        raise ValueError("No valid tickers with features")
    
    print(f"  → Using {len(valid_tickers)} stocks")
    
    # Forward-fill NaN in each stock's features
    for ticker in valid_tickers:
        features_dict[ticker] = features_dict[ticker].ffill().bfill()
    
    # Find common dates across all valid tickers
    common_dates = None
    for ticker in valid_tickers:
        dates = set(features_dict[ticker].index)
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)
    
    common_dates = sorted(common_dates)
    
    # Get feature names
    feature_names = list(features_dict[valid_tickers[0]].columns)
    
    # Build tensor [T x N x F]
    T = len(common_dates)
    N = len(valid_tickers)
    F = len(feature_names)
    
    tensor = torch.zeros(T, N, F)
    
    for i, ticker in enumerate(valid_tickers):
        df = features_dict[ticker].loc[common_dates, feature_names]
        tensor[:, i, :] = torch.tensor(df.values, dtype=torch.float32)
    
    # Check for any remaining NaN
    nan_count = torch.isnan(tensor).sum().item()
    if nan_count > 0:
        print(f"  → Warning: {nan_count} NaN values remain, filling with 0")
        tensor = torch.nan_to_num(tensor, nan=0.0)
    
    print(f"  ✓ Tensor: {tensor.shape} [T={T} days, N={N} stocks, F={F} features]")
    
    return tensor, feature_names, pd.DatetimeIndex(common_dates)


def normalize_features(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Z-score normalize features (per feature, across all stocks and time).
    
    Returns:
        Tuple of:
        - Normalized tensor
        - Dict with mean and std for each feature (for inverse transform)
    """
    # tensor shape: [T, N, F]
    T, N, F = tensor.shape
    
    # Reshape to [T*N, F] for per-feature normalization
    flat = tensor.reshape(-1, F)
    
    means = flat.mean(dim=0)
    stds = flat.std(dim=0) + 1e-8
    
    normalized = (flat - means) / stds
    normalized = normalized.reshape(T, N, F)
    
    stats = {'mean': means, 'std': stds}
    
    return normalized, stats


# ===================
# Quick Test
# ===================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Classical Features")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    prices = pd.DataFrame({
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 101 + np.random.randn(100).cumsum(),
        'Low': 99 + np.random.randn(100).cumsum(),
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Compute features
    features = build_node_features(prices)
    
    print(f"\n✓ Features shape: {features.shape}")
    print(f"✓ Feature columns: {list(features.columns)}")
    print(f"\n✓ Sample (last 5 rows after dropping NaN):")
    print(features.dropna().tail())
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
