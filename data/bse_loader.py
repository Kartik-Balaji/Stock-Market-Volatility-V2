"""
CausalFolio - BSE Data Loader
==============================
Fetches stock prices from Yahoo Finance for BSE stocks.

Usage (in Colab):
    from data.bse_loader import get_prices, get_tickers, get_sector_mapping
    
    tickers = get_tickers()
    data = get_prices(tickers, "2023-01-01", "2023-12-31")
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
from data.universe_loader import get_universe_tickers, get_sector_mapping as get_universe_sectors

# Try to import yfinance (installed in Colab)
try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    yf = None


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from config.yaml
    
    Args:
        config_path: Optional path to config file. 
                     If None, looks in ../config/config.yaml
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default: look relative to this file
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_tickers(config: Optional[dict] = None) -> List[str]:
    """
    Get list of stock tickers dynamically from the universe loader.
    """
    return get_universe_tickers(config)


def get_sector_mapping(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Get mapping of ticker -> sector via universe loader.
    """
    return get_universe_sectors(config)


def get_prices(
    tickers: List[str],
    start: str,
    end: str,
    progress: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV data for given tickers from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols (e.g., ['TCS.BO', 'RELIANCE.BO'])
        start: Start date string 'YYYY-MM-DD'
        end: End date string 'YYYY-MM-DD'
        progress: Show download progress bar
    
    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV)
        For single ticker, returns simple OHLCV DataFrame
    """
    if yf is None:
        raise ImportError("yfinance not installed")
        
    config = load_config()
    batch_size = config.get('api_limits', {}).get('yfinance_batch_size', 50)
    delay_seconds = config.get('api_limits', {}).get('yfinance_delay_seconds', 2)
    max_retries = config.get('api_limits', {}).get('yfinance_max_retries', 3)
    
    all_data = []
    
    # Process in batches
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:min(i + batch_size, len(tickers))]
        
        if progress:
            print(f"Fetching batch {i//batch_size + 1}/{len(tickers)//batch_size + 1} ({len(batch_tickers)} tickers)...")
            
        for attempt in range(max_retries):
            try:
                batch_data = yf.download(
                    batch_tickers,
                    start=start,
                    end=end,
                    progress=False,
                    group_by='ticker',
                    auto_adjust=False
                )
                
                # Check if any ticker has all NaNs, try .NS fallback
                for tk in batch_tickers:
                    has_data = False
                    if isinstance(batch_data.columns, pd.MultiIndex):
                        if tk in batch_data.columns.levels[0] and len(batch_data[tk].dropna()) > 10:
                            has_data = True
                        elif tk in batch_data.columns.levels[1] and len(batch_data.xs(tk, level=1, axis=1).dropna()) > 10:
                            has_data = True
                    elif len(batch_data.dropna()) > 10:
                        has_data = True
                        
                    if not has_data and tk.endswith('.BO'):
                        ns_tk = tk.replace('.BO', '.NS')
                        try:
                            ns_data = yf.download(ns_tk, start=start, end=end, progress=False, auto_adjust=False)
                            if len(ns_data.dropna()) > 10:
                                if len(batch_tickers) == 1:
                                    batch_data = ns_data
                                else:
                                    # Rename columns to .BO
                                    ns_data.columns = pd.MultiIndex.from_product([[tk], ns_data.columns])
                                    batch_data = pd.concat([batch_data, ns_data], axis=1)
                        except Exception:
                            pass
                            
                all_data.append(batch_data)
                break # Success
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(delay_seconds * (attempt + 1))
        
        # Delay between batches to prevent rate limits
        if i + batch_size < len(tickers):
            time.sleep(delay_seconds)
            
    if not all_data:
        return pd.DataFrame()
        
    # Combine batches horizontally
    if len(all_data) == 1:
        data = all_data[0]
    else:
        # yfinance returns single level columns if only 1 ticker in batch. 
        # We need to handle merging these properly.
        processed_data = []
        for bd, tks in zip(all_data, [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]):
            if len(tks) == 1 and not isinstance(bd.columns, pd.MultiIndex):
                # Convert to multiindex
                bd.columns = pd.MultiIndex.from_product([tks, bd.columns])
            processed_data.append(bd)
        data = pd.concat(processed_data, axis=1)
    
    return data


def get_single_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single stock.
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    if yf is None:
        raise ImportError("yfinance not installed")
    
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    
    return data


def get_latest_prices(tickers: List[str], lookback_days: int = 5) -> pd.DataFrame:
    """
    Fetch recent prices for daily update.
    
    Args:
        tickers: List of ticker symbols
        lookback_days: Number of days to fetch
    
    Returns:
        DataFrame with recent OHLCV data
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 7)  # Buffer for weekends
    
    return get_prices(
        tickers,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        progress=False
    )


def validate_data(data: pd.DataFrame) -> Dict[str, any]:
    """
    Check data quality and return summary.
    
    Returns:
        Dict with:
        - total_rows: Number of trading days
        - missing_pct: Percentage of missing values
        - tickers_with_issues: Tickers with >5% missing data
    """
    if isinstance(data.columns, pd.MultiIndex):
        # Check column structure - tickers could be at level 0 or level 1
        level_0 = data.columns.get_level_values(0).unique().tolist()
        level_1 = data.columns.get_level_values(1).unique().tolist()
        
        # Determine which level has tickers (more values = tickers)
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        if any(col in level_0 for col in price_cols):
            # New structure: (Price, Ticker)
            tickers = level_1
            get_ticker_data = lambda t: data.xs(t, level=1, axis=1)
        else:
            # Old structure: (Ticker, Price)
            tickers = level_0
            get_ticker_data = lambda t: data[t]
        
        issues = []
        for ticker in tickers:
            try:
                ticker_data = get_ticker_data(ticker)
                if 'Close' in ticker_data.columns:
                    missing_pct = ticker_data['Close'].isna().mean() * 100
                    if missing_pct > 5:
                        issues.append(ticker)
            except:
                continue
        
        return {
            'total_rows': len(data),
            'total_tickers': len(tickers),
            'missing_pct': round(data.isna().mean().mean() * 100, 2),
            'tickers_with_issues': issues,
            'date_range': (str(data.index.min()), str(data.index.max()))
        }
    else:
        # Single ticker case
        return {
            'total_rows': len(data),
            'missing_pct': round(data.isna().mean().mean() * 100, 2),
            'date_range': (str(data.index.min()), str(data.index.max()))
        }


# ===================
# Quick Test (run directly)
# ===================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing BSE Data Loader")
    print("=" * 50)
    
    # Load config
    config = load_config()
    print(f"\n✓ Config loaded")
    
    # Get tickers
    tickers = get_tickers(config)
    print(f"✓ Found {len(tickers)} tickers: {tickers}")
    
    # Get sector mapping
    sectors = get_sector_mapping(config)
    print(f"✓ Sectors: {list(set(sectors.values()))}")
    
    # Test single stock fetch
    print(f"\n→ Fetching TCS.BO data (Dec 2024)...")
    data = get_single_stock("TCS.BO", "2024-12-01", "2024-12-27")
    print(f"✓ Got {len(data)} rows")
    print(data.tail(3))
    
    # Validate
    validation = validate_data(data)
    print(f"\n✓ Validation: {validation}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
