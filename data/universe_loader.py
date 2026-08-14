"""
CausalFolio - Dynamic Universe Loader
======================================
Dynamically fetches stock constituents for major indices (NIFTY 50, 100, 500)
from official NSE India CSV endpoints, bypassing Wikipedia scrapers entirely.
"""

import pandas as pd
import warnings
from typing import List, Dict, Optional
import os
import yaml
from pathlib import Path

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        try:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        except NameError:
            config_path = Path.cwd() / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fetch_nifty_symbols(index_name: str = "NIFTY_100") -> pd.DataFrame:
    """
    Fetches the constituents for a given NIFTY index from local offline CSVs.
    Returns a DataFrame containing 'Symbol' and 'Industry' columns.
    """
    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd() / "data"
        
    files = {
        "NIFTY_100": base_dir / "ind_nifty100list.csv",
        "NIFTY_500": base_dir / "ind_nifty500list.csv"
    }
    
    if index_name not in files:
        raise ValueError(f"Unknown index {index_name}. Supported: {list(files.keys())}")
        
    csv_path = files[index_name]
    
    try:
        # Read the offline local CSV
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading local NSE CSV at {csv_path}: {e}")
        return pd.DataFrame()

def get_universe_tickers(config: Optional[dict] = None) -> List[str]:
    """
    Returns the final list of tickers with the correct exchange suffix.
    """
    if config is None:
        config = load_config()
        
    index_name = config.get('universe', {}).get('index', 'NIFTY_100')
    suffix = config.get('universe', {}).get('exchange_suffix', '.BO')
    
    df = fetch_nifty_symbols(index_name)
    
    if df.empty or 'Symbol' not in df.columns:
        print("Warning: NSE CSV fetch failed. Using minimal fallback.")
        return [f"TCS{suffix}", f"RELIANCE{suffix}"]
        
    symbols = df['Symbol'].dropna().astype(str).tolist()
    
    # Clean spacing and append suffix
    tickers = [f"{sym.strip()}{suffix}" for sym in symbols]
    return tickers

def get_sector_mapping(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Extracts the official Industry sectors from the NSE CSVs.
    """
    if config is None:
        config = load_config()
    
    index_name = config.get('universe', {}).get('index', 'NIFTY_100')
    suffix = config.get('universe', {}).get('exchange_suffix', '.BO')
    
    df = fetch_nifty_symbols(index_name)
    mapping = {}
    
    if not df.empty and 'Symbol' in df.columns and 'Industry' in df.columns:
        for _, row in df.iterrows():
            sym = str(row['Symbol']).strip()
            mapping[f"{sym}{suffix}"] = str(row['Industry']).strip()
    else:
        # Fallback if scraping fails
        tickers = get_universe_tickers(config)
        for ticker in tickers:
            mapping[ticker] = 'Unknown'
            
    return mapping

if __name__ == "__main__":
    print("Testing Stable Official NSE Universe Loader...")
    tickers = get_universe_tickers()
    print(f"Loaded {len(tickers)} tickers. First 5: {tickers[:5]}")
    sectors = get_sector_mapping()
    print(f"Loaded {len(sectors)} sector mappings. First 5: {list(sectors.items())[:5]}")
