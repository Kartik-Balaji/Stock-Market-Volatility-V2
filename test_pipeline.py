"""
CausalFolio - Data Pipeline Test Script
========================================
Run this in Google Colab to test all data collection/preprocessing modules.

Instructions:
1. Upload the 'ml' folder to Colab or mount Google Drive
2. Run: %cd /content/drive/MyDrive/your_path/ml
3. Run: !pip install -r requirements.txt
4. Run this script: !python test_pipeline.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CausalFolio - Data Pipeline Test")
print("=" * 60)

# ===================
# Test 1: Imports
# ===================
print("\n[1/5] Testing imports...")
try:
    from data.bse_loader import load_config, get_tickers, get_sector_mapping, get_prices
    from data.news_scraper import get_news_headlines
    from features.classical import build_node_features, build_multi_stock_features, features_to_tensor
    from features.graph_builder import build_graph, get_graph_stats
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# ===================
# Test 2: Config
# ===================
print("\n[2/5] Testing config loading...")
try:
    config = load_config()
    tickers = get_tickers(config)
    sectors = get_sector_mapping(config)
    
    print(f"  ✓ Config loaded")
    print(f"  ✓ Tickers: {tickers}")
    print(f"  ✓ Sectors: {list(set(sectors.values()))}")
except Exception as e:
    print(f"  ✗ Config error: {e}")
    sys.exit(1)

# ===================
# Test 3: Price Data
# ===================
print("\n[3/5] Testing price data fetch...")
try:
    # Fetch 6 months of data
    data = get_prices(tickers, "2024-06-01", "2024-12-20")
    
    # Check shape
    print(f"  ✓ Data shape: {data.shape}")
    print(f"  ✓ Date range: {data.index.min()} to {data.index.max()}")
    
    # Show sample
    print(f"  ✓ Sample (first ticker, last 3 rows):")
    if hasattr(data.columns, 'get_level_values'):
        first_ticker = data.columns.get_level_values(0)[0]
        print(data[first_ticker][['Close', 'Volume']].tail(3))
    else:
        print(data[['Close', 'Volume']].tail(3))
except Exception as e:
    print(f"  ✗ Price data error: {e}")
    import traceback
    traceback.print_exc()

# ===================
# Test 4: Features
# ===================
print("\n[4/5] Testing feature computation...")
try:
    features_dict = build_multi_stock_features(data, tickers)
    
    print(f"  ✓ Features computed for {len(features_dict)} stocks")
    
    # Show feature columns
    first_ticker = list(features_dict.keys())[0]
    print(f"  ✓ Feature columns: {list(features_dict[first_ticker].columns)}")
    
    # Convert to tensor
    tensor, feature_names, dates = features_to_tensor(features_dict, tickers)
    print(f"  ✓ Tensor shape: {tensor.shape} [T x N_stocks x N_features]")
    print(f"  ✓ Valid dates: {len(dates)} (after dropping NaN)")
    
except Exception as e:
    print(f"  ✗ Feature error: {e}")
    import traceback
    traceback.print_exc()

# ===================
# Test 5: Graph
# ===================
print("\n[5/5] Testing graph construction...")
try:
    edge_index, weights = build_graph(
        features_dict, tickers, sectors,
        corr_threshold=0.3
    )
    
    stats = get_graph_stats(edge_index, len(tickers))
    print(f"  ✓ Graph stats: {stats}")
    
except Exception as e:
    print(f"  ✗ Graph error: {e}")
    import traceback
    traceback.print_exc()

# ===================
# Summary
# ===================
print("\n" + "=" * 60)
print("DATA PIPELINE TEST COMPLETE")
print("=" * 60)
print(f"""
Summary:
- Tickers: {len(tickers)} stocks
- Data: {data.shape[0]} trading days
- Features: {tensor.shape[2]} per stock
- Graph: {stats['num_edges']} edges, density={stats['density']}

Next steps:
1. Test news scraping: get_news_headlines("TCS.BO")
2. Proceed to Phase 2: Model implementation
""")
