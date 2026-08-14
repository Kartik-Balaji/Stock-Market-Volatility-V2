---
description: CausalFolio development workflow - complete step-by-step implementation from architecture
---

# CausalFolio Complete Development Workflow

## Current Status Summary
| Component | Status | File |
|-----------|--------|------|
| GNN (GATv2) | ✅ Built | `models/gnn.py` |
| TCN | ✅ Built | `models/tcn.py` |
| FinBERT Sentiment | ✅ Built | `models/sentiment.py` |
| News Scraping (3 sources) | ✅ Built | `data/news_scraper.py` |
| Walk-Forward Backtest | ✅ Built | `scripts/backtest.py` |
| TDA Features | 🔲 Pending | `features/topological.py` |
| Blackwell Regime | 🔲 Pending | `models/blackwell.py` |
| Purged CV | 🔲 Pending | `validation/purged_cv.py` |
| Risk Metrics | 🔲 Pending | `validation/metrics.py` |
| Position Limits | 🔲 Pending | `risk/position_limits.py` |
| Transaction Costs | 🔲 Pending | `risk/costs.py` |
| VaR/CVaR | 🔲 Pending | `risk/var_model.py` |
| Decision Trace | 🔲 Pending | `explain/decision_trace.py` |
| Cointegration Tests | 🔲 Pending | `validation/cointegration.py` |
| Slippage Model | 🔲 Pending | `risk/slippage.py` |

---

## PHASE 2: VALIDATION FRAMEWORK

### Step 2.1: Run Walk-Forward Backtest (Created)
```python
# In Colab:
exec(open('/content/drive/MyDrive/ml/scripts/backtest.py').read())
results, metrics = run_walkforward_backtest()
```
**Output:** Direction accuracy, MAE, correlation

### Step 2.2: Create Purged Cross-Validation
**File:** `validation/purged_cv.py`
```python
def purged_kfold_split(n_samples, k=5, purge_gap=5):
    """5-day gap to prevent data leakage."""
```

### Step 2.3: Create Performance Metrics Module
**File:** `validation/metrics.py`
- Sharpe ratio (risk-adjusted return)
- Sortino ratio (downside-only risk)
- Max drawdown (worst loss from peak)
- VaR 95% (value at risk)
- CVaR (conditional VaR / expected shortfall)
- Alpha & Beta (vs Nifty 50)

### Step 2.4: Add Residual Diagnostics
**Tests to implement:**
- Durbin-Watson (autocorrelation)
- Shapiro-Wilk (normality)
- Ljung-Box (serial correlation)
- ARCH test (heteroskedasticity)

---

## PHASE 3: RISK MANAGEMENT

### Step 3.1: Position Limits
**File:** `risk/position_limits.py`
```python
LIMITS = {
    'max_gross_exposure': 1.5,    # 150% total
    'max_net_exposure': 0.3,      # 30% directional
    'max_single_position': 0.1,   # 10% per stock
    'max_sector_exposure': 0.3,   # 30% per sector
}
```

### Step 3.2: Transaction Costs (BSE-specific)
**File:** `risk/costs.py`
- Spread: 10 bps half-spread
- Brokerage: 5 bps
- STT: 2.5 bps (sell only)
- Exchange: 1 bps
- GST: 18% of brokerage

### Step 3.3: VaR/CVaR Model
**File:** `risk/var_model.py`
```python
def check_var_limit(positions, returns_history, var_limit=0.05):
    """Scale down if portfolio VaR exceeds limit."""
```

### Step 3.4: Slippage Model (Kyle's Lambda)
**File:** `risk/slippage.py`
```python
def estimate_slippage(order_size, avg_volume, volatility):
    impact = 0.1 * sqrt(order_size / avg_volume) * volatility
```

### Step 3.5: Liquidity Filters
Filter out illiquid stocks:
- Min avg volume: 1M shares
- Min avg turnover: ₹10Cr

---

## PHASE 4: ADVANCED MODEL COMPONENTS

### Step 4.1: TDA Features (Topological Data Analysis)
**File:** `features/topological.py`
**Dependency:** `pip install ripser persim`
```python
def compute_tda_features(returns, window=20):
    # Takens delay embedding
    # Vietoris-Rips complex
    # Extract Betti numbers, persistence
```
**Features:** betti_0, betti_1, max_persistence

### Step 4.2: Blackwell Regime Detection
**File:** `models/blackwell.py`
**Alternative:** Start with simpler HMM (3-state)
```python
class BlackwellApproach:
    def detect_regime(self, returns, volatility):
        # Returns: 'BULL', 'BEAR', or 'SIDEWAYS'
```

### Step 4.3: Cointegration Tests
**File:** `validation/cointegration.py`
```python
from statsmodels.tsa.stattools import coint
def find_cointegrated_pairs(prices, threshold=0.05):
    # For pairs trading signals
```

---

## PHASE 5: EXPLAINABILITY

### Step 5.1: Decision Trace Logging
**File:** `explain/decision_trace.py`
Log for EVERY trade:
- Regime (Bull/Bear/Sideways)
- GNN signal
- TCN forecast
- Sentiment score
- Gate decision
- Position change
- Cost estimate
- Actual outcome (filled later)

### Step 5.2: Simplicity Bias Check
```python
def select_best_model(results):
    """If 2 models within 5%, choose simpler one."""
```

---

## PHASE 6: FULL MODEL INTEGRATION

### Step 6.1: Create Model Full (Model A)
**File:** `models/model_full.py`
Combine: TDA + GNN + TCN + Blackwell + FinBERT

### Step 6.2: Ablation Study
Compare 3 variants:
| Model | Components | Expected Edge |
|-------|------------|---------------|
| A (Full) | TDA + GNN + TCN + Blackwell + FinBERT | +3-5% Sharpe |
| B (Minimal) | GNN + TCN + FinBERT | Baseline |
| C (TDA Only) | TDA + GNN + TCN + FinBERT | +1-2% vs B |

---

## PHASE 7: PRODUCTION

### Step 7.1: Expand Stock Universe
- Add 40+ more BSE stocks
- Update `config/config.yaml`
- Proper sector classification

### Step 7.2: Retrain with Full Data
- 5 years of data
- More stocks
- Fine-tune hyperparameters

---

## Quick Commands

// turbo-all

### Run daily prediction
```python
exec(open('/content/drive/MyDrive/ml/scripts/daily_update.py').read())
predictions = quick_predict()
```

### Run price forecast
```python
exec(open('/content/drive/MyDrive/ml/scripts/predict_price.py').read())
predictions = predict_prices_main()
```

### Run walk-forward backtest
```python
exec(open('/content/drive/MyDrive/ml/scripts/backtest.py').read())
results, metrics = run_walkforward_backtest()
```
