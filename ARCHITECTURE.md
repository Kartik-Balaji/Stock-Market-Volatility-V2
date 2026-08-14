# CausalFolio v2: Full Topological Causal Portfolio Intelligence

## Architecture Overview

**Full Model A** + Institutional Quant Practices

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  BSE Prices ──► Classical Features ──► Returns, Vol, RSI, Momentum          │
│                         ↓                                                    │
│  Price Windows ──► TDA Persistence ──► Betti Numbers, Landscapes            │
│                                                                              │
│  News (3 sources) ──► FinBERT ──► Sentiment Scores [-1, +1]                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐             │
│  │   GNN    │───►│   TCN    │───►│ Blackwell │───►│  Fusion  │             │
│  │ (GATv2)  │    │(Temporal)│    │ (Regime)  │    │  Layer   │             │
│  └──────────┘    └──────────┘    └───────────┘    └──────────┘             │
│       ↑                                                 ↑                   │
│  TDA Features                                    Sentiment Gate             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Purged Cross-Validation ──► Walk-Forward Testing ──► Performance Metrics  │
│        (5-day gap)              (Rolling windows)       (Sharpe, VaR)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RISK LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Position Sizing ──► Leverage Control ──► VaR/CVaR Check ──► Sector Limits │
│     (Kelly)           (Max 150%)          (95% conf)        (Max 30%)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXECUTION LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Transaction Costs ──► Slippage Model ──► Decision Trace ──► Final Signal  │
│    (STT, GST)          (Kyle's λ)         (Full logging)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Full Model Components

| Component | Status | Description |
|-----------|--------|-------------|
| **TDA** | 🔲 To Build | Topological features (Betti numbers, persistence) |
| **GNN (GATv2)** | ✅ Built | Cross-stock relationships via attention |
| **TCN** | ✅ Built | Temporal patterns, 127-day receptive field |
| **Blackwell** | 🔲 To Build | Regime detection (Bull/Bear/Sideways) |
| **FinBERT** | ✅ Built | Sentiment from yfinance, Google News, NewsAPI |
| **Purged CV** | 🔲 To Build | Time-series safe cross-validation |
| **Walk-Forward** | 🔲 To Build | Rolling window validation |
| **Risk Metrics** | 🔲 To Build | Sharpe, Sortino, VaR, CVaR |
| **Decision Trace** | 🔲 To Build | Full explainability logging |

---

## 1. TDA Features (Topological Data Analysis)

**Purpose:** Capture market structure patterns invisible to traditional features.

```python
# TDA Pipeline
Price Windows → Point Cloud → Vietoris-Rips Complex → Persistence Diagram → Features

# Key Features:
# - Betti-0: Connected components (market fragmentation)
# - Betti-1: Loops/cycles (mean reversion signals)
# - Persistence: Feature stability (noise vs signal)
# - Landscape: Vectorized topology for ML input
```

**Implementation:**
```python
from ripser import ripser
from persim import plot_diagrams

def compute_tda_features(returns, window=20):
    """Compute topological features from price data."""
    # Create delay embedding (Takens)
    delay = 5
    embedding = np.array([returns[i:i+window] for i in range(0, len(returns)-window, delay)])
    
    # Compute persistence
    diagrams = ripser(embedding, maxdim=1)['dgms']
    
    # Extract features
    h0 = diagrams[0]  # Connected components
    h1 = diagrams[1]  # 1-cycles
    
    return {
        'betti_0': len(h0),
        'betti_1': len(h1),
        'max_persistence_0': (h0[:, 1] - h0[:, 0]).max() if len(h0) > 0 else 0,
        'max_persistence_1': (h1[:, 1] - h1[:, 0]).max() if len(h1) > 0 else 0,
    }
```

---

## 2. Blackwell Regime Detection

**Purpose:** Identify market regimes (Bull/Bear/Sideways) to adjust predictions.

```python
class BlackwellApproach:
    """
    Blackwell's approachability for regime detection.
    
    Unlike HMM (assumes stationary), Blackwell handles:
    - Non-stationary markets
    - Regime transitions
    - Uncertainty in regime boundaries
    """
    
    def __init__(self, n_regimes=3):
        self.regimes = ['BULL', 'BEAR', 'SIDEWAYS']
        
    def detect_regime(self, returns, volatility, sentiment):
        """Classify current market regime."""
        # Regime rules (can be learned or rule-based)
        avg_return = returns[-20:].mean()
        avg_vol = volatility[-20:].mean()
        
        if avg_return > 0.001 and avg_vol < 0.02:
            return 'BULL', 0.8
        elif avg_return < -0.001 and avg_vol > 0.025:
            return 'BEAR', 0.7
        else:
            return 'SIDEWAYS', 0.6
```

**Alternative:** Start with HMM if Blackwell is too complex.

---

## 3. Validation Framework

### Purged Cross-Validation
```python
def purged_kfold_split(n_samples, k=5, purge_gap=5):
    """Time-series CV with purge gap to prevent leakage."""
    fold_size = n_samples // k
    
    for fold in range(k):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        
        # Purge: Remove samples too close to test set
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[max(0, test_start - purge_gap):min(n_samples, test_end + purge_gap)] = False
        
        yield np.where(train_mask)[0], np.arange(test_start, test_end)
```

### Walk-Forward Testing
```python
def walk_forward_split(n_samples, initial_train=252, test_size=21, step=21):
    """Expanding window walk-forward."""
    splits = []
    
    for train_end in range(initial_train, n_samples - test_size, step):
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + test_size)
        splits.append((train_idx, test_idx))
    
    return splits
```

---

## 4. Performance Metrics

```python
def compute_all_metrics(returns, benchmark_returns=None, rf=0.05/252):
    """Comprehensive performance metrics."""
    
    # Returns-based
    sharpe = (returns.mean() - rf) / returns.std() * np.sqrt(252)
    sortino = (returns.mean() - rf) / returns[returns < 0].std() * np.sqrt(252)
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Risk
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Market-relative (if benchmark provided)
    if benchmark_returns is not None:
        covar = np.cov(returns, benchmark_returns)[0, 1]
        beta = covar / np.var(benchmark_returns)
        alpha = returns.mean() - (rf + beta * (benchmark_returns.mean() - rf))
    else:
        beta, alpha = None, None
    
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'beta': beta,
        'alpha': alpha
    }
```

---

## 5. Risk Management

### Position Limits
```python
LIMITS = {
    'max_gross_exposure': 1.5,    # 150% total
    'max_net_exposure': 0.3,      # 30% directional
    'max_single_position': 0.1,   # 10% per stock
    'max_sector_exposure': 0.3,   # 30% per sector
}

def apply_risk_limits(positions, sectors):
    """Enforce all risk limits."""
    # Clip individual positions
    positions = np.clip(positions, -LIMITS['max_single_position'], LIMITS['max_single_position'])
    
    # Scale for gross exposure
    gross = np.abs(positions).sum()
    if gross > LIMITS['max_gross_exposure']:
        positions *= LIMITS['max_gross_exposure'] / gross
    
    return positions
```

### VaR Check
```python
def check_var_limit(positions, returns_history, var_limit=0.05):
    """Ensure portfolio VaR is within limits."""
    portfolio_returns = (returns_history * positions).sum(axis=1)
    var_95 = np.percentile(portfolio_returns, 5)
    
    if abs(var_95) > var_limit:
        scale = var_limit / abs(var_95)
        return positions * scale
    
    return positions
```

---

## 6. Transaction Costs (BSE-specific)

```python
def calculate_execution_cost(trade_value, is_sell=False):
    """BSE transaction costs."""
    costs = {
        'spread': trade_value * 0.0010,        # 10 bps half-spread
        'brokerage': trade_value * 0.0005,     # 5 bps
        'stt': trade_value * 0.00025 if is_sell else 0,  # 2.5 bps (sell only)
        'exchange': trade_value * 0.0001,      # 1 bps
        'gst': 0  # Calculated below
    }
    costs['gst'] = costs['brokerage'] * 0.18   # 18% GST on brokerage
    
    return sum(costs.values())
```

---

## 7. Decision Trace (Explainability)

```python
@dataclass
class TradeDecision:
    """Log every decision for debugging and research."""
    timestamp: datetime
    ticker: str
    
    # Model outputs
    tda_signal: float          # Topological pattern strength
    gnn_embedding: np.ndarray  # Cross-stock signal
    tcn_forecast: float        # Temporal prediction
    blackwell_regime: str      # Bull/Bear/Sideways
    sentiment_score: float     # FinBERT output
    
    # Combined
    raw_signal: float          # Before risk adjustment
    risk_adjusted_signal: float # After limits
    
    # Execution
    position_change: float
    estimated_cost: float
    estimated_slippage: float
    
    # Outcome (filled later)
    actual_return: float = None
    
def log_decision(decision: TradeDecision, db):
    """Store for analysis."""
    db.insert('decisions', asdict(decision))
```

---

## 8. Implementation Phases

| Phase | Components | Priority | Status |
|-------|------------|----------|--------|
| **Phase 1** | GNN + TCN + FinBERT (Minimal) | ✅ Done | Complete |
| **Phase 2** | Return Prediction Training | 🔴 High | In Progress |
| **Phase 3** | Walk-Forward + Risk Metrics | 🔴 High | Next |
| **Phase 4** | TDA Features | 🟡 Medium | Planned |
| **Phase 5** | Blackwell Regime | 🟡 Medium | Planned |
| **Phase 6** | Decision Trace + Costs | 🟡 Medium | Planned |
| **Phase 7** | Full Model A Integration | 🔴 High | Final |

---

## 9. Model Comparison (Ablation)

| Variant | Components | Expected Edge |
|---------|------------|---------------|
| **Model A (Full)** | TDA + GNN + TCN + Blackwell + FinBERT | +3-5% Sharpe |
| **Model B (Minimal)** | GNN + TCN + FinBERT | ✅ Baseline (working) |
| **Model C (TDA Only)** | TDA + GNN + TCN + FinBERT | +1-2% vs B |

---

## 10. Simplicity Bias

> **Rule:** If two models perform within 5% → choose simpler.

```python
def select_best_model(results):
    """Choose simplest model within performance threshold."""
    best_sharpe = max(r['sharpe'] for r in results)
    threshold = best_sharpe * 0.95  # 5% tolerance
    
    # Sort by complexity (parameters)
    sorted_results = sorted(results, key=lambda x: x['params'])
    
    for r in sorted_results:
        if r['sharpe'] >= threshold:
            return r['model']  # Simplest that meets threshold
```

---

## File Structure

```
ml/
├── data/
│   ├── bse_loader.py
│   └── news_scraper.py          ✅ (3 sources)
├── features/
│   ├── classical.py             ✅ (+ forward returns)
│   ├── topological.py           🔲 (TDA)
│   └── graph_builder.py         ✅
├── models/
│   ├── model_full.py            🔲 (Model A)
│   ├── model_minimal.py         ✅ (Model B)
│   ├── model_tda.py             🔲 (Model C)
│   ├── gnn.py                   ✅
│   ├── tcn.py                   ✅
│   ├── blackwell.py             🔲
│   └── sentiment.py             ✅
├── validation/
│   ├── purged_cv.py             🔲
│   ├── walk_forward.py          🔲
│   └── metrics.py               🔲
├── risk/
│   ├── position_limits.py       🔲
│   ├── var_model.py             🔲
│   └── costs.py                 🔲
├── explain/
│   └── decision_trace.py        🔲
└── scripts/
    ├── train_initial.py         ✅
    ├── train_returns.py         ✅
    ├── daily_update.py          ✅
    ├── predict_price.py         ✅
    └── backtest.py              🔲
```
