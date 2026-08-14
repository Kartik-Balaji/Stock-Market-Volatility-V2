# CausalFolio

CausalFolio is a spatial-temporal equity intelligence system for Indian markets (BSE/NIFTY universes). It combines graph-based stock relationship learning, temporal modeling, and news sentiment to produce short-horizon directional and volatility forecasts.

The current repository includes:
- Data and feature pipelines for BSE/NIFTY universes.
- A working Model B (minimal stack): GNN + TCN + sentiment fusion.
- Training, daily fine-tuning, prediction, and walk-forward backtest scripts.
- A FastAPI backend and React dashboard frontend.
- Architecture plans for future Model A components (TDA, regime layer, richer risk/validation stack).

## 1) What This Project Tries To Solve

Traditional single-stock models miss two major realities:
- Stocks move in networks, not isolation.
- Market behavior shifts across time and regimes.

CausalFolio addresses this by:
- Learning cross-stock dependencies with a Graph Neural Network (GNN).
- Learning temporal structure with a Temporal Convolutional Network (TCN).
- Injecting market narrative context through FinBERT-based sentiment.

Primary target use case:
- Short-term (about 5-day) directional classification (`DOWN`, `SIDEWAYS`, `UP`) plus volatility estimation for a basket of stocks.

## 2) High-Level Architecture

Current implemented model path:
1. OHLCV + volume history is fetched for a configured universe.
2. Classical features are engineered per stock.
3. A sparse stock graph is built from correlations (and optionally sector links).
4. GNN produces per-stock embeddings at each timestep.
5. TCN models sequential patterns over time.
6. Dual heads output:
   - Volatility (positive scalar)
   - Direction logits (`DOWN`, `SIDEWAYS`, `UP`)
7. Sentiment can be fused into both heads.

Planned/partial architecture (documented in `ARCHITECTURE.md`):
- Topological features (TDA)
- Regime detection (Blackwell/HMM-like layer)
- Purged CV and richer risk layer
- Decision-trace explainability module

## 3) Repository Structure

```
ml/
  config/
    config.yaml                 # Main runtime/training config

  data/
    bse_loader.py               # Batched yfinance loader with retry/backoff
    universe_loader.py          # Universe + sector mapping from local NIFTY CSVs
    news_scraper.py             # Multi-source news collection
    ind_nifty100list.csv
    ind_nifty500list.csv

  features/
    classical.py                # Returns/volatility/RSI/momentum/forward returns
    graph_builder.py            # Correlation + sector graph construction
    graph_builder.py            # Sparsification via max edges per node

  models/
    gnn.py                      # GATv2 graph encoder (+ fallback SimpleGNN)
    tcn.py                      # Causal dilated TCN blocks
    sentiment.py                # FinBERT and rule-based sentiment analyzers
    model_minimal.py            # Integrated Model B + training utilities
    model_minimal.py            # Dual-output: volatility + direction classes

  scripts/
    train_initial.py            # Main initial training entrypoint
    daily_update.py             # Daily fine-tuning and prediction
    predict_price.py            # One-shot prediction report
    backtest.py                 # Walk-forward backtest

  web_backend/
    main.py                     # FastAPI service (currently simulated inference output)

  web_frontend/
    src/App.jsx                 # React dashboard with ticker search + chart view
    package.json                # Vite/React/Recharts stack

  test_pipeline.py              # Data pipeline smoke test
  ARCHITECTURE.md               # Conceptual architecture and roadmap
  blueprint.md                  # Scaling and system roles blueprint
  notes.md                      # Strategy notes
```

## 4) Model Details (Implemented)

### 4.1 Graph Module (`models/gnn.py`)
- Uses `GATv2Conv` from `torch-geometric` when available.
- Supports multi-layer attention and optional batched forward pass.
- Includes fallback `SimpleGNN` when `torch-geometric` is unavailable.

### 4.2 Temporal Module (`models/tcn.py`)
- Uses causal, dilated convolutions.
- Residual blocks with normalization and dropout.
- Receptive field grows exponentially with dilation (`1, 2, 4, ...`).

### 4.3 Sentiment Module (`models/sentiment.py`)
- Primary: `ProsusAI/finbert` via Hugging Face `transformers`.
- Fallback: keyword-based rule model if transformer stack is not present.
- Produces score range roughly `[-1, +1]` (positive minus negative probability).

### 4.4 Integrated Model (`models/model_minimal.py`)
- Backbone: GNN -> TCN hidden states.
- Heads:
  - Volatility head (`Softplus` output)
  - Direction classification head (3 logits)
- Optional late-fusion sentiment layers for both outputs.
- Training module includes:
  - Combined loss (MSE + CrossEntropy)
  - Class weighting support
  - LR scheduling and early stopping
  - Checkpoint save/load

## 5) Data and Feature Pipeline

### 5.1 Universe Selection
Controlled by `config/config.yaml`:
- `universe.index`: `NIFTY_50`, `NIFTY_100`, or `NIFTY_500`
- `universe.exchange_suffix`: `.BO` or `.NS`

`data/universe_loader.py` loads constituents from local CSV snapshots.

### 5.2 Price Loading
`data/bse_loader.py`:
- Batched `yfinance` downloads
- Retry + delay behavior from config
- Handles varying `yfinance` MultiIndex formats

### 5.3 Feature Engineering
`features/classical.py` computes:
- Log returns (`1d`, `5d`, `20d`)
- Rolling realized volatility (`5d`, `20d`, `60d`)
- RSI
- Momentum
- Volume ratio
- Optional forward returns (targets)

### 5.4 Graph Construction
`features/graph_builder.py`:
- Correlation-based edges (thresholded absolute correlation)
- Optional sector edges
- Sparsification using `max_edges_per_node` (important for larger universes)

## 6) Training, Updating, Predicting, Backtesting

### 6.1 Initial Training
Runs full pipeline and writes checkpoint.

Command:
```bash
python scripts/train_initial.py
```

What it does:
- Loads universe from config
- Downloads historical data
- Builds features and graph
- Computes volatility and direction labels
- Trains model and saves checkpoint (`checkpoints/causalfolio_v3.pt`)

### 6.2 Daily Update
Fine-tunes on recent data and generates new forecast output.

Command:
```bash
python scripts/daily_update.py
```

### 6.3 One-Shot Prediction Report
Generates directional and price-range style output.

Command:
```bash
python scripts/predict_price.py
```

### 6.4 Walk-Forward Backtest
Evaluates rolling prediction behavior over historical segments.

Command:
```bash
python scripts/backtest.py
```

## 7) Web Stack

## 7.1 Backend (FastAPI)
File: `web_backend/main.py`

Run:
```bash
python web_backend/main.py
```

Default endpoint base: `http://localhost:8000/api`

Available endpoints:
- `GET /api/health`
- `GET /api/tickers`
- `GET /api/predict/{ticker}`

Important status note:
- The backend currently returns simulated prediction payloads (realistic mock series) rather than directly calling the model checkpoint. The startup path already prepares for real model loading.

## 7.2 Frontend (React + Vite)
Directory: `web_frontend/`

Install and run:
```bash
cd web_frontend
npm install
npm run dev
```

Frontend features:
- Symbol search and selection
- Forecast direction badges
- Sentiment and topological info cards
- Historical vs forecast line chart (`recharts`)

## 8) Setup Instructions (Windows / General)

### 8.1 Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Git (optional)

### 8.2 Python Environment
From repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional (if needed for your CUDA/PyTorch setup):
- Install PyTorch and `torch-geometric` compatible wheels according to your local CUDA/CPU environment.

### 8.3 Basic Pipeline Smoke Test
```bash
python test_pipeline.py
```

## 9) Configuration Guide

Main config file: `config/config.yaml`

Most important groups:
- `data`: market and date windows
- `universe`: which index/universe to run
- `api_limits`: yfinance batch/retry/delay
- `model`: GNN/TCN dimensions and dropout
- `training`: epochs, lr, batch size, regularization
- `paths`: checkpoint/log/cache folders
- `news`: source list and delays

Typical scale-up workflow:
1. Start with `NIFTY_100` and moderate hidden dims.
2. Validate memory footprint and graph size.
3. Move to `NIFTY_500` with stricter sparsification (`max_edges_per_node`).

## 10) Known Gaps and Practical Notes

- Some architecture docs describe advanced components that are planned but not fully implemented in code (TDA, Blackwell regime layer, purged CV modules as standalone package, full risk stack).
- `web_backend/main.py` currently serves simulated forecast data; production model inference integration is still pending.
- News API key values are currently hardcoded in scripts in this repo. For production or sharing, move keys to environment variables.
- Model and script comments still mention earlier naming in a few places; runtime paths now center around `causalfolio_v3.pt` and direction classification.

## 11) Suggested End-to-End Run Order

1. Install Python dependencies.
2. Run `python test_pipeline.py`.
3. Run `python scripts/train_initial.py`.
4. Run `python scripts/predict_price.py`.
5. Optionally run `python scripts/backtest.py`.
6. Start backend: `python web_backend/main.py`.
7. Start frontend from `web_frontend`: `npm install && npm run dev`.

## 12) Documentation Map

- `ARCHITECTURE.md`: system vision and phased architecture.
- `blueprint.md`: scaling and role-based execution blueprint.
- `notes.md`: strategic modeling notes and rationale.

If you are extending this project, start by aligning implementation decisions with `config/config.yaml`, then keep module boundaries clean:
- `data` for retrieval
- `features` for transformations
- `models` for learnable components
- `scripts` for orchestration
- `web_backend`/`web_frontend` for serving and UX
