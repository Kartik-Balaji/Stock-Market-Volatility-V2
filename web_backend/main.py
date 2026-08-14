import os
import sys
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features.classical import build_multi_stock_features, apply_normalization  # noqa: E402
from features.graph_builder import build_graph  # noqa: E402
from models.model_minimal import CausalFolioMinimal  # noqa: E402
from scripts.predict_price import build_features_tensor_strict  # noqa: E402

CONFIG = {
    "universe": os.environ.get("CF_UNIVERSE", "NIFTY_100"),
    "max_tickers": int(os.environ.get("CF_MAX_TICKERS", "100")),
    "lookback_days": int(os.environ.get("CF_LOOKBACK_DAYS", "250")),
    "checkpoint_dir": str(BASE_DIR / "checkpoints"),
    "model_name": os.environ.get("CF_MODEL_NAME", "causalfolio_v3.pt"),
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CausalFolio Inference API",
    description="Serves real predictions from the CausalFolio GNN+TCN+FinBERT model",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelContainer:
    """Holds the loaded model + full-universe inference results in memory."""

    def __init__(self):
        self.model = None
        self.tickers = []
        self.results = {}
        self.ready = False
        self.lock = threading.Lock()
        self.error = None


model_state = ModelContainer()


# ---------------------------------------------------------------------------
# Background training job control
# ---------------------------------------------------------------------------
class TrainJob:
    def __init__(self):
        self.running = False
        self.last_status = "idle"  # idle | running | success | error
        self.message = ""
        self.log = []
        self.started_at = None
        self.finished_at = None
        self.lock = threading.Lock()

    def to_dict(self):
        with self.lock:
            return {
                "running": self.running,
                "status": self.last_status,
                "message": self.message,
                "log": self.log[-40:],
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


train_job = TrainJob()


def _run_train_job():
    import io
    import contextlib

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            from scripts.daily_update import run_daily_update
            run_daily_update()
        with train_job.lock:
            train_job.running = False
            train_job.last_status = "success"
            train_job.message = "Training completed successfully."
            train_job.log = buf.getvalue().splitlines()
            train_job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:  # noqa: BLE001
        with train_job.lock:
            train_job.running = False
            train_job.last_status = "error"
            train_job.message = str(exc)
            train_job.log = buf.getvalue().splitlines()[-40:]
            train_job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Reload the inference engine so the newly trained weights are served.
    with model_state.lock:
        model_state.ready = False
        model_state.error = None
    threading.Thread(target=_ensure_loaded, daemon=True).start()


@app.post("/api/train")
def start_train():
    if train_job.running:
        return {
            "started": False,
            "detail": "A training job is already running.",
            **train_job.to_dict(),
        }
    with train_job.lock:
        train_job.running = True
        train_job.last_status = "running"
        train_job.message = "Training started."
        train_job.log = []
        train_job.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_job.finished_at = None
    threading.Thread(target=_run_train_job, daemon=True).start()
    return {"started": True, **train_job.to_dict()}


@app.get("/api/train/status")
def train_status():
    return train_job.to_dict()


# ---------------------------------------------------------------------------


def _get_close(data, ticker):
    """Return the Close price Series for a ticker regardless of MultiIndex orientation."""
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            return data[("Close", ticker)]
        if (ticker, "Close") in data.columns:
            return data[(ticker, "Close")]
    return data["Close"][ticker]


def _load_engine():
    """Build model + full-universe inference. Runs once, guarded by a lock."""
    from data.bse_loader import get_prices
    from data.universe_loader import get_universe_tickers, get_sector_mapping, load_config

    cfg = load_config(os.path.join(BASE_DIR, "config", "config.yaml"))
    cfg["universe"]["index"] = CONFIG["universe"]

    print("=" * 50)
    print("Initializing CausalFolio Real Inference Engine...")
    print("=" * 50)

    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["model_name"])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} not found. Train a model first."
        )

    all_tickers = get_universe_tickers(cfg)
    tickers = all_tickers[: CONFIG["max_tickers"]]
    sectors = get_sector_mapping(cfg)
    sectors = {t: sectors.get(t, "Unknown") for t in tickers}
    print(f"  Universe: {len(tickers)} tickers")

    end = datetime.now()
    start = end - pd.DateOffset(days=CONFIG["lookback_days"] + 60)
    data = get_prices(tickers, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), progress=False)
    if data is None or data.empty:
        raise RuntimeError("No price data fetched. Check network / yfinance.")

    features_dict = build_multi_stock_features(data, tickers)
    tensor, fnames, dates, aligned_tickers, aligned_features = build_features_tensor_strict(
        features_dict, tickers
    )
    if len(aligned_tickers) == 0:
        raise RuntimeError("No tickers had sufficient feature data.")
    print(f"  Tensor: {tensor.shape}  Tickers: {len(aligned_tickers)}")

    sector_map = {t: sectors.get(t, "Unknown") for t in aligned_tickers}
    edge_index, _ = build_graph(
        aligned_features, aligned_tickers, sector_map,
        corr_threshold=0.3, max_edges_per_node=5,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {})

    model = CausalFolioMinimal(
        num_features=model_config.get("num_features", 10),
        num_stocks=len(aligned_tickers),
        gnn_hidden=model_config.get("gnn_hidden", 32),
        gnn_output=model_config.get("gnn_output", 16),
        tcn_hidden=model_config.get("tcn_hidden", 32),
        tcn_layers=model_config.get("tcn_layers", 4),
        dropout=model_config.get("dropout", 0.2),
        use_sentiment=True,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    feature_mean = model_config.get("feature_mean")
    feature_std = model_config.get("feature_std")
    feature_names_config = model_config.get("feature_names")
    if feature_mean is not None and feature_std is not None:
        from features.classical import apply_preprocessing
        tensor = apply_preprocessing(
            tensor.float(),
            feature_names_config if feature_names_config is not None else fnames,
            feature_mean,
            feature_std,
            market_neutralized=model_config.get("market_neutralized", False),
        )

    sentiment = torch.zeros(len(aligned_tickers), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(tensor.to(device), edge_index.to(device), sentiment.to(device))
        vol_all = outputs["volatility"][-1].cpu()
        dir_logits = outputs["direction"][-1].cpu()
        dir_classes = torch.argmax(dir_logits, dim=-1).numpy()

    base_expected = {0: -2.5, 1: 0.0, 2: 2.5}
    directions = ["DOWN", "SIDEWAYS", "UP"]

    graph_cc = None
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        e = edge_index.cpu().numpy()
        mat = coo_matrix((np.ones(e.shape[1]), (e[0], e[1])),
                         shape=(len(aligned_tickers), len(aligned_tickers)))
        graph_cc = int(connected_components(mat, directed=False)[0])
    except Exception:
        graph_cc = 1

    now = pd.Timestamp(datetime.now()).normalize()
    results = {}
    for i, t in enumerate(aligned_tickers):
        close = _get_close(data, t).dropna()
        if close.empty:
            continue
        vol_i = float(vol_all[i])
        cls = int(dir_classes[i])
        expected_move = base_expected.get(cls, 0.0)
        current_price = float(close.iloc[-1])

        hist_dates = close.index.strftime("%Y-%m-%d").tolist()[-60:]
        hist_prices = [float(x) for x in close.values.tolist()[-60:]]

        fdates = [(now + pd.Timedelta(days=d + 1)).strftime("%Y-%m-%d") for d in range(5)]
        drift = expected_move / 100.0  # total 5-day drift
        fstep = drift / 5.0
        fprice = current_price
        forecast_prices = []
        for _ in range(5):
            fprice = fprice * (1.0 + fstep)
            forecast_prices.append(float(fprice))

        # Simple momentum-based sentiment proxy so the UI bar is meaningful.
        if len(hist_prices) >= 5:
            sent_proxy = max(-1.0, min(1.0, (hist_prices[-1] / hist_prices[-6] - 1.0) * 20.0))
        else:
            sent_proxy = 0.0

        results[t] = {
            "ticker": t,
            "current_price": round(current_price, 2),
            "forecast_signal": directions[cls],
            "expected_move_pct": round(expected_move, 2),
            "volatility": round(vol_i, 4),
            "sentiment_score": round(sent_proxy, 3),
            "tda_betti_0": graph_cc,
            "history": {"dates": hist_dates, "prices": [round(p, 2) for p in hist_prices]},
            "forecast": {"dates": fdates, "prices": [round(p, 2) for p in forecast_prices]},
        }

    model_state.model = model
    model_state.tickers = aligned_tickers
    model_state.results = results
    model_state.ready = True
    print(f"✓ Engine ready. Predictions cached for {len(results)} tickers.")


def _ensure_loaded():
    if model_state.ready:
        return
    with model_state.lock:
        if model_state.ready:
            return
        try:
            _load_engine()
        except Exception as exc:  # noqa: BLE001
            model_state.error = str(exc)
            raise


@app.on_event("startup")
async def startup():
    # Kick off loading in the background so the port is available immediately.
    threading.Thread(target=_ensure_loaded, daemon=True).start()


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model_state.ready,
        "tickers_loaded": len(model_state.tickers),
        "error": model_state.error,
    }


@app.get("/api/tickers")
def get_available_tickers():
    if not model_state.ready and model_state.error:
        raise HTTPException(status_code=503, detail=f"Engine not loaded: {model_state.error}")
    return {"tickers": model_state.tickers}


@app.get("/api/predict/{ticker}")
def get_prediction(ticker: str):
    if not model_state.ready:
        if model_state.error:
            raise HTTPException(status_code=503, detail=f"Engine failed to load: {model_state.error}")
        try:
            _ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Engine failed to load: {exc}")
        if not model_state.ready:
            raise HTTPException(status_code=503, detail="Engine still loading, retry in a moment.")

    normalized = ticker.upper()
    if normalized not in model_state.results:
        found = None
        for k in model_state.results:
            if k.upper() == normalized:
                found = k
                break
        if found is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in loaded universe.")
        normalized = found

    return model_state.results[normalized]


@app.get("/api/overview")
def get_overview():
    """Return the cached prediction for every loaded ticker (for the market table)."""
    if not model_state.ready:
        if model_state.error:
            raise HTTPException(status_code=503, detail=f"Engine failed to load: {model_state.error}")
        try:
            _ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Engine failed to load: {exc}")
        if not model_state.ready:
            raise HTTPException(status_code=503, detail="Engine still loading, retry in a moment.")

    rows = []
    for t in model_state.tickers:
        r = model_state.results.get(t)
        if r is None:
            continue
        rows.append({
            "ticker": t,
            "current_price": r["current_price"],
            "forecast_signal": r["forecast_signal"],
            "expected_move_pct": r["expected_move_pct"],
            "volatility": r["volatility"],
            "sentiment_score": r["sentiment_score"],
        })
    # Most-bullish first.
    order = {"UP": 0, "SIDEWAYS": 1, "DOWN": 2}
    rows.sort(key=lambda r: (order.get(r["forecast_signal"], 3), -abs(r["expected_move_pct"])))
    return {"count": len(rows), "universe": CONFIG["universe"], "rows": rows}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)