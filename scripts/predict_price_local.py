"""
CausalFolio - Local Price Prediction
====================================
Runs price predictions using a local checkpoint from pastmodels.
No external news or transformer downloads are used.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import yaml
import pandas as pd


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CONFIG = {
    "config_path": str(BASE_DIR / "config" / "config.yaml"),
    "model_dir": str(BASE_DIR / "checkpoints"),
    "lookback_days": 250,
}


def setup() -> None:
    print("=" * 60)
    print("CausalFolio - Local Price Prediction")
    print("=" * 60)


def load_config() -> dict:
    with open(CONFIG["config_path"], "r") as f:
        return yaml.safe_load(f)


def normalize_ticker(ticker: str, suffix: str) -> str:
    if ticker.endswith(suffix):
        return ticker
    return f"{ticker}{suffix}"


def load_stocks(target_ticker: str | None = None) -> tuple[list[str], dict[str, str]]:
    from data.universe_loader import get_universe_tickers, get_sector_mapping

    config = load_config()
    tickers = get_universe_tickers(config)
    sectors = get_sector_mapping(config)

    if target_ticker:
        suffix = config.get("universe", {}).get("exchange_suffix", ".BO")
        target = normalize_ticker(target_ticker, suffix)
        return [target], {target: sectors.get(target, "Unknown")}

    return tickers, sectors


def fetch_data(tickers: list[str], days: int = 250, progress: bool = True) -> pd.DataFrame:
    from data.bse_loader import get_prices

    end = datetime.now()
    start = end - timedelta(days=days + 10)
    return get_prices(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=progress,
    )


def compute_price_indicators(data: pd.DataFrame, tickers: list[str]) -> dict[str, dict[str, float]]:
    indicators: dict[str, dict[str, float]] = {}

    for ticker in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            level_0 = [str(x) for x in data.columns.get_level_values(0).unique()]
            level_1 = [str(x) for x in data.columns.get_level_values(1).unique()]

            alt_ticker = ticker.replace('.BO', '.NS') if ticker.endswith('.BO') else ticker.replace('.NS', '.BO')
            base_symbol = ticker.split('.')[0]
            
            matched_col = None
            is_level_0 = False
            for candidate in [ticker, alt_ticker, base_symbol]:
                if candidate in level_0:
                    matched_col = candidate
                    is_level_0 = True
                    break
                elif candidate in level_1:
                    matched_col = candidate
                    is_level_0 = False
                    break

            if matched_col is not None:
                if is_level_0:
                    close = data[matched_col]["Close"]
                else:
                    close = data.xs(matched_col, level=1, axis=1)["Close"]
            else:
                continue
        else:
            close = data["Close"]

        close = close.dropna()
        if len(close) < 50:
            continue

        current_price = close.iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        momentum_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        if current_price > sma_20 > sma_50:
            trend_signal = "BULLISH"
        elif current_price < sma_20 < sma_50:
            trend_signal = "BEARISH"
        else:
            trend_signal = "NEUTRAL"

        indicators[ticker] = {
            "current_price": float(current_price),
            "sma_20": float(sma_20),
            "sma_50": float(sma_50),
            "rsi": float(rsi),
            "momentum_5d": float(momentum_5d),
            "trend_signal": trend_signal,
        }

    return indicators


def version_key(name: str) -> tuple[int, ...]:
    match = re.search(r"v(\d+(?:\.\d+)*)", name)
    if not match:
        return tuple()
    return tuple(int(x) for x in match.group(1).split("."))


def resolve_checkpoint(model_name: str | None = None) -> Path:
    model_dir = Path(CONFIG["model_dir"])
    if model_name:
        checkpoint = model_dir / model_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    default_cp = model_dir / "causalfolio_v3.pt"
    if default_cp.exists():
        return default_cp

    candidates = [p for p in model_dir.glob("causalfolio_v*.pt") if not p.name.endswith("_old.pt")]
    if not candidates:
        candidates = list(model_dir.glob("causalfolio_v*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    candidates.sort(key=lambda p: version_key(p.name), reverse=True)
    return candidates[0]


def get_predictions(
    tickers: list[str],
    sectors: dict[str, str],
    data: pd.DataFrame,
    model_name: str | None = None,
) -> tuple[list[str], list[float], list[int], dict[str, float]]:
    from features.classical import build_multi_stock_features, features_to_tensor
    from features.graph_builder import build_graph
    from models.model_minimal import CausalFolioMinimal

    features_dict = build_multi_stock_features(data, tickers)
    valid_tickers = [t for t in tickers if t in features_dict and len(features_dict[t]) > 0]
    if not valid_tickers:
        raise ValueError("No valid tickers with features. Check price data.")

    tensor, _, _, valid_tickers = features_to_tensor(features_dict, valid_tickers)
    edge_index, _ = build_graph(
        features_dict,
        valid_tickers,
        sectors,
        corr_threshold=0.3,
        max_edges_per_node=5,
    )

    sentiment_scores = {t: 0.0 for t in valid_tickers}
    sentiment = torch.tensor([sentiment_scores[t] for t in valid_tickers], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = resolve_checkpoint(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = checkpoint.get("config", {})
    model = CausalFolioMinimal(
        num_features=cfg.get("num_features", 10),
        num_stocks=len(valid_tickers),
        gnn_hidden=cfg.get("gnn_hidden", 32),
        gnn_output=cfg.get("gnn_output", 16),
        tcn_hidden=cfg.get("tcn_hidden", 32),
        tcn_layers=cfg.get("tcn_layers", 4),
        dropout=cfg.get("dropout", 0.2),
        use_sentiment=cfg.get("use_sentiment", True),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    f_mean = cfg.get("feature_mean")
    f_std = cfg.get("feature_std")
    f_names = cfg.get("feature_names")
    m_neutral = cfg.get("market_neutralized", False)
    
    from features.classical import apply_preprocessing
    tensor = apply_preprocessing(
        tensor.float(),
        f_names if f_names is not None else list(features_dict[valid_tickers[0]].columns),
        f_mean,
        f_std,
        market_neutralized=m_neutral
    )

    with torch.no_grad():
        outputs = model(tensor.to(device), edge_index.to(device), sentiment.to(device))
        vol = outputs["volatility"][-1].cpu().numpy().flatten()
        if "returns" in outputs:
            ret_preds = outputs["returns"][-1].cpu().numpy().flatten()
        else:
            ret_preds = outputs["direction"][-1].cpu().numpy().flatten()

    return valid_tickers, vol.tolist(), ret_preds.tolist(), sentiment_scores


def predict_prices(
    tickers: list[str],
    indicators: dict[str, dict[str, float]],
    vol: list[float],
    ret_preds: list[float],
    sentiment: dict[str, float],
    target_ticker: str | None = None,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    if target_ticker:
        print(f"Step 4: Price Predictions (Local Model) for {target_ticker}")
    else:
        print("Step 4: Price Predictions (Local Model)")
    print("=" * 60)

    predictions: list[dict[str, float | str]] = []
    for i, ticker in enumerate(tickers):
        if target_ticker and ticker != target_ticker:
            continue

        if ticker not in indicators:
            continue

        ind = indicators[ticker]
        v = float(vol[i])
        s = float(sentiment.get(ticker, 0.0))
        raw_ret = float(ret_preds[i])

        current = float(ind["current_price"])

        model_move_pct = raw_ret * 100.0
        expected_move_pct = model_move_pct + (s * 3.0) + (ind["momentum_5d"] * 0.05)
        expected_move_pct = max(-15.0, min(15.0, expected_move_pct))

        target = current * (1.0 + expected_move_pct / 100.0)

        if expected_move_pct > 0.5:
            dir_str = "UP"
        elif expected_move_pct < -0.5:
            dir_str = "DOWN"
        else:
            dir_str = "SIDEWAYS"

        confidence = 50.0 + (abs(expected_move_pct) * 4.0) + ((1.0 - min(v, 0.4)) * 25.0)
        confidence = min(confidence, 92.0)
        if expected_move_pct > 0 and ind["trend_signal"] == "BULLISH":
            confidence += 4.0
        if expected_move_pct < 0 and ind["trend_signal"] == "BEARISH":
            confidence += 4.0

        five_day_vol = v / 7.0

        predictions.append(
            {
                "Ticker": ticker,
                "Current_Price": current,
                "Direction": dir_str,
                "Target_Price": target,
                "Expected_Move": expected_move_pct,
                "Support": current * (1.0 - five_day_vol),
                "Resistance": current * (1.0 + five_day_vol),
                "Volatility": v,
                "Confidence": confidence,
            }
        )

    predictions.sort(key=lambda x: -float(x["Confidence"]))

    print("\nPRICE PREDICTIONS (Next 5 Days):")
    print("=" * 90)
    print(f"{'Ticker':<12} {'Price':>10} {'Direction':>10} {'Target':>10} {'Move':>8} {'Support':>10} {'Resist':>10} {'Conf':>6}")
    print("-" * 90)

    for p in predictions:
        print(
            f"{p['Ticker']:<12} {p['Current_Price']:>10.2f} {p['Direction']:>10} "
            f"{p['Target_Price']:>10.2f} {p['Expected_Move']:+.1f}%    "
            f"{p['Support']:>10.2f} {p['Resistance']:>10.2f} {p['Confidence']:>5.1f}%"
        )

    return pd.DataFrame(predictions)


def predict_prices_main(
    target_ticker: str | None = None,
    model_name: str | None = None,
    lookback_days: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    setup()
    tickers, sectors = load_stocks(target_ticker)
    data = fetch_data(tickers, days=lookback_days or CONFIG["lookback_days"], progress=progress)
    indicators = compute_price_indicators(data, tickers)
    valid_tickers, vol, dir_classes, sentiment = get_predictions(
        tickers, sectors, data, model_name=model_name
    )
    return predict_prices(valid_tickers, indicators, vol, dir_classes, sentiment, target_ticker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local CausalFolio price prediction")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker (e.g. TCS.BO)")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint filename in pastmodels")
    parser.add_argument("--lookback-days", type=int, default=CONFIG["lookback_days"])
    parser.add_argument("--no-progress", action="store_true", help="Disable download progress")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_prices_main(
        target_ticker=args.ticker,
        model_name=args.model,
        lookback_days=args.lookback_days,
        progress=not args.no_progress,
    )
