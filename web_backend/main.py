from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# Initialize FastAPI App
app = FastAPI(
    title="CausalFolio Inference API",
    description="Serves predictions from the CausalFolio topological model",
    version="1.0.0"
)

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow React/Next.js frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder for the loaded model to keep it in memory
class ModelContainer:
    def __init__(self):
        self.model = None
        self.tickers = []
        self.features_dict = {}

model_state = ModelContainer()

@app.on_event("startup")
async def load_model_on_startup():
    """
    Loads the PyTorch model and latest cached data into memory ONCE at startup.
    This prevents reloading the heavy 500-stock checkpoint on every API call.
    """
    print("=" * 50)
    print("Initializing CausalFolio Backend Engine...")
    print("=" * 50)
    
    # In a real environment, you load the .pt weights here:
    # model_state.model = torch.load('checkpoints/causalfolio_v3.pt')
    
    # For now, we simulate having the ticker list loaded
    from data.universe_loader import get_universe_tickers
    model_state.tickers = get_universe_tickers()
    
    print(f"✓ Backend ready. Armed to serve predictions for {len(model_state.tickers)} stocks.")

@app.get("/api/health")
def health_check():
    return {"status": "online", "model_loaded": model_state.tickers is not None}

@app.get("/api/tickers")
def get_available_tickers():
    """Returns the list of 100/500 strings the user can search for."""
    return {"tickers": model_state.tickers}

@app.get("/api/predict/{ticker}")
def get_prediction(ticker: str):
    """
    Returns historical pricing and future forecast for a given stock.
    Normally this would invoke `model(features_dict[ticker])`.
    """
    ticker = ticker.upper()
    if ticker not in model_state.tickers:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in model universe.")
        
    # Simulated Inference Output (Replacing this with actual model.forward() later)
    # We generate a realistic looking historical + predictive JSON payload
    today = datetime.now()
    dates = [(today - pd.Timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
    future_dates = [(today + pd.Timedelta(days=x)).strftime('%Y-%m-%d') for x in range(1, 6)]
    
    # Random historical walk
    base_price = 2500.0 + np.random.randn() * 500
    hist_prices = [base_price]
    for _ in range(29):
        hist_prices.append(hist_prices[-1] * (1 + np.random.randn() * 0.015))
        
    # Forecasted trajectory (5-days out)
    forecast_signal = np.random.choice(["UP", "DOWN", "SIDEWAYS"])
    forecast_prices = [hist_prices[-1]]
    trend = 0.02 if forecast_signal == "UP" else (-0.02 if forecast_signal == "DOWN" else 0.0)
    
    for _ in range(5):
        forecast_prices.append(forecast_prices[-1] * (1 + trend + np.random.randn() * 0.005))
    
    return {
        "ticker": ticker,
        "current_price": round(hist_prices[-1], 2),
        "forecast_signal": forecast_signal,
        "sentiment_score": round(np.random.uniform(-1, 1), 2),
        "tda_betti_0": int(np.random.randint(1, 5)), # Topological feature
        "history": {
            "dates": dates,
            "prices": [round(p, 2) for p in hist_prices]
        },
        "forecast": {
            "dates": future_dates,
            "prices": [round(p, 2) for p in forecast_prices[1:]] # Skip day 0
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
