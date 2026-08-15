# CausalFolio - Major Changes Log

- **Data Pipeline Isolation**: Decoupled feature filling from forward targets in `features_to_tensor` to prevent data/target leakage.
- **Model Architecture Restoration**: Reverted Head 2 back to continuous return regression with MSE loss, restored TCN hidden dim to 256, and added sign accuracy tracking in `TrainingModule`.
- **Training Pipeline Overhaul**: Upgraded `train_initial.py` to train on continuous returns, cleanly mask unclosed forward windows, save normalization statistics, and report directional sign accuracy alongside conviction thresholds.
- **Backtest Calibration & Metric Granularity**: Updated `scripts/backtest.py` with calibrated momentum fusion, 3-state direction evaluations (UP/DOWN/SIDEWAYS), and granular sign tracking.
- **Robust Ticker & Exchange Fallback**: Added automatic `.NS` fallback in `data/bse_loader.py` and fuzzy multi-index column matching in `features/classical.py` and `scripts/predict_price_local.py` for fault-tolerant single-stock and universe inference.
- **Local Prediction Validation**: Fixed checkpoint resolution in `scripts/predict_price_local.py` to point directly to `checkpoints/causalfolio_v3.pt` with full support for individual ticker runs.
