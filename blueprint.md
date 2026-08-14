# CausalFolio Scaling & Web App Blueprint

To prevent context exhaustion and ensure focused execution, this blueprint divides the scaling to 100/500 stocks and the web app creation into distinct, sequential "Roles" or "Phases". 

When executing, we will switch into the mindset of each role, complete its specific checklists, verify its outputs, and only then proceed to the next role.

---

## Role 1: Data Engineer
**Objective:** Establish robust, rate-limited pipelines to download and format large-scale universe data without triggering API bans or OOM (Out of Memory) crashes.

1.  **Define the Universe:** Create `data/universe_loader.py` to maintain a static or dynamic list of the BSE Top 100/500 ticker symbols. Link this to `config.yaml`.
2.  **Robust Price Fetching:** Refactor `data/bse_loader.py` to batch `yfinance` requests (e.g., 50 tickers at a time), implement exponential backoff on `429 Too Many Requests` errors, and cache raw downloads to disk.
3.  **Throttled News Scraping:** Update `data/news_scraper.py` to strictly enforce the `delay_seconds` configuration to prevent IP bans. Implement resuming capabilities if the scraper fails mid-run across 500 stocks.
4.  **Verification:** Run the pipelines locally on the Top 100 universe. Verify that CSV/Parquet files are written correctly, rate limits aren't breached, and memory usage remains stable.

---

## Role 2: Quantitative Researcher / ML Engineer
**Objective:** Scale the Feature Engineering and Model Architecture to handle massive spatial density (the 500-node graph) and increased state complexity.

1.  **Graph Sparsification:** Modify `features/graph_builder.py`. A fully connected 500-node graph ($250,000$ edges) is mathematically correct but computationally devastating. Implement k-Nearest Neighbors (k-NN) or a hard correlation threshold to sparsify the adjacency matrix so message-passing in the GNN remains tractable.
2.  **Capacity Tuning:** Update `config.yaml` to increase `gnn_hidden_dim` and `tcn_hidden_dim` to provide the network enough capacity to internalize the larger BSE 100/500 state space. 
3.  **Training Pipeline Adaptation:** Refactor `scripts/train_initial.py` to handle the larger batched dataset. Add memory profiling logs. Ensure the loss function adequately handles extreme outliers present in the wider universe.
4.  **Verification:** Execute a test training run on 6 months of the Top 100 data. Ensure the PyTorch/CUDA environment does not throw OOM exceptions. 

---

## Role 3: Backend Systems Engineer
**Objective:** Expose the trained quantitative model via a low-latency, stateless API.

1.  **API Framework:** Create a new `web_backend/` directory. Initialize a FastAPI (or Flask) application.
2.  **Model Loading Strategy:** Design the backend to load the `Model B` weight file (e.g., `model_checkpoint.pt`) into GPU/CPU memory once upon startup.
3.  **Inference Endpoint:** Refactor logic from `scripts/predict_price.py` into a modular inference engine. Create endpoints (e.g., `GET /api/predict/{ticker}`) that return historically predicted vs actual prices, the forward forecast, and the active FinBERT/TDA signals in JSON format.
4.  **Verification:** Send HTTP requests to the local server (via `curl` or Postman equivalents) and verify JSON structure and response times.

---

## Role 4: Frontend UI Developer
**Objective:** Build a stunning, interactive predictive visualization layer for the user.

1.  **Initialize Application:** Create a new `web_frontend/` directory. Initialize a modern web stack (e.g., Next.js/React or Vanilla HTML/CSS/JS with Vite).
2.  **Aesthetics & Design System:** Implement a highly premium interface (dark mode, glassmorphism, dynamic animations) suitable for a quantitative hedge fund terminal.
3.  **Search & Selection:** Build a searchable dashboard allowing the user to select any of the 100/500 processed BSE stocks.
4.  **Interactive Visualization:** Integrate a premium charting library (e.g., Recharts, Chart.js, or lightweight TradingView). Fetch data from the `web_backend` and overlay Historical Prices with the Model's Forecast trajectory. 
5.  **Verification:** Click through the UI. Ensure charts render quickly, data matches the JSON payloads, and the aesthetic meets the requested "wow factor".

---

## Role 5: Full System Verifier (Final Phase)
**Objective:** End-to-end testing and the final massive data run.

1.  **Integration Test:** Select a random stock in the frontend UI and ensure data flows perfectly from the raw CSV cache → Inference Engine → Backend API → Frontend Chart.
2.  **The Grand Retrain:** Once the entire architecture is proven stable, act as the final operator to initiate the comprehensive data fetch and `train_initial.py` execution for the full 2020-2023+ timeline across the targeted 100/500 universe.
