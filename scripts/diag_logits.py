import os, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from data.universe_loader import get_universe_tickers, get_sector_mapping
from data.bse_loader import get_prices
import yaml

with open(BASE_DIR / 'config' / 'config.yaml') as f:
    CFG = yaml.safe_load(f)

tickers = get_universe_tickers(CFG)[:20]
end = datetime.now()
start = end - timedelta(days=3 * 365)
data = get_prices(tickers, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
print("data", data.shape)

from features.classical import build_multi_stock_features, market_neutralize, normalize_features
from features.graph_builder import build_graph
from models.model_minimal import CausalFolioMinimal, TrainingModule

import models.gnn, models.tcn, models.model_minimal
from models.gnn import HAS_GEOMETRIC
print("HAS_GEOMETRIC", HAS_GEOMETRIC)

def get_close(data, t):
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        l0 = set(cols.get_level_values(0))
        l1 = set(cols.get_level_values(1))
        if t in l0:
            return data[t]['Close']
        if t in l1:
            return data.xs(t, level=1, axis=1)['Close']
    return data['Close'][t]

initial, test, step, n_periods, retrain_epochs = 252, 5, 5, 12, 15
all_rows = []
for period in range(n_periods):
    train_end = initial + period * step
    test_end = train_end + test
    if test_end >= len(data):
        break
    train_data = data.iloc[:train_end + 1]
    valid = []
    for t in tickers:
        c = get_close(train_data, t)
        if c is None:
            continue
        if int(c.dropna().shape[0]) >= int(np.ceil(len(train_data) * 0.9)):
            valid.append(t)
    if not valid:
        continue
    fd = build_multi_stock_features(train_data, valid, include_forward_returns=True)
    cleaned = {}
    for t in valid:
        df = fd.get(t)
        if df is None:
            continue
        df = df.dropna()
        if not df.empty:
            cleaned[t] = df
    if not cleaned:
        continue
    common = sorted(set.intersection(*[set(df.index) for df in cleaned.values()]))
    fnames_all = list(next(iter(cleaned.values())).columns)
    model_cols = [c for c in fnames_all if not c.startswith('forward_return')]
    tl, vol_l, ret_l = [], [], []
    aticks = []
    for t in valid:
        if t not in cleaned:
            continue
        alg = cleaned[t].loc[common]
        if alg.isna().any().any():
            continue
        tl.append(torch.tensor(alg[model_cols].values, dtype=torch.float32))
        vol_l.append(alg['vol_5d'].values)
        ret_l.append(alg['forward_return_5d'].values)
        aticks.append(t)
    tensor = torch.stack(tl, dim=1)
    sector_map = {t: 'X' for t in aticks}
    edge_index, _ = build_graph(cleaned, aticks, sector_map, corr_threshold=0.3, max_edges_per_node=5)

    T, N = tensor.shape[0], len(aticks)
    vol_targets = torch.tensor(np.stack(vol_l, axis=1), dtype=torch.float32).unsqueeze(-1)
    ret_matrix = np.stack(ret_l, axis=1)
    thr = 2.0 / 100.0
    labels = np.ones_like(ret_matrix, dtype=np.int64)
    labels[ret_matrix < -thr] = 0
    labels[ret_matrix > thr] = 2
    lab = torch.tensor(labels, dtype=torch.long)
    counts = np.bincount(labels.flatten(), minlength=3).astype(float)
    total = counts.sum()
    w = np.where(counts > 0, total / (3.0 * counts), 1.0)
    w = w / w.sum() * 3.0
    cw = torch.tensor(w, dtype=torch.float32)

    sentiment = torch.zeros(N)
    tensor = market_neutralize(tensor, model_cols)
    tensor, stats = normalize_features(tensor)
    model = CausalFolioMinimal(
        num_features=len(model_cols), num_stocks=N,
        gnn_hidden=32, gnn_output=16, tcn_hidden=64, tcn_layers=3,
        dropout=0.3, use_sentiment=True)
    trainer = TrainingModule(model, learning_rate=1e-3, weight_decay=1e-4, device='cpu')
    trainer.train(tensor, edge_index, vol_targets, lab, cw, sentiment,
                  epochs=retrain_epochs, val_split=0.15, early_stopping=10, verbose=False)
    model.eval()
    with torch.no_grad():
        out = model(tensor, edge_index, sentiment)
    logits = out['direction']
    probs = torch.softmax(logits, dim=-1).numpy()

    # actual returns for the test period for each stock
    test_date = data.index[train_end]
    for i, t in enumerate(aticks):
        close = get_close(data, t)
        if close is None or train_end + test >= len(close):
            continue
        a = float(close.iloc[train_end])
        b = float(close.iloc[train_end + test])
        act = (b - a) / a * 100.0
        p = probs[-1, i]
        all_rows.append({
            'date': test_date, 'ticker': t,
            'p_down': p[0], 'p_side': p[1], 'p_up': p[2],
            'actual': act,
        })
    print(f"period {period} done")

df = pd.DataFrame(all_rows)
df.to_csv(BASE_DIR / 'diag_logits.csv', index=False)
print("rows", len(df))
thr2 = 2.0
df['true'] = np.where(df['actual'] > thr2, 2, np.where(df['actual'] < -thr2, 0, 1))
df['argmax'] = df[['p_down', 'p_side', 'p_up']].values.argmax(axis=1)
df['correct'] = df['true'] == df['argmax']
print("argmax acc", df['correct'].mean())
print("baseline (always side)", (df['true'] == 1).mean())

for margin in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    df['maxp'] = df[['p_down', 'p_side', 'p_up']].max(axis=1)
    df['conf'] = df['maxp'] >= (1.0 / 3 + margin)
    sub = df[df['conf']]
    acc = sub['correct'].mean() if len(sub) else float('nan')
    print(f"margin {margin}: confident {len(sub)}/{len(df)} acc {acc:.3f}")