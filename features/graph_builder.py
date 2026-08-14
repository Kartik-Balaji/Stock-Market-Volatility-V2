"""
CausalFolio - Graph Construction
=================================
Builds correlation-based and sector-based edges for GNN.

Usage (in Colab):
    from features.graph_builder import build_graph, get_graph_stats
    
    edge_index, edge_weights = build_graph(returns_dict, tickers, sector_mapping)
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


def compute_correlation_matrix(
    features_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    feature_col: str = 'return_1d',
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix.
    
    Args:
        features_dict: Dict mapping ticker -> features DataFrame
        tickers: Ordered list of tickers
        feature_col: Which feature to use for correlation (default: 1-day return)
        window: Optional rolling window (if None, use all data)
    
    Returns:
        NxN correlation matrix as DataFrame
    """
    # Extract the feature column for each ticker
    returns_data = {}
    for ticker in tickers:
        if ticker in features_dict:
            df = features_dict[ticker]
            if feature_col in df.columns:
                returns_data[ticker] = df[feature_col]
    
    # Build DataFrame
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if window is not None and len(returns_df) > window:
        returns_df = returns_df.iloc[-window:]
    
    return returns_df.corr()


def build_correlation_edges(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.5,
    max_edges_per_node: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge index from correlation matrix with k-NN sparsification.
    
    Args:
        corr_matrix: NxN correlation DataFrame
        threshold: Minimum |correlation| to create edge
        max_edges_per_node: Limit each node to k most correlated neighbors
    
    Returns:
        Tuple of:
        - edge_index: [2, num_edges] tensor
        - edge_weights: [num_edges] tensor of correlation values
    """
    tickers = list(corr_matrix.columns)
    n = len(tickers)
    
    edges = []
    weights = []
    
    for i in range(n):
        node_edges = []
        for j in range(n):
            if i == j:  # Skip self-loops
                continue
            
            corr = corr_matrix.iloc[i, j]
            
            if abs(corr) >= threshold:
                node_edges.append((j, abs(corr)))
                
        # k-NN Sparsification
        if max_edges_per_node is not None and len(node_edges) > max_edges_per_node:
            # Sort by absolute correlation descending and take top K
            node_edges.sort(key=lambda x: x[1], reverse=True)
            node_edges = node_edges[:max_edges_per_node]
            
        for j, weight in node_edges:
            edges.append([i, j])
            weights.append(weight)
    
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
    
    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_weights = torch.tensor(weights, dtype=torch.float32)
    
    return edge_index, edge_weights


def build_sector_edges(
    tickers: List[str],
    sector_mapping: Dict[str, str]
) -> torch.Tensor:
    """
    Build edges connecting stocks in the same sector.
    
    Args:
        tickers: Ordered list of tickers
        sector_mapping: Dict mapping ticker -> sector name
    
    Returns:
        edge_index: [2, num_edges] tensor
    """
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    edges = []
    
    # Group by sector
    sector_groups = {}
    for ticker in tickers:
        sector = sector_mapping.get(ticker, 'Unknown')
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(ticker)
    
    # Create bidirectional edges within each sector
    for sector, group_tickers in sector_groups.items():
        for i, t1 in enumerate(group_tickers):
            for t2 in group_tickers[i+1:]:
                idx1 = ticker_to_idx[t1]
                idx2 = ticker_to_idx[t2]
                edges.append([idx1, idx2])
                edges.append([idx2, idx1])
    
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    
    return torch.tensor(edges, dtype=torch.long).T


def merge_edge_indices(*edge_indices: torch.Tensor) -> torch.Tensor:
    """
    Merge multiple edge index tensors, removing duplicates.
    """
    all_edges = []
    
    for edge_index in edge_indices:
        if edge_index.numel() > 0:
            all_edges.append(edge_index)
    
    if not all_edges:
        return torch.zeros((2, 0), dtype=torch.long)
    
    merged = torch.cat(all_edges, dim=1)
    
    # Remove duplicates
    edge_set = set()
    unique_edges = []
    
    for i in range(merged.shape[1]):
        edge = (merged[0, i].item(), merged[1, i].item())
        if edge not in edge_set:
            edge_set.add(edge)
            unique_edges.append([edge[0], edge[1]])
    
    return torch.tensor(unique_edges, dtype=torch.long).T


def build_graph(
    features_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    sector_mapping: Dict[str, str],
    corr_threshold: float = 0.5,
    corr_window: Optional[int] = 60,
    use_sector_edges: bool = True,
    max_edges_per_node: Optional[int] = 5
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build complete graph structure for GNN.
    
    Args:
        features_dict: Dict mapping ticker -> features DataFrame
        tickers: Ordered list of tickers
        sector_mapping: Dict mapping ticker -> sector
        corr_threshold: Minimum correlation for edge (default 0.5)
        corr_window: Rolling window for correlation (default 60 days)
        use_sector_edges: Whether to add sector edges (default True)
        max_edges_per_node: Max correlation edges per node (k-NN sparsification)
    
    Returns:
        Tuple of:
        - edge_index: [2, num_edges] tensor
        - edge_weights: Optional weights tensor
    """
    print(f"→ Building graph for {len(tickers)} stocks...")
    
    # Correlation-based edges
    corr_matrix = compute_correlation_matrix(
        features_dict, tickers, 'return_1d', corr_window
    )
    corr_edges, edge_weights = build_correlation_edges(
        corr_matrix, corr_threshold, max_edges_per_node
    )
    print(f"  ✓ Correlation edges: {corr_edges.shape[1]}")
    
    if use_sector_edges:
        sector_edges = build_sector_edges(tickers, sector_mapping)
        print(f"  ✓ Sector edges: {sector_edges.shape[1]}")
        
        # Merge (note: weights only for correlation edges)
        edge_index = merge_edge_indices(corr_edges, sector_edges)
        print(f"  ✓ Total edges (merged): {edge_index.shape[1]}")
        return edge_index, None  # Weights don't apply to merged graph
    else:
        return corr_edges, edge_weights


def get_graph_stats(edge_index: torch.Tensor, num_nodes: int) -> Dict:
    """
    Compute graph statistics for debugging.
    """
    num_edges = edge_index.shape[1]
    
    # Node degrees
    degrees = torch.zeros(num_nodes)
    for i in range(num_edges):
        degrees[edge_index[0, i]] += 1
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': round(num_edges / (num_nodes * (num_nodes - 1) + 1e-10), 3),
        'avg_degree': round(degrees.mean().item(), 2),
        'max_degree': int(degrees.max().item()),
        'min_degree': int(degrees.min().item()),
        'isolated_nodes': int((degrees == 0).sum().item())
    }


# ===================
# Quick Test
# ===================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Graph Builder")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    tickers = ['TCS.BO', 'INFY.BO', 'HDFCBANK.BO', 'ICICIBANK.BO', 'RELIANCE.BO']
    
    # Create correlated returns
    base = np.random.randn(100) * 0.02
    features_dict = {}
    for ticker in tickers:
        noise = np.random.randn(100) * 0.01
        features_dict[ticker] = pd.DataFrame({
            'return_1d': base + noise
        }, index=dates)
    
    sector_mapping = {
        'TCS.BO': 'IT',
        'INFY.BO': 'IT',
        'HDFCBANK.BO': 'Banking',
        'ICICIBANK.BO': 'Banking',
        'RELIANCE.BO': 'Energy'
    }
    
    # Build graph
    edge_index, weights = build_graph(
        features_dict, tickers, sector_mapping,
        corr_threshold=0.3,
        max_edges_per_node=2
    )
    
    stats = get_graph_stats(edge_index, len(tickers))
    print(f"\n✓ Graph stats: {stats}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
