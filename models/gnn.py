"""
CausalFolio - Graph Neural Network (GNN)
=========================================
Uses Graph Attention Network v2 (GATv2) to learn stock relationships.

Key concepts:
- Nodes = Stocks
- Edges = Relationships (correlation + sector)
- Attention = Learn which connections matter more

Usage:
    from models.gnn import StockGNN
    
    model = StockGNN(input_dim=10, hidden_dim=32, output_dim=16)
    embeddings = model(node_features, edge_index)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# Check if torch_geometric is available
try:
    from torch_geometric.nn import GATv2Conv
    HAS_GEOMETRIC = True
except ImportError:
    print("Warning: torch_geometric not installed. Run: pip install torch-geometric")
    HAS_GEOMETRIC = False


class StockGNN(nn.Module):
    """
    Graph Attention Network for stock relationship learning.
    
    Uses GATv2 which fixes attention mechanism issues in original GAT.
    Multiple attention heads learn different types of relationships.
    
    Architecture:
        Input [N, F] → GATv2 Layer 1 → ReLU → Dropout 
                     → GATv2 Layer 2 → Output [N, H]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Number of input features per node (10 in our case)
            hidden_dim: Hidden dimension for GATv2 layers
            output_dim: Output embedding dimension per node
            num_heads: Number of attention heads
            num_layers: Number of GATv2 layers
            dropout: Dropout probability
        """
        super().__init__()
        
        if not HAS_GEOMETRIC:
            raise ImportError("torch_geometric required. Install with: pip install torch-geometric")
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GATv2 layers
        self.convs = nn.ModuleList()
        
        # First layer: input_dim → hidden_dim
        self.convs.append(
            GATv2Conv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True  # Concatenate attention heads
            )
        )
        
        # Middle layers: hidden_dim * num_heads → hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Final layer: hidden_dim * num_heads → output_dim
        if num_layers > 1:
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=output_dim,
                    heads=1,  # Single head for final layer
                    dropout=dropout,
                    concat=False
                )
            )
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads) 
            for _ in range(num_layers - 1)
        ])
        
        # Store config
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_heads': num_heads,
            'num_layers': num_layers
        }
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features [N, F] or [batch, N, F]
            edge_index: Edge indices [2, E]
            return_attention: Whether to return attention weights
        
        Returns:
            Node embeddings [N, output_dim] or [batch, N, output_dim]
        """
        # Handle batched input [batch, N, F]
        if x.dim() == 3:
            # Force requires_grad so checkpoint triggers backprop to GNN parameters
            if self.training and not x.requires_grad and x.is_floating_point():
                x.requires_grad_(True)
                
            batch_size, num_nodes, num_features = x.shape
            outputs = []
            attentions = []
            
            for b in range(batch_size):
                if return_attention:
                    out, attn = self._forward_single(x[b], edge_index, True)
                    attentions.append(attn)
                else:
                    if self.training:
                        out = cp.checkpoint(self._forward_single, x[b], edge_index, use_reentrant=False)
                    else:
                        out = self._forward_single(x[b], edge_index)
                outputs.append(out)
            
            result = torch.stack(outputs, dim=0)
            
            if return_attention:
                return result, attentions
            return result
        else:
            return self._forward_single(x, edge_index, return_attention)
    
    def _forward_single(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """Process single graph (no batch dimension)."""
        attention_weights = []
        
        for i, conv in enumerate(self.convs):
            if return_attention:
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                x = conv(x, edge_index)
            
            # Apply layer norm and activation for all but last layer
            if i < len(self.convs) - 1:
                x = self.layer_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention:
            return x, attention_weights
        return x
    
    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> list:
        """
        Get attention weights for interpretability.
        
        Returns list of attention weight tensors for each layer.
        Higher attention = stronger learned relationship.
        """
        _, attention_weights = self.forward(x, edge_index, return_attention=True)
        return attention_weights


class SimpleGNN(nn.Module):
    """
    Simplified GNN without torch_geometric dependency.
    Uses basic message passing with learned weights.
    
    Use this as fallback if torch_geometric installation fails.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = dropout
        
        # Attention weight matrix
        self.attention = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Simple message passing with attention.
        
        Args:
            x: Node features [N, F] or [batch, N, F]
            edge_index: Edge indices [2, E]
        """
        # Handle batched input
        if x.dim() == 3:
            batch_outputs = []
            for b in range(x.shape[0]):
                out = self._forward_single(x[b], edge_index)
                batch_outputs.append(out)
            return torch.stack(batch_outputs, dim=0)
        
        return self._forward_single(x, edge_index)
    
    def _forward_single(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Process single graph."""
        for i, layer in enumerate(self.layers):
            # Linear transform
            x = layer(x)
            
            if i < len(self.layers) - 1:
                # Message passing: aggregate neighbor features
                x = self._aggregate(x, edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def _aggregate(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor messages with mean pooling."""
        num_nodes = x.shape[0]
        
        # Initialize output with self
        out = x.clone()
        
        # Count neighbors for normalization
        neighbor_count = torch.zeros(num_nodes, device=x.device)
        
        # Sum neighbor features
        src, dst = edge_index
        for i in range(edge_index.shape[1]):
            out[dst[i]] = out[dst[i]] + x[src[i]]
            neighbor_count[dst[i]] += 1
        
        # Normalize by neighbor count
        neighbor_count = neighbor_count.clamp(min=1).unsqueeze(1)
        out = out / neighbor_count
        
        return out


# ===================
# Quick Test
# ===================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing GNN Components")
    print("=" * 50)
    
    # Create sample data
    num_nodes = 10  # 10 stocks
    num_features = 10  # 10 features per stock
    batch_size = 5
    
    # Random features
    x = torch.randn(batch_size, num_nodes, num_features)
    
    # Sample edge index (some connections)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 3],  # source
        [1, 2, 3, 4, 0, 0, 1, 2]   # target
    ], dtype=torch.long)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    
    # Test StockGNN (if torch_geometric available)
    if HAS_GEOMETRIC:
        print("\nTesting StockGNN (GATv2)...")
        model = StockGNN(input_dim=10, hidden_dim=32, output_dim=16)
        out = model(x, edge_index)
        print(f"  Output shape: {out.shape}")
        print(f"  Expected: [{batch_size}, {num_nodes}, 16]")
        
        # Test attention retrieval
        _, attns = model(x[0], edge_index, return_attention=True)
        print(f"  Attention weights from {len(attns)} layers")
    else:
        print("\nSkipping StockGNN (torch_geometric not installed)")
    
    # Test SimpleGNN (fallback)
    print("\nTesting SimpleGNN (fallback)...")
    simple_model = SimpleGNN(input_dim=10, hidden_dim=32, output_dim=16)
    out = simple_model(x, edge_index)
    print(f"  Output shape: {out.shape}")
    
    print("\n" + "=" * 50)
    print("GNN tests complete!")
    print("=" * 50)
