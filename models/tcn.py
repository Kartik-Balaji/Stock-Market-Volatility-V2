"""
CausalFolio - Temporal Convolutional Network (TCN)
===================================================
Uses dilated causal convolutions to learn temporal patterns.

Key concepts:
- Causal: Only uses past data (no future leakage)
- Dilated: Exponentially expanding receptive field (1, 2, 4, 8...)
- Residual: Skip connections for stable training

Usage:
    from models.tcn import TemporalConvNet, StockTCN
    
    model = StockTCN(input_dim=16, hidden_dim=32, output_dim=1)
    predictions = model(gnn_embeddings)  # [T, N, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that only uses past information.
    
    For a kernel of size K and dilation D:
    - Pads (K-1)*D on the left
    - No future information leaks into predictions
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time]
        Returns:
            [batch, channels, time] (same length)
        """
        # Left-pad to maintain causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Single TCN block with:
    - Two causal convolutions
    - ReLU activations
    - Dropout
    - Residual connection
    
    Architecture:
        Input → Conv → ReLU → Dropout → Conv → ReLU → Dropout → + → Output
          └────────────────── Residual ──────────────────────────┘
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (if dimensions don't match)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time]
        Returns:
            [batch, channels, time]
        """
        # First conv block
        out = self.conv1(x)
        out = out.transpose(1, 2)  # [batch, time, channels] for LayerNorm
        out = self.norm1(out)
        out = out.transpose(1, 2)  # [batch, channels, time]
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual is not None:
            x = self.residual(x)
        
        return F.relu(out + x)


class TemporalConvNet(nn.Module):
    """
    Full Temporal Convolutional Network.
    
    Stacks multiple TCN blocks with exponentially increasing dilation:
    - Block 1: dilation = 1 (sees 1 timestep)
    - Block 2: dilation = 2 (sees 2 timesteps)
    - Block 3: dilation = 4 (sees 4 timesteps)
    - ...
    
    With 4 blocks and kernel_size=3: receptive field = 2^4 * 2 = 32 days
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            
            self.blocks.append(
                TCNBlock(in_ch, hidden_dim, kernel_size, dilation, dropout)
            )
        
        # Store config
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'kernel_size': kernel_size,
            'receptive_field': (kernel_size - 1) * sum(2**i for i in range(num_layers)) + 1
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, time, features] or [batch, features, time]
        Returns:
            [batch, hidden_dim, time]
        """
        # Convert to [batch, features, time] if needed
        if x.dim() == 3:
            input_dim = self.config['input_dim']
            if x.shape[1] == input_dim:
                pass
            elif x.shape[2] == input_dim:
                x = x.transpose(1, 2)
            else:
                raise ValueError(
                    f"Expected feature dim {input_dim} in axis 1 or 2, got shape {tuple(x.shape)}"
                )
        
        for block in self.blocks:
            x = block(x)
        
        return x


class StockTCN(nn.Module):
    """
    TCN for stock prediction.
    
    Takes GNN embeddings [T, N, H] and outputs predictions [T, N, out_dim].
    Processes each stock independently through the TCN.
    
    Architecture:
        GNN Embeddings [T, N, H] 
        → Reshape to [N, H, T]  (each stock is a "batch")
        → TCN → [N, hidden, T]
        → Output projection → [N, out_dim, T]
        → Reshape to [T, N, out_dim]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: Dimension of GNN embeddings (16 in our case)
            hidden_dim: TCN hidden dimension
            output_dim: Prediction dimension (1 for volatility)
            num_layers: Number of TCN blocks
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.config = {
            **self.tcn.config,
            'output_dim': output_dim
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: GNN embeddings [T, N, H] where:
               T = time steps (138 days)
               N = number of stocks (10)
               H = embedding dimension (16)
        
        Returns:
            Predictions [T, N, output_dim]
        """
        T, N, H = x.shape
        
        # Reshape: [T, N, H] → [N, H, T] (each stock is a batch sample)
        x = x.permute(1, 2, 0)  # [N, H, T]
        
        # Apply TCN
        x = self.tcn(x)  # [N, hidden, T]
        
        # Reshape for output projection: [N, hidden, T] → [N, T, hidden]
        x = x.permute(0, 2, 1)  # [N, T, hidden]
        
        # Output projection
        x = self.output_proj(x)  # [N, T, output_dim]
        
        # Reshape back: [N, T, output_dim] → [T, N, output_dim]
        x = x.permute(1, 0, 2)  # [T, N, output_dim]
        
        return x
    
    def get_receptive_field(self) -> int:
        """Return how many past days the model can see."""
        return self.config['receptive_field']


# ===================
# Quick Test
# ===================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing TCN Components")
    print("=" * 50)
    
    # Simulate GNN output
    T = 138  # 138 days
    N = 10   # 10 stocks
    H = 16   # 16-dim embeddings from GNN
    
    gnn_output = torch.randn(T, N, H)
    print(f"\nGNN output (input to TCN): {gnn_output.shape}")
    
    # Test StockTCN
    print("\nTesting StockTCN...")
    model = StockTCN(input_dim=16, hidden_dim=32, output_dim=1, num_layers=4)
    
    print(f"Model config: {model.config}")
    print(f"Receptive field: {model.get_receptive_field()} days")
    
    # Forward pass
    predictions = model(gnn_output)
    print(f"\nInput:  {gnn_output.shape} [T, N, H]")
    print(f"Output: {predictions.shape} [T, N, 1]")
    
    # Check output range
    print(f"\nOutput stats:")
    print(f"  Mean: {predictions.mean().item():.4f}")
    print(f"  Std:  {predictions.std().item():.4f}")
    
    print("\n" + "=" * 50)
    print("TCN tests complete! ✅")
    print("=" * 50)
