"""
CausalFolio - Model B (Minimal)
================================
Complete model combining GNN + TCN + FinBERT Sentiment.

Architecture:
    Node Features → GNN → Graph Embeddings
                          ↓
                         TCN → Volatility Prediction (Head 1)
                          ↓
                         TCN → Return Prediction (Head 2)
                          ↓
                    + Sentiment → Final Forecast & Risk Score

Usage:
    from models.model_minimal import CausalFolioMinimal
    
    model = CausalFolioMinimal(num_features=10, num_stocks=10)
    predictions = model(features, edge_index, sentiment_scores)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
import sys
import os

# Handle both relative and absolute imports
try:
    from .gnn import StockGNN
    from .tcn import StockTCN
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from gnn import StockGNN
    from tcn import StockTCN


class CausalFolioMinimal(nn.Module):
    """
    Complete Model B: GNN + TCN + Sentiment Fusion with DUAL CONTINUOUS OUTPUT.
    
    Outputs:
    1. Volatility predictions (realized volatility estimation)
    2. Forward Return predictions (direction + magnitude forecast)
    
    Components:
    1. GNN: Learns cross-stock spatial relationships
    2. TCN: Learns temporal sequence dynamics (shared backbone)
    3. Dual Heads: volatility_head + return_head
    4. Sentiment: Late fusion on both heads
    """
    
    def __init__(
        self,
        num_features: int = 10,
        num_stocks: int = 10,
        gnn_hidden: int = 128,
        gnn_output: int = 64,
        tcn_hidden: int = 256,
        tcn_layers: int = 5,
        output_dim: int = 1,
        dropout: float = 0.3,
        use_sentiment: bool = True
    ):
        super().__init__()
        
        self.num_stocks = num_stocks
        self.use_sentiment = use_sentiment
        self.tcn_hidden = tcn_hidden
        
        # GNN for spatial (cross-stock) patterns
        self.gnn = StockGNN(
            input_dim=num_features,
            hidden_dim=gnn_hidden,
            output_dim=gnn_output,
            num_heads=2,
            num_layers=2,
            dropout=dropout
        )
        
        # TCN for temporal patterns - outputs hidden states
        self.tcn = StockTCN(
            input_dim=gnn_output,
            hidden_dim=tcn_hidden,
            output_dim=tcn_hidden,
            num_layers=tcn_layers,
            dropout=dropout
        )
        
        # ========== DUAL OUTPUT HEADS ==========
        # Head 1: Volatility prediction (positive values)
        self.volatility_head = nn.Sequential(
            nn.Linear(tcn_hidden, tcn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_hidden // 2, output_dim),
            nn.Softplus()
        )
        
        # Head 2: Continuous Forward Return prediction (can be +/-)
        self.return_head = nn.Sequential(
            nn.Linear(tcn_hidden, tcn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_hidden // 2, output_dim)
        )
        
        # Sentiment fusion layers
        if use_sentiment:
            self.vol_sentiment_fusion = nn.Sequential(
                nn.Linear(output_dim + 1, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, output_dim),
                nn.Softplus()
            )
            self.ret_sentiment_fusion = nn.Sequential(
                nn.Linear(output_dim + 1, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, output_dim)
            )
        
        # Store config
        self.config = {
            'num_features': num_features,
            'num_stocks': num_stocks,
            'gnn_hidden': gnn_hidden,
            'gnn_output': gnn_output,
            'tcn_hidden': tcn_hidden,
            'tcn_layers': tcn_layers,
            'dropout': dropout,
            'output_dim': output_dim,
            'use_sentiment': use_sentiment,
            'classification_output': False
        }
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        Forward pass through complete model.
        
        Args:
            x: Node features [T, N, F]
            edge_index: Graph edges [2, E]
            sentiment: Optional sentiment scores [T, N] or [N]
            return_dict: If True, return dict with 'volatility' and 'returns'
        
        Returns:
            Dict with keys:
                'volatility': [T, N, 1]
                'returns': [T, N, 1]
        """
        T, N, F = x.shape
        
        # Step 1: GNN (spatial patterns)
        gnn_out = self.gnn(x, edge_index)
        
        # Step 2: TCN (temporal patterns)
        tcn_hidden_out = self.tcn(gnn_out)
        
        # Step 3: Dual output heads
        vol_out = self.volatility_head(tcn_hidden_out)
        ret_out = self.return_head(tcn_hidden_out)
        
        # Step 4: Sentiment fusion
        if self.use_sentiment and sentiment is not None:
            if sentiment.dim() == 1:
                sentiment = sentiment.unsqueeze(0).unsqueeze(-1).expand(T, -1, -1)
            elif sentiment.dim() == 2:
                sentiment = sentiment.unsqueeze(-1)
            
            vol_combined = torch.cat([vol_out, sentiment], dim=-1)
            vol_out = self.vol_sentiment_fusion(vol_combined)
            
            ret_combined = torch.cat([ret_out, sentiment], dim=-1)
            ret_out = self.ret_sentiment_fusion(ret_combined)
        
        if return_dict:
            return {
                'volatility': vol_out,
                'returns': ret_out
            }
        else:
            return vol_out, ret_out
    
    def get_receptive_field(self) -> int:
        """Get TCN receptive field in days."""
        return self.tcn.get_receptive_field()


class TrainingModule:
    """
    Training utilities for CausalFolioMinimal with Continuous Return & Volatility Heads.
    """
    
    def __init__(
        self,
        model: CausalFolioMinimal,
        learning_rate: float = 0.0005,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        vol_loss_weight: float = 1.0,
        return_loss_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.vol_loss_weight = vol_loss_weight
        self.return_loss_weight = return_loss_weight
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.vol_criterion = nn.MSELoss()
        self.return_criterion = nn.MSELoss()
    
    def train_epoch(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        vol_targets: torch.Tensor,
        ret_targets: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
        batch_size: int = 32
    ) -> Tuple[float, float, float, float]:
        """
        Train for one epoch with dual continuous targets.
        """
        self.model.train()
        
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        vol_targets = vol_targets.to(self.device)
        ret_targets = ret_targets.to(self.device)
        if sentiment is not None:
            sentiment = sentiment.to(self.device)
        
        T = features.shape[0]
        total_loss = 0.0
        total_vol_loss = 0.0
        total_ret_loss = 0.0
        correct_signs = 0
        total_points = 0
        num_batches = 0
        
        rf = self.model.get_receptive_field()
        
        for start in range(rf, T, batch_size):
            end = min(start + batch_size, T)
            
            batch_features = features[start - rf:end]
            batch_vol_targets = vol_targets[start:end]
            batch_ret_targets = ret_targets[start:end]
            
            if sentiment is not None:
                batch_sentiment = sentiment if sentiment.dim() == 1 else sentiment[start:end]
            else:
                batch_sentiment = None
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_features, edge_index, batch_sentiment)
            
            vol_preds = outputs['volatility'][rf:]
            ret_preds = outputs['returns'][rf:]
            
            vol_loss = self.vol_criterion(vol_preds, batch_vol_targets)
            ret_loss = self.return_criterion(ret_preds, batch_ret_targets)
            
            loss = self.vol_loss_weight * vol_loss + self.return_loss_weight * ret_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Directional sign accuracy on training batch
            with torch.no_grad():
                sign_match = ((ret_preds > 0) == (batch_ret_targets > 0)).sum().item()
                correct_signs += sign_match
                total_points += batch_ret_targets.numel()
            
            total_loss += loss.item()
            total_vol_loss += vol_loss.item()
            total_ret_loss += ret_loss.item()
            num_batches += 1
            
            del outputs, vol_preds, ret_preds, loss, vol_loss, ret_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_vol = total_vol_loss / max(num_batches, 1)
        avg_ret = total_ret_loss / max(num_batches, 1)
        sign_acc = (correct_signs / max(total_points, 1)) * 100.0
        
        return avg_loss, avg_vol, avg_ret, sign_acc
    
    @torch.no_grad()
    def validate(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        vol_targets: torch.Tensor,
        ret_targets: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
        warmup: int = 0
    ) -> Tuple[float, float, float, float, Dict[str, torch.Tensor]]:
        """
        Validate model on out-of-sample data.
        """
        self.model.eval()
        
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        vol_targets = vol_targets.to(self.device)
        ret_targets = ret_targets.to(self.device)
        if sentiment is not None:
            sentiment = sentiment.to(self.device)
        
        outputs = self.model(features, edge_index, sentiment)
        
        vol_preds = outputs['volatility'][warmup:]
        ret_preds = outputs['returns'][warmup:]
        
        vol_loss = self.vol_criterion(vol_preds, vol_targets)
        ret_loss = self.return_criterion(ret_preds, ret_targets)
        
        combined_loss = self.vol_loss_weight * vol_loss + self.return_loss_weight * ret_loss
        
        sign_match = ((ret_preds > 0) == (ret_targets > 0)).float().mean().item() * 100.0
        
        return combined_loss.item(), vol_loss.item(), ret_loss.item(), sign_match, {
            'volatility': outputs['volatility'].cpu(),
            'returns': outputs['returns'].cpu()
        }
    
    def train(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        vol_targets: torch.Tensor,
        ret_targets: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
        epochs: int = 100,
        val_split: float = 0.15,
        early_stopping: int = 25,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with temporal train/validation split.
        """
        T = features.shape[0]
        val_size = int(T * val_split)
        train_size = T - val_size
        rf = self.model.get_receptive_field()
        
        train_features = features[:train_size]
        train_vol_targets = vol_targets[:train_size]
        train_ret_targets = ret_targets[:train_size]
        
        val_features = features[max(0, train_size - rf):]
        val_vol_targets = vol_targets[train_size:]
        val_ret_targets = ret_targets[train_size:]
        val_warmup = min(rf, train_size)
        
        if sentiment is not None:
            if sentiment.dim() == 1:
                train_sentiment = sentiment
                val_sentiment = sentiment
            else:
                train_sentiment = sentiment[:train_size]
                val_sentiment = sentiment[max(0, train_size - rf):]
        else:
            train_sentiment = None
            val_sentiment = None
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_sign_acc': [],
            'val_sign_acc': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'best_val_sign_acc': 0.0
        }
        
        best_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_vol, train_ret, train_sign_acc = self.train_epoch(
                train_features, edge_index, train_vol_targets, train_ret_targets, train_sentiment
            )
            
            val_loss, val_vol, val_ret, val_sign_acc, _ = self.validate(
                val_features, edge_index, val_vol_targets, val_ret_targets, val_sentiment,
                warmup=val_warmup
            )
            
            self.scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_sign_acc'].append(train_sign_acc)
            history['val_sign_acc'].append(val_sign_acc)
            
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_val_sign_acc'] = val_sign_acc
                history['best_epoch'] = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.6f} (sign_acc={train_sign_acc:.1f}%), "
                      f"val_loss={val_loss:.6f} (val_sign_acc={val_sign_acc:.1f}%), lr={lr:.6f}")
            
            if patience_counter >= early_stopping:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        if verbose:
            print(f"\n✓ Best model restored from epoch {history['best_epoch']+1} "
                  f"with val_loss={history['best_val_loss']:.6f}, val_sign_acc={history['best_val_sign_acc']:.1f}%")
        
        return history
    
    def save_checkpoint(self, path: str, extra_config: Optional[Dict] = None):
        """Save model checkpoint with all normalization metadata."""
        config = dict(self.model.config)
        if extra_config:
            config.update(extra_config)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': config
        }, path)
        print(f"✓ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"✓ Checkpoint loaded from {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CausalFolioMinimal (Regression Head)")
    print("=" * 60)
    
    T, N, F = 138, 10, 10
    features = torch.randn(T, N, F)
    edge_index = torch.randint(0, N, (2, 30), dtype=torch.long)
    sentiment = torch.randn(N)
    
    model = CausalFolioMinimal(
        num_features=F,
        num_stocks=N,
        gnn_hidden=128,
        gnn_output=64,
        tcn_hidden=256,
        tcn_layers=5,
        use_sentiment=True
    )
    
    outputs = model(features, edge_index, sentiment)
    print(f"Volatility output shape: {outputs['volatility'].shape}")
    print(f"Returns output shape: {outputs['returns'].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Model test passed!")
