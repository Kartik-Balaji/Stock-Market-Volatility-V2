"""
CausalFolio - Model B (Minimal)
================================
Complete model combining GNN + TCN + FinBERT Sentiment.

Architecture:
    Node Features → GNN → Graph Embeddings
                          ↓
                         TCN → Volatility Prediction
                          ↓
                    + Sentiment → Final Risk Score

Usage:
    from models.model_minimal import CausalFolioMinimal
    
    model = CausalFolioMinimal(num_features=10, num_stocks=10)
    risk_scores = model(features, edge_index, sentiment_scores)
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
    # For direct file execution, add parent to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from gnn import StockGNN
    from tcn import StockTCN


class CausalFolioMinimal(nn.Module):
    """
    Complete Model B: GNN + TCN + Sentiment Fusion with DUAL OUTPUT.
    
    This model outputs BOTH:
    1. Volatility predictions (for confidence/range estimation)
    2. Return predictions (for direction/magnitude)
    
    Components:
    1. GNN: Learns cross-stock relationships
    2. TCN: Learns temporal patterns (shared backbone)
    3. Dual Heads: volatility_head + return_head
    4. Sentiment: Incorporated via late fusion on both heads
    
    Output: Dict with 'volatility' and 'returns' predictions
    """
    
    def __init__(
        self,
        num_features: int = 10,
        num_stocks: int = 10,
        gnn_hidden: int = 32,
        gnn_output: int = 16,
        tcn_hidden: int = 32,
        tcn_layers: int = 4,
        output_dim: int = 1,
        dropout: float = 0.3,
        use_sentiment: bool = True
    ):
        """
        Args:
            num_features: Number of input features per stock (10 classical features)
            num_stocks: Number of stocks in the portfolio
            gnn_hidden: Hidden dimension for GNN
            gnn_output: Output dimension from GNN (input to TCN)
            tcn_hidden: Hidden dimension for TCN
            tcn_layers: Number of TCN blocks
            output_dim: Output dimension per head (1 for each)
            dropout: Dropout probability
            use_sentiment: Whether to incorporate sentiment scores
        """
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
        
        # TCN for temporal patterns - outputs HIDDEN states (not final predictions)
        # We pass output_dim=tcn_hidden to get hidden states, then use separate heads
        self.tcn = StockTCN(
            input_dim=gnn_output,
            hidden_dim=tcn_hidden,
            output_dim=tcn_hidden,  # Output hidden states, not predictions
            num_layers=tcn_layers,
            dropout=dropout
        )
        
        # ========== DUAL OUTPUT HEADS ==========
        # Head 1: Volatility prediction (how much it will move)
        self.volatility_head = nn.Sequential(
            nn.Linear(tcn_hidden, tcn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_hidden // 2, output_dim),
            nn.Softplus()  # Volatility must be positive
        )
        
        # Head 2: DIRECTION CLASSIFICATION (3 classes: DOWN, SIDEWAYS, UP)
        # Changed from regression to classification to avoid zero-prediction collapse
        self.num_classes = 3
        self.direction_head = nn.Sequential(
            nn.Linear(tcn_hidden, tcn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_hidden // 2, self.num_classes)  # 3 classes
            # No softmax here - will use CrossEntropyLoss which includes it
        )
        
        # Sentiment fusion layer (applies to both heads)
        if use_sentiment:
            # Sentiment adjustment for volatility
            self.vol_sentiment_fusion = nn.Sequential(
                nn.Linear(output_dim + 1, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, output_dim),
                nn.Softplus()  # Keep volatility positive
            )
            # Sentiment adjustment for direction (3 classes)
            self.direction_sentiment_fusion = nn.Sequential(
                nn.Linear(self.num_classes + 1, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, self.num_classes)
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
            'classification_output': True,  # v3: direction is classification
            'num_classes': 3  # DOWN=0, SIDEWAYS=1, UP=2
        }
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        Forward pass through complete model with DUAL OUTPUT.
        
        Args:
            x: Node features [T, N, F] where:
               T = time steps (days)
               N = number of stocks
               F = number of features
            edge_index: Graph edges [2, E]
            sentiment: Optional sentiment scores [T, N] or [N]
            return_dict: If True, return dict. If False, return tuple for backward compat.
        
        Returns:
            Dict with keys:
                'volatility': [T, N, 1] - Predicted volatility (positive values)
                'direction': [T, N, 3] - Logits for DOWN/SIDEWAYS/UP classification
            Or tuple: (volatility, direction) if return_dict=False
        """
        T, N, F = x.shape
        
        # Step 1: GNN (spatial patterns)
        # Input: [T, N, F] → Output: [T, N, gnn_output]
        gnn_out = self.gnn(x, edge_index)
        
        # Step 2: TCN (temporal patterns) - outputs hidden states
        # Input: [T, N, gnn_output] → Output: [T, N, tcn_hidden]
        tcn_hidden_out = self.tcn(gnn_out)
        
        # Step 3: Dual output heads
        vol_out = self.volatility_head(tcn_hidden_out)  # [T, N, 1]
        dir_logits = self.direction_head(tcn_hidden_out)  # [T, N, 3] - logits for 3 classes
        
        # Step 4: Sentiment fusion (if enabled)
        if self.use_sentiment and sentiment is not None:
            # Ensure sentiment has right shape [T, N, 1]
            if sentiment.dim() == 1:
                # [N] → [T, N, 1] (broadcast same sentiment across time)
                sentiment = sentiment.unsqueeze(0).unsqueeze(-1).expand(T, -1, -1)
            elif sentiment.dim() == 2:
                # [T, N] → [T, N, 1]
                sentiment = sentiment.unsqueeze(-1)
            
            # Fuse sentiment into volatility
            vol_combined = torch.cat([vol_out, sentiment], dim=-1)
            vol_out = self.vol_sentiment_fusion(vol_combined)
            
            # Fuse sentiment into direction logits
            # Expand sentiment to match logits shape [T, N, 1] → broadcast
            dir_with_sent = torch.cat([dir_logits, sentiment], dim=-1)  # [T, N, 4]
            dir_logits = self.direction_sentiment_fusion(dir_with_sent)  # [T, N, 3]
        
        if return_dict:
            return {
                'volatility': vol_out,
                'direction': dir_logits  # Changed from 'returns' to 'direction'
            }
        else:
            # Backward compatibility: return volatility only (like old model)
            return vol_out
    
    def get_gnn_attention(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> List:
        """Get GNN attention weights for interpretability."""
        return self.gnn.get_attention_weights(x, edge_index)
    
    def get_receptive_field(self) -> int:
        """Get TCN receptive field (days of history used)."""
        return self.tcn.get_receptive_field()


class TrainingModule:
    """
    Training utilities for CausalFolioMinimal with DUAL OUTPUT.
    
    Handles:
    - Dual loss computation (volatility MSE + direction CrossEntropy)
    - Training loop
    - Validation
    - Checkpointing
    """
    
    def __init__(
        self,
        model: CausalFolioMinimal,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        vol_loss_weight: float = 1.0,
        direction_loss_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.vol_loss_weight = vol_loss_weight
        self.direction_loss_weight = direction_loss_weight
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Loss functions: MSE for volatility, CrossEntropy for direction (weights set dynamically in train)
        self.vol_criterion = nn.MSELoss()
        self.direction_criterion = None # Initialized in train() when weights are available
    
    def compute_targets(
        self, 
        features_dict: Dict[str, 'pd.DataFrame'],
        tickers: List[str],
        target_col: str = 'vol_5d'
    ) -> torch.Tensor:
        """
        Compute training targets from features.
        
        Args:
            features_dict: Dict of ticker → DataFrame with features
            tickers: List of tickers (defines order)
            target_col: Which feature to predict (e.g., 'vol_5d' for 5-day volatility)
        
        Returns:
            Targets tensor [T, N, 1]
        """
        # Get common dates
        common_dates = None
        for t in tickers:
            if t in features_dict:
                dates = set(features_dict[t].index)
                if common_dates is None:
                    common_dates = dates
                else:
                    common_dates = common_dates.intersection(dates)
        
        common_dates = sorted(common_dates)
        
        # Extract target for each stock
        targets = []
        for t in tickers:
            if t in features_dict:
                target = features_dict[t].loc[common_dates, target_col].values
                targets.append(target)
        
        # Stack: [N, T] → [T, N, 1]
        targets = np.stack(targets, axis=0).T  # [T, N]
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        
        return targets
    
    def train_epoch(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        vol_targets: torch.Tensor,
        direction_labels: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch with DUAL TARGETS.
        
        Args:
            features: Input features [T, N, F]
            edge_index: Graph edges [2, E]
            vol_targets: Volatility target values [T, N, 1]
            direction_labels: Direction class labels [T, N] (0=DOWN, 1=SIDEWAYS, 2=UP)
            sentiment: Optional sentiment [T, N] or [N]
            batch_size: Batch size (number of time steps per batch)
        
        Returns:
            Average combined loss for the epoch
        """
        self.model.train()
        
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        vol_targets = vol_targets.to(self.device)
        direction_labels = direction_labels.to(self.device)
        if sentiment is not None:
            sentiment = sentiment.to(self.device)
        
        T = features.shape[0]
        total_loss = 0.0
        total_vol_loss = 0.0
        total_dir_loss = 0.0
        num_batches = 0
        
        # Receptive field determines minimum lookback
        rf = self.model.get_receptive_field()
        
        # Process in batches
        for start in range(rf, T, batch_size):
            end = min(start + batch_size, T)
            
            # Get batch
            batch_features = features[start - rf:end]  # Include lookback
            batch_vol_targets = vol_targets[start:end]
            batch_dir_labels = direction_labels[start:end]  # [batch_T, N]
            
            if sentiment is not None:
                if sentiment.dim() == 1:
                    batch_sentiment = sentiment
                else:
                    batch_sentiment = sentiment[start:end]
            else:
                batch_sentiment = None
            
            # Forward pass - returns dict with 'volatility' and 'direction'
            self.optimizer.zero_grad()
            outputs = self.model(batch_features, edge_index, batch_sentiment)
            
            # Only use predictions from the non-lookback part
            vol_preds = outputs['volatility'][rf:]  # [batch_T, N, 1]
            dir_logits = outputs['direction'][rf:]  # [batch_T, N, 3]
            
            # Compute volatility loss (MSE)
            vol_loss = self.vol_criterion(vol_preds, batch_vol_targets)
            
            # Compute direction loss (CrossEntropy)
            # CrossEntropyLoss expects: logits [batch, classes], labels [batch]
            # We have: [batch_T, N, 3] and [batch_T, N]
            # Reshape: [batch_T * N, 3] and [batch_T * N]
            batch_T, N, C = dir_logits.shape
            dir_logits_flat = dir_logits.view(-1, C)  # [batch_T * N, 3]
            dir_labels_flat = batch_dir_labels.view(-1).long()  # [batch_T * N]
            dir_loss = self.direction_criterion(dir_logits_flat, dir_labels_flat)
            
            # Combined weighted loss
            loss = self.vol_loss_weight * vol_loss + self.direction_loss_weight * dir_loss
            
            # Extract scalars to avoid holding the computation graph
            loss_val = loss.item()
            vol_loss_val = vol_loss.item()
            dir_loss_val = dir_loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss_val
            total_vol_loss += vol_loss_val
            total_dir_loss += dir_loss_val
            num_batches += 1
            
            # Explicitly free VRAM
            del outputs, vol_preds, dir_logits, dir_logits_flat, dir_labels_flat, loss, vol_loss, dir_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(num_batches, 1)
        # Store component losses for logging
        self._last_vol_loss = total_vol_loss / max(num_batches, 1)
        self._last_dir_loss = total_dir_loss / max(num_batches, 1)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        vol_targets: torch.Tensor,
        direction_labels: torch.Tensor,
        sentiment: Optional[torch.Tensor] = None,
        warmup: int = 0
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Validate the model with DUAL TARGETS (classification).
        
        Args:
            warmup: Number of leading timesteps whose predictions are ignored
                    (receptive field rows where the TCN lacks full context).
        
        Returns:
            Tuple of (combined_loss, predictions_dict)
        """
        self.model.eval()
        
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        vol_targets = vol_targets.to(self.device)
        direction_labels = direction_labels.to(self.device)
        if sentiment is not None:
            sentiment = sentiment.to(self.device)
        
        outputs = self.model(features, edge_index, sentiment)
        
        # Volatility loss (MSE) - skip warmup rows from predictions only;
        # targets are already aligned by the caller (train passes targets[train_size:]).
        vol_preds = outputs['volatility'][warmup:]
        vol_loss = self.vol_criterion(vol_preds, vol_targets)
        
        # Direction loss (CrossEntropy) - skip warmup rows from predictions only
        dir_logits = outputs['direction'][warmup:]  # [T, N, 3]
        T, N, C = dir_logits.shape
        dir_logits_flat = dir_logits.view(-1, C)
        dir_labels_flat = direction_labels.view(-1).long()
        
        if self.direction_criterion is None:
            self.direction_criterion = nn.CrossEntropyLoss()
            
        dir_loss = self.direction_criterion(dir_logits_flat, dir_labels_flat)
        
        combined_loss = self.vol_loss_weight * vol_loss + self.direction_loss_weight * dir_loss
        
        return combined_loss.item(), {
            'volatility': outputs['volatility'].cpu(),
            'direction': outputs['direction'].cpu()  # Changed from 'returns'
        }
    
    def train(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        vol_targets: torch.Tensor,
        direction_labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        sentiment: Optional[torch.Tensor] = None,
        epochs: int = 100,
        val_split: float = 0.2,
        early_stopping: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with DUAL TARGETS (Classification).
        
        Args:
            features: Input features [T, N, F]
            edge_index: Graph edges [2, E]
            vol_targets: Volatility target values [T, N, 1]
            direction_labels: Direction class labels [T, N] (0=DOWN, 1=SIDEWAYS, 2=UP)
            class_weights: Optional class weights for CrossEntropyLoss [3]
            sentiment: Optional sentiment scores
            epochs: Number of epochs
            val_split: Validation split ratio
            early_stopping: Patience for early stopping
            verbose: Print progress
        
        Returns:
            Dict with training history
        """
        T = features.shape[0]
        val_size = int(T * val_split)
        train_size = T - val_size
        rf = self.model.get_receptive_field()
        
        # Setup weighted loss if weights provided
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.direction_criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif self.direction_criterion is None:
            self.direction_criterion = nn.CrossEntropyLoss()
        
        # Split data (temporal split - no shuffle)
        train_features = features[:train_size]
        train_vol_targets = vol_targets[:train_size]
        train_dir_labels = direction_labels[:train_size]
        # Include rf-lookback context before the validation window so the TCN
        # has full causal context at the start of the validation set.
        val_features = features[max(0, train_size - rf):]
        val_vol_targets = vol_targets[train_size:]
        val_dir_labels = direction_labels[train_size:]
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
            'train_vol_loss': [],
            'train_dir_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        best_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(
                train_features, edge_index, train_vol_targets, train_dir_labels, train_sentiment
            )
            
            # Validate (with warmup removal so val loss is fair)
            val_loss, _ = self.validate(
                val_features, edge_index, val_vol_targets, val_dir_labels, val_sentiment,
                warmup=val_warmup
            )
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Check for improvement
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Verbose output
            if verbose and (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, "
                      f"val_loss={val_loss:.6f}, lr={lr:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        if verbose:
            print(f"\nBest model at epoch {history['best_epoch']+1} "
                  f"with val_loss={history['best_val_loss']:.6f}")
        
        return history
    
    def save_checkpoint(self, path: str, extra_config: Optional[Dict] = None):
        """Save model checkpoint."""
        config = dict(self.model.config)
        if extra_config:
            config.update(extra_config)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': config
        }, path)
        print(f"✓ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"✓ Checkpoint loaded from {path}")


# ===================
# Quick Test
# ===================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing CausalFolioMinimal")
    print("=" * 60)
    
    # Create sample data
    T = 138  # days
    N = 10   # stocks
    F = 10   # features
    
    features = torch.randn(T, N, F)
    edge_index = torch.randint(0, N, (2, 30), dtype=torch.long)
    sentiment = torch.randn(N)  # One sentiment per stock
    
    print(f"\nInput shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Sentiment: {sentiment.shape}")
    
    # Create model
    model = CausalFolioMinimal(
        num_features=F,
        num_stocks=N,
        gnn_hidden=32,
        gnn_output=16,
        tcn_hidden=32,
        tcn_layers=4,
        use_sentiment=True
    )
    
    print(f"\nModel config: {model.config}")
    print(f"Receptive field: {model.get_receptive_field()} days")
    
    # Forward pass
    output = model(features, edge_index, sentiment)
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [{T}, {N}, 1]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("CausalFolioMinimal tests complete! ✅")
    print("=" * 60)
