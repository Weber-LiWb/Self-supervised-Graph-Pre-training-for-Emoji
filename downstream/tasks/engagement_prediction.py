#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import logging

from .base_downstream_task import BaseDownstreamTask

logger = logging.getLogger(__name__)


class EngagementPredictionHead(nn.Module):
    """
    Neural network head for predicting engagement scores from post embeddings.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final prediction layer (outputs single engagement score)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1]
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: Post embeddings [batch_size, embedding_dim]
            
        Returns:
            Engagement scores [batch_size, 1]
        """
        return self.mlp(embeddings)


class EngagementPredictionTask(BaseDownstreamTask):
    """
    Downstream task for predicting engagement scores based on post content.
    Uses pre-trained post embeddings and learns a prediction head.
    """
    
    def __init__(
        self, 
        checkpoint_path: str,
        device: str = 'cuda:0',
        dgl_graphs_file: Optional[str] = None,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3
    ):
        super().__init__(checkpoint_path, device, dgl_graphs_file)
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Will be set up after we know the embedding dimension
        self.task_head = None
        self.optimizer = None
        
    def setup_task_head(self, embedding_dim: int):
        """Setup the engagement prediction head."""
        self.task_head = EngagementPredictionHead(
            input_dim=embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.task_head.parameters(), 
            lr=self.learning_rate
        )
        
        logger.info(f"Engagement prediction head initialized with input_dim={embedding_dim}")
    
    def prepare_training_data(
        self, 
        post_embeddings: torch.Tensor,
        engagement_scores: torch.Tensor,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training and validation data.
        
        Args:
            post_embeddings: Post embeddings [num_posts, embedding_dim]
            engagement_scores: Ground truth engagement scores [num_posts]
            test_size: Fraction of data for validation
            random_state: Random seed for splitting
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # Ensure scores are normalized to [0, 1]
        if engagement_scores.max() > 1.0:
            engagement_scores = engagement_scores / engagement_scores.max()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            post_embeddings.cpu().numpy(),
            engagement_scores.cpu().numpy(),
            test_size=test_size,
            random_state=random_state
        )
        
        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
    
    def train(
        self,
        post_embeddings: torch.Tensor,
        engagement_scores: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the engagement prediction model.
        
        Args:
            post_embeddings: Post embeddings [num_posts, embedding_dim]
            engagement_scores: Ground truth engagement scores [num_posts]
            num_epochs: Number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        # Setup task head if not already done
        if self.task_head is None:
            self.setup_task_head(post_embeddings.shape[1])
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_training_data(
            post_embeddings, engagement_scores
        )
        
        # Move to device
        X_train, X_val = X_train.to(self.device), X_val.to(self.device)
        y_train, y_val = y_train.to(self.device), y_val.to(self.device)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            # Training phase
            self.task_head.train()
            train_losses = []
            
            # Create batches
            num_batches = (len(X_train) + batch_size - 1) // batch_size
            indices = torch.randperm(len(X_train))
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                self.optimizer.zero_grad()
                predictions = self.task_head(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            self.task_head.eval()
            with torch.no_grad():
                val_predictions = self.task_head(X_val).squeeze()
                val_loss = criterion(val_predictions, y_val).item()
                
                # Additional metrics
                val_mae = F.l1_loss(val_predictions, y_val).item()
                val_r2 = r2_score(y_val.cpu().numpy(), val_predictions.cpu().numpy())
            
            # Record history
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = self.task_head.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val MAE: {val_mae:.4f}, "
                    f"Val R²: {val_r2:.4f}"
                )
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.task_head.load_state_dict(best_model_state)
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        return history
    
    def predict(self, post_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict engagement scores for posts.
        
        Args:
            post_embeddings: Post embeddings [num_posts, embedding_dim]
            
        Returns:
            Predicted engagement scores [num_posts]
        """
        if self.task_head is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.task_head.eval()
        post_embeddings = post_embeddings.to(self.device)
        
        with torch.no_grad():
            predictions = self.task_head(post_embeddings).squeeze()
        
        return predictions.cpu()
    
    def evaluate(
        self, 
        post_embeddings: torch.Tensor,
        engagement_scores: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            post_embeddings: Post embeddings [num_posts, embedding_dim]
            engagement_scores: Ground truth engagement scores [num_posts]
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(post_embeddings)
        
        # Ensure ground truth is normalized
        if engagement_scores.max() > 1.0:
            engagement_scores = engagement_scores / engagement_scores.max()
        
        # Calculate metrics
        mse = mean_squared_error(engagement_scores.cpu(), predictions.cpu())
        mae = mean_absolute_error(engagement_scores.cpu(), predictions.cpu())
        r2 = r2_score(engagement_scores.cpu(), predictions.cpu())
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def generate_post_embeddings(self, batch_size: int = 32) -> torch.Tensor:
        """Generate embeddings for all posts in the graph."""
        # Post type corresponds to the last element in etype
        post_etype = ('emoji', 'ein', 'post')
        post_metapath = ['ein', 'hase']
        
        return self.generate_embeddings(post_etype, post_metapath, batch_size)
    
    def predict_from_content(self, post_content: str) -> float:
        """
        Predict engagement for a single post content.
        This is a simplified version - in practice, you'd need to
        process the content through the graph construction pipeline.
        
        Args:
            post_content: Text content of the post
            
        Returns:
            Predicted engagement score
        """
        # This is a placeholder - actual implementation would require
        # converting post_content to graph representation first
        logger.warning("predict_from_content requires graph construction pipeline")
        return 0.5  # Placeholder