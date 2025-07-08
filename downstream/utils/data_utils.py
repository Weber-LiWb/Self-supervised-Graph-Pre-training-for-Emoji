#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
import pickle
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_checkpoint_embeddings(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load pre-computed embeddings from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with embedding tensors
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load embeddings from {checkpoint_path}: {e}")
        return {}


def create_synthetic_engagement_data(
    num_posts: int = 1000,
    embedding_dim: int = 128,
    engagement_range: Tuple[float, float] = (0.0, 1.0),
    noise_level: float = 0.1,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic engagement data for testing.
    
    Args:
        num_posts: Number of posts to generate
        embedding_dim: Dimension of embeddings
        engagement_range: Range of engagement scores
        noise_level: Amount of noise to add
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (post_embeddings, engagement_scores)
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Generate random embeddings
    post_embeddings = torch.randn(num_posts, embedding_dim)
    post_embeddings = torch.nn.functional.normalize(post_embeddings, p=2, dim=1)
    
    # Create engagement scores based on some properties of embeddings
    # Use a simple linear combination of features with noise
    weights = torch.randn(embedding_dim) * 0.1
    base_scores = torch.mm(post_embeddings, weights.unsqueeze(1)).squeeze()
    
    # Normalize to engagement range
    min_score, max_score = base_scores.min(), base_scores.max()
    normalized_scores = (base_scores - min_score) / (max_score - min_score)
    engagement_scores = normalized_scores * (engagement_range[1] - engagement_range[0]) + engagement_range[0]
    
    # Add noise
    noise = torch.randn_like(engagement_scores) * noise_level
    engagement_scores = torch.clamp(engagement_scores + noise, engagement_range[0], engagement_range[1])
    
    logger.info(f"Generated {num_posts} synthetic posts with engagement scores")
    logger.info(f"Engagement score range: {engagement_scores.min():.3f} - {engagement_scores.max():.3f}")
    
    return post_embeddings, engagement_scores


def save_embeddings(embeddings: Dict[str, torch.Tensor], save_path: str):
    """Save embeddings to file."""
    torch.save(embeddings, save_path)
    logger.info(f"Embeddings saved to {save_path}")


def load_embeddings(load_path: str) -> Dict[str, torch.Tensor]:
    """Load embeddings from file."""
    embeddings = torch.load(load_path, map_location='cpu')
    logger.info(f"Embeddings loaded from {load_path}")
    return embeddings