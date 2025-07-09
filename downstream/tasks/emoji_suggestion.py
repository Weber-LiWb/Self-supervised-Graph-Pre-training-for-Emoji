#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score, average_precision_score
import logging

from .base_downstream_task import BaseDownstreamTask

logger = logging.getLogger(__name__)


class EmojiSuggestionTask(BaseDownstreamTask):
    """
    Downstream task for suggesting emojis based on post content.
    Uses similarity between post and emoji embeddings in the learned space.
    """
    
    def __init__(
        self, 
        checkpoint_path: str,
        device: str = 'cuda:0',
        dgl_graphs_file: Optional[str] = None,
        similarity_metric: str = 'cosine',
        temperature: float = 1.0
    ):
        super().__init__(checkpoint_path, device, dgl_graphs_file)
        
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        
        # Embedding matrices (will be populated during setup)
        self.post_embeddings = None
        self.emoji_embeddings = None
        self.emoji_vocab = None  # Mapping from emoji index to emoji string
        
    def setup_task_head(self, **kwargs):
        """
        Emoji suggestion doesn't need a trainable head - uses similarity directly.
        """
        logger.info("Emoji suggestion uses similarity-based ranking - no trainable head needed")
    
    def generate_all_embeddings(self, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for both posts and emojis.
        
        Args:
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with 'post' and 'emoji' embeddings
        """
        embeddings = {}
        
        # Generate post embeddings
        logger.info("Generating post embeddings...")
        post_etype = ('emoji', 'ein', 'post')
        post_metapath = ['ein', 'hase']
        embeddings['post'] = self.generate_embeddings(post_etype, post_metapath, batch_size)
        
        # Generate emoji embeddings
        logger.info("Generating emoji embeddings...")
        emoji_etype = ('post', 'hase', 'emoji')  
        emoji_metapath = ['hase', 'ein']
        embeddings['emoji'] = self.generate_embeddings(emoji_etype, emoji_metapath, batch_size)
        
        logger.info(f"Generated {embeddings['post'].shape[0]} post embeddings and "
                   f"{embeddings['emoji'].shape[0]} emoji embeddings")
        
        return embeddings
    
    def setup_embeddings(
        self, 
        post_embeddings: torch.Tensor,
        emoji_embeddings: torch.Tensor,
        emoji_vocab: Optional[Dict[int, str]] = None
    ):
        """
        Setup the embedding matrices for similarity computation.
        
        Args:
            post_embeddings: Post embeddings [num_posts, embedding_dim]
            emoji_embeddings: Emoji embeddings [num_emojis, embedding_dim]
            emoji_vocab: Mapping from emoji index to emoji string
        """
        self.post_embeddings = F.normalize(post_embeddings, p=2, dim=1)
        self.emoji_embeddings = F.normalize(emoji_embeddings, p=2, dim=1)
        self.emoji_vocab = emoji_vocab or {i: f"emoji_{i}" for i in range(len(emoji_embeddings))}
        
        logger.info(f"Setup embeddings: {len(self.post_embeddings)} posts, "
                   f"{len(self.emoji_embeddings)} emojis")
    
    def compute_similarity(
        self, 
        post_emb: torch.Tensor, 
        emoji_emb: torch.Tensor,
        metric: str = None
    ) -> torch.Tensor:
        """
        Compute similarity between post and emoji embeddings.
        
        Args:
            post_emb: Post embeddings [num_posts, dim] or [dim]
            emoji_emb: Emoji embeddings [num_emojis, dim]
            metric: Similarity metric ('cosine', 'dot', 'euclidean')
            
        Returns:
            Similarity matrix [num_posts, num_emojis] or [num_emojis]
        """
        if metric is None:
            metric = self.similarity_metric
        
        if post_emb.dim() == 1:
            post_emb = post_emb.unsqueeze(0)
        
        if metric == 'cosine':
            # Assuming embeddings are already normalized
            similarities = torch.mm(post_emb, emoji_emb.t())
        elif metric == 'dot':
            similarities = torch.mm(post_emb, emoji_emb.t())
        elif metric == 'euclidean':
            # Negative distance (higher is more similar)
            distances = torch.cdist(post_emb, emoji_emb, p=2)
            similarities = -distances
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        # Apply temperature scaling
        similarities = similarities / self.temperature
        
        return similarities.squeeze() if similarities.shape[0] == 1 else similarities
    
    def suggest_emojis(
        self, 
        post_embedding: torch.Tensor,
        top_k: int = 5,
        return_scores: bool = False
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Suggest top-k emojis for a given post embedding.
        
        Args:
            post_embedding: Single post embedding [embedding_dim]
            top_k: Number of emojis to suggest
            return_scores: Whether to return similarity scores
            
        Returns:
            List of suggested emojis, optionally with scores
        """
        if self.emoji_embeddings is None:
            raise ValueError("Embeddings not setup. Call setup_embeddings() first.")
        
        # Compute similarities
        similarities = self.compute_similarity(post_embedding, self.emoji_embeddings)
        
        # Get top-k
        top_k = min(top_k, len(similarities))
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        # Convert to emoji strings
        suggested_emojis = [self.emoji_vocab[idx.item()] for idx in top_indices]
        
        if return_scores:
            return suggested_emojis, top_scores.tolist()
        else:
            return suggested_emojis
    
    def batch_suggest_emojis(
        self, 
        post_embeddings: torch.Tensor,
        top_k: int = 5
    ) -> List[List[str]]:
        """
        Suggest emojis for a batch of posts.
        
        Args:
            post_embeddings: Batch of post embeddings [batch_size, embedding_dim]
            top_k: Number of emojis to suggest per post
            
        Returns:
            List of emoji suggestions for each post
        """
        if self.emoji_embeddings is None:
            raise ValueError("Embeddings not setup. Call setup_embeddings() first.")
        
        # Compute similarities for all posts at once
        similarities = self.compute_similarity(post_embeddings, self.emoji_embeddings)
        
        # Get top-k for each post
        top_k = min(top_k, similarities.shape[1])
        _, top_indices = torch.topk(similarities, top_k, dim=1)
        
        # Convert to emoji strings
        batch_suggestions = []
        for post_idx in range(len(post_embeddings)):
            suggested_emojis = [self.emoji_vocab[idx.item()] 
                              for idx in top_indices[post_idx]]
            batch_suggestions.append(suggested_emojis)
        
        return batch_suggestions
    
    def train(self, **kwargs):
        """
        Emoji suggestion is unsupervised - no training needed.
        Uses pre-trained embeddings directly.
        """
        logger.info("Emoji suggestion is unsupervised - no training phase needed")
        return {}
    
    def evaluate(
        self,
        test_posts: torch.Tensor,
        test_emojis: List[List[str]],
        top_k_list: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate emoji suggestion performance.
        
        Args:
            test_posts: Test post embeddings [num_posts, embedding_dim]
            test_emojis: Ground truth emojis for each post
            top_k_list: List of k values for evaluation
            
        Returns:
            Dictionary of metrics for each k
        """
        if self.emoji_embeddings is None:
            raise ValueError("Embeddings not setup. Call setup_embeddings() first.")
        
        # Create reverse emoji vocabulary for evaluation
        emoji_to_idx = {emoji: idx for idx, emoji in self.emoji_vocab.items()}
        
        metrics = {}
        for metric_name in ['precision', 'recall', 'f1', 'ndcg']:
            metrics[metric_name] = {}
        
        for k in top_k_list:
            precisions, recalls, f1s, ndcgs = [], [], [], []
            
            for i, post_emb in enumerate(test_posts):
                # Get suggestions
                suggested_emojis = self.suggest_emojis(post_emb, top_k=k)
                
                # Convert to indices for evaluation
                true_emoji_indices = []
                for emoji in test_emojis[i]:
                    if emoji in emoji_to_idx:
                        true_emoji_indices.append(emoji_to_idx[emoji])
                
                if not true_emoji_indices:
                    continue  # Skip posts with no valid emojis
                
                suggested_indices = []
                for emoji in suggested_emojis:
                    if emoji in emoji_to_idx:
                        suggested_indices.append(emoji_to_idx[emoji])
                
                # Calculate metrics
                if suggested_indices:
                    # Precision, Recall, F1
                    true_set = set(true_emoji_indices)
                    pred_set = set(suggested_indices)
                    
                    precision = len(true_set & pred_set) / len(pred_set)
                    recall = len(true_set & pred_set) / len(true_set)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
                    
                    # NDCG
                    relevance_scores = [1 if idx in true_set else 0 for idx in suggested_indices]
                    if sum(relevance_scores) > 0:
                        ideal_scores = [1] * min(len(true_set), k) + [0] * max(0, k - len(true_set))
                        ndcg = ndcg_score([ideal_scores], [relevance_scores])
                        ndcgs.append(ndcg)
            
            # Store average metrics
            metrics['precision'][k] = np.mean(precisions) if precisions else 0
            metrics['recall'][k] = np.mean(recalls) if recalls else 0
            metrics['f1'][k] = np.mean(f1s) if f1s else 0
            metrics['ndcg'][k] = np.mean(ndcgs) if ndcgs else 0
        
        return metrics
    
    def create_emoji_retrieval_index(self):
        """
        Create an efficient index for emoji retrieval (optional optimization).
        For large emoji vocabularies, this could use approximate nearest neighbor search.
        """
        if self.emoji_embeddings is None:
            raise ValueError("Embeddings not setup. Call setup_embeddings() first.")
        
        # For now, we use exact search, but this could be optimized with FAISS
        logger.info("Using exact similarity search. Consider FAISS for large emoji vocabularies.")
    
    def suggest_emojis_with_explanation(
        self,
        post_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Suggest emojis with similarity scores as explanations.
        
        Args:
            post_embedding: Single post embedding
            top_k: Number of suggestions
            
        Returns:
            List of dictionaries with emoji and score
        """
        suggested_emojis, scores = self.suggest_emojis(
            post_embedding, top_k, return_scores=True
        )
        
        return [
            {'emoji': emoji, 'score': score}
            for emoji, score in zip(suggested_emojis, scores)
        ]