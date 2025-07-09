#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import logging

logger = logging.getLogger(__name__)


def evaluate_ranking(
    predictions: List[List[str]],
    ground_truth: List[List[str]], 
    top_k_list: List[int] = [1, 3, 5, 10]
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate ranking performance for emoji suggestion.
    
    Args:
        predictions: List of predicted emoji lists for each post
        ground_truth: List of true emoji lists for each post  
        top_k_list: List of k values to evaluate
        
    Returns:
        Dictionary of metrics for each k value
    """
    metrics = {}
    for metric in ['precision', 'recall', 'f1', 'ndcg']:
        metrics[metric] = {}
    
    for k in top_k_list:
        precisions, recalls, f1s, ndcgs = [], [], [], []
        
        for pred, true in zip(predictions, ground_truth):
            if not true:  # Skip if no ground truth
                continue
                
            pred_k = pred[:k]  # Top-k predictions
            true_set = set(true)
            pred_set = set(pred_k)
            
            # Precision, Recall, F1
            if pred_set:
                precision = len(true_set & pred_set) / len(pred_set)
                recall = len(true_set & pred_set) / len(true_set)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall) 
                f1s.append(f1)
                
                # NDCG
                relevance = [1 if item in true_set else 0 for item in pred_k]
                if sum(relevance) > 0:
                    ideal_relevance = [1] * min(len(true_set), k)
                    if len(ideal_relevance) < k:
                        ideal_relevance.extend([0] * (k - len(ideal_relevance)))
                    
                    ndcg = ndcg_score([ideal_relevance], [relevance])
                    ndcgs.append(ndcg)
        
        # Store average metrics
        metrics['precision'][k] = np.mean(precisions) if precisions else 0
        metrics['recall'][k] = np.mean(recalls) if recalls else 0  
        metrics['f1'][k] = np.mean(f1s) if f1s else 0
        metrics['ndcg'][k] = np.mean(ndcgs) if ndcgs else 0
    
    return metrics


def calculate_engagement_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics for engagement prediction.
    
    Args:
        predictions: Predicted engagement scores
        ground_truth: True engagement scores
        
    Returns:
        Dictionary of regression metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truth, predictions)
    
    # Additional metrics
    mape = np.mean(np.abs((ground_truth - predictions) / np.maximum(ground_truth, 1e-8))) * 100
    
    return {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def print_evaluation_results(metrics: Dict, task_name: str = "Task"):
    """
    Pretty print evaluation results.
    
    Args:
        metrics: Dictionary of metrics
        task_name: Name of the task for logging
    """
    logger.info(f"\n=== {task_name} Evaluation Results ===")
    
    if 'precision' in metrics:  # Ranking metrics
        for k in sorted(metrics['precision'].keys()):
            logger.info(f"Top-{k}:")
            logger.info(f"  Precision: {metrics['precision'][k]:.4f}")
            logger.info(f"  Recall:    {metrics['recall'][k]:.4f}")
            logger.info(f"  F1:        {metrics['f1'][k]:.4f}")
            logger.info(f"  NDCG:      {metrics['ndcg'][k]:.4f}")
    else:  # Regression metrics
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")