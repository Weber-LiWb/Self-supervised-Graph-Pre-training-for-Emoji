#!/usr/bin/env python
# encoding: utf-8

"""
Comprehensive Demo for Reproducing Downstream Tasks from EMOJI Paper

This script demonstrates how to use the pre-trained GNN checkpoint to reproduce
the two main downstream tasks:
1. Engagement Prediction
2. Emoji Suggestion

Usage:
    python reproduce_downstream_demo.py --checkpoint moco_True_linkpred_True/current.pth --device cuda:0
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
from typing import Dict, List

# Add the project root to Python path
sys.path.append('/workspace')

from downstream.tasks import EngagementPredictionTask, EmojiSuggestionTask
from downstream.utils.data_utils import create_synthetic_engagement_data
from downstream.utils.evaluation_utils import print_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_engagement_prediction(
    checkpoint_path: str,
    device: str,
    num_synthetic_posts: int = 500
):
    """
    Demonstrate engagement prediction using the pre-trained GNN.
    
    Args:
        checkpoint_path: Path to the pre-trained checkpoint
        device: Device to run on
        num_synthetic_posts: Number of synthetic posts for demo
    """
    logger.info("\n" + "="*60)
    logger.info("ENGAGEMENT PREDICTION DEMONSTRATION")
    logger.info("="*60)
    
    # Initialize the engagement prediction task
    logger.info("🔧 Initializing Engagement Prediction Task...")
    engagement_task = EngagementPredictionTask(
        checkpoint_path=checkpoint_path,
        device=device,
        learning_rate=1e-3,
        hidden_dims=[64, 32]  # Custom architecture for the prediction head
    )
    
    # For demo purposes, we'll create synthetic data
    # In real usage, you would generate embeddings from your actual posts
    logger.info("📊 Creating synthetic training data...")
    embedding_dim = engagement_task.checkpoint_args.hidden_size
    post_embeddings, engagement_scores = create_synthetic_engagement_data(
        num_posts=num_synthetic_posts,
        embedding_dim=embedding_dim,
        random_seed=42
    )
    
    # Split data for training and testing
    split_idx = int(0.8 * len(post_embeddings))
    train_embeddings = post_embeddings[:split_idx]
    train_scores = engagement_scores[:split_idx]
    test_embeddings = post_embeddings[split_idx:]
    test_scores = engagement_scores[split_idx:]
    
    logger.info(f"📚 Training set: {len(train_embeddings)} posts")
    logger.info(f"🧪 Test set: {len(test_embeddings)} posts")
    
    # Train the engagement prediction model
    logger.info("🚀 Training engagement prediction model...")
    history = engagement_task.train(
        post_embeddings=train_embeddings,
        engagement_scores=train_scores,
        num_epochs=50,
        batch_size=32,
        early_stopping_patience=10,
        verbose=True
    )
    
    # Evaluate on test set
    logger.info("📈 Evaluating on test set...")
    test_metrics = engagement_task.evaluate(test_embeddings, test_scores)
    print_evaluation_results(test_metrics, "Engagement Prediction")
    
    # Make some example predictions
    logger.info("🎯 Making example predictions...")
    sample_embeddings = test_embeddings[:5]
    sample_true_scores = test_scores[:5]
    sample_predictions = engagement_task.predict(sample_embeddings)
    
    logger.info("\n📋 Sample Predictions:")
    for i, (true_score, pred_score) in enumerate(zip(sample_true_scores, sample_predictions)):
        logger.info(f"  Post {i+1}: True={true_score:.3f}, Predicted={pred_score:.3f}, "
                   f"Error={abs(true_score - pred_score):.3f}")
    
    # Save the trained model
    model_save_path = "engagement_prediction_model.pth"
    engagement_task.save_model(model_save_path)
    logger.info(f"💾 Model saved to {model_save_path}")
    
    return engagement_task, test_metrics


def demonstrate_emoji_suggestion(
    checkpoint_path: str,
    device: str,
    num_synthetic_posts: int = 200,
    num_synthetic_emojis: int = 50
):
    """
    Demonstrate emoji suggestion using the pre-trained GNN.
    
    Args:
        checkpoint_path: Path to the pre-trained checkpoint  
        device: Device to run on
        num_synthetic_posts: Number of synthetic posts for demo
        num_synthetic_emojis: Number of synthetic emojis for demo
    """
    logger.info("\n" + "="*60)
    logger.info("EMOJI SUGGESTION DEMONSTRATION") 
    logger.info("="*60)
    
    # Initialize the emoji suggestion task
    logger.info("🔧 Initializing Emoji Suggestion Task...")
    emoji_task = EmojiSuggestionTask(
        checkpoint_path=checkpoint_path,
        device=device,
        similarity_metric='cosine',
        temperature=1.0
    )
    
    # For demo purposes, create synthetic embeddings
    # In real usage, you would generate these from actual posts and emojis
    logger.info("📊 Creating synthetic embeddings...")
    embedding_dim = emoji_task.checkpoint_args.hidden_size
    
    # Create synthetic post and emoji embeddings
    post_embeddings = torch.randn(num_synthetic_posts, embedding_dim)
    emoji_embeddings = torch.randn(num_synthetic_emojis, embedding_dim)
    
    # Normalize embeddings
    post_embeddings = torch.nn.functional.normalize(post_embeddings, p=2, dim=1)
    emoji_embeddings = torch.nn.functional.normalize(emoji_embeddings, p=2, dim=1)
    
    # Create a simple emoji vocabulary
    emoji_vocab = {i: f"emoji_{i:02d}" for i in range(num_synthetic_emojis)}
    
    # Setup embeddings in the task
    logger.info("⚙️ Setting up embeddings...")
    emoji_task.setup_embeddings(
        post_embeddings=post_embeddings,
        emoji_embeddings=emoji_embeddings,
        emoji_vocab=emoji_vocab
    )
    
    # Demonstrate emoji suggestion for individual posts
    logger.info("🎯 Making emoji suggestions...")
    sample_posts = post_embeddings[:5]
    
    logger.info("\n📋 Sample Emoji Suggestions:")
    for i, post_emb in enumerate(sample_posts):
        suggestions = emoji_task.suggest_emojis_with_explanation(post_emb, top_k=5)
        logger.info(f"  Post {i+1}:")
        for j, suggestion in enumerate(suggestions):
            logger.info(f"    {j+1}. {suggestion['emoji']} (score: {suggestion['score']:.3f})")
    
    # Demonstrate batch suggestion
    logger.info("\n🚀 Batch emoji suggestion...")
    batch_suggestions = emoji_task.batch_suggest_emojis(sample_posts, top_k=3)
    for i, suggestions in enumerate(batch_suggestions):
        logger.info(f"  Post {i+1}: {', '.join(suggestions)}")
    
    # Create synthetic evaluation data
    logger.info("📊 Creating synthetic evaluation data...")
    test_posts = post_embeddings[-20:]  # Last 20 posts for testing
    
    # Create synthetic ground truth (random emojis for each post)
    np.random.seed(42)
    test_emojis = []
    for _ in range(len(test_posts)):
        num_true_emojis = np.random.randint(1, 4)  # 1-3 emojis per post
        true_emojis = np.random.choice(list(emoji_vocab.values()), 
                                     size=num_true_emojis, replace=False).tolist()
        test_emojis.append(true_emojis)
    
    # Evaluate the model
    logger.info("📈 Evaluating emoji suggestion performance...")
    eval_metrics = emoji_task.evaluate(
        test_posts=test_posts,
        test_emojis=test_emojis,
        top_k_list=[1, 3, 5]
    )
    print_evaluation_results(eval_metrics, "Emoji Suggestion")
    
    return emoji_task, eval_metrics


def demonstrate_integration_workflow(
    checkpoint_path: str,
    device: str
):
    """
    Demonstrate an integrated workflow using both tasks.
    
    Args:
        checkpoint_path: Path to the pre-trained checkpoint
        device: Device to run on
    """
    logger.info("\n" + "="*60)
    logger.info("INTEGRATED WORKFLOW DEMONSTRATION")
    logger.info("="*60)
    
    logger.info("🔄 This demonstrates how both tasks can work together...")
    logger.info("💡 In a real application, you would:")
    logger.info("   1. Load your actual post-emoji-word graph")
    logger.info("   2. Generate embeddings for posts and emojis")
    logger.info("   3. Train engagement prediction on historical data")
    logger.info("   4. Use emoji suggestion to enhance posts")
    logger.info("   5. Predict engagement for optimized content")
    
    # For demo, we'll create a simple synthetic example
    embedding_dim = 128  # Assuming this from typical checkpoint
    
    # Simulate a new post
    new_post_embedding = torch.randn(1, embedding_dim)
    new_post_embedding = torch.nn.functional.normalize(new_post_embedding, p=2, dim=1)
    
    logger.info("\n📝 Analyzing a new post...")
    logger.info("1. Original post embedding generated")
    
    # Note: In practice, you would load trained models here
    logger.info("2. Predicting original engagement score... (would use trained model)")
    original_engagement = 0.65  # Simulated
    logger.info(f"   📊 Original engagement prediction: {original_engagement:.3f}")
    
    logger.info("3. Suggesting emojis to enhance the post... (would use emoji model)")
    suggested_emojis = ["😍", "💯", "🔥", "✨", "💫"]  # Simulated
    logger.info(f"   🎯 Suggested emojis: {', '.join(suggested_emojis)}")
    
    logger.info("4. Predicting engagement with emoji enhancement... (would use trained model)")
    enhanced_engagement = 0.78  # Simulated
    logger.info(f"   📈 Enhanced engagement prediction: {enhanced_engagement:.3f}")
    
    improvement = enhanced_engagement - original_engagement
    logger.info(f"   🚀 Predicted improvement: +{improvement:.3f} ({improvement/original_engagement*100:.1f}%)")


def main():
    """Main function to run the demonstrations."""
    parser = argparse.ArgumentParser(
        description="Reproduce downstream tasks from EMOJI paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default checkpoint
    python reproduce_downstream_demo.py --checkpoint moco_True_linkpred_True/current.pth
    
    # Run on CPU
    python reproduce_downstream_demo.py --checkpoint moco_True_linkpred_True/current.pth --device cpu
    
    # Run only engagement prediction
    python reproduce_downstream_demo.py --checkpoint moco_True_linkpred_True/current.pth --task engagement
        """
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="moco_True_linkpred_True/current.pth",
        help="Path to the pre-trained GNN checkpoint"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to run on (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["all", "engagement", "emoji", "integration"],
        default="all",
        help="Which task(s) to demonstrate"
    )
    parser.add_argument(
        "--num-posts",
        type=int,
        default=500,
        help="Number of synthetic posts for demonstration"
    )
    parser.add_argument(
        "--num-emojis",
        type=int, 
        default=50,
        help="Number of synthetic emojis for demonstration"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.isfile(args.checkpoint):
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        logger.error("   Please ensure the checkpoint path is correct.")
        return
    
    # Check device availability
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info("🎬 Starting Downstream Tasks Demonstration")
    logger.info(f"📁 Checkpoint: {args.checkpoint}")
    logger.info(f"💻 Device: {args.device}")
    
    results = {}
    
    try:
        # Run demonstrations based on task selection
        if args.task in ["all", "engagement"]:
            engagement_task, engagement_metrics = demonstrate_engagement_prediction(
                args.checkpoint, args.device, args.num_posts
            )
            results["engagement"] = engagement_metrics
        
        if args.task in ["all", "emoji"]:
            emoji_task, emoji_metrics = demonstrate_emoji_suggestion(
                args.checkpoint, args.device, args.num_posts, args.num_emojis
            )
            results["emoji"] = emoji_metrics
        
        if args.task in ["all", "integration"]:
            demonstrate_integration_workflow(args.checkpoint, args.device)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATION SUMMARY")
        logger.info("="*60)
        
        if "engagement" in results:
            metrics = results["engagement"]
            logger.info(f"📊 Engagement Prediction Performance:")
            logger.info(f"   • R² Score: {metrics.get('r2', 0):.3f}")
            logger.info(f"   • RMSE: {metrics.get('rmse', 0):.3f}")
            logger.info(f"   • MAE: {metrics.get('mae', 0):.3f}")
        
        if "emoji" in results:
            metrics = results["emoji"]
            logger.info(f"🎯 Emoji Suggestion Performance (Top-5):")
            precision_5 = metrics.get('precision', {}).get(5, 0)
            recall_5 = metrics.get('recall', {}).get(5, 0)
            f1_5 = metrics.get('f1', {}).get(5, 0)
            logger.info(f"   • Precision@5: {precision_5:.3f}")
            logger.info(f"   • Recall@5: {recall_5:.3f}")
            logger.info(f"   • F1@5: {f1_5:.3f}")
        
        logger.info("\n✅ Demonstration completed successfully!")
        logger.info("\n💡 Next Steps:")
        logger.info("   1. Replace synthetic data with your actual Xiaohongshu dataset")
        logger.info("   2. Fine-tune the models on your specific data")
        logger.info("   3. Integrate into your content optimization pipeline")
        logger.info("   4. Evaluate on real engagement metrics")
        
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()