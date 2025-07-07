#!/usr/bin/env python
# encoding: utf-8
"""
XHS Engagement Optimization Pipeline
Combines graph pre-training model predictions with LLM-based iterative optimization
"""

import argparse
import logging
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import openai
from pathlib import Path
import dgl
import re

# Import project modules
from gcc.models import GraphEncoder
from gcc.datasets.graph_dataset_new import NodeClassificationDatasetv2
from gcc.datasets.data_util import batcher
from xhs_graph_processor import XHSGraphProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Results from one optimization iteration"""
    iteration: int
    original_content: str
    optimized_content: str
    original_score: float
    predicted_score: float
    emoji_suggestions: List[str]
    improvement: float

@dataclass
class XHSPost:
    """XHS post data structure"""
    content: str
    engagement_score: Optional[float] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None

class EngagementPredictor:
    """Predicts engagement scores using the pre-trained graph model"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.args = None
        self.graph_processor = XHSGraphProcessor()
        self.engagement_predictor_head = None
        self._load_model()
        self._setup_engagement_predictor()
    
    def _load_model(self):
        """Load the pre-trained model from checkpoint"""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.args = checkpoint["opt"]
        
        # Initialize model architecture
        self.model = GraphEncoder(
            positional_embedding_size=self.args.positional_embedding_size,
            max_degree=self.args.max_degree,
            degree_embedding_size=self.args.degree_embedding_size,
            output_dim=self.args.hidden_size,
            node_hidden_dim=self.args.hidden_size,
            node_feat_dim=self.args.node_feat_dim,
            num_layers=self.args.num_layer,
            norm=self.args.norm,
            degree_input=True,
        )
        
        # Load trained weights
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _setup_engagement_predictor(self):
        """Setup engagement prediction head"""
        # Simple MLP for engagement prediction
        self.engagement_predictor_head = torch.nn.Sequential(
            torch.nn.Linear(self.args.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        # Initialize with some reasonable weights
        # In practice, this would be trained on actual engagement data
        with torch.no_grad():
            for layer in self.engagement_predictor_head:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
    
    def predict_engagement(self, post_content: str) -> float:
        """Predict engagement score using the pre-trained model (no BERT)"""
        try:
            # Get post embedding using the pre-trained model directly
            post_embedding = self.graph_processor.get_post_embedding_from_pretrained_model(
                post_content, self.model
            )
            
            # Use the learned embedding to predict engagement
            with torch.no_grad():
                post_embedding = post_embedding.unsqueeze(0).to(self.device)
                engagement_score = self.engagement_predictor_head(post_embedding)
                
            # Log the embedding quality for debugging
            embedding_norm = torch.norm(post_embedding).item()
            logger.debug(f"Post embedding norm: {embedding_norm:.3f}")
                
            return engagement_score.item()
            
        except Exception as e:
            logger.error(f"Error in model-based engagement prediction: {e}")
            logger.info("Falling back to heuristic prediction")
            return self._heuristic_engagement_score(post_content)
    
    def _heuristic_engagement_score(self, post_content: str) -> float:
        """Simple heuristic engagement predictor as fallback"""
        score = 0.3  # Base score
        
        # Check for emojis
        emoji_count = len(self.graph_processor.extract_emojis(post_content))
        score += min(emoji_count * 0.1, 0.3)  # Up to 0.3 bonus for emojis
        
        # Check for length (optimal around 50-100 chars)
        text_length = len(post_content)
        if 50 <= text_length <= 100:
            score += 0.2
        elif 20 <= text_length <= 150:
            score += 0.1
        
        # Check for question marks (engagement prompts)
        if '?' in post_content or '？' in post_content:
            score += 0.1
        
        # Check for exclamation marks (enthusiasm)
        if '!' in post_content or '！' in post_content:
            score += 0.1
        
        return min(score, 1.0)
    
    def generate_emoji_suggestions(self, post_content: str, top_k: int = 5) -> List[str]:
        """Generate optimal emoji suggestions using the pre-trained model (no BERT)"""
        try:
            # Use the graph processor's method that leverages the pre-trained model
            suggestions = self.graph_processor.find_similar_emojis_using_model(
                post_content, self.model, top_k
            )
            
            if suggestions:
                logger.info(f"Generated {len(suggestions)} emoji suggestions using pre-trained model")
                return suggestions
            else:
                logger.warning("No suggestions from model, using fallback")
                return self._fallback_emoji_suggestions(post_content, top_k)
            
        except Exception as e:
            logger.error(f"Error in model-based emoji generation: {e}")
            return self._fallback_emoji_suggestions(post_content, top_k)
    
    def _fallback_emoji_suggestions(self, post_content: str, top_k: int = 5) -> List[str]:
        """Fallback emoji suggestions using simpler semantic matching"""
        logger.info("Using fallback emoji suggestion method")
        
        # Extract current emojis
        current_emojis = set(self.graph_processor.extract_emojis(post_content))
        
        # Use basic content analysis with a smaller set of common emojis
        common_emojis = ["😊", "❤️", "👍", "🔥", "✨", "💯", "🎉", "😍", "👏", "💪"]
        suggestions = [emoji for emoji in common_emojis if emoji not in current_emojis]
        
        return suggestions[:top_k]

    def _load_emoji_vocabulary_from_checkpoint(self) -> List[str]:
        """
        Extract emoji vocabulary from the checkpoint if available
        This would contain the actual emojis the model was trained on
        """
        try:
            # Check if the checkpoint contains emoji vocabulary
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            
            # Look for emoji vocabulary in the checkpoint
            if 'emoji_vocab' in checkpoint:
                emoji_vocab = checkpoint['emoji_vocab']
                logger.info(f"Loaded {len(emoji_vocab)} emojis from checkpoint vocabulary")
                return list(emoji_vocab.keys())
            
            # Look for other possible vocabulary keys
            for key in ['vocab', 'emoji_to_id', 'emoji_dict']:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    potential_emojis = [k for k in checkpoint[key].keys() if self.graph_processor.emoji_pattern.match(k)]
                    if potential_emojis:
                        logger.info(f"Found {len(potential_emojis)} emojis in checkpoint['{key}']")
                        return potential_emojis
            
            logger.warning("No emoji vocabulary found in checkpoint, using default set")
            return None
            
        except Exception as e:
            logger.error(f"Error loading emoji vocabulary from checkpoint: {e}")
            return None

class LLMOptimizer:
    """Uses LLM to optimize content based on engagement predictions"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def create_optimization_prompt(self, 
                                 original_content: str,
                                 current_score: float,
                                 target_score: float,
                                 emoji_suggestions: List[str],
                                 iteration: int) -> str:
        """Create optimization prompt for LLM"""
        
        prompt = f"""
# XHS Content Optimization Task

You are an expert social media content optimizer for Xiaohongshu (XHS). Your goal is to optimize post content to achieve higher engagement.

## Current Situation:
- **Original Content**: "{original_content}"
- **Current Engagement Score**: {current_score:.3f}
- **Target Score**: {target_score:.3f}
- **Optimization Round**: {iteration}

## Available Emoji Suggestions:
{', '.join(emoji_suggestions)}

## Optimization Guidelines:
1. **Maintain Core Message**: Keep the original meaning and intent
2. **Emoji Placement**: Use emojis strategically for emotional impact and readability
3. **XHS Style**: Make it engaging, authentic, and suitable for XHS audience
4. **Length**: Keep it concise but impactful
5. **Emotional Resonance**: Enhance emotional connection with readers

## Your Task:
Rewrite the content to achieve higher engagement while following these principles:
- Use 2-4 emojis from the suggestions (consider semantic relevance and positioning)
- Enhance readability and emotional appeal
- Make it more shareable and comment-worthy
- Maintain authenticity

## Output Format:
Return ONLY the optimized content with emojis, nothing else.

## Optimized Content:
"""
        return prompt
    
    def optimize_content(self, 
                        original_content: str,
                        current_score: float,
                        target_score: float,
                        emoji_suggestions: List[str],
                        iteration: int) -> str:
        """Optimize content using LLM"""
        
        prompt = self.create_optimization_prompt(
            original_content, current_score, target_score, emoji_suggestions, iteration
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            optimized_content = response.choices[0].message.content.strip()
            return optimized_content
            
        except Exception as e:
            logger.error(f"LLM optimization failed: {e}")
            return original_content

class XHSEngagementOptimizer:
    """Main optimization pipeline"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 openai_api_key: str,
                 device: str = "cpu"):
        
        self.predictor = EngagementPredictor(checkpoint_path, device)
        self.llm_optimizer = LLMOptimizer(openai_api_key)
        self.optimization_history: List[OptimizationResult] = []
    
    def optimize_post(self, 
                     post: XHSPost,
                     target_score: float = 0.8,
                     max_iterations: int = 5,
                     min_improvement: float = 0.01) -> List[OptimizationResult]:
        """
        Iteratively optimize a post until target score or max iterations
        
        Args:
            post: XHS post to optimize
            target_score: Target engagement score (0-1)
            max_iterations: Maximum optimization iterations
            min_improvement: Minimum improvement required to continue
            
        Returns:
            List of optimization results for each iteration
        """
        
        current_content = post.content
        results = []
        
        logger.info(f"Starting optimization for post: {current_content[:50]}...")
        logger.info(f"Target score: {target_score}, Max iterations: {max_iterations}")
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration} ---")
            
            # 1. Predict current engagement
            current_score = self.predictor.predict_engagement(current_content)
            logger.info(f"Current engagement score: {current_score:.3f}")
            
            # Check if target achieved
            if current_score >= target_score:
                logger.info(f"Target score achieved! Final score: {current_score:.3f}")
                break
            
            # 2. Generate emoji suggestions
            emoji_suggestions = self.predictor.generate_emoji_suggestions(current_content)
            logger.info(f"Emoji suggestions: {emoji_suggestions}")
            
            # 3. Optimize with LLM
            logger.info("Optimizing content with LLM...")
            optimized_content = self.llm_optimizer.optimize_content(
                original_content=post.content,
                current_score=current_score,
                target_score=target_score,
                emoji_suggestions=emoji_suggestions,
                iteration=iteration
            )
            
            # 4. Predict new score
            new_score = self.predictor.predict_engagement(optimized_content)
            improvement = new_score - current_score
            
            logger.info(f"Optimized content: {optimized_content}")
            logger.info(f"New score: {new_score:.3f} (improvement: {improvement:+.3f})")
            
            # 5. Record results
            result = OptimizationResult(
                iteration=iteration,
                original_content=post.content,
                optimized_content=optimized_content,
                original_score=current_score,
                predicted_score=new_score,
                emoji_suggestions=emoji_suggestions,
                improvement=improvement
            )
            results.append(result)
            
            # 6. Check improvement threshold
            if improvement < min_improvement:
                logger.info(f"Improvement below threshold ({min_improvement}). Stopping.")
                break
            
            # 7. Update current content for next iteration
            current_content = optimized_content
        
        logger.info(f"\nOptimization complete! Total iterations: {len(results)}")
        return results
    
    def batch_optimize(self, 
                      posts: List[XHSPost],
                      target_score: float = 0.8,
                      max_iterations: int = 5) -> Dict[int, List[OptimizationResult]]:
        """Optimize multiple posts in batch"""
        
        batch_results = {}
        
        for i, post in enumerate(posts):
            logger.info(f"\n{'='*50}")
            logger.info(f"Optimizing post {i+1}/{len(posts)}")
            logger.info(f"{'='*50}")
            
            results = self.optimize_post(post, target_score, max_iterations)
            batch_results[i] = results
        
        return batch_results
    
    def save_results(self, results: Dict, output_path: str):
        """Save optimization results to JSON"""
        
        # Convert results to serializable format
        serializable_results = {}
        for post_id, post_results in results.items():
            serializable_results[post_id] = [
                {
                    "iteration": r.iteration,
                    "original_content": r.original_content,
                    "optimized_content": r.optimized_content,
                    "original_score": r.original_score,
                    "predicted_score": r.predicted_score,
                    "emoji_suggestions": r.emoji_suggestions,
                    "improvement": r.improvement
                }
                for r in post_results
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser("XHS Engagement Optimization Pipeline")
    
    # Model arguments
    parser.add_argument("--checkpoint-path", type=str, 
                       default="moco_True_linkpred_True/current.pth",
                       help="Path to pre-trained model checkpoint")

    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run model on")
    
    # Optimization arguments
    parser.add_argument("--openai-api-key", type=str, required=True,
                       help="OpenAI API key for LLM optimization")
    parser.add_argument("--target-score", type=float, default=0.8,
                       help="Target engagement score")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum optimization iterations")
    parser.add_argument("--min-improvement", type=float, default=0.01,
                       help="Minimum improvement threshold")
    
    # Data arguments
    parser.add_argument("--input-posts", type=str,
                       help="JSON file with XHS posts to optimize")
    parser.add_argument("--output-results", type=str, 
                       default="optimization_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = XHSEngagementOptimizer(
        checkpoint_path=args.checkpoint_path,
        openai_api_key=args.openai_api_key,
        device=args.device
    )
    
    # Example posts for demonstration
    if args.input_posts:
        with open(args.input_posts, 'r', encoding='utf-8') as f:
            posts_data = json.load(f)
        posts = [XHSPost(content=p["content"]) for p in posts_data]
    else:
        # Demo posts
        posts = [
            XHSPost(content="今天去了一个很棒的咖啡店，咖啡很香，环境也很好"),
            XHSPost(content="分享一个超好用的护肤品，用了一个月皮肤变好了很多"),
            XHSPost(content="周末和朋友们一起爬山，风景真的太美了")
        ]
    
    # Run optimization
    results = optimizer.batch_optimize(
        posts=posts,
        target_score=args.target_score,
        max_iterations=args.max_iterations
    )
    
    # Save results
    optimizer.save_results(results, args.output_results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    
    for post_id, post_results in results.items():
        if post_results:
            final_result = post_results[-1]
            print(f"\nPost {post_id + 1}:")
            print(f"  Original: {final_result.original_content[:50]}...")
            print(f"  Optimized: {final_result.optimized_content}")
            print(f"  Score improvement: {final_result.original_score:.3f} → {final_result.predicted_score:.3f}")
            print(f"  Iterations: {len(post_results)}")

if __name__ == "__main__":
    main() 