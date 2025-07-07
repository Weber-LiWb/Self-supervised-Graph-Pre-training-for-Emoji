#!/usr/bin/env python
# encoding: utf-8
"""
XHS Engagement Optimization Pipeline V2 - ENHANCED VERSION
Fixes Feature Representation Gap and Emoji Vocabulary Limitation

Key Improvements:
1. Uses XHSGraphProcessorV2 with real database vocabularies
2. Proper TF-IDF features matching training data
3. Real emoji vocabulary from XHS database
4. Better engagement prediction using learned embeddings
5. Enhanced emoji suggestion system
"""

import argparse
import logging
import json
import torch
import numpy as np
import os
import sqlite3
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
from xhs_graph_processor_v2 import XHSGraphProcessorV2

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
    feature_quality_score: float  # New: quality of feature representation

@dataclass
class XHSPost:
    """XHS post data structure"""
    content: str
    engagement_score: Optional[float] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None

class EngagementPredictorV2:
    """Enhanced engagement predictor with proper feature representation"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu", db_path: str = "xhs_data.db"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.args = None
        self.graph_processor = XHSGraphProcessorV2(db_path=db_path)
        self.engagement_predictor_head = None
        
        # Load vocabularies
        if not self.graph_processor.load_vocabularies_from_file():
            logger.info("Loading vocabularies from database...")
        
        self._load_model()
        
        # Log vocabulary stats
        vocab_stats = self.graph_processor.get_vocabulary_stats()
        logger.info(f"Loaded vocabularies: {vocab_stats}")
    
    def _load_model(self):
        """Load the pre-trained model from checkpoint"""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.args = checkpoint["opt"]
        
        # Handle missing node_feat_dim attribute (default is 768)
        if not hasattr(self.args, 'node_feat_dim'):
            self.args.node_feat_dim = 768
            logger.info("Added missing node_feat_dim=768 to checkpoint args")
        
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
    
    def _calculate_content_engagement_features(self, post_content: str) -> float:
        """Calculate content-based engagement features to complement model embeddings"""
        score = 0.0
        
        # Emoji analysis
        emojis = self.graph_processor.extract_emojis(post_content)
        emoji_count = len(emojis)
        if emoji_count > 0:
            emoji_bonus = min(emoji_count * 0.1, 0.3)
            unique_emojis = len(set(emojis))
            diversity_bonus = (unique_emojis / max(emoji_count, 1)) * 0.1
            score += emoji_bonus + diversity_bonus
        
        # Text length analysis (optimal range for XHS)
        text_length = len(post_content)
        if 30 <= text_length <= 120:
            score += 0.3
        elif 15 <= text_length <= 200:
            score += 0.2
        elif text_length > 200:
            score += 0.1
        
        # Engagement triggers
        if '?' in post_content or '？' in post_content:
            score += 0.15
        
        exclamation_count = post_content.count('!') + post_content.count('！')
        if exclamation_count > 0:
            score += min(exclamation_count * 0.05, 0.1)
        
        # Vocabulary coverage (how well the post matches our training vocabulary)
        words = self.graph_processor.extract_words(post_content)
        if words:
            vocab_coverage = sum(1 for w in words if w in self.graph_processor.word_vocab) / len(words)
            score += vocab_coverage * 0.2
        
        return min(score, 1.0)
    
    def predict_engagement(self, post_content: str) -> Tuple[float, float]:
        """
        Predict engagement score using the pre-trained model directly
        Returns: (engagement_score, feature_quality_score)
        """
        try:
            # Get post embedding using the pre-trained model
            post_embedding = self.graph_processor.get_post_embedding_from_pretrained_model(
                post_content, self.model
            )
            
            # Calculate feature quality score
            feature_quality = self._calculate_feature_quality(post_content, post_embedding)
            
            # Use the pre-trained model embedding to predict engagement
            # The model was trained for graph-level tasks, so we can use the embedding norm
            # and semantic richness as engagement indicators
            with torch.no_grad():
                # Normalize embedding and use its properties for engagement prediction
                embedding_norm = torch.norm(post_embedding).item()
                embedding_mean = torch.mean(post_embedding).item()
                embedding_std = torch.std(post_embedding).item()
                
                # Combine embedding properties with content features for engagement score
                # Higher norm indicates richer representation
                norm_score = min(embedding_norm / 50.0, 1.0)  # Normalize to [0,1]
                
                # Add content-based features
                content_score = self._calculate_content_engagement_features(post_content)
                
                # Weighted combination: 60% model embedding + 40% content features
                engagement_score = 0.6 * norm_score + 0.4 * content_score
                engagement_score = max(0.1, min(engagement_score, 1.0))  # Clamp to [0.1, 1.0]
                
            return engagement_score, feature_quality
            
        except Exception as e:
            logger.error(f"Error in model-based engagement prediction: {e}")
            return self._heuristic_engagement_score(post_content), 0.5
    
    def _calculate_feature_quality(self, post_content: str, embedding: torch.Tensor) -> float:
        """Calculate quality score of feature representation"""
        try:
            # Check if embedding is meaningful (not mostly zeros)
            non_zero_ratio = (embedding != 0).float().mean().item()
            
            # Check embedding norm (should be reasonable)
            norm = torch.norm(embedding).item()
            norm_score = min(norm / 10.0, 1.0)
            
            # Check vocabulary coverage
            words = self.graph_processor.extract_words(post_content)
            emojis = self.graph_processor.extract_emojis(post_content)
            
            word_coverage = sum(1 for w in words if w in self.graph_processor.word_vocab) / max(len(words), 1)
            emoji_coverage = sum(1 for e in emojis if e in self.graph_processor.emoji_vocab) / max(len(emojis), 1)
            
            # Combine scores
            quality_score = (non_zero_ratio * 0.3 + norm_score * 0.3 + 
                           word_coverage * 0.2 + emoji_coverage * 0.2)
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.debug(f"Error calculating feature quality: {e}")
            return 0.5
    
    def _heuristic_engagement_score(self, post_content: str) -> float:
        """Enhanced heuristic engagement predictor"""
        score = 0.3  # Base score
        
        # Emoji analysis
        emojis = self.graph_processor.extract_emojis(post_content)
        emoji_count = len(emojis)
        
        # Bonus for emojis, but with diminishing returns
        if emoji_count > 0:
            emoji_bonus = min(emoji_count * 0.08, 0.25)
            # Extra bonus for diverse emojis
            unique_emojis = len(set(emojis))
            diversity_bonus = (unique_emojis / max(emoji_count, 1)) * 0.05
            score += emoji_bonus + diversity_bonus
        
        # Text length analysis (optimal range)
        text_length = len(post_content)
        if 30 <= text_length <= 120:
            score += 0.25
        elif 15 <= text_length <= 200:
            score += 0.15
        elif text_length > 200:
            score += 0.05  # Penalty for too long
        
        # Engagement prompts
        if '?' in post_content or '？' in post_content:
            score += 0.12
        
        # Enthusiasm indicators
        exclamation_count = post_content.count('!') + post_content.count('！')
        if exclamation_count > 0:
            score += min(exclamation_count * 0.05, 0.15)
        
        # Word vocabulary coverage bonus
        words = self.graph_processor.extract_words(post_content)
        if words:
            vocab_coverage = sum(1 for w in words if w in self.graph_processor.word_vocab) / len(words)
            score += vocab_coverage * 0.1
        
        return min(score, 1.0)
    
    def generate_emoji_suggestions(self, post_content: str, top_k: int = 5) -> List[str]:
        """Generate enhanced emoji suggestions using the pre-trained model"""
        try:
            # Use the improved processor with real vocabulary
            suggestions = self.graph_processor.find_similar_emojis_using_model(
                post_content, self.model, top_k
            )
            
            if suggestions:
                logger.info(f"Generated {len(suggestions)} emoji suggestions using enhanced model")
                return suggestions
            else:
                logger.warning("No suggestions from model, using enhanced fallback")
                return self._enhanced_fallback_emoji_suggestions(post_content, top_k)
            
        except Exception as e:
            logger.error(f"Error in model-based emoji generation: {e}")
            return self._enhanced_fallback_emoji_suggestions(post_content, top_k)
    
    def _enhanced_fallback_emoji_suggestions(self, post_content: str, top_k: int = 5) -> List[str]:
        """
        Enhanced fallback emoji suggestions using ONLY real XHS emoji vocabulary
        NO HARD-CODED EMOJIS - uses database vocabulary with semantic analysis
        """
        logger.info("Using enhanced fallback emoji suggestion method with real XHS vocabulary")
        
        # Extract current emojis
        current_emojis = set(self.graph_processor.extract_emojis(post_content))
        
        # Get available emojis from the actual XHS database vocabulary
        available_emojis = list(self.graph_processor.emoji_vocab.keys())
        
        if not available_emojis:
            logger.warning("No emoji vocabulary available from database!")
            return []
        
        logger.info(f"Working with {len(available_emojis)} real emojis from XHS database")
        
        # Remove emojis already in the post
        candidate_emojis = [emoji for emoji in available_emojis if emoji not in current_emojis]
        
        if not candidate_emojis:
            logger.info("All available emojis are already in the post")
            return []
        
        # Use simple frequency-based selection from the database
        # The most frequent emojis in XHS are likely the most engaging
        suggestions = candidate_emojis[:top_k]
        
        logger.info(f"Selected {len(suggestions)} emoji suggestions from real XHS vocabulary")
        return suggestions

class LLMOptimizerV2:
    """Enhanced LLM optimizer with better prompting"""
    
    def __init__(self, api_key: str, model_name: str = "glm-4-plus", base_url: str = "https://open.bigmodel.cn/api/paas/v4/"):
        # Support both OpenAI and Zhipu models
        if "glm" in model_name.lower():
            # Zhipu GLM model
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            # OpenAI model
            self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def create_optimization_prompt(self, 
                                 original_content: str,
                                 current_score: float,
                                 target_score: float,
                                 iteration: int,
                                 feature_quality: float) -> str:
        """Create enhanced optimization prompt without separate emoji suggestions"""
        
        # Determine optimization strategy based on feature quality
        if feature_quality > 0.8:
            strategy_note = "The content has high semantic richness. Focus on subtle improvements."
        elif feature_quality > 0.6:
            strategy_note = "The content has good semantic foundation. Enhance key elements."
        else:
            strategy_note = "The content needs semantic enrichment. Add meaningful vocabulary."
        
        prompt = f"""
# XHS Content Optimization Task - Advanced

You are an expert XHS content optimizer with deep understanding of Chinese social media engagement patterns.

## Current Analysis:
- **Original Content**: "{original_content}"
- **Current Engagement Score**: {current_score:.3f}/1.0
- **Target Score**: {target_score:.3f}/1.0
- **Feature Quality**: {feature_quality:.3f}/1.0
- **Optimization Round**: {iteration}/5
- **Strategy**: {strategy_note}

## Optimization Framework:

### 1. Content Enhancement Priorities:
- **Semantic Richness**: Use vocabulary that resonates with XHS users
- **Emotional Connection**: Create authentic emotional hooks
- **Visual Appeal**: Strategic Unicode emoji placement for readability
- **Engagement Triggers**: Include elements that encourage interaction

### 2. XHS Platform Optimization:
- **Length**: 50-120 characters for optimal engagement
- **Structure**: Clear, scannable format with emojis as visual breaks
- **Tone**: Authentic, relatable, slightly casual but polished
- **Call-to-Action**: Subtle encouragement for likes/comments

### 3. Technical Requirements:
- Add 2-4 relevant Unicode emojis directly into the text
- Place emojis naturally within the content, not as separate suggestions
- Maintain the core message and authenticity
- Ensure smooth reading flow
- Balance text and emojis effectively

### 4. Quality Metrics to Optimize:
- Vocabulary richness (use diverse, meaningful words)
- Emotional resonance (create connection with readers)
- Visual structure (emojis as punctuation and emphasis)
- Shareability factor (make it worth sharing)

## Your Task:
Transform the content to achieve higher engagement while maintaining authenticity and leveraging platform-specific best practices. Add Unicode emojis directly into the text content.

## Optimized Content:
"""
        return prompt
    
    def optimize_content(self, 
                        original_content: str,
                        current_score: float,
                        target_score: float,
                        iteration: int,
                        feature_quality: float = 0.5) -> str:
        """Optimize content using enhanced LLM prompting"""
        
        prompt = self.create_optimization_prompt(
            original_content, current_score, target_score, 
            iteration, feature_quality
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
                top_p=0.9
            )
            
            optimized_content = response.choices[0].message.content.strip()
            
            # Clean up the response (remove any extra formatting)
            if "优化内容:" in optimized_content:
                optimized_content = optimized_content.split("优化内容:")[-1].strip()
            if "Optimized Content:" in optimized_content:
                optimized_content = optimized_content.split("Optimized Content:")[-1].strip()
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"LLM optimization failed: {e}")
            return original_content

class XHSEngagementOptimizerV2:
    """Enhanced optimization pipeline with improved feature representation"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 openai_api_key: str,
                 device: str = "cpu",
                 db_path: str = "xhs_data.db"):
        
        self.predictor = EngagementPredictorV2(checkpoint_path, device, db_path)
        self.llm_optimizer = LLMOptimizerV2(openai_api_key)
        self.optimization_history: List[OptimizationResult] = []
    
    def optimize_post(self, 
                     post: XHSPost,
                     target_score: float = 0.8,
                     max_iterations: int = 5,
                     min_improvement: float = 0.01) -> List[OptimizationResult]:
        """
        Enhanced iterative optimization with feature quality tracking
        """
        
        current_content = post.content
        results = []
        
        logger.info(f"🚀 Starting enhanced optimization for post: {current_content[:50]}...")
        logger.info(f"🎯 Target score: {target_score}, Max iterations: {max_iterations}")
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration} ---")
            
            # 1. Predict current engagement with feature quality
            current_score, feature_quality = self.predictor.predict_engagement(current_content)
            logger.info(f"📊 Current engagement: {current_score:.3f}, Feature quality: {feature_quality:.3f}")
            
            # Check if target achieved
            if current_score >= target_score:
                logger.info(f"🎉 Target score achieved! Final score: {current_score:.3f}")
                break
            
            # 2. Optimize with enhanced LLM (emojis added directly to content)
            logger.info("🤖 Optimizing content with enhanced LLM...")
            optimized_content = self.llm_optimizer.optimize_content(
                original_content=post.content,
                current_score=current_score,
                target_score=target_score,
                iteration=iteration,
                feature_quality=feature_quality
            )
            
            # 3. Predict new score
            new_score, new_feature_quality = self.predictor.predict_engagement(optimized_content)
            improvement = new_score - current_score
            
            logger.info(f"✨ Optimized content: {optimized_content}")
            logger.info(f"📈 New score: {new_score:.3f} (improvement: {improvement:+.3f})")
            logger.info(f"🔧 Feature quality: {feature_quality:.3f} → {new_feature_quality:.3f}")
            
            # 4. Record results
            result = OptimizationResult(
                iteration=iteration,
                original_content=post.content,
                optimized_content=optimized_content,
                original_score=current_score,
                predicted_score=new_score,
                emoji_suggestions=[],  # No separate emoji suggestions - emojis added directly to content
                improvement=improvement,
                feature_quality_score=new_feature_quality
            )
            results.append(result)
            
            # 5. Check improvement threshold
            if improvement < min_improvement and improvement >= 0:
                logger.info(f"⏹️ Improvement below threshold ({min_improvement}). Stopping.")
                break
            
            # 6. Update current content for next iteration
            current_content = optimized_content
        
        logger.info(f"\n🏁 Enhanced optimization complete! Total iterations: {len(results)}")
        return results
    
    def batch_optimize(self, 
                      posts: List[XHSPost],
                      target_score: float = 0.8,
                      max_iterations: int = 5) -> Dict[int, List[OptimizationResult]]:
        """Optimize multiple posts in batch with enhanced tracking"""
        
        batch_results = {}
        
        for i, post in enumerate(posts):
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 Optimizing post {i+1}/{len(posts)}")
            logger.info(f"{'='*60}")
            
            results = self.optimize_post(post, target_score, max_iterations)
            batch_results[i] = results
        
        return batch_results
    
    def save_results(self, results: Dict, output_path: str):
        """Save enhanced optimization results to JSON"""
        
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
                    "improvement": r.improvement,
                    "feature_quality_score": r.feature_quality_score
                }
                for r in post_results
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Enhanced results saved to {output_path}")
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        vocab_stats = self.predictor.graph_processor.get_vocabulary_stats()
        
        return {
            'model_loaded': self.predictor.model is not None,
            'vocabularies': vocab_stats,
            'device': str(self.predictor.device),
            'checkpoint_path': self.predictor.checkpoint_path
        }

def create_prediction_only_optimizer(checkpoint_path: str, device: str = "cpu", db_path: str = "xhs_data.db"):
    """Create a prediction-only optimizer when no API key is available"""
    
    class PredictionOnlyOptimizer:
        def __init__(self, checkpoint_path: str, device: str, db_path: str):
            self.predictor = EngagementPredictorV2(checkpoint_path, device, db_path)
            
        def optimize_post(self, post, target_score=0.8, max_iterations=5):
            """Mock optimization that only does prediction"""
            logger.info("🔍 Running prediction-only mode (no LLM optimization)")
            
            # Just predict the current score
            score, quality = self.predictor.predict_engagement(post.content)
            emoji_suggestions = self.predictor.generate_emoji_suggestions(post.content)
            
            logger.info(f"📊 Content: {post.content}")
            logger.info(f"📈 Engagement Score: {score:.3f}")
            logger.info(f"🔧 Feature Quality: {quality:.3f}")
            logger.info(f"🎭 Emoji Suggestions: {emoji_suggestions}")
            
            # Return empty results since no optimization was performed
            return []
        
        def batch_optimize(self, posts, target_score=0.8, max_iterations=5):
            """Batch prediction without optimization"""
            results = {}
            for i, post in enumerate(posts):
                logger.info(f"\n📝 Analyzing post {i+1}/{len(posts)}")
                self.optimize_post(post, target_score, max_iterations)
                results[i] = []  # No optimization results
            return results
        
        def save_results(self, results, output_path):
            """Save prediction results"""
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"note": "Prediction-only mode, no optimization performed"}, 
                         f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Prediction results noted in {output_path}")
        
        def get_system_stats(self):
            """Get system statistics"""
            vocab_stats = self.predictor.graph_processor.get_vocabulary_stats()
            return {
                'model_loaded': self.predictor.model is not None,
                'vocabularies': vocab_stats,
                'device': str(self.predictor.device),
                'checkpoint_path': self.predictor.checkpoint_path,
                'mode': 'prediction_only'
            }
    
    return PredictionOnlyOptimizer(checkpoint_path, device, db_path)

def load_api_key(api_key_arg: str = None, api_key_file: str = "api_key.txt") -> Optional[str]:
    """
    Load OpenAI API key from multiple sources in priority order:
    1. Command line argument
    2. Environment variable FDU_API_KEY
    3. API key file (default: api_key.txt)
    """
    
    # 1. Command line argument (highest priority)
    if api_key_arg:
        logger.info("🔑 Using API key from command line argument")
        return api_key_arg
    
    # 2. Environment variable
    env_key = os.environ.get('FDU_API_KEY')
    if env_key:
        logger.info("🔑 Using API key from FDU_API_KEY environment variable")
        return env_key
    
    # 3. API key file
    try:
        if os.path.exists(api_key_file):
            with open(api_key_file, 'r', encoding='utf-8') as f:
                file_key = f.read().strip()
                if file_key:
                    logger.info(f"🔑 Using API key from file: {api_key_file}")
                    return file_key
        else:
            logger.debug(f"API key file not found: {api_key_file}")
    except Exception as e:
        logger.warning(f"⚠️  Error reading API key file {api_key_file}: {e}")
    
    # No API key found
    logger.warning("🔑 No API key found from any source")
    return None

def create_api_key_file_template(api_key_file: str = "api_key.txt"):
    """Create a template API key file"""
    try:
        with open(api_key_file, 'w', encoding='utf-8') as f:
            f.write("# Replace this line with your OpenAI API key\n")
            f.write("# Example: sk-your-openai-api-key-here\n")
            f.write("\n")
        
        logger.info(f"📝 Created API key template file: {api_key_file}")
        logger.info(f"Please edit {api_key_file} and add your OpenAI API key")
        
    except Exception as e:
        logger.error(f"❌ Error creating API key file template: {e}")

def load_posts_from_database(db_path: str, post_numbers: List[int]) -> List[XHSPost]:
    """Load specific posts from database by their numbers"""
    posts = []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Convert post numbers to SQL placeholders
        placeholders = ','.join(['?' for _ in post_numbers])
        
        query = f"""
        SELECT note_id, title, content 
        FROM note_info 
        WHERE rowid IN ({placeholders})
        ORDER BY rowid
        """
        
        cursor.execute(query, post_numbers)
        results = cursor.fetchall()
        
        for note_id, title, content in results:
            # Combine title and content for full post text
            full_content = f"{title} {content}".strip() if title and content else (title or content or "")
            
            if full_content:
                posts.append(XHSPost(content=full_content))
                logger.info(f"📝 Loaded post {note_id}: {full_content[:50]}...")
        
        conn.close()
        logger.info(f"✅ Loaded {len(posts)} posts from database")
        
    except Exception as e:
        logger.error(f"❌ Error loading posts from database: {e}")
        
    return posts

def browse_database_posts(db_path: str, limit: int = 20) -> List[Tuple[int, str]]:
    """Browse posts from database for selection"""
    posts = []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT rowid, note_id, title, content 
        FROM note_info 
        WHERE (title IS NOT NULL AND title != '') OR (content IS NOT NULL AND content != '')
        ORDER BY rowid
        LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        
        for rowid, note_id, title, content in results:
            # Combine title and content
            full_content = f"{title} {content}".strip() if title and content else (title or content or "")
            
            if full_content:
                posts.append((rowid, full_content))
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error browsing database: {e}")
        
    return posts

def interactive_post_selection(db_path: str, browse_limit: int = 50) -> List[int]:
    """Interactive post selection from database"""
    logger.info("🔍 Interactive Post Selection Mode")
    logger.info("=" * 50)
    
    # Browse available posts
    posts = browse_database_posts(db_path, limit=browse_limit)
    
    if not posts:
        logger.error("❌ No posts found in database")
        return []
    
    # Display posts for selection
    print("\n📋 Available Posts:")
    print("-" * 80)
    
    for i, (rowid, content) in enumerate(posts, 1):
        preview = content[:60] + "..." if len(content) > 60 else content
        print(f"{i:2d}. [ID: {rowid:4d}] {preview}")
    
    print("-" * 80)
    print(f"Showing {len(posts)} posts (limit: {browse_limit})")
    print("💡 Tip: Use --browse-limit to see more/fewer posts")
    
    # Get user selection
    while True:
        try:
            selection = input("\n🎯 Enter post numbers to optimize (comma-separated, e.g., '1,3,5'): ").strip()
            
            if not selection:
                logger.info("No selection made. Exiting.")
                return []
            
            # Parse selection
            selected_indices = [int(x.strip()) for x in selection.split(',')]
            
            # Validate indices
            valid_indices = []
            for idx in selected_indices:
                if 1 <= idx <= len(posts):
                    valid_indices.append(idx)
                else:
                    logger.warning(f"⚠️  Invalid post number: {idx}")
            
            if not valid_indices:
                logger.error("❌ No valid post numbers selected")
                continue
            
            # Convert to database row IDs
            selected_rowids = [posts[idx-1][0] for idx in valid_indices]
            
            # Show selected posts
            print(f"\n✅ Selected {len(selected_rowids)} posts:")
            for idx in valid_indices:
                rowid, content = posts[idx-1]
                preview = content[:60] + "..." if len(content) > 60 else content
                print(f"  • [ID: {rowid}] {preview}")
            
            # Confirm selection
            confirm = input(f"\n🤔 Proceed with these {len(selected_rowids)} posts? (y/n): ").strip().lower()
            
            if confirm == 'y':
                return selected_rowids
            else:
                logger.info("Selection cancelled. Please choose again.")
                continue
                
        except ValueError:
            logger.error("❌ Invalid input. Please enter numbers separated by commas.")
        except KeyboardInterrupt:
            logger.info("\n👋 Selection cancelled by user.")
            return []

def get_database_stats(db_path: str) -> Dict:
    """Get database statistics"""
    stats = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count total posts
        cursor.execute("SELECT COUNT(*) FROM note_info WHERE (title IS NOT NULL AND title != '') OR (content IS NOT NULL AND content != '')")
        stats['total_posts'] = cursor.fetchone()[0]
        
        # Count posts with both title and content
        cursor.execute("SELECT COUNT(*) FROM note_info WHERE title IS NOT NULL AND title != '' AND content IS NOT NULL AND content != ''")
        stats['posts_with_title_and_content'] = cursor.fetchone()[0]
        
        # Sample some post IDs
        cursor.execute("SELECT rowid FROM note_info WHERE (title IS NOT NULL AND title != '') OR (content IS NOT NULL AND content != '') ORDER BY rowid LIMIT 10")
        sample_ids = [row[0] for row in cursor.fetchall()]
        stats['sample_post_ids'] = sample_ids
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error getting database stats: {e}")
        stats = {'error': str(e)}
    
    return stats

def main():
    parser = argparse.ArgumentParser("XHS Engagement Optimization Pipeline V2")
    
    # Model arguments
    parser.add_argument("--checkpoint-path", type=str, 
                       default="moco_True_linkpred_True/current.pth",
                       help="Path to pre-trained model checkpoint")
    parser.add_argument("--db-path", type=str, default="xhs_data.db",
                       help="Path to XHS database")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run model on")
    
    # Optimization arguments
    parser.add_argument("--openai-api-key", type=str, 
                       help="OpenAI API key for LLM optimization (or set FDU_API_KEY env var)")
    parser.add_argument("--target-score", type=float, default=0.8,
                       help="Target engagement score")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum optimization iterations")
    parser.add_argument("--min-improvement", type=float, default=0.01,
                       help="Minimum improvement threshold")
    
    # Data arguments
    parser.add_argument("--input-posts", type=str,
                       help="JSON file with XHS posts to optimize")
    parser.add_argument("--post-numbers", type=str,
                       help="Comma-separated post numbers from database (e.g., '1,5,10,25')")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode to select posts from database")
    parser.add_argument("--browse-limit", type=int, default=50,
                       help="Number of posts to show in interactive mode (default: 50)")
    parser.add_argument("--auto-select", type=int,
                       help="Automatically select first N posts from database (e.g., --auto-select 100)")
    parser.add_argument("--output-results", type=str, 
                       default="optimization_results_v2.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Simple API key loading - same as demo
    api_key = os.environ.get('FDU_API_KEY')
    openai_api_key = api_key
    
    # Override with command line argument if provided
    if args.openai_api_key:
        openai_api_key = args.openai_api_key
        logger.info("🔑 Using API key from command line argument")
    elif openai_api_key:
        logger.info("🔑 Using API key from FDU_API_KEY environment variable")
    else:
        logger.warning("⚠️  No OpenAI API key found!")
        logger.info("You can either:")
        logger.info("1. Set FDU_API_KEY environment variable")
        logger.info("2. Use --openai-api-key argument")
        logger.info("3. Continue without LLM optimization (prediction only)")
        
        response = input("Continue without LLM optimization? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Exiting. Please set API key and try again.")
            return
    
    # Initialize enhanced optimizer
    logger.info("🚀 Initializing XHS Engagement Optimizer V2...")
    
    if openai_api_key:
        optimizer = XHSEngagementOptimizerV2(
            checkpoint_path=args.checkpoint_path,
            openai_api_key=openai_api_key,
            device=args.device,
            db_path=args.db_path
        )
    else:
        # Create prediction-only optimizer
        optimizer = create_prediction_only_optimizer(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            db_path=args.db_path
        )
    
    # Print system stats
    stats = optimizer.get_system_stats()
    logger.info(f"📊 System Stats: {stats}")
    
    # Show database stats
    db_stats = get_database_stats(args.db_path)
    logger.info(f"📊 Database Stats: {db_stats}")
    
    # Determine post source
    posts = []
    
    if args.input_posts:
        # Load from JSON file
        logger.info(f"📂 Loading posts from file: {args.input_posts}")
        with open(args.input_posts, 'r', encoding='utf-8') as f:
            posts_data = json.load(f)
        posts = [XHSPost(content=p["content"]) for p in posts_data]
        
    elif args.post_numbers:
        # Load specific posts by numbers
        logger.info(f"🔢 Loading specific posts from database: {args.post_numbers}")
        post_numbers = [int(x.strip()) for x in args.post_numbers.split(',')]
        posts = load_posts_from_database(args.db_path, post_numbers)
        
    elif args.auto_select:
        # Auto-select first N posts
        logger.info(f"🤖 Auto-selecting first {args.auto_select} posts from database...")
        post_numbers = list(range(1, args.auto_select + 1))
        posts = load_posts_from_database(args.db_path, post_numbers)
        
    elif args.interactive:
        # Interactive selection
        logger.info(f"🎯 Starting interactive post selection (showing {args.browse_limit} posts)...")
        selected_rowids = interactive_post_selection(args.db_path, args.browse_limit)
        
        if selected_rowids:
            posts = load_posts_from_database(args.db_path, selected_rowids)
        else:
            logger.info("No posts selected. Exiting.")
            return
            
    else:
        # Enhanced demo posts (fallback)
        logger.info("📝 Using demo posts (no specific selection made)")
        posts = [
            XHSPost(content="今天去了一个很棒的咖啡店，咖啡很香，环境也很好"),
            XHSPost(content="分享一个超好用的护肤品，用了一个月皮肤变好了很多"),
            XHSPost(content="周末和朋友们一起爬山，风景真的太美了"),
            XHSPost(content="最近入手了一件很喜欢的裙子，颜色和款式都很棒"),
            XHSPost(content="今天学会了一道新菜，味道还不错，下次再改进一下")
        ]
    
    if not posts:
        logger.error("❌ No posts to optimize. Please check your selection.")
        return
    
    logger.info(f"✅ Ready to optimize {len(posts)} posts")
    
    # Run enhanced optimization
    results = optimizer.batch_optimize(
        posts=posts,
        target_score=args.target_score,
        max_iterations=args.max_iterations
    )
    
    # Save results
    optimizer.save_results(results, args.output_results)
    
    # Print enhanced summary
    print(f"\n{'='*80}")
    print("🎉 ENHANCED OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    for post_id, post_results in results.items():
        if post_results:
            final_result = post_results[-1]
            print(f"\n📝 Post {post_id + 1}:")
            print(f"  📄 Original: {final_result.original_content[:50]}...")
            print(f"  ✨ Optimized: {final_result.optimized_content}")
            print(f"  📈 Score: {final_result.original_score:.3f} → {final_result.predicted_score:.3f}")
            print(f"  🔧 Feature Quality: {final_result.feature_quality_score:.3f}")
            print(f"  🔄 Iterations: {len(post_results)}")
            print(f"  🎭 Final Emojis: {final_result.emoji_suggestions}")

if __name__ == "__main__":
    main() 