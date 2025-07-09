#!/usr/bin/env python
# encoding: utf-8

"""
LLM-Based Content Optimizer for Xiaohongshu Posts

This script combines engagement prediction and emoji suggestion with LLM optimization
to iteratively improve post content by optimizing emoji usage without changing text.

Workflow:
1. Predict engagement score for original post
2. Generate emoji suggestions based on content 
3. Use LLM to optimize emoji placement without changing text
4. Predict engagement for optimized version
5. Iterate until threshold reached or max iterations

Usage:
    python llm_content_optimizer.py --checkpoint moco_True_linkpred_True/current.pth --content "Your post content here"
"""

import argparse
import logging
import os
import sys
import time
import json
from typing import Dict, List, Tuple, Optional, Any
import re

# Add project root to path
sys.path.append('/workspace')

from downstream.tasks import EngagementPredictionTask, EmojiSuggestionTask
from downstream.utils.data_utils import create_synthetic_engagement_data

# LLM Integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMContentOptimizer:
    """
    Iterative content optimizer using engagement prediction, emoji suggestion, and LLM.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda:0',
        llm_provider: str = 'openai',
        llm_model: str = 'gpt-3.5-turbo',
        api_key: Optional[str] = None,
        max_iterations: int = 5,
        engagement_threshold: float = 0.8,
        temperature: float = 0.7
    ):
        """
        Initialize the content optimizer.
        
        Args:
            checkpoint_path: Path to pre-trained GNN checkpoint
            device: Device for model inference
            llm_provider: LLM provider ('openai', 'anthropic', 'mock')
            llm_model: Specific model to use
            api_key: API key for LLM service
            max_iterations: Maximum optimization iterations
            engagement_threshold: Target engagement score
            temperature: LLM generation temperature
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.engagement_threshold = engagement_threshold
        self.temperature = temperature
        
        # Initialize downstream tasks
        logger.info("🔧 Initializing downstream tasks...")
        self._init_downstream_tasks()
        
        # Initialize LLM
        logger.info(f"🤖 Initializing LLM: {llm_provider}")
        self._init_llm()
        
        # Optimization history
        self.optimization_history = []
        
    def _init_downstream_tasks(self):
        """Initialize engagement prediction and emoji suggestion tasks."""
        # For demo purposes, we'll simulate these tasks since they need actual graph data
        # In production, you would initialize them properly with your data
        
        # Create synthetic embeddings for demo
        logger.info("📊 Creating synthetic embeddings for demo...")
        embedding_dim = 128  # Typical embedding dimension
        
        # Simulate post and emoji embeddings
        import torch
        post_embeddings = torch.randn(100, embedding_dim)
        emoji_embeddings = torch.randn(50, embedding_dim)
        post_embeddings = torch.nn.functional.normalize(post_embeddings, p=2, dim=1)
        emoji_embeddings = torch.nn.functional.normalize(emoji_embeddings, p=2, dim=1)
        
        # Create emoji vocabulary
        emoji_vocab = {
            0: "😍", 1: "💯", 2: "🔥", 3: "✨", 4: "💫", 5: "❤️", 6: "👏", 7: "🎉",
            8: "💕", 9: "🌟", 10: "😊", 11: "💖", 12: "🥰", 13: "😘", 14: "💪",
            15: "🎊", 16: "🌈", 17: "☀️", 18: "🌸", 19: "🦄", 20: "💎", 21: "🍭",
            22: "🎈", 23: "🌺", 24: "🧚", 25: "🎀", 26: "💝", 27: "🌙", 28: "⭐",
            29: "🌻", 30: "🦋", 31: "🍀", 32: "🌷", 33: "🎯", 34: "💐", 35: "🎵",
            36: "🍰", 37: "🎂", 38: "🍓", 39: "🥳", 40: "😎", 41: "💃", 42: "🕺",
            43: "🎨", 44: "📸", 45: "✈️", 46: "🏖️", 47: "🌊", 48: "🏔️", 49: "🗺️"
        }
        
        # Initialize tasks (simplified for demo)
        self.engagement_task = self._create_mock_engagement_task(embedding_dim)
        self.emoji_task = self._create_mock_emoji_task(emoji_embeddings, emoji_vocab)
        
        logger.info("✅ Downstream tasks initialized")
        
    def _create_mock_engagement_task(self, embedding_dim: int):
        """Create a mock engagement prediction task for demo."""
        class MockEngagementTask:
            def __init__(self, embedding_dim):
                import torch
                # Simple mock predictor
                self.weights = torch.randn(embedding_dim) * 0.1
                
            def predict_from_content(self, content: str) -> float:
                # Mock prediction based on content features
                content_score = 0.5  # Base score
                
                # Boost for emojis
                emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]', content))
                content_score += min(emoji_count * 0.05, 0.2)
                
                # Boost for length
                if 50 <= len(content) <= 150:
                    content_score += 0.1
                
                # Boost for engagement words
                engagement_words = ['推荐', '好用', '必买', '种草', '分享', '超棒', '爱了', '绝绝子']
                for word in engagement_words:
                    if word in content:
                        content_score += 0.05
                
                # Add some randomness but keep it realistic
                import random
                random.seed(hash(content) % 1000)
                noise = random.uniform(-0.1, 0.1)
                
                return max(0.1, min(1.0, content_score + noise))
        
        return MockEngagementTask(embedding_dim)
    
    def _create_mock_emoji_task(self, emoji_embeddings, emoji_vocab):
        """Create a mock emoji suggestion task for demo."""
        class MockEmojiTask:
            def __init__(self, emoji_embeddings, emoji_vocab):
                self.emoji_vocab = emoji_vocab
                self.content_emoji_map = {
                    # Beauty/Fashion
                    'beauty': ["💄", "💅", "✨", "💫", "🌟"],
                    'fashion': ["👗", "👠", "💎", "✨", "👜"],
                    'skincare': ["🧴", "💧", "✨", "😍", "💕"],
                    
                    # Food
                    'food': ["🍰", "🍓", "😋", "💕", "🎉"],
                    'coffee': ["☕", "😍", "💕", "✨", "☀️"],
                    'dessert': ["🍰", "🧁", "🍭", "😍", "💕"],
                    
                    # Lifestyle
                    'travel': ["✈️", "🏖️", "📸", "🌊", "🗺️"],
                    'fitness': ["💪", "🔥", "💯", "🏃", "💦"],
                    'study': ["📚", "✏️", "💪", "🔥", "⭐"],
                    
                    # Emotions
                    'happy': ["😊", "🥰", "💕", "🎉", "✨"],
                    'love': ["❤️", "💕", "😘", "💖", "🥰"],
                    'excited': ["🎉", "🔥", "💯", "😍", "⭐"],
                }
                
            def suggest_emojis_for_content(self, content: str, top_k: int = 5) -> List[str]:
                content_lower = content.lower()
                suggested = []
                
                # Content-based suggestions
                for category, emojis in self.content_emoji_map.items():
                    if any(keyword in content_lower for keyword in self._get_keywords(category)):
                        suggested.extend(emojis[:2])
                
                # Default popular emojis
                if not suggested:
                    suggested = ["😍", "💯", "✨", "💕", "🔥"]
                
                # Remove duplicates while preserving order
                seen = set()
                unique_suggested = []
                for emoji in suggested:
                    if emoji not in seen:
                        seen.add(emoji)
                        unique_suggested.append(emoji)
                
                return unique_suggested[:top_k]
                
            def _get_keywords(self, category: str) -> List[str]:
                keyword_map = {
                    'beauty': ['美妆', '化妆', '口红', '粉底', '美容'],
                    'fashion': ['穿搭', '时尚', '衣服', '搭配', '服装'],
                    'skincare': ['护肤', '面膜', '精华', '乳液', '保湿'],
                    'food': ['美食', '好吃', '味道', '餐厅', '料理'],
                    'coffee': ['咖啡', '拿铁', '卡布', '星巴克'],
                    'dessert': ['甜品', '蛋糕', '甜点', '糖果'],
                    'travel': ['旅行', '旅游', '景点', '度假', '出行'],
                    'fitness': ['健身', '运动', '锻炼', '减肥', '塑形'],
                    'study': ['学习', '考试', '复习', '笔记', '知识'],
                    'happy': ['开心', '快乐', '高兴', '幸福'],
                    'love': ['喜欢', '爱', '心动', '恋爱'],
                    'excited': ['激动', '兴奋', '期待', '惊喜']
                }
                return keyword_map.get(category, [])
        
        return MockEmojiTask(emoji_embeddings, emoji_vocab)
    
    def _init_llm(self):
        """Initialize the LLM client."""
        if self.llm_provider == 'openai':
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI not available, using mock LLM")
                self.llm_provider = 'mock'
                return
            
            if self.api_key:
                openai.api_key = self.api_key
            else:
                # Try to get from environment
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    openai.api_key = api_key
                else:
                    logger.warning("No OpenAI API key found, using mock LLM")
                    self.llm_provider = 'mock'
                    
        elif self.llm_provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                logger.warning("Anthropic not available, using mock LLM")
                self.llm_provider = 'mock'
                return
                
            if not self.api_key:
                self.api_key = os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                logger.warning("No Anthropic API key found, using mock LLM")
                self.llm_provider = 'mock'
                
        logger.info(f"✅ LLM initialized: {self.llm_provider}")
    
    def _predict_engagement(self, content: str) -> float:
        """Predict engagement score for content."""
        try:
            score = self.engagement_task.predict_from_content(content)
            return float(score)
        except Exception as e:
            logger.warning(f"Engagement prediction failed: {e}, using fallback")
            return 0.5  # Fallback score
    
    def _suggest_emojis(self, content: str, top_k: int = 5) -> List[str]:
        """Get emoji suggestions for content."""
        try:
            return self.emoji_task.suggest_emojis_for_content(content, top_k)
        except Exception as e:
            logger.warning(f"Emoji suggestion failed: {e}, using fallback")
            return ["😍", "💯", "✨", "💕", "🔥"][:top_k]
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        if self.llm_provider == 'mock':
            return self._mock_llm_response(prompt)
        
        try:
            if self.llm_provider == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
                
            elif self.llm_provider == 'anthropic':
                client = anthropic.Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model=self.llm_model,
                    max_tokens=500,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._mock_llm_response(prompt)
    
    def _mock_llm_response(self, prompt: str) -> str:
        """Mock LLM response for demo purposes."""
        # Extract original content and suggested emojis from prompt
        content_match = re.search(r'原始内容：(.+?)(?=\n|$)', prompt)
        emoji_match = re.search(r'建议的表情符号：(.+?)(?=\n|$)', prompt)
        
        if not content_match:
            return "Unable to process the content."
        
        original_content = content_match.group(1).strip()
        suggested_emojis = emoji_match.group(1).strip() if emoji_match else "😍✨💕"
        
        # First, remove any existing emojis from the content to avoid duplication
        clean_content = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]', '', original_content).strip()
        
        # Simple mock optimization: add emojis strategically
        sentences = clean_content.split('。')
        optimized_sentences = []
        
        emoji_list = re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]', suggested_emojis)
        if not emoji_list:
            emoji_list = ["😍", "✨", "💕"]
        
        # Ensure we have enough unique emojis
        if len(emoji_list) < 3:
            default_emojis = ["😍", "✨", "💕", "💯", "🔥"]
            for emoji in default_emojis:
                if emoji not in emoji_list:
                    emoji_list.append(emoji)
                if len(emoji_list) >= 3:
                    break
        
        emoji_used = set()
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                # Add emoji to some sentences, ensuring no duplicates
                emoji_to_use = None
                if i == 0:  # First sentence
                    emoji_to_use = emoji_list[0] if emoji_list[0] not in emoji_used else None
                elif i == len(sentences) - 2 and len(sentences) > 1:  # Last meaningful sentence
                    emoji_to_use = next((e for e in emoji_list[1:] if e not in emoji_used), None)
                elif len(sentence) > 20:  # Long sentences
                    emoji_to_use = next((e for e in emoji_list if e not in emoji_used), None)
                
                if emoji_to_use:
                    sentence = f"{sentence} {emoji_to_use}"
                    emoji_used.add(emoji_to_use)
                
                optimized_sentences.append(sentence)
        
        optimized_content = '。'.join(optimized_sentences)
        if not optimized_content.endswith('。') and clean_content.endswith('。'):
            optimized_content += '。'
            
        return optimized_content
    
    def _create_optimization_prompt(self, content: str, suggested_emojis: List[str], iteration: int) -> str:
        """Create the LLM prompt for emoji optimization."""
        emoji_str = ''.join(suggested_emojis)
        
        prompt = f"""你是一个专业的小红书内容优化师，擅长通过合理使用表情符号来提升内容的互动性和吸引力。

任务：优化以下内容的表情符号使用，但不能改变任何文字内容。

原始内容：{content}

建议的表情符号：{emoji_str}

优化要求：
1. 不能修改、删除或添加任何文字
2. 只能调整表情符号的位置和使用
3. 可以使用建议的表情符号，也可以使用其他合适的表情符号
4. 表情符号应该增强内容的情感表达和视觉吸引力
5. 避免过度使用表情符号，保持自然和谐
6. 考虑小红书用户的使用习惯

当前是第{iteration + 1}轮优化，请提供更有吸引力的表情符号搭配。

请直接返回优化后的内容，不要包含任何解释或说明。"""

        return prompt
    
    def optimize_content(
        self, 
        original_content: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize content through iterative LLM-based emoji enhancement.
        
        Args:
            original_content: Original post content
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("🚀 Starting content optimization...")
        logger.info(f"📝 Original content: {original_content}")
        
        # Initialize optimization tracking
        current_content = original_content.strip()
        optimization_log = []
        
        # Get initial engagement score
        initial_score = self._predict_engagement(current_content)
        logger.info(f"📊 Initial engagement score: {initial_score:.3f}")
        
        optimization_log.append({
            'iteration': 0,
            'content': current_content,
            'engagement_score': initial_score,
            'suggested_emojis': [],
            'improvement': 0.0,
            'timestamp': time.time()
        })
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            logger.info(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get emoji suggestions
            suggested_emojis = self._suggest_emojis(current_content, top_k=5)
            logger.info(f"🎯 Suggested emojis: {' '.join(suggested_emojis)}")
            
            # Create optimization prompt
            prompt = self._create_optimization_prompt(current_content, suggested_emojis, iteration)
            
            # Get LLM optimization
            logger.info("🤖 Optimizing with LLM...")
            optimized_content = self._call_llm(prompt)
            
            if not optimized_content or optimized_content == current_content:
                logger.info("⏸️ No further optimization suggested")
                break
            
            # Predict engagement for optimized content
            optimized_score = self._predict_engagement(optimized_content)
            improvement = optimized_score - initial_score
            
            logger.info(f"📈 Optimized engagement score: {optimized_score:.3f}")
            logger.info(f"⬆️ Improvement: {improvement:+.3f}")
            
            if verbose:
                logger.info(f"📝 Optimized content: {optimized_content}")
            
            # Record iteration
            optimization_log.append({
                'iteration': iteration + 1,
                'content': optimized_content,
                'engagement_score': optimized_score,
                'suggested_emojis': suggested_emojis,
                'improvement': improvement,
                'timestamp': time.time()
            })
            
            # Check if threshold reached
            if optimized_score >= self.engagement_threshold:
                logger.info(f"🎉 Target engagement threshold ({self.engagement_threshold:.3f}) reached!")
                break
            
            # Check if improvement is significant
            if improvement < 0.01 and iteration > 0:
                logger.info("⏸️ Minimal improvement detected, stopping optimization")
                break
            
            # Update current content for next iteration
            current_content = optimized_content
        
        # Compile final results
        final_log = optimization_log[-1]
        total_improvement = final_log['engagement_score'] - initial_score
        
        results = {
            'original_content': original_content,
            'optimized_content': final_log['content'],
            'initial_score': initial_score,
            'final_score': final_log['engagement_score'],
            'total_improvement': total_improvement,
            'iterations_used': len(optimization_log) - 1,
            'optimization_log': optimization_log,
            'threshold_reached': final_log['engagement_score'] >= self.engagement_threshold,
            'success': total_improvement > 0
        }
        
        # Log final summary
        logger.info("\n" + "="*60)
        logger.info("📊 OPTIMIZATION SUMMARY")
        logger.info("="*60)
        logger.info(f"📝 Original: {original_content}")
        logger.info(f"✨ Optimized: {final_log['content']}")
        logger.info(f"📈 Score: {initial_score:.3f} → {final_log['engagement_score']:.3f}")
        logger.info(f"⬆️ Improvement: {total_improvement:+.3f} ({total_improvement/initial_score*100:+.1f}%)")
        logger.info(f"🔄 Iterations: {results['iterations_used']}")
        logger.info(f"🎯 Success: {'✅' if results['success'] else '❌'}")
        
        return results
    
    def batch_optimize(
        self, 
        contents: List[str],
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Optimize multiple pieces of content.
        
        Args:
            contents: List of content strings to optimize
            save_results: Whether to save results to file
            
        Returns:
            List of optimization results
        """
        logger.info(f"📦 Starting batch optimization for {len(contents)} items...")
        
        results = []
        for i, content in enumerate(contents):
            logger.info(f"\n📄 Processing item {i+1}/{len(contents)}")
            try:
                result = self.optimize_content(content, verbose=False)
                results.append(result)
                
                # Brief summary
                improvement = result['total_improvement']
                logger.info(f"✅ Item {i+1}: {improvement:+.3f} improvement")
                
            except Exception as e:
                logger.error(f"❌ Item {i+1} failed: {e}")
                results.append({'error': str(e), 'original_content': content})
        
        # Save results if requested
        if save_results:
            timestamp = int(time.time())
            filename = f"batch_optimization_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Results saved to: {filename}")
        
        # Summary statistics
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            avg_improvement = sum(r['total_improvement'] for r in successful_results) / len(successful_results)
            success_rate = len([r for r in successful_results if r['success']]) / len(successful_results)
            
            logger.info(f"\n📊 Batch Summary:")
            logger.info(f"   • Total items: {len(contents)}")
            logger.info(f"   • Successful: {len(successful_results)}")
            logger.info(f"   • Average improvement: {avg_improvement:+.3f}")
            logger.info(f"   • Success rate: {success_rate:.1%}")
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="LLM-based content optimizer for Xiaohongshu posts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Optimize single post
    python llm_content_optimizer.py --content "今天试了这个新面膜，效果真的很不错，皮肤变得水润有光泽。"
    
    # Use custom parameters
    python llm_content_optimizer.py --content "Your content" --max-iterations 3 --threshold 0.9
    
    # Use OpenAI API
    python llm_content_optimizer.py --content "Your content" --llm-provider openai --api-key sk-xxx
    
    # Batch optimize from file
    python llm_content_optimizer.py --batch-file posts.txt --save-results
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="moco_True_linkpred_True/current.pth",
        help="Path to pre-trained GNN checkpoint"
    )
    parser.add_argument(
        "--device", 
        type=str,
        default="cuda:0",
        help="Device for model inference"
    )
    parser.add_argument(
        "--content",
        type=str,
        help="Content to optimize (for single optimization)"
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        help="File containing multiple contents to optimize (one per line)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=['openai', 'anthropic', 'mock'],
        default='mock',
        help="LLM provider to use"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-3.5-turbo",
        help="Specific LLM model to use"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LLM service"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum optimization iterations"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Target engagement threshold"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM generation temperature"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save optimization results to file"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.content and not args.batch_file:
        logger.error("❌ Must provide either --content or --batch-file")
        return
    
    if args.content and args.batch_file:
        logger.error("❌ Cannot use both --content and --batch-file")
        return
    
    # Check checkpoint
    if not os.path.isfile(args.checkpoint):
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    try:
        # Initialize optimizer
        optimizer = LLMContentOptimizer(
            checkpoint_path=args.checkpoint,
            device=args.device,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            api_key=args.api_key,
            max_iterations=args.max_iterations,
            engagement_threshold=args.threshold,
            temperature=args.temperature
        )
        
        if args.content:
            # Single content optimization
            result = optimizer.optimize_content(args.content)
            
            if args.save_results:
                timestamp = int(time.time())
                filename = f"optimization_result_{timestamp}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Result saved to: {filename}")
        
        elif args.batch_file:
            # Batch optimization
            if not os.path.isfile(args.batch_file):
                logger.error(f"❌ Batch file not found: {args.batch_file}")
                return
            
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                contents = [line.strip() for line in f if line.strip()]
            
            if not contents:
                logger.error("❌ No content found in batch file")
                return
            
            results = optimizer.batch_optimize(contents, save_results=args.save_results)
        
        logger.info("🎉 Optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()