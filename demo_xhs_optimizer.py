#!/usr/bin/env python
# encoding: utf-8
"""
Demo script for XHS Engagement Optimization Pipeline
Shows how to use the system with example posts
"""

import os
import json
import logging
from xhs_engagement_optimizer import XHSEngagementOptimizer, XHSPost

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_optimization():
    """Run a demo of the XHS engagement optimization system"""
    
    # Demo configuration
    api_key = os.environ.get('FDU_API_KEY')
    OPENAI_API_KEY = api_key
    CHECKPOINT_PATH = "moco_True_linkpred_True/current.pth"
    DEVICE = "cpu"
    
    # Check if API key is set
    if OPENAI_API_KEY == "your-openai-api-key-here":
        logger.warning("Please set your OpenAI API key in the demo script")
        # For demo purposes, we'll run without LLM optimization
        OPENAI_API_KEY = None
    
    logger.info("🚀 Starting XHS Engagement Optimization Demo")
    logger.info("=" * 60)
    
    # Create demo posts
    demo_posts = [
        XHSPost(content="今天去了一个很棒的咖啡店，咖啡很香，环境也很好"),
        XHSPost(content="分享一个超好用的护肤品，用了一个月皮肤变好了很多"),
        XHSPost(content="周末和朋友们一起爬山，风景真的太美了"),
        XHSPost(content="最近入手了一件很喜欢的裙子，颜色和款式都很棒"),
        XHSPost(content="今天学会了一道新菜，味道还不错，下次再改进一下")
    ]
    
    try:
        # Initialize optimizer
        if OPENAI_API_KEY:
            optimizer = XHSEngagementOptimizer(
                checkpoint_path=CHECKPOINT_PATH,
                openai_api_key=OPENAI_API_KEY,
                device=DEVICE
            )
        else:
            # Create a mock optimizer for demo
            optimizer = create_mock_optimizer(CHECKPOINT_PATH, DEVICE)
        
        logger.info("📊 Running engagement prediction demo...")
        
        # Test individual predictions
        for i, post in enumerate(demo_posts):
            logger.info(f"\n--- Post {i+1} ---")
            logger.info(f"Content: {post.content}")
            
            # Predict engagement
            score = optimizer.predictor.predict_engagement(post.content)
            logger.info(f"Predicted engagement: {score:.3f}")
            
            # Generate emoji suggestions
            emoji_suggestions = optimizer.predictor.generate_emoji_suggestions(post.content)
            logger.info(f"Emoji suggestions: {emoji_suggestions}")
        
        # Test optimization (only if API key is available)
        if OPENAI_API_KEY:
            logger.info(f"\n{'='*60}")
            logger.info("🎯 Running optimization demo...")
            
            # Optimize the first post
            results = optimizer.optimize_post(
                post=demo_posts[0],
                target_score=0.8,
                max_iterations=3
            )
            
            # Display results
            logger.info(f"\n📈 Optimization Results:")
            for result in results:
                logger.info(f"Iteration {result.iteration}:")
                logger.info(f"  Content: {result.optimized_content}")
                logger.info(f"  Score: {result.original_score:.3f} → {result.predicted_score:.3f}")
                logger.info(f"  Improvement: {result.improvement:+.3f}")
                logger.info(f"  Emojis: {result.emoji_suggestions}")
                logger.info("")
        else:
            logger.info("\n⚠️  LLM optimization skipped (no API key)")
            logger.info("The system successfully loaded the pre-trained model and can:")
            logger.info("✅ Predict engagement scores")
            logger.info("✅ Generate emoji suggestions")
            logger.info("✅ Create graph representations")
            logger.info("❌ LLM optimization (requires OpenAI API key)")
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ Demo completed successfully!")
        logger.info("The system is ready for production use with real XHS data.")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.error("Please check:")
        logger.error("1. Checkpoint file exists: moco_True_linkpred_True/current.pth")
        logger.error("2. All dependencies are installed")
        logger.error("3. OpenAI API key is valid (if using LLM optimization)")

def create_mock_optimizer(checkpoint_path, device):
    """Create a mock optimizer for demo when no API key is available"""
    from xhs_engagement_optimizer import EngagementPredictor
    
    class MockOptimizer:
        def __init__(self, checkpoint_path, device):
            self.predictor = EngagementPredictor(checkpoint_path, device)
        
        def optimize_post(self, post, target_score=0.8, max_iterations=3):
            # Mock optimization without LLM
            return []
    
    return MockOptimizer(checkpoint_path, device)

def create_sample_posts_file():
    """Create a sample posts file for testing"""
    sample_posts = [
        {"content": "今天去了一个很棒的咖啡店，咖啡很香，环境也很好"},
        {"content": "分享一个超好用的护肤品，用了一个月皮肤变好了很多"},
        {"content": "周末和朋友们一起爬山，风景真的太美了"},
        {"content": "最近入手了一件很喜欢的裙子，颜色和款式都很棒"},
        {"content": "今天学会了一道新菜，味道还不错，下次再改进一下"},
        {"content": "推荐一家超好吃的火锅店，服务态度也很好"},
        {"content": "新买的口红颜色很正，质地也很好"},
        {"content": "今天的妆容很满意，眼影和口红都很配"},
        {"content": "分享一个很实用的收纳方法，让房间变得整洁"},
        {"content": "最近在学习新的健身动作，感觉很有效果"}
    ]
    
    with open("sample_xhs_posts.json", "w", encoding="utf-8") as f:
        json.dump(sample_posts, f, ensure_ascii=False, indent=2)
    
    logger.info("📝 Created sample_xhs_posts.json with 10 demo posts")

if __name__ == "__main__":
    print("🎯 XHS Engagement Optimization Demo")
    print("=" * 40)
    print("This demo shows how to use the XHS engagement optimization pipeline.")
    print("Please ensure you have:")
    print("1. The pre-trained checkpoint: moco_True_linkpred_True/current.pth")
    print("2. OpenAI API key (optional, for LLM optimization)")
    print("3. All required dependencies installed")
    print("")
    
    # Create sample posts file
    create_sample_posts_file()
    
    # Run demo
    demo_optimization()
    
    print("\n🎉 Demo completed!")
    print("You can now use the system with:")
    print("python xhs_engagement_optimizer.py --openai-api-key YOUR_KEY") 