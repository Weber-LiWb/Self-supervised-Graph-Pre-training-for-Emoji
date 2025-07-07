#!/usr/bin/env python
# encoding: utf-8
"""
Demo: Post Selection Methods for XHS Optimizer V2
Shows different ways to select posts for optimization
"""

import os
import logging
from xhs_engagement_optimizer_v2 import get_database_stats, browse_database_posts, load_posts_from_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_post_selection_methods():
    """Demonstrate different post selection methods"""
    
    logger.info("🎯 XHS Post Selection Methods Demo")
    logger.info("=" * 50)
    
    db_path = "xhs_data.db"
    
    # 1. Show database statistics
    logger.info("\n📊 Database Statistics:")
    stats = get_database_stats(db_path)
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # 2. Browse available posts
    logger.info("\n📋 Browsing Available Posts (first 10):")
    posts = browse_database_posts(db_path, limit=10)
    
    for i, (rowid, content) in enumerate(posts, 1):
        preview = content[:60] + "..." if len(content) > 60 else content
        print(f"{i:2d}. [ID: {rowid:4d}] {preview}")
    
    # 3. Show how to load specific posts
    logger.info(f"\n🔢 Example: Loading specific posts by ID")
    if posts:
        # Take first 3 post IDs as example
        example_ids = [rowid for rowid, _ in posts[:3]]
        logger.info(f"Loading posts with IDs: {example_ids}")
        
        selected_posts = load_posts_from_database(db_path, example_ids)
        logger.info(f"✅ Successfully loaded {len(selected_posts)} posts")
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 Demo Complete!")

def show_usage_examples():
    """Show command-line usage examples"""
    
    print("\n🚀 XHS Optimizer V2 - Post Selection Usage Examples")
    print("=" * 60)
    
    print("\n1️⃣  Interactive Mode (choose posts interactively):")
    print("   python xhs_engagement_optimizer_v2.py --interactive")
    print("   python xhs_engagement_optimizer_v2.py --interactive --browse-limit 100")
    
    print("\n2️⃣  Specific Post Numbers (comma-separated):")
    print("   python xhs_engagement_optimizer_v2.py --post-numbers '1,5,10,25'")
    
    print("\n3️⃣  Load from JSON file:")
    print("   python xhs_engagement_optimizer_v2.py --input-posts sample_posts.json")
    
    print("\n4️⃣  API Key Options (multiple ways):")
    print("   # Environment variable")
    print("   export FDU_API_KEY='your-api-key'")
    print("   python xhs_engagement_optimizer_v2.py --interactive")
    print("   ")
    print("   # API key file (default: api_key.txt)")
    print("   echo 'sk-your-key-here' > api_key.txt")
    print("   python xhs_engagement_optimizer_v2.py --interactive")
    print("   ")
    print("   # Custom API key file")
    print("   python xhs_engagement_optimizer_v2.py --api-key-file my_key.txt --interactive")
    
    print("\n5️⃣  Prediction-only mode (no API key needed):")
    print("   python xhs_engagement_optimizer_v2.py --post-numbers '1,2,3'")
    print("   # Will prompt to continue without LLM optimization")
    
    print("\n6️⃣  Full optimization with custom settings:")
    print("   python xhs_engagement_optimizer_v2.py \\")
    print("     --post-numbers '10,20,30' \\")
    print("     --browse-limit 200 \\")
    print("     --target-score 0.85 \\")
    print("     --max-iterations 3 \\")
    print("     --output-results my_results.json")
    
    print("\n📝 Notes:")
    print("  • Post numbers refer to database row IDs")
    print("  • --browse-limit controls how many posts to show in interactive mode")
    print("  • API key priority: argument > env var > api_key.txt file")
    print("  • System works in prediction-only mode without API key")
    print("  • Default browse limit is 50 posts")

if __name__ == "__main__":
    print("🎯 XHS Post Selection Demo")
    print("=" * 30)
    print("This demo shows how to select specific posts for optimization")
    print("")
    
    # Run the demo
    demo_post_selection_methods()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 Ready to use XHS Optimizer V2 with custom post selection!") 