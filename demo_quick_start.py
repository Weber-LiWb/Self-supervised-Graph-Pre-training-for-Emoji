#!/usr/bin/env python
# encoding: utf-8
"""
Quick Start Demo for XHS Optimizer V2
Shows how easy it is to use with environment API key and custom post selection
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_quick_start():
    """Demonstrate the quickest way to get started"""
    
    print("🚀 XHS Optimizer V2 - Quick Start Guide")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.environ.get('FDU_API_KEY')
    
    if api_key:
        print("✅ API Key Status: Found in environment (FDU_API_KEY)")
        print(f"   Key preview: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("⚠️  API Key Status: Not found in environment")
        print("   You can still run in prediction-only mode!")
    
    print("\n🎯 Ready-to-Run Commands:")
    print("-" * 30)
    
    print("\n1️⃣  Interactive Mode (Recommended):")
    print("   python xhs_engagement_optimizer_v2.py --interactive")
    print("   → Browse and select posts visually")
    
    print("\n2️⃣  Quick Test with Specific Posts:")
    print("   python xhs_engagement_optimizer_v2.py --post-numbers '1,5,10'")
    print("   → Optimize posts 1, 5, and 10 from database")
    
    print("\n3️⃣  Custom Browse Size:")
    print("   python xhs_engagement_optimizer_v2.py --interactive --browse-limit 20")
    print("   → Show only 20 posts for quicker selection")
    
    print("\n4️⃣  Large Selection:")
    print("   python xhs_engagement_optimizer_v2.py --interactive --browse-limit 200")
    print("   → Browse through 200 posts for more variety")
    
    print("\n5️⃣  Full Optimization Pipeline:")
    print("   python xhs_engagement_optimizer_v2.py \\")
    print("     --post-numbers '100,200,300' \\")
    print("     --target-score 0.9 \\")
    print("     --max-iterations 5")
    print("   → Complete optimization with high target score")
    
    print("\n🔧 What Happens Automatically:")
    print("  ✅ API key loaded from FDU_API_KEY environment variable")
    print("  ✅ Real XHS vocabulary (10K+ words, 5K+ emojis) loaded from database")
    print("  ✅ Pre-trained model checkpoint loaded automatically")
    print("  ✅ Results saved to optimization_results_v2.json")
    
    print("\n📊 Database Info:")
    print("  • Total posts available: 1.9M+")
    print("  • Posts with content: 1.85M+")
    print("  • You can select any post by its row ID")
    
    if api_key:
        print("\n🎉 You're all set! Just run any command above to start optimizing!")
    else:
        print("\n💡 Without API key, you can still:")
        print("  • Get engagement predictions")
        print("  • See emoji suggestions")
        print("  • Analyze post features")
        print("  • Browse database posts")

def show_interactive_preview():
    """Show what interactive mode looks like"""
    
    print("\n🎮 Interactive Mode Preview:")
    print("=" * 40)
    print("When you run --interactive, you'll see:")
    print()
    print("📋 Available Posts:")
    print("─" * 80)
    print(" 1. [ID:    1] 肌肉酸痛药 从中学时代开始，我的肩颈一直不好...")
    print(" 2. [ID:    2] 广源良菜瓜水 此生不回购款 广源良的菜瓜水...")
    print(" 3. [ID:    3] origins菌菇水 origins 菌菇水 最近用完好多空瓶...")
    print("...")
    print("50. [ID:   50] 今天分享一个超好用的护肤品...")
    print("─" * 80)
    print("Showing 50 posts (limit: 50)")
    print("💡 Tip: Use --browse-limit to see more/fewer posts")
    print()
    print("🎯 Enter post numbers to optimize (comma-separated, e.g., '1,3,5'): ")
    print()
    print("Then the system will:")
    print("  1. Load your selected posts from database")
    print("  2. Predict engagement scores")
    print("  3. Generate emoji suggestions")
    print("  4. Optimize content with LLM")
    print("  5. Save results to JSON file")

if __name__ == "__main__":
    demo_quick_start()
    show_interactive_preview()
    
    print("\n" + "="*60)
    print("🎯 Next Steps:")
    print("1. Try: python xhs_engagement_optimizer_v2.py --interactive")
    print("2. Select a few posts you want to optimize")
    print("3. Watch the magic happen! ✨") 