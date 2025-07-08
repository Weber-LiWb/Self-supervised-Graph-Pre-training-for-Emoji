#!/usr/bin/env python
# encoding: utf-8

"""
Demo script for LLM Content Optimizer

This script demonstrates the iterative content optimization workflow that combines:
1. Engagement prediction from the pre-trained GNN
2. Emoji suggestion based on content analysis  
3. LLM-based optimization to improve emoji placement
4. Iterative refinement until engagement threshold is reached

Usage:
    python demo_llm_optimizer.py
"""

import sys
import os

# Add workspace to path
sys.path.append('/workspace')

from downstream.llm_content_optimizer import LLMContentOptimizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_single_optimization():
    """Demonstrate optimization of a single post."""
    print("\n" + "="*80)
    print("🎯 DEMO: Single Post Optimization")
    print("="*80)
    
    # Example Chinese post content (typical Xiaohongshu style)
    test_content = "今天试了这个新面膜，效果真的很不错，皮肤变得水润有光泽。推荐给大家，值得入手。"
    
    print(f"📝 Original Content: {test_content}")
    
    # Initialize optimizer with mock models (for demo)
    optimizer = LLMContentOptimizer(
        checkpoint_path="moco_True_linkpred_True/current.pth",
        device="cpu",  # Use CPU for demo
        llm_provider="mock",  # Use mock LLM for demo
        max_iterations=3,
        engagement_threshold=0.75,
        temperature=0.7
    )
    
    # Run optimization
    result = optimizer.optimize_content(test_content)
    
    # Print detailed results
    print(f"\n📊 Final Results:")
    print(f"   Original Score: {result['initial_score']:.3f}")
    print(f"   Final Score: {result['final_score']:.3f}")
    print(f"   Improvement: {result['total_improvement']:+.3f}")
    print(f"   Iterations: {result['iterations_used']}")
    print(f"   Success: {'✅' if result['success'] else '❌'}")
    
    return result


def demo_batch_optimization():
    """Demonstrate batch optimization of multiple posts."""
    print("\n" + "="*80)
    print("📦 DEMO: Batch Post Optimization")
    print("="*80)
    
    # Example posts in different categories
    test_contents = [
        "这款口红颜色真的绝了，显白又有气质，强烈推荐。",
        "今天的穿搭分享，简约风格很适合日常出街。",
        "周末和朋友去了这家新开的咖啡店，环境很棒。",
        "分享一下最近在用的护肤品，效果不错值得回购。",
        "健身第30天打卡，坚持真的会看到变化。"
    ]
    
    print(f"📋 Processing {len(test_contents)} posts...")
    
    # Initialize optimizer
    optimizer = LLMContentOptimizer(
        checkpoint_path="moco_True_linkpred_True/current.pth",
        device="cpu",
        llm_provider="mock",
        max_iterations=2,  # Fewer iterations for batch demo
        engagement_threshold=0.8,
        temperature=0.7
    )
    
    # Run batch optimization
    results = optimizer.batch_optimize(test_contents, save_results=False)
    
    # Print summary
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        avg_improvement = sum(r['total_improvement'] for r in successful_results) / len(successful_results)
        success_count = len([r for r in successful_results if r['success']])
        
        print(f"\n📈 Batch Results Summary:")
        print(f"   Total Posts: {len(test_contents)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Improved: {success_count}")
        print(f"   Average Improvement: {avg_improvement:+.3f}")
        print(f"   Success Rate: {success_count/len(successful_results):.1%}")
    
    return results


def demo_workflow_explanation():
    """Explain the optimization workflow."""
    print("\n" + "="*80)
    print("🔧 OPTIMIZATION WORKFLOW EXPLANATION")
    print("="*80)
    
    workflow_steps = [
        ("1. Content Analysis", "Parse the original post content and extract features"),
        ("2. Engagement Prediction", "Use pre-trained GNN to predict initial engagement score"),
        ("3. Emoji Suggestion", "Analyze content and suggest relevant emojis"),
        ("4. LLM Optimization", "Use LLM to optimize emoji placement without changing text"),
        ("5. Re-evaluation", "Predict engagement score for optimized content"),
        ("6. Iteration Decision", "Continue if below threshold, stop if goal reached"),
        ("7. Final Output", "Return optimized content with improvement metrics")
    ]
    
    for step, description in workflow_steps:
        print(f"   {step}: {description}")
    
    print(f"\n🎯 Key Features:")
    print(f"   • Uses your pre-trained GNN checkpoint for engagement prediction")
    print(f"   • Content-aware emoji suggestions based on semantic analysis")
    print(f"   • LLM preserves original text while optimizing emoji usage")
    print(f"   • Iterative refinement until engagement threshold reached")
    print(f"   • Comprehensive tracking and evaluation metrics")


def demo_integration_with_existing_code():
    """Show how this integrates with existing XHSOpt code."""
    print("\n" + "="*80)
    print("🔗 INTEGRATION WITH EXISTING CODE")
    print("="*80)
    
    print("This LLM optimizer improves upon the XHSOpt/ implementation by:")
    print("")
    print("📈 Advantages over XHSOpt/:")
    print("   1. Uses actual pre-trained GNN embeddings instead of heuristics")
    print("   2. Proper engagement prediction with learned representations")
    print("   3. Content-aware emoji suggestions based on semantic similarity")
    print("   4. Systematic iterative optimization with stopping criteria")
    print("   5. Comprehensive evaluation and tracking")
    print("")
    print("🔄 Workflow Comparison:")
    print("   XHSOpt/: Heuristic engagement → Hardcoded emojis → LLM optimization")
    print("   New:     GNN engagement → Semantic emoji suggestion → LLM optimization")
    print("")
    print("💡 Integration Points:")
    print("   • Replace XHSOpt engagement prediction with downstream/tasks/engagement_prediction.py")
    print("   • Replace hardcoded emoji lists with downstream/tasks/emoji_suggestion.py")
    print("   • Keep LLM optimization logic but with better input features")
    print("   • Add proper evaluation metrics and iterative refinement")


def main():
    """Run all demonstration scenarios."""
    print("🎬 LLM Content Optimizer Demonstration")
    print("This demo shows how to combine GNN-based predictions with LLM optimization")
    
    try:
        # Run demos
        demo_workflow_explanation()
        demo_integration_with_existing_code()
        
        # Check if checkpoint exists for actual demos
        checkpoint_path = "moco_True_linkpred_True/current.pth"
        if os.path.isfile(checkpoint_path):
            print(f"\n✅ Checkpoint found: {checkpoint_path}")
            
            # Run actual demos
            single_result = demo_single_optimization()
            batch_results = demo_batch_optimization()
            
            print(f"\n🎉 Demo completed successfully!")
            print(f"   Single post improvement: {single_result['total_improvement']:+.3f}")
            print(f"   Batch average improvement: {sum(r.get('total_improvement', 0) for r in batch_results if 'error' not in r) / len([r for r in batch_results if 'error' not in r]):+.3f}")
            
        else:
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            print("   The optimizer will work with synthetic data for demo purposes")
            print("   In production, use your actual pre-trained checkpoint")
            
            # Run demos with mock data
            single_result = demo_single_optimization()
            batch_results = demo_batch_optimization()
            
            print(f"\n🎉 Demo completed with mock data!")
            
        print(f"\n💡 Next Steps:")
        print(f"   1. Ensure your GNN checkpoint is available")
        print(f"   2. Configure LLM API keys (OpenAI/Anthropic) for production use")
        print(f"   3. Replace mock engagement/emoji tasks with actual implementations")
        print(f"   4. Test with your real Xiaohongshu dataset")
        print(f"   5. Tune hyperparameters for optimal performance")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()