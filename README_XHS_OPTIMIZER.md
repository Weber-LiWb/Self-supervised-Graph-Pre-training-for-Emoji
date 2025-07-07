# 🚀 XHS Engagement Optimization Pipeline

**An AI-powered iterative content optimization system for Xiaohongshu (XHS) posts that combines graph neural network predictions with LLM-based content generation.**

## 🎯 Overview

This system implements an innovative pipeline that:

1. **📊 Predicts engagement scores** using a pre-trained graph neural network
2. **🎨 Generates optimal emoji suggestions** based on semantic relationships
3. **🔄 Iteratively optimizes content** using LLM prompting until reaching target engagement
4. **📈 Provides actionable insights** for content creators and social media managers

## 🏗️ Architecture

```
XHS Post → Graph Neural Network → Engagement Prediction
                                        ↓
Emoji Generation ← Content Analysis → LLM Optimization
        ↓                                     ↓
   Iterative Loop ←←←←←← Convergence Check ←←←←←
```

### Core Components

- **`EngagementPredictor`**: Uses pre-trained graph model to predict engagement scores
- **`LLMOptimizer`**: Leverages GPT-4 for intelligent content optimization
- **`XHSGraphProcessor`**: Converts posts to heterogeneous graph format
- **`XHSEngagementOptimizer`**: Main pipeline orchestrator

## 🔬 Technical Foundation

Based on the research paper "Unleashing the Power of Emojis in Texts via Self-supervised Graph Pre-Training", this system uses:

- **Heterogeneous Graph Structure**: Post ↔ Word ↔ Emoji relationships
- **Self-supervised Pre-training**: MoCo + Link Prediction tasks
- **Semantic Emoji Understanding**: Context-aware emoji placement
- **Iterative Optimization**: Convergence-based content improvement

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install torch dgl transformers openai jieba

# Ensure you have the pre-trained checkpoint
ls moco_True_linkpred_True/current.pth
```

### 2. Basic Usage

```python
from xhs_engagement_optimizer import XHSEngagementOptimizer, XHSPost

# Initialize optimizer
optimizer = XHSEngagementOptimizer(
    checkpoint_path="moco_True_linkpred_True/current.pth",
    openai_api_key="your-openai-api-key",
    device="cpu"  # or "cuda"
)

# Create a post
post = XHSPost(content="今天去了一个很棒的咖啡店，咖啡很香，环境也很好")

# Optimize until target engagement
results = optimizer.optimize_post(
    post=post,
    target_score=0.8,  # Target engagement score (0-1)
    max_iterations=5   # Maximum optimization rounds
)

# View results
for result in results:
    print(f"Iteration {result.iteration}:")
    print(f"  Content: {result.optimized_content}")
    print(f"  Score: {result.predicted_score:.3f}")
    print(f"  Improvement: {result.improvement:+.3f}")
```

### 3. V2 Enhanced Usage (Recommended)

The V2 optimizer supports multiple ways to select posts and uses real XHS database vocabulary:

```bash
# Interactive post selection (recommended)
python xhs_engagement_optimizer_v2.py --interactive

# Select specific posts by database ID
python xhs_engagement_optimizer_v2.py --post-numbers '1,5,10,25'

# Use environment API key (automatic detection)
export FDU_API_KEY='your-openai-api-key'
python xhs_engagement_optimizer_v2.py --interactive

# Full optimization with custom settings
python xhs_engagement_optimizer_v2.py \
    --post-numbers '10,20,30' \
    --target-score 0.85 \
    --max-iterations 3 \
    --output-results my_results.json

# Prediction-only mode (no API key needed)
python xhs_engagement_optimizer_v2.py --post-numbers '1,2,3'
```

### 4. V1 Command Line Usage (Legacy)

```bash
# Run with custom posts file
python xhs_engagement_optimizer.py \
    --openai-api-key sk-your-key \
    --input-posts sample_posts.json \
    --target-score 0.8 \
    --max-iterations 5

# Run demo
python demo_xhs_optimizer.py
```

## 📊 Features

### 🆕 V2 Enhancements (Latest)
- **🎯 Manual Post Selection** - Choose specific posts from 1.9M+ database entries
- **🎮 Interactive Mode** - Visual post browser with preview and selection
- **🔤 Real Vocabulary** - 10,000+ words and 5,000+ emojis from actual XHS data
- **🌟 No Hard-coding** - Eliminated all hard-coded emoji lists
- **🔧 Better Features** - Proper TF-IDF matching original training data
- **🔑 Auto API Key** - Automatic detection from FDU_API_KEY environment
- **📊 Quality Metrics** - Feature quality assessment and monitoring
- **🎛️ Prediction Mode** - Works without API key for analysis only

### ✅ Core Engagement Prediction
- **Graph-based modeling** of post-word-emoji relationships
- **Pre-trained embeddings** from self-supervised learning
- **Multi-factor scoring** considering content, emojis, and structure

### ✅ Emoji Generation
- **Semantic matching** based on content analysis
- **Database-driven suggestions** from real XHS emoji usage
- **Context-sensitive placement** recommendations

### ✅ LLM Optimization
- **Iterative refinement** using GPT-4
- **XHS-specific prompting** for platform optimization
- **Convergence-based stopping** criteria

### ✅ Production Ready
- **Batch processing** for multiple posts
- **Error handling** with graceful fallbacks
- **Comprehensive logging** and monitoring
- **JSON export** for results analysis

## 📈 Example Optimization

**Original Post:**
```
今天去了一个很棒的咖啡店，咖啡很香，环境也很好
```

**Iteration 1 (Score: 0.45 → 0.62):**
```
今天去了一个超棒的咖啡店☕，咖啡真的很香😊，环境也特别好✨
```

**Iteration 2 (Score: 0.62 → 0.78):**
```
今天发现了一家宝藏咖啡店☕！咖啡香气扑鼻😍，环境超级温馨✨，完美的下午时光💕
```

**Final (Score: 0.78 → 0.85):**
```
分享一家绝美咖啡店☕！香浓咖啡配绝佳环境😍，每一口都是享受✨，姐妹们快来打卡吧💕🔥
```

## 🎯 Optimization Strategies

The system applies multiple optimization techniques:

### Content Enhancement
- **Emotional amplification** (好 → 超好, 很棒 → 绝美)
- **Engagement triggers** (分享, 推荐, 快来)
- **Community language** (姐妹们, 宝藏, 打卡)

### Emoji Optimization
- **Strategic placement** for maximum emotional impact
- **Semantic relevance** to content themes
- **Quantity balancing** (2-4 emojis optimal)

### Structural Improvement
- **Hook creation** (发现了, 分享)
- **Call-to-action** (快来, 试试)
- **Social proof** (绝美, 完美)

## 🛠️ Customization

### Custom Emoji Categories
```python
custom_emojis = {
    'beauty': ['💄', '💅', '✨', '💖'],
    'food': ['🍰', '☕', '🥘', '🍓'],
    'travel': ['✈️', '🏖️', '📷', '🌍']
}

# Add to EngagementPredictor.generate_emoji_suggestions()
```

### Custom Scoring Weights
```python
def custom_engagement_score(post_content):
    score = 0.3  # Base
    
    # Custom factors
    if '种草' in post_content: score += 0.2  # Planting grass
    if '测评' in post_content: score += 0.15  # Review
    if '教程' in post_content: score += 0.1   # Tutorial
    
    return min(score, 1.0)
```

### Custom LLM Prompts
```python
def create_custom_prompt(content, score, target):
    return f"""
    优化这个小红书内容以提高互动率：
    原内容：{content}
    当前评分：{score}
    目标评分：{target}
    
    要求：
    1. 保持真实性
    2. 增加互动性
    3. 符合小红书风格
    """
```

## 📊 Performance Metrics

The system tracks multiple engagement indicators:

- **Engagement Score**: 0-1 composite score
- **Emoji Effectiveness**: Semantic relevance scoring
- **Content Quality**: Readability and appeal metrics
- **Optimization Speed**: Iterations to convergence

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
export HF_CACHE_DIR="/path/to/cache"  # For Transformers
export TORCH_HOME="/path/to/torch"    # For PyTorch models
```

### Model Configuration
```python
# GPU optimization
optimizer = XHSEngagementOptimizer(
    checkpoint_path="moco_True_linkpred_True/current.pth",
    device="cuda",  # Use GPU
    openai_api_key=api_key
)

# Custom parameters
results = optimizer.optimize_post(
    post=post,
    target_score=0.85,      # Higher target
    max_iterations=10,      # More iterations
    min_improvement=0.005   # Finer convergence
)
```

## 🚨 Error Handling

The system includes robust error handling:

- **Model loading failures**: Graceful degradation to heuristics
- **API failures**: Retry logic with exponential backoff
- **Graph construction errors**: Fallback to simplified representations
- **Memory issues**: Batch size adaptation

## 📝 Input Formats

### Single Post
```python
post = XHSPost(
    content="你的内容",
    engagement_score=0.6,  # Optional: actual score
    likes=100,            # Optional: metrics
    comments=20,
    shares=5
)
```

### Batch Posts (JSON)
```json
[
    {"content": "第一条内容"},
    {"content": "第二条内容"},
    {"content": "第三条内容"}
]
```

### CSV Format
```csv
content,likes,comments,shares
今天的穿搭分享,150,30,10
美食推荐来啦,200,45,15
```

## 🔍 Monitoring & Analytics

### Real-time Monitoring
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor optimization progress
for result in results:
    print(f"Score progression: {result.original_score:.3f} → {result.predicted_score:.3f}")
    print(f"Emoji changes: {result.emoji_suggestions}")
```

### Analytics Dashboard
```python
# Generate optimization report
report = optimizer.generate_report(results)
print(f"Average improvement: {report['avg_improvement']:.3f}")
print(f"Best performing emojis: {report['top_emojis']}")
print(f"Optimization efficiency: {report['iterations_per_score']:.2f}")
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research paper: "Unleashing the Power of Emojis in Texts via Self-supervised Graph Pre-Training"
- Inspired by [GCC: Graph Contrastive Coding](https://github.com/THUDM/GCC)
- Built with [DGL](https://dgl.ai/), [PyTorch](https://pytorch.org/), and [Transformers](https://huggingface.co/transformers/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

---

**🎉 Start optimizing your XHS content today and boost engagement with AI-powered insights!**

## ⚡ **Key Technical Innovation**

### **✅ Leveraging Learned Embeddings (Correct Approach)**

The system now properly uses the **pre-trained graph model's learned representations**:

1. **Post Embeddings**: Uses the trained GNN to generate semantic embeddings for posts
2. **Emoji Embeddings**: Extracts emoji embeddings learned during pre-training  
3. **Similarity Computation**: Computes cosine similarity between learned post and emoji embeddings
4. **Semantic Ranking**: Ranks emojis by their learned semantic relationship to the post content

### **❌ What We Avoided (Previous Mistakes)**

- ~~Hard-coded emoji categories~~ 
- ~~Manual keyword-to-emoji mapping~~
- ~~Rule-based engagement scoring~~
- ~~Ignoring the sophisticated relationships learned during training~~

## Model Integration Details

### **Graph Neural Network Integration**

The system leverages your successful pre-training experiment through:

```python
# Get post embedding using learned relationships
post_embedding = model(graph, edge_type)

# Get emoji embeddings using the same learned representations  
emoji_embedding = model(emoji_graph, edge_type)

# Compute semantic similarity using learned embeddings
similarity = torch.cosine_similarity(post_embedding, emoji_embedding)
```

### **Edge Type Priority**

The system uses edge types in order of importance (based on your research):
1. `('emoji', 'ein', 'post')` - Direct emoji-post relationships
2. `('post', 'hasw', 'word')` - Post-word relationships  
3. `('word', 'withe', 'emoji')` - Word-emoji co-occurrence
4. Direct BERT features as fallback

### **Learned vs Hard-coded Relationships**

| Aspect | ❌ Hard-coded (Wrong) | ✅ Learned (Correct) |
|--------|---------------------|---------------------|
| Emoji Selection | Manual categories | Cosine similarity of embeddings |
| Engagement Prediction | Rule-based scoring | Neural network on learned features |
| Text-Emoji Relationships | Keyword matching | Graph neural network learned associations |
| Semantic Understanding | Static rules | Dynamic learned representations | 