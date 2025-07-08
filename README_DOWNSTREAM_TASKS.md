# Downstream Tasks for EMOJI Paper Implementation

This implementation reproduces the two main downstream tasks described in the EMOJI paper using the pre-trained GNN checkpoint:

1. **Engagement Prediction** - Predicting post engagement scores based on content
2. **Emoji Suggestion** - Recommending emojis to enhance post content

## 🎯 Overview

The implementation uses the pre-trained Graph Contrastive Coding (GCC) model checkpoint to extract meaningful embeddings for posts and emojis, then applies task-specific heads for each downstream application.

## 📁 Project Structure

```
downstream/
├── tasks/
│   ├── __init__.py
│   ├── base_downstream_task.py      # Base class for all downstream tasks
│   ├── engagement_prediction.py     # Engagement prediction implementation
│   └── emoji_suggestion.py          # Emoji suggestion implementation
├── utils/
│   ├── __init__.py
│   ├── data_utils.py               # Data loading and synthetic data generation
│   └── evaluation_utils.py         # Evaluation metrics and utilities
├── reproduce_downstream_demo.py    # Complete demonstration script
└── README_DOWNSTREAM_TASKS.md     # This documentation
```

## 🚀 Quick Start

### Prerequisites

Ensure you have the required dependencies:
- PyTorch
- DGL (Deep Graph Library)
- scikit-learn
- numpy

### Running the Demo

```bash
# Run complete demonstration with synthetic data
python reproduce_downstream_demo.py --checkpoint moco_True_linkpred_True/current.pth

# Run only engagement prediction
python reproduce_downstream_demo.py --checkpoint moco_True_linkpred_True/current.pth --task engagement

# Run only emoji suggestion  
python reproduce_downstream_demo.py --checkpoint moco_True_linkpred_True/current.pth --task emoji

# Run on CPU
python reproduce_downstream_demo.py --checkpoint moco_True_linkpred_True/current.pth --device cpu
```

## 📊 Task 1: Engagement Prediction

### Approach

The engagement prediction task uses the pre-trained GNN to generate post embeddings, then trains a neural network head to predict engagement scores.

**Architecture:**
- **Input**: Post embeddings from pre-trained GNN (typically 128-512 dimensions)
- **Head**: Multi-layer perceptron with configurable hidden layers
- **Output**: Single engagement score in [0, 1] range
- **Loss**: Mean Squared Error (MSE)

### Usage

```python
from downstream.tasks import EngagementPredictionTask

# Initialize task
task = EngagementPredictionTask(
    checkpoint_path="moco_True_linkpred_True/current.pth",
    device="cuda:0",
    hidden_dims=[64, 32],  # MLP architecture
    learning_rate=1e-3
)

# Train on your data
history = task.train(
    post_embeddings=your_post_embeddings,    # [num_posts, embedding_dim]
    engagement_scores=your_engagement_scores, # [num_posts]
    num_epochs=100,
    batch_size=32
)

# Make predictions
predictions = task.predict(new_post_embeddings)

# Evaluate
metrics = task.evaluate(test_embeddings, test_scores)
```

### Expected Outputs

The engagement prediction task outputs:
- **Engagement Score**: Float in [0, 1] representing predicted engagement level
- **Metrics**: R², RMSE, MAE for model evaluation

### Fine-tuning Options

1. **Architecture**: Adjust `hidden_dims` for the prediction head
2. **Regularization**: Modify `dropout` parameter
3. **Training**: Tune `learning_rate`, `batch_size`, and `num_epochs`
4. **Early Stopping**: Configure `early_stopping_patience`

## 🎯 Task 2: Emoji Suggestion

### Approach

The emoji suggestion task uses similarity-based ranking between post and emoji embeddings in the learned representation space.

**Architecture:**
- **Input**: Post embeddings and emoji embeddings from pre-trained GNN
- **Method**: Cosine similarity (or dot product/euclidean distance)
- **Output**: Ranked list of top-k emoji suggestions
- **Features**: Temperature scaling for similarity scores

### Usage

```python
from downstream.tasks import EmojiSuggestionTask

# Initialize task
task = EmojiSuggestionTask(
    checkpoint_path="moco_True_linkpred_True/current.pth",
    device="cuda:0",
    similarity_metric='cosine',  # 'cosine', 'dot', 'euclidean'
    temperature=1.0
)

# Setup embeddings (computed from your graph data)
task.setup_embeddings(
    post_embeddings=your_post_embeddings,     # [num_posts, embedding_dim]
    emoji_embeddings=your_emoji_embeddings,   # [num_emojis, embedding_dim]
    emoji_vocab=your_emoji_vocabulary         # {idx: emoji_str}
)

# Get suggestions for a single post
suggestions = task.suggest_emojis(post_embedding, top_k=5)

# Get suggestions with scores
suggestions_with_scores = task.suggest_emojis_with_explanation(post_embedding, top_k=5)

# Batch suggestions
batch_suggestions = task.batch_suggest_emojis(post_embeddings_batch, top_k=5)

# Evaluate on test data
metrics = task.evaluate(test_posts, test_emojis, top_k_list=[1, 3, 5, 10])
```

### Expected Outputs

The emoji suggestion task outputs:
- **Emoji List**: Ranked list of suggested emojis (e.g., `["😍", "💯", "🔥"]`)
- **Scores**: Similarity scores for each suggestion
- **Metrics**: Precision@k, Recall@k, F1@k, NDCG@k for evaluation

### Customization Options

1. **Similarity Metric**: Choose between 'cosine', 'dot', or 'euclidean'
2. **Temperature Scaling**: Adjust temperature for similarity score calibration
3. **Top-k**: Configure number of suggestions returned
4. **Filtering**: Add domain-specific emoji filtering logic

## 🔧 Integration with Existing Checkpoint

### Loading the Pre-trained Model

The implementation automatically handles checkpoint loading:

```python
# The BaseDownstreamTask class handles this automatically
task = EngagementPredictionTask(checkpoint_path="moco_True_linkpred_True/current.pth")

# Access checkpoint arguments
print(task.checkpoint_args.hidden_size)  # Embedding dimension
print(task.checkpoint_args.num_layer)    # Number of GNN layers
```

### Embedding Generation

For real data usage, you need to:

1. **Prepare Graph Data**: Convert your posts/emojis into DGL graph format
2. **Generate Embeddings**: Use the pre-trained model to extract embeddings
3. **Apply Tasks**: Use the embeddings with the downstream task implementations

```python
# Generate embeddings for posts
post_etype = ('emoji', 'ein', 'post')
post_metapath = ['ein', 'hase']
post_embeddings = task.generate_embeddings(post_etype, post_metapath)

# Generate embeddings for emojis  
emoji_etype = ('post', 'hase', 'emoji')
emoji_metapath = ['hase', 'ein']
emoji_embeddings = task.generate_embeddings(emoji_etype, emoji_metapath)
```

## 📈 Evaluation Metrics

### Engagement Prediction Metrics

- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

### Emoji Suggestion Metrics

- **Precision@k**: Fraction of suggested emojis that are relevant
- **Recall@k**: Fraction of relevant emojis that are suggested
- **F1@k**: Harmonic mean of precision and recall
- **NDCG@k**: Normalized Discounted Cumulative Gain (ranking quality)

## 🔍 Expected Performance

### Synthetic Data Results (Demo)

The demo script uses synthetic data to demonstrate functionality:

- **Engagement Prediction**: R² ≈ 0.6-0.8 (varies with synthetic data complexity)
- **Emoji Suggestion**: Precision@5 ≈ 0.2-0.4 (depends on emoji vocabulary size)

### Real Data Expectations

With actual Xiaohongshu data, you should expect:

- **Engagement Prediction**: Performance depends on data quality and feature richness
- **Emoji Suggestion**: Performance depends on emoji-post co-occurrence patterns

## 🚧 Limitations and Considerations

1. **Data Dependency**: Performance heavily depends on the quality of the original graph construction
2. **Domain Adaptation**: May require fine-tuning for different social media platforms
3. **Embedding Quality**: Results depend on the pre-trained GNN checkpoint quality
4. **Cold Start**: New posts/emojis not seen during pre-training may have poor embeddings

## 🛠 Extending the Implementation

### Adding New Tasks

Create a new task by inheriting from `BaseDownstreamTask`:

```python
from downstream.tasks.base_downstream_task import BaseDownstreamTask

class NewDownstreamTask(BaseDownstreamTask):
    def setup_task_head(self, **kwargs):
        # Setup your task-specific components
        pass
    
    def train(self, **kwargs):
        # Implement training logic
        pass
    
    def evaluate(self, **kwargs):
        # Implement evaluation logic
        pass
```

### Custom Evaluation Metrics

Add new metrics to `evaluation_utils.py`:

```python
def custom_metric(predictions, ground_truth):
    # Your custom metric implementation
    return metric_value
```

## 📝 Citation

If you use this implementation in your research, please cite the original EMOJI paper and this implementation.

## 🤝 Contributing

Feel free to submit issues and pull requests to improve the implementation.

## 📧 Support

For questions about the implementation, please check the code comments and this documentation first. The demo script provides comprehensive examples of usage.