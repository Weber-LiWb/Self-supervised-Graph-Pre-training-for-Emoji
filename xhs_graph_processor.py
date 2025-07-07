#!/usr/bin/env python
# encoding: utf-8
"""
XHS Graph Data Processor
Converts XHS posts to graph format compatible with the pre-trained model
"""

import torch
import dgl
import numpy as np
import re
import jieba
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class XHSGraphProcessor:
    """Processes XHS posts into heterogeneous graphs"""
    
    def __init__(self):
        """
        Initialize processor for working with pre-trained model
        NO BERT - we work with the existing feature space
        """
        
        # Emoji pattern for extraction
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        # Initialize vocabulary mappings
        self.word_vocab = {}
        self.emoji_vocab = {}
        self.post_vocab = {}
        
    def extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text"""
        return self.emoji_pattern.findall(text)
    
    def extract_words(self, text: str) -> List[str]:
        """Extract words from Chinese text using jieba"""
        # Remove emojis first
        clean_text = self.emoji_pattern.sub('', text)
        # Tokenize Chinese text
        words = list(jieba.cut(clean_text.strip()))
        # Filter out empty strings and punctuation
        words = [w for w in words if w.strip() and len(w) > 1]
        return words
    
    def create_simple_features(self, posts: List[str]) -> torch.Tensor:
        """
        Create simple, consistent features without BERT
        This is a placeholder - ideally we'd use the same method as training
        """
        # Create simple bag-of-words style features for posts
        # This is much simpler than BERT but at least consistent
        post_features = []
        
        for post in posts:
            # Create a simple feature vector based on content properties
            features = torch.zeros(768)  # Same dimension as expected
            
            # Basic content features
            features[0] = len(post)  # Text length
            features[1] = len(self.extract_words(post))  # Word count
            features[2] = len(self.extract_emojis(post))  # Emoji count
            
            # Simple hash-based features for content
            hash_val = hash(post) % 765  # Use remaining dimensions
            features[3 + hash_val] = 1.0
            
            post_features.append(features)
        
        return torch.stack(post_features)
    
    def build_vocabulary(self, posts: List[str]):
        """Build vocabulary mappings for posts, words, and emojis"""
        all_words = set()
        all_emojis = set()
        
        for i, post in enumerate(posts):
            # Add post to vocabulary
            self.post_vocab[post] = len(self.post_vocab)
            
            # Extract and add words
            words = self.extract_words(post)
            all_words.update(words)
            
            # Extract and add emojis
            emojis = self.extract_emojis(post)
            all_emojis.update(emojis)
        
        # Build word vocabulary
        for word in all_words:
            if word not in self.word_vocab:
                self.word_vocab[word] = len(self.word_vocab)
        
        # Build emoji vocabulary
        for emoji in all_emojis:
            if emoji not in self.emoji_vocab:
                self.emoji_vocab[emoji] = len(self.emoji_vocab)
        
        logger.info(f"Vocabulary built: {len(self.post_vocab)} posts, "
                   f"{len(self.word_vocab)} words, {len(self.emoji_vocab)} emojis")
    
    def create_heterogeneous_graph(self, posts: List[str]) -> dgl.DGLGraph:
        """
        Create heterogeneous graph compatible with the pre-trained model
        Uses simple features instead of BERT to maintain compatibility
        """
        
        # Build vocabularies
        self.build_vocabulary(posts)
        
        # Initialize node counts
        num_posts = len(self.post_vocab)
        num_words = len(self.word_vocab)
        num_emojis = len(self.emoji_vocab)
        
        logger.info(f"Creating graph with {num_posts} posts, {num_words} words, {num_emojis} emojis")
        
        # Create simple node features (no BERT!)
        post_features = self.create_simple_features(posts)
        
        # Simple features for words (hash-based)
        word_features = []
        for word in self.word_vocab.keys():
            features = torch.zeros(768)
            features[0] = len(word)
            hash_val = hash(word) % 767
            features[1 + hash_val] = 1.0
            word_features.append(features)
        
        word_features = torch.stack(word_features) if word_features else torch.zeros(0, 768)
        
        # Random features for emojis (as in original paper)
        emoji_features = torch.randn(num_emojis, 768) if num_emojis > 0 else torch.zeros(0, 768)
        
        # Collect edges
        post_word_edges = []
        post_emoji_edges = []
        word_emoji_edges = []
        
        for post_text in posts:
            post_id = self.post_vocab[post_text]
            
            # Extract words and emojis
            words = self.extract_words(post_text)
            emojis = self.extract_emojis(post_text)
            
            # Post-word edges
            for word in words:
                if word in self.word_vocab:
                    word_id = self.word_vocab[word]
                    post_word_edges.append((post_id, word_id))
            
            # Post-emoji edges
            for emoji in emojis:
                if emoji in self.emoji_vocab:
                    emoji_id = self.emoji_vocab[emoji]
                    post_emoji_edges.append((post_id, emoji_id))
            
            # Word-emoji co-occurrence edges
            for word in words:
                if word in self.word_vocab:
                    word_id = self.word_vocab[word]
                    for emoji in emojis:
                        if emoji in self.emoji_vocab:
                            emoji_id = self.emoji_vocab[emoji]
                            word_emoji_edges.append((word_id, emoji_id))
        
        # Create heterogeneous graph data dict
        graph_data = {}
        
        # Add edges if they exist
        if post_word_edges:
            post_ids, word_ids = zip(*post_word_edges)
            graph_data[('post', 'hasw', 'word')] = (torch.tensor(post_ids), torch.tensor(word_ids))
            graph_data[('word', 'hasw_rev', 'post')] = (torch.tensor(word_ids), torch.tensor(post_ids))
        
        if post_emoji_edges:
            post_ids, emoji_ids = zip(*post_emoji_edges)
            graph_data[('post', 'hase', 'emoji')] = (torch.tensor(post_ids), torch.tensor(emoji_ids))
            graph_data[('emoji', 'ein', 'post')] = (torch.tensor(emoji_ids), torch.tensor(post_ids))
        
        if word_emoji_edges:
            word_ids, emoji_ids = zip(*word_emoji_edges)
            graph_data[('word', 'withe', 'emoji')] = (torch.tensor(word_ids), torch.tensor(emoji_ids))
            graph_data[('emoji', 'by', 'word')] = (torch.tensor(emoji_ids), torch.tensor(word_ids))
        
        # Create the heterogeneous graph
        if graph_data:
            g = dgl.heterograph(graph_data)
        else:
            # Create empty graph with node types
            g = dgl.heterograph({
                ('post', 'dummy', 'post'): ([], []),
                ('word', 'dummy', 'word'): ([], []),
                ('emoji', 'dummy', 'emoji'): ([], [])
            })
        
        # Set node features
        if num_posts > 0:
            g.nodes['post'].data['feat'] = post_features
        if num_words > 0:
            g.nodes['word'].data['feat'] = word_features
        if num_emojis > 0:
            g.nodes['emoji'].data['feat'] = emoji_features
        
        logger.info(f"Graph created successfully: {g}")
        return g
    
    def get_post_embedding_from_pretrained_model(self, post: str, model) -> torch.Tensor:
        """
        Get embedding for a post using the pre-trained model
        This works with the model's learned representations
        """
        try:
            # Create a minimal graph with just this post
            graph = self.create_heterogeneous_graph([post])
            
            # Use the pre-trained model to get embeddings
            with torch.no_grad():
                # Try different edge types
                if ('emoji', 'ein', 'post') in graph.canonical_etypes and graph.num_edges(('emoji', 'ein', 'post')) > 0:
                    embedding = model(graph, ('emoji', 'ein', 'post'))
                elif ('post', 'hasw', 'word') in graph.canonical_etypes and graph.num_edges(('post', 'hasw', 'word')) > 0:
                    embedding = model(graph, ('post', 'hasw', 'word'))
                else:
                    # Fallback: use the post's feature vector directly
                    if graph.num_nodes('post') > 0:
                        embedding = graph.nodes['post'].data['feat'][0]
                    else:
                        embedding = torch.zeros(768)
                        
                return embedding.squeeze()
                
        except Exception as e:
            logger.error(f"Error getting embedding with model: {e}")
            # Fallback to simple feature vector
            return self.create_simple_features([post])[0]
    
    def find_similar_emojis_using_model(self, post_content: str, model, top_k: int = 5) -> List[str]:
        """
        Find emojis semantically similar to the post using the pre-trained model
        This leverages the learned relationships without BERT
        """
        try:
            # Get post embedding using the pre-trained model
            post_embedding = self.get_post_embedding_from_pretrained_model(post_content, model)
            
            # Common emojis that the model might have learned about
            candidate_emojis = [
                '😊', '😍', '🥰', '😘', '🤗', '😄', '😆', '🤩', '✨', '💖',
                '🔥', '💯', '🎉', '🚀', '⚡', '🌟', '🎊', '🙌', '👏', '💪',
                '🌸', '🌺', '🌻', '🌷', '🌹', '🍃', '🌿', '🌱', '🌾', '🌼',
                '🍰', '🧁', '🍓', '🥭', '🍑', '🍊', '🥑', '🍯', '🍪', '☕',
                '👗', '👠', '💄', '💅', '👜', '💍', '🎀', '🕶️', '👑', '💫',
                '✈️', '🏖️', '🌊', '🗺️', '📷', '🎒', '🏨', '🌍', '🚗', '🛫',
                '👍', '👌', '💝', '🎁', '📝', '💡', '🔮', '🎯', '🌈', '⭐'
            ]
            
            # Remove emojis already in the post
            current_emojis = set(self.extract_emojis(post_content))
            candidate_emojis = [emoji for emoji in candidate_emojis if emoji not in current_emojis]
            
            # For each emoji, get its embedding and compute similarity
            emoji_similarities = []
            
            for emoji in candidate_emojis:
                try:
                    emoji_embedding = self.get_post_embedding_from_pretrained_model(emoji, model)
                    
                    # Compute cosine similarity
                    similarity = torch.cosine_similarity(
                        post_embedding.unsqueeze(0), 
                        emoji_embedding.unsqueeze(0)
                    ).item()
                    
                    emoji_similarities.append((emoji, similarity))
                    
                except Exception as e:
                    logger.debug(f"Failed to get embedding for emoji {emoji}: {e}")
                    continue
            
            # Sort by similarity and return top k
            emoji_similarities.sort(key=lambda x: x[1], reverse=True)
            top_emojis = [emoji for emoji, _ in emoji_similarities[:top_k]]
            
            logger.debug(f"Top emoji similarities: {emoji_similarities[:top_k]}")
            return top_emojis
            
        except Exception as e:
            logger.error(f"Error finding similar emojis: {e}")
            return [] 