#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 xhs_data.db 构建 DGL 图数据，用于利用预训练权重
"""

import sqlite3
import dgl
import torch
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import pickle
import os
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, db_path: str = "xhs_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # 词汇表
        self.post_vocab = {}  # post_id -> node_index
        self.emoji_vocab = {}  # emoji -> node_index
        self.word_vocab = {}  # word -> node_index
        
        # 边数据
        self.edges = {
            ('emoji', 'ein', 'post'): [],
            ('post', 'hase', 'emoji'): [],
            ('post', 'hasw', 'word'): [],
            ('word', 'win', 'post'): [],
            ('word', 'withe', 'emoji'): [],
            ('emoji', 'by', 'word'): []
        }
        
    def extract_emojis(self, text: str) -> List[str]:
        """从文本中提取emoji"""
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.findall(text)
    
    def extract_words(self, text: str) -> List[str]:
        """从文本中提取关键词（简化版，可以改进）"""
        # 移除emoji和特殊字符
        clean_text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]', '', text)
        # 简单分词（可以用jieba等更好的分词器）
        words = re.findall(r'[\u4e00-\u9fff]+', clean_text)
        # 过滤长度
        words = [w for w in words if 2 <= len(w) <= 10]
        return words
    
    def build_vocabularies(self, sample_size: int = 50000):
        """构建词汇表"""
        logger.info("🔨 构建词汇表...")
        
        query = """
        SELECT note_id, content, liked_count, collected_count, comments_count 
        FROM note_info 
        WHERE content IS NOT NULL AND content != ''
        ORDER BY (liked_count + collected_count + comments_count) DESC
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, self.conn, params=(sample_size,))
        logger.info(f"📊 加载了 {len(df)} 条数据")
        
        # 统计emoji和词频
        emoji_counter = Counter()
        word_counter = Counter()
        
        for idx, row in df.iterrows():
            content = row['content']
            
            # 提取emojis
            emojis = self.extract_emojis(content)
            emoji_counter.update(emojis)
            
            # 提取words
            words = self.extract_words(content)
            word_counter.update(words)
            
            # 构建post词汇表
            if row['note_id'] not in self.post_vocab:
                self.post_vocab[row['note_id']] = len(self.post_vocab)
        
        # 选择高频emoji和word
        top_emojis = [emoji for emoji, count in emoji_counter.most_common(200) if count >= 10]
        top_words = [word for word, count in word_counter.most_common(1000) if count >= 5]
        
        # 构建emoji和word词汇表
        for emoji in top_emojis:
            if emoji not in self.emoji_vocab:
                self.emoji_vocab[emoji] = len(self.emoji_vocab)
                
        for word in top_words:
            if word not in self.word_vocab:
                self.word_vocab[word] = len(self.word_vocab)
        
        logger.info(f"✅ 词汇表构建完成:")
        logger.info(f"  📝 Posts: {len(self.post_vocab)}")
        logger.info(f"  😊 Emojis: {len(self.emoji_vocab)}")
        logger.info(f"  📚 Words: {len(self.word_vocab)}")
        
        return df
    
    def build_edges(self, df: pd.DataFrame):
        """构建边关系"""
        logger.info("🔗 构建边关系...")
        
        for idx, row in df.iterrows():
            post_id = row['note_id']
            content = row['content']
            
            if post_id not in self.post_vocab:
                continue
                
            post_idx = self.post_vocab[post_id]
            
            # 提取emojis和words
            emojis = self.extract_emojis(content)
            words = self.extract_words(content)
            
            # Post-Emoji edges
            for emoji in emojis:
                if emoji in self.emoji_vocab:
                    emoji_idx = self.emoji_vocab[emoji]
                    
                    # emoji -> post
                    self.edges[('emoji', 'ein', 'post')].append((emoji_idx, post_idx))
                    # post -> emoji  
                    self.edges[('post', 'hase', 'emoji')].append((post_idx, emoji_idx))
            
            # Post-Word edges
            for word in words:
                if word in self.word_vocab:
                    word_idx = self.word_vocab[word]
                    
                    # post -> word
                    self.edges[('post', 'hasw', 'word')].append((post_idx, word_idx))
                    # word -> post
                    self.edges[('word', 'win', 'post')].append((word_idx, post_idx))
            
            # Word-Emoji co-occurrence (同一个post中的word和emoji)
            for word in words:
                if word in self.word_vocab:
                    word_idx = self.word_vocab[word]
                    for emoji in emojis:
                        if emoji in self.emoji_vocab:
                            emoji_idx = self.emoji_vocab[emoji]
                            
                            # word -> emoji
                            self.edges[('word', 'withe', 'emoji')].append((word_idx, emoji_idx))
                            # emoji -> word
                            self.edges[('emoji', 'by', 'word')].append((emoji_idx, word_idx))
        
        # 去重
        for etype in self.edges:
            self.edges[etype] = list(set(self.edges[etype]))
            
        logger.info(f"✅ 边关系构建完成:")
        for etype, edge_list in self.edges.items():
            logger.info(f"  {etype}: {len(edge_list)} edges")
    
    def create_dgl_graph(self) -> dgl.DGLHeteroGraph:
        """创建DGL异构图"""
        logger.info("🏗️ 创建DGL图...")
        
        # 准备边数据
        edge_dict = {}
        for etype, edge_list in self.edges.items():
            if edge_list:
                src_nodes, dst_nodes = zip(*edge_list)
                edge_dict[etype] = (torch.tensor(src_nodes), torch.tensor(dst_nodes))
        
        # 节点数量
        num_nodes_dict = {
            'post': len(self.post_vocab),
            'emoji': len(self.emoji_vocab), 
            'word': len(self.word_vocab)
        }
        
        # 创建图
        g = dgl.heterograph(edge_dict, num_nodes_dict)
        
        # 添加节点特征（简单的随机初始化，实际中可以用BERT等）
        g.nodes['post'].data['feat'] = torch.randn(num_nodes_dict['post'], 768)
        g.nodes['emoji'].data['feat'] = torch.randn(num_nodes_dict['emoji'], 768)
        g.nodes['word'].data['feat'] = torch.randn(num_nodes_dict['word'], 768)
        
        logger.info(f"✅ DGL图创建完成:")
        logger.info(f"  节点: {g.num_nodes()}")
        logger.info(f"  边: {g.num_edges()}")
        
        return g
    
    def save_graph_and_vocabs(self, g: dgl.DGLHeteroGraph, output_dir: str = "graph_data"):
        """保存图和词汇表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图
        graph_path = os.path.join(output_dir, "graph.bin")
        dgl.save_graphs(graph_path, [g])
        logger.info(f"💾 图已保存: {graph_path}")
        
        # 保存词汇表
        vocabs = {
            'post_vocab': self.post_vocab,
            'emoji_vocab': self.emoji_vocab,
            'word_vocab': self.word_vocab
        }
        
        vocab_path = os.path.join(output_dir, "vocabularies.pkl")
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocabs, f)
        logger.info(f"💾 词汇表已保存: {vocab_path}")
        
        return graph_path, vocab_path

def main():
    builder = GraphBuilder()
    
    # Step 1: 构建词汇表
    df = builder.build_vocabularies(sample_size=50000)
    
    # Step 2: 构建边关系
    builder.build_edges(df)
    
    # Step 3: 创建DGL图
    g = builder.create_dgl_graph()
    
    # Step 4: 保存
    graph_path, vocab_path = builder.save_graph_and_vocabs(g)
    
    logger.info("🎉 图构建完成！")
    logger.info(f"📁 图文件: {graph_path}")
    logger.info(f"📁 词汇表: {vocab_path}")
    
    return graph_path, vocab_path

if __name__ == "__main__":
    main() 