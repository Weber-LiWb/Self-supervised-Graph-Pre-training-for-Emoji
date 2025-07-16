#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在预生成的embeddings基础上训练downstream tasks的轻量级头部
完全利用预训练权重，不重新训练GNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import sqlite3
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
import argparse
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EngagementPredictor(nn.Module):
    """轻量级engagement预测头部"""
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = None, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [384, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 确保输出在[0,1]
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(embeddings)

class EmojiSuggester:
    """基于embedding相似度的emoji建议器"""
    def __init__(self, emoji_embeddings: torch.Tensor, emoji_vocab: Dict[int, str]):
        self.emoji_embeddings = F.normalize(emoji_embeddings, p=2, dim=1)
        self.emoji_vocab = emoji_vocab
        self.idx_to_emoji = {idx: emoji for emoji, idx in emoji_vocab.items()}
        
    def suggest_emojis(self, post_embedding: torch.Tensor, top_k: int = 5) -> List[str]:
        """为post embedding建议top-k emojis"""
        post_emb = F.normalize(post_embedding.unsqueeze(0), p=2, dim=1)
        
        # 计算相似度
        similarities = torch.mm(post_emb, self.emoji_embeddings.t()).squeeze()
        
        # 获取top-k
        top_k = min(top_k, len(similarities))
        _, top_indices = torch.topk(similarities, top_k)
        
        # 转换为emoji字符串
        suggested_emojis = [self.idx_to_emoji[idx.item()] for idx in top_indices]
        return suggested_emojis

class DownstreamTrainer:
    def __init__(self, embeddings_dir: str, db_path: str = "xhs_data.db"):
        self.embeddings_dir = embeddings_dir
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # 加载embeddings和词汇表
        self._load_embeddings()
        
    def _load_embeddings(self):
        """加载预生成的embeddings"""
        logger.info("📂 加载预生成的embeddings...")
        
        # 加载embeddings
        self.post_embeddings = torch.load(os.path.join(self.embeddings_dir, "post_embeddings.pt"))
        self.emoji_embeddings = torch.load(os.path.join(self.embeddings_dir, "emoji_embeddings.pt"))
        
        # 加载词汇表
        with open(os.path.join(self.embeddings_dir, "vocabularies.pkl"), 'rb') as f:
            vocabs = pickle.load(f)
        
        self.post_vocab = vocabs['post_vocab']
        self.emoji_vocab = vocabs['emoji_vocab']
        
        logger.info(f"✅ Embeddings加载完成:")
        logger.info(f"  📝 Post embeddings: {self.post_embeddings.shape}")
        logger.info(f"  😊 Emoji embeddings: {self.emoji_embeddings.shape}")
        
    def prepare_engagement_data(self, sample_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备engagement预测的训练数据"""
        logger.info("📊 准备engagement预测数据...")
        
        # 从数据库获取engagement数据（查询整表然后过滤）
        post_ids_set = set(self.post_vocab.keys())
        
        logger.info(f"  📊 查询所有engagement数据然后过滤 {len(post_ids_set)} 个posts...")
        
        try:
            # 查询整个表格
            query = """
            SELECT note_id, liked_count, collected_count, comments_count 
            FROM note_info 
            WHERE liked_count > 0 OR collected_count > 0 OR comments_count > 0
            """
            df_all = pd.read_sql_query(query, self.conn)
            logger.info(f"  📊 从数据库获取了 {len(df_all)} 条记录")
            
            # 在Python中过滤出我们需要的posts
            df = df_all[df_all['note_id'].isin(post_ids_set)]
            logger.info(f"  ✅ 匹配到 {len(df)} 条有用的记录")
            
            if len(df) == 0:
                logger.error("❌ 没有匹配的engagement数据")
                return None, None
                
        except Exception as e:
            logger.error(f"❌ 数据库查询失败: {e}")
            return None, None
        
        # 计算engagement score (归一化)
        df['engagement'] = df['liked_count'] + df['collected_count'] + df['comments_count']
        df['engagement_normalized'] = df['engagement'] / df['engagement'].max()
        
        # 匹配embedding indices
        valid_data = []
        for _, row in df.iterrows():
            if row['note_id'] in self.post_vocab:
                post_idx = self.post_vocab[row['note_id']]
                if post_idx < len(self.post_embeddings):
                    valid_data.append((post_idx, row['engagement_normalized']))
        
        if len(valid_data) < 100:
            logger.warning(f"⚠️ 只有 {len(valid_data)} 个有效样本，可能不足以训练")
        
        # 转换为tensor
        post_indices, engagement_scores = zip(*valid_data)
        X = self.post_embeddings[list(post_indices)]
        y = torch.tensor(engagement_scores, dtype=torch.float32)
        
        logger.info(f"✅ Engagement数据准备完成: {len(valid_data)} 个样本")
        return X, y
    
    def train_engagement_predictor(self, 
                                 X: torch.Tensor, 
                                 y: torch.Tensor,
                                 epochs: int = 100,
                                 lr: float = 1e-3,
                                 test_size: float = 0.2) -> Dict:
        """训练engagement预测器"""
        logger.info("🎯 训练engagement预测器...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=test_size, random_state=42
        )
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        # 创建模型
        model = EngagementPredictor(input_dim=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 训练
        best_loss = float('inf')
        history = {'train_loss': [], 'test_loss': []}
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            optimizer.zero_grad()
            
            predictions = model(X_train).squeeze()
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test).squeeze()
                test_loss = criterion(test_predictions, y_test).item()
            
            history['train_loss'].append(loss.item())
            history['test_loss'].append(test_loss)
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1:3d}: Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 最终评估
        model.eval()
        with torch.no_grad():
            final_predictions = model(X_test).squeeze()
            mse = mean_squared_error(y_test.numpy(), final_predictions.numpy())
            mae = mean_absolute_error(y_test.numpy(), final_predictions.numpy())
            r2 = r2_score(y_test.numpy(), final_predictions.numpy())
        
        results = {
            'model': model,
            'history': history,
            'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
            'best_loss': best_loss
        }
        
        logger.info(f"✅ 训练完成! MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return results
    
    def create_emoji_suggester(self) -> EmojiSuggester:
        """创建emoji建议器"""
        logger.info("😊 创建emoji建议器...")
        
        suggester = EmojiSuggester(self.emoji_embeddings, self.emoji_vocab)
        logger.info(f"✅ Emoji建议器创建完成，支持 {len(self.emoji_vocab)} 个emojis")
        
        return suggester
    
    def save_models(self, engagement_model: nn.Module, output_dir: str = "trained_models"):
        """保存训练好的模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存engagement预测器
        engagement_path = os.path.join(output_dir, "engagement_predictor.pt")
        torch.save(engagement_model.state_dict(), engagement_path)
        logger.info(f"💾 Engagement预测器已保存: {engagement_path}")
        
        # 保存配置信息
        config = {
            'embedding_dim': self.post_embeddings.shape[1],
            'num_posts': len(self.post_vocab),
            'num_emojis': len(self.emoji_vocab)
        }
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"💾 配置文件已保存: {config_path}")

def main():
    parser = argparse.ArgumentParser("训练轻量级downstream tasks")
    parser.add_argument("--embeddings", type=str, default="embeddings", help="embeddings目录")
    parser.add_argument("--db", type=str, default="xhs_data.db", help="数据库路径")
    parser.add_argument("--output", type=str, default="trained_models", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.embeddings):
        logger.error(f"❌ Embeddings目录不存在: {args.embeddings}")
        logger.error("请先运行 generate_embeddings.py 生成embeddings")
        return
    
    if not os.path.exists(args.db):
        logger.error(f"❌ 数据库不存在: {args.db}")
        return
    
    # 创建训练器
    trainer = DownstreamTrainer(args.embeddings, args.db)
    
    # 训练engagement预测器
    X, y = trainer.prepare_engagement_data()
    engagement_results = trainer.train_engagement_predictor(
        X, y, epochs=args.epochs, lr=args.lr
    )
    
    # 创建emoji建议器
    emoji_suggester = trainer.create_emoji_suggester()
    
    # 保存模型
    trainer.save_models(engagement_results['model'], args.output)
    
    # 简单测试
    logger.info("\n🧪 简单测试:")
    model = engagement_results['model']
    model.eval()
    
    # 测试engagement预测
    with torch.no_grad():
        sample_post_emb = X[0:1]
        predicted_engagement = model(sample_post_emb).item()
        logger.info(f"  📊 样本engagement预测: {predicted_engagement:.3f}")
    
    # 测试emoji建议
    suggested_emojis = emoji_suggester.suggest_emojis(X[0], top_k=5)
    logger.info(f"  😊 样本emoji建议: {' '.join(suggested_emojis)}")
    
    logger.info("🎉 所有downstream tasks训练完成！")
    logger.info(f"📁 模型已保存到: {args.output}")

if __name__ == "__main__":
    main() 