#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用预训练的GNN权重生成post和emoji embeddings
简化版本：直接使用图中的节点特征，避免复杂的采样
"""

import torch
import dgl
import pickle
import os
import argparse
from typing import Dict, Tuple
import logging
import numpy as np

from gcc.models import GraphEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, checkpoint_path: str, graph_path: str, vocab_path: str, device: str = 'cpu'):
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path
        self.vocab_path = vocab_path
        self.device = torch.device(device) if torch.cuda.is_available() and device != 'cpu' else torch.device('cpu')
        
        # 加载预训练权重
        self.args, self.model = self._load_checkpoint()
        
        # 加载词汇表
        with open(vocab_path, 'rb') as f:
            self.vocabs = pickle.load(f)
        
        # 加载图
        self.graph = self._load_graph()
        
        logger.info(f"✅ 加载完成:")
        logger.info(f"  📝 Posts: {len(self.vocabs['post_vocab'])}")
        logger.info(f"  😊 Emojis: {len(self.vocabs['emoji_vocab'])}")
        logger.info(f"  📚 Words: {len(self.vocabs['word_vocab'])}")
        logger.info(f"  🔗 图节点: post={self.graph.num_nodes('post')}, emoji={self.graph.num_nodes('emoji')}, word={self.graph.num_nodes('word')}")
        
    def _load_checkpoint(self) -> Tuple:
        """加载预训练权重"""
        logger.info(f"🔄 加载预训练权重: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        args = checkpoint["opt"]
        
        # 处理可能缺失的参数
        if not hasattr(args, 'node_feat_dim'):
            args.node_feat_dim = 768  # 默认值，与train.py保持一致
            logger.info("  ⚠️ 添加缺失参数: node_feat_dim = 768")
        
        # 创建模型
        model = GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_degree=args.max_degree,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            node_feat_dim=args.node_feat_dim,
            num_layers=args.num_layer,
            norm=args.norm,
            degree_input=True,
        )
        
        # 加载权重
        model.load_state_dict(checkpoint["model"])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"✅ 模型加载成功 (epoch {checkpoint['epoch']})")
        logger.info(f"  📏 Hidden size: {args.hidden_size}")
        logger.info(f"  🏗️ Architecture: {args.model} ({args.num_layer} layers)")
        
        del checkpoint
        return args, model
        
    def _load_graph(self):
        """加载图数据"""
        logger.info(f"📊 加载图数据: {self.graph_path}")
        graphs, _ = dgl.load_graphs(self.graph_path)
        graph = graphs[0].to(self.device)
        return graph
    
    def generate_node_embeddings(self, node_type: str, batch_size: int = 1000):
        """
        为指定节点类型生成embeddings
        使用图的全局结构，不进行子图采样
        """
        logger.info(f"🎯 生成 {node_type} embeddings...")
        
        num_nodes = self.graph.num_nodes(node_type)
        if num_nodes == 0:
            logger.warning(f"⚠️ {node_type} 节点数为0，跳过")
            return torch.empty(0, self.args.hidden_size)
        
        # 创建单节点的简单"子图"用于提取embeddings
        all_embeddings = []
        
        with torch.no_grad():
            for start_idx in range(0, num_nodes, batch_size):
                end_idx = min(start_idx + batch_size, num_nodes)
                batch_indices = list(range(start_idx, end_idx))
                
                # 为每个节点创建一个简单的单节点图
                batch_embeddings = []
                for node_idx in batch_indices:
                    # 创建只包含这个节点的简单图
                    node_subgraph = self._create_simple_node_graph(node_type, node_idx)
                    
                    if node_subgraph is not None:
                        # 使用模型提取embedding
                        # 需要提供edtype参数，根据原图的边类型确定
                        edtype = self.graph.canonical_etypes[0]  # 使用第一个边类型作为默认
                        embedding = self.model(node_subgraph, edtype)
                        # 取第一个节点的embedding（因为我们只有一个节点）
                        batch_embeddings.append(embedding[0])
                    else:
                        # 如果无法创建子图，使用零向量
                        batch_embeddings.append(torch.zeros(self.args.hidden_size, device=self.device))
                
                if batch_embeddings:
                    all_embeddings.append(torch.stack(batch_embeddings))
                
                if (start_idx // batch_size) % 10 == 0:
                    logger.info(f"  📈 已处理 {end_idx}/{num_nodes} 个{node_type}节点")
        
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            logger.info(f"✅ {node_type} embeddings 完成: {embeddings.shape}")
            return embeddings
        else:
            logger.warning(f"⚠️ 无法生成 {node_type} embeddings")
            return torch.empty(0, self.args.hidden_size)
    
    def _create_simple_node_graph(self, node_type: str, node_idx: int):
        """
        为单个节点创建一个简单的图，用于embedding提取
        直接使用节点特征，避免复杂的采样
        """
        try:
            # 直接使用节点特征创建最小图
            if 'feat' in self.graph.nodes[node_type].data:
                # 直接使用原图中该节点的特征
                node_feat = self.graph.nodes[node_type].data['feat'][node_idx]
                
                # 创建一个包含单节点和自环的最小图
                edge_dict = {('temp', 'self', 'temp'): ([0], [0])}
                minimal_graph = dgl.heterograph(edge_dict, num_nodes_dict={'temp': 1}, device=self.device)
                minimal_graph.nodes['temp'].data['feat'] = node_feat.unsqueeze(0)
                minimal_graph.nodes['temp'].data['seed'] = torch.zeros(1, dtype=torch.long, device=self.device)
                
                # 转换为同构图
                homo_graph = dgl.to_homogeneous(minimal_graph, ndata=['feat', 'seed'])
                
                # 添加位置embedding
                if not hasattr(self.args, 'positional_embedding_size'):
                    self.args.positional_embedding_size = 32
                    
                pos_emb_size = getattr(self.args, 'positional_embedding_size', 32)
                homo_graph.ndata['pos_undirected'] = torch.randn(homo_graph.num_nodes(), pos_emb_size, device=self.device)
                
                return homo_graph
            else:
                logger.warning(f"节点 {node_type}[{node_idx}] 没有特征，跳过")
                return None
            
        except Exception as e:
            logger.error(f"创建图失败 {node_type}[{node_idx}]: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return None
    
    def generate_all_embeddings(self, batch_size: int = 1000, output_dir: str = "embeddings"):
        """生成所有类型的embeddings"""
        logger.info("🚀 开始生成所有embeddings...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings = {}
        
        # 生成各种类型的embeddings
        node_types = ['post', 'emoji', 'word']
        for node_type in node_types:
            if self.graph.num_nodes(node_type) > 0:
                emb = self.generate_node_embeddings(node_type, batch_size)
                embeddings[f'{node_type}_embeddings'] = emb
                
                # 保存到文件
                torch.save(emb, os.path.join(output_dir, f"{node_type}_embeddings.pt"))
                logger.info(f"💾 {node_type} embeddings 已保存: {emb.shape}")
        
        # 复制词汇表到输出目录
        vocab_output_path = os.path.join(output_dir, "vocabularies.pkl")
        with open(vocab_output_path, 'wb') as f:
            pickle.dump(self.vocabs, f)
        logger.info(f"💾 词汇表已复制到: {vocab_output_path}")
        
        logger.info("🎉 所有embeddings生成完成！")
        return embeddings


def main():
    parser = argparse.ArgumentParser("使用预训练权重生成embeddings")
    parser.add_argument("--checkpoint", type=str, default="moco_True_linkpred_True/current.pth", help="预训练权重路径")
    parser.add_argument("--graph", type=str, default="graph_data/graph.bin", help="图数据路径")
    parser.add_argument("--vocab", type=str, default="graph_data/vocabularies.pkl", help="词汇表路径")
    parser.add_argument("--output", type=str, default="embeddings", help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1000, help="批次大小")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for path in [args.checkpoint, args.graph, args.vocab]:
        if not os.path.exists(path):
            logger.error(f"❌ 文件不存在: {path}")
            return
    
    # 生成embeddings
    generator = EmbeddingGenerator(
        checkpoint_path=args.checkpoint,
        graph_path=args.graph,
        vocab_path=args.vocab,
        device=args.device
    )
    
    embeddings = generator.generate_all_embeddings(
        batch_size=args.batch_size,
        output_dir=args.output
    )
    
    logger.info("✨ 完成！现在可以使用这些embeddings进行downstream tasks了。")


if __name__ == "__main__":
    main() 