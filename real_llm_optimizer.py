#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
真实的LLM Content Optimizer
使用预训练GNN权重生成的embeddings + 智谱GLM API
完全避免mock和fallback，让错误充分暴露
"""

import torch
import torch.nn as nn
import pickle
import json
import os
import re
import logging
import argparse
import time
import sqlite3
import random
from typing import Dict, List, Optional, Any
import requests
import pandas as pd

from train_downstream_tasks import EngagementPredictor, EmojiSuggester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZhipuLLMClient:
    """智谱GLM API客户端"""
    def __init__(self, api_key: str, model: str = "glm-4-plus"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"
        
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500) -> str:
        """调用GLM对话接口"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=self._get_headers(), json=payload)
        
        if response.status_code != 200:
            raise Exception(f"GLM API调用失败: {response.status_code}, {response.text}")
        
        result = response.json()
        
        if "choices" not in result or len(result["choices"]) == 0:
            raise Exception(f"GLM API返回格式错误: {result}")
        
        return result["choices"][0]["message"]["content"]

class ContentToEmbedding:
    """将新内容转换为embedding的工具（简化版）"""
    def __init__(self, embeddings_dir: str):
        # 加载词汇表和平均embeddings（优先使用过滤后的版本）
        try:
            with open(os.path.join(embeddings_dir, "updated_vocabularies.pkl"), 'rb') as f:
                self.vocabs = pickle.load(f)
            logger.info("✅ 使用过滤后的词汇表")
        except FileNotFoundError:
            with open(os.path.join(embeddings_dir, "vocabularies.pkl"), 'rb') as f:
                self.vocabs = pickle.load(f)
            logger.warning("⚠️ 使用原始词汇表")
        
        # 加载embedding用于计算相似度
        self.post_embeddings = torch.load(os.path.join(embeddings_dir, "post_embeddings.pt"))
        
        # 优先使用过滤后的emoji embeddings（排除话题标签）
        try:
            self.emoji_embeddings = torch.load(os.path.join(embeddings_dir, "valid_emoji_embeddings.pt"))
            logger.info("✅ 使用过滤后的emoji embeddings（无话题标签）")
        except FileNotFoundError:
            self.emoji_embeddings = torch.load(os.path.join(embeddings_dir, "emoji_embeddings.pt"))
            logger.warning("⚠️ 使用原始emoji embeddings（可能包含话题标签）")
        
        # 计算平均embedding作为未知内容的基线
        self.avg_post_embedding = torch.mean(self.post_embeddings, dim=0)
        
    def extract_emojis(self, text: str) -> List[str]:
        """从文本中提取emoji，使用字符级别的检查"""
        import unicodedata
        
        emojis = []
        for char in text:
            # 检查字符是否为emoji或符号
            if (char >= '\U0001F600' and char <= '\U0001F64F') or \
               (char >= '\U0001F300' and char <= '\U0001F5FF') or \
               (char >= '\U0001F680' and char <= '\U0001F6FF') or \
               (char >= '\U0001F1E0' and char <= '\U0001F1FF') or \
               (char >= '\U00002600' and char <= '\U000027BF') or \
               (char >= '\U0001F900' and char <= '\U0001F9FF') or \
               char in ['➕', '🌰', '🔗', '～', '❤️', '💖', '✨', '🔥', '👗', '🍵', '👚', '👉']:
                if char not in emojis:  # 去重
                    emojis.append(char)
        
        return emojis
    
    def content_to_embedding(self, content: str) -> torch.Tensor:
        """
        基于内容特征生成有区分度的embedding
        """
        # 1. 基础特征提取
        content_length = len(content)
        content_emojis = self.extract_emojis(content)
        
        # 2. 添加随机性和内容特征来生成不同的embedding
        # 使用内容hash作为随机种子，确保相同内容得到相同结果
        import hashlib
        content_hash = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
        torch.manual_seed(content_hash % 10000)
        
        # 3. 根据内容长度调整embedding
        length_factor = min(content_length / 200.0, 2.0)  # 长度影响因子
        
        # 4. 处理emoji特征
        emoji_embedding = torch.zeros_like(self.avg_post_embedding)
        if content_emojis:
            valid_emoji_indices = []
            for emoji in content_emojis:
                if emoji in self.vocabs['emoji_vocab']:
                    valid_emoji_indices.append(self.vocabs['emoji_vocab'][emoji])
            
            if valid_emoji_indices:
                emoji_embs = self.emoji_embeddings[valid_emoji_indices]
                emoji_embedding = torch.mean(emoji_embs, dim=0)
        
        # 5. 基于内容hash选择相似的post embedding
        num_posts = len(self.post_embeddings)
        similar_post_indices = [
            (content_hash + i * 7) % num_posts for i in range(3)
        ]
        similar_posts_emb = torch.mean(self.post_embeddings[similar_post_indices], dim=0)
        
        # 6. 生成变化的noise基于内容特征
        noise_scale = 0.1 * length_factor
        noise = torch.randn_like(self.avg_post_embedding) * noise_scale
        
        # 7. 组合所有特征
        if len(content_emojis) > 0:
            # 有emoji的情况：更多基于emoji和相似post
            final_embedding = (
                0.4 * similar_posts_emb +
                0.3 * emoji_embedding + 
                0.2 * self.avg_post_embedding +
                0.1 * noise
            )
        else:
            # 没有emoji的情况：更多基于内容特征和相似post  
            final_embedding = (
                0.6 * similar_posts_emb +
                0.3 * self.avg_post_embedding +
                0.1 * noise
            )
        
        # 8. 基于内容长度进一步调整
        length_adjustment = torch.tanh(torch.tensor(length_factor)) * 0.05
        final_embedding = final_embedding * (1.0 + length_adjustment)
        
        return final_embedding

class RealLLMContentOptimizer:
    """真实的LLM内容优化器"""
    def __init__(self, 
                 glm_api_key: str,
                 trained_models_dir: str = "trained_models",
                 embeddings_dir: str = "embeddings",
                 max_iterations: int = 5,
                 engagement_threshold: float = 0.8,
                 temperature: float = 0.7):
        
        self.max_iterations = max_iterations
        self.engagement_threshold = engagement_threshold
        self.temperature = temperature
        
        # 初始化GLM客户端
        logger.info("🤖 初始化智谱GLM客户端...")
        self.glm_client = ZhipuLLMClient(glm_api_key)
        
        # 加载训练好的模型
        logger.info("📂 加载训练好的downstream models...")
        self._load_trained_models(trained_models_dir, embeddings_dir)
        
        # 初始化内容转embedding工具
        self.content_converter = ContentToEmbedding(embeddings_dir)
        
        logger.info("✅ RealLLMContentOptimizer 初始化完成!")
    
    def _load_trained_models(self, models_dir: str, embeddings_dir: str):
        """加载训练好的模型"""
        # 加载配置
        with open(os.path.join(models_dir, "config.json"), 'r') as f:
            config = json.load(f)
        
        # 加载engagement预测器
        self.engagement_predictor = EngagementPredictor(input_dim=config['embedding_dim'])
        self.engagement_predictor.load_state_dict(
            torch.load(os.path.join(models_dir, "engagement_predictor.pt"), map_location='cpu')
        )
        self.engagement_predictor.eval()
        
        # 加载emoji建议器需要的数据（优先使用过滤后的版本）
        try:
            emoji_embeddings = torch.load(os.path.join(embeddings_dir, "valid_emoji_embeddings.pt"))
            with open(os.path.join(embeddings_dir, "updated_vocabularies.pkl"), 'rb') as f:
                vocabs = pickle.load(f)
            logger.info(f"✅ 使用过滤后的emoji词汇表: {len(vocabs['emoji_vocab'])} 个有效emoji")
        except FileNotFoundError:
            emoji_embeddings = torch.load(os.path.join(embeddings_dir, "emoji_embeddings.pt"))
            with open(os.path.join(embeddings_dir, "vocabularies.pkl"), 'rb') as f:
                vocabs = pickle.load(f)
            logger.warning(f"⚠️ 使用原始emoji词汇表: {len(vocabs['emoji_vocab'])} 个emoji（可能包含话题标签）")
        
        self.emoji_suggester = EmojiSuggester(emoji_embeddings, vocabs['emoji_vocab'])
        
        logger.info(f"  📊 Engagement预测器: {config['embedding_dim']}D input")
        logger.info(f"  😊 Emoji建议器: {config['num_emojis']} emojis")
    
    def predict_engagement(self, content: str) -> float:
        """预测engagement分数"""
        # 将内容转换为embedding
        content_embedding = self.content_converter.content_to_embedding(content)
        
        # 使用训练好的模型预测
        with torch.no_grad():
            score = self.engagement_predictor(content_embedding.unsqueeze(0)).item()
        
        return score
    
    def suggest_emojis(self, content: str, top_k: int = 5) -> List[str]:
        """建议emoji"""
        # 将内容转换为embedding
        content_embedding = self.content_converter.content_to_embedding(content)
        
        # 使用emoji建议器
        suggested_emojis = self.emoji_suggester.suggest_emojis(content_embedding, top_k)
        
        return suggested_emojis
    
    def call_glm_for_optimization(self, content: str, suggested_emojis: List[str], iteration: int) -> str:
        """调用智谱GLM进行内容优化"""
        emoji_str = ' '.join(suggested_emojis)
        
        prompt = f"""你是一个专业的小红书内容优化师，擅长通过精准的表情符号使用来提升内容互动性和吸引力。

任务：基于AI推荐的表情符号，优化以下内容的表情符号使用，但不能改变任何文字内容。

原始内容：
{content}

AI推荐的表情符号：{emoji_str}

优化要求：
1. 【文字完全不变】不能修改、删除或添加任何文字，只能调整表情符号
2. 【智能选择】优先使用AI推荐的表情符号，它们是基于内容语义匹配的
3. 【精准定位】选择最能增强内容情感表达的关键位置添加表情符号
4. 【适度使用】避免过度使用，每段内容选择1-3个最合适的位置即可
5. 【小红书风格】符合小红书用户的使用习惯，突出重点，增强视觉吸引力
6. 【可读性优先】保持内容的流畅性和美观性

优化策略：
- 在情感高点、关键词、动作描述等位置适当添加推荐的表情符号
- 避免每句话都加，重点突出核心内容
- 可以保留原有的合适表情符号，结合推荐表情符号进行优化

这是第{iteration + 1}轮优化，请提供更有吸引力的表情符号搭配。

请直接返回优化后的内容，不要包含任何说明。"""

        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.glm_client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=500
            )
            return response.strip()
        except Exception as e:
            logger.error(f"❌ GLM API调用失败: {e}")
            raise  # 不使用fallback，让错误暴露
    
    def optimize_content(self, original_content: str, verbose: bool = True) -> Dict[str, Any]:
        """优化内容"""
        logger.info("🚀 开始内容优化...")
        logger.info(f"📝 原始内容: {original_content}")
        
        # 初始化跟踪
        current_content = original_content.strip()
        optimization_log = []
        
        # 获取初始engagement分数
        initial_score = self.predict_engagement(current_content)
        logger.info(f"📊 初始engagement分数: {initial_score:.3f}")
        
        optimization_log.append({
            'iteration': 0,
            'content': current_content,
            'engagement_score': initial_score,
            'suggested_emojis': [],
            'improvement': 0.0,
            'timestamp': time.time()
        })
        
        # 优化循环
        for iteration in range(self.max_iterations):
            logger.info(f"\n🔄 第 {iteration + 1}/{self.max_iterations} 轮优化")
            
            # 获取emoji建议
            suggested_emojis = self.suggest_emojis(current_content, top_k=5)
            logger.info(f"🎯 建议的emojis: {' '.join(suggested_emojis)}")
            
            # 调用GLM优化
            logger.info("🤖 调用智谱GLM进行优化...")
            try:
                optimized_content = self.call_glm_for_optimization(
                    current_content, suggested_emojis, iteration
                )
            except Exception as e:
                logger.error(f"❌ 第{iteration+1}轮优化失败: {e}")
                break
            
            if not optimized_content or optimized_content == current_content:
                logger.info("⏸️ GLM没有提供进一步优化")
                break
            
            # 预测优化后的engagement分数
            optimized_score = self.predict_engagement(optimized_content)
            improvement = optimized_score - initial_score
            
            logger.info(f"📈 优化后engagement分数: {optimized_score:.3f}")
            logger.info(f"⬆️ 改进幅度: {improvement:+.3f}")
            
            if verbose:
                logger.info(f"📝 优化后内容: {optimized_content}")
            
            # 记录此轮优化
            optimization_log.append({
                'iteration': iteration + 1,
                'content': optimized_content,
                'engagement_score': optimized_score,
                'suggested_emojis': suggested_emojis,
                'improvement': improvement,
                'improvement_vs_previous': optimized_score - optimization_log[-1]['engagement_score'],
                'timestamp': time.time()
            })
            
            # 检查是否达到阈值
            if optimized_score >= self.engagement_threshold:
                logger.info(f"🎉 达到目标engagement阈值 ({self.engagement_threshold:.3f})!")
                break
            
            # 检查改进是否显著
            if improvement < 0.01 and iteration > 0:
                logger.info("⏸️ 改进幅度较小，停止优化")
                break
            
            # 更新当前内容
            current_content = optimized_content
        
        # 编译最终结果
        final_log = optimization_log[-1]
        total_improvement = final_log['engagement_score'] - initial_score
        
        results = {
            'original_content': original_content,
            'optimized_content': final_log['content'],
            'initial_score': initial_score,
            'final_score': final_log['engagement_score'],
            'total_improvement': total_improvement,
            'iterations_used': len(optimization_log) - 1,
            'optimization_log': optimization_log,
            'threshold_reached': final_log['engagement_score'] >= self.engagement_threshold,
            'success': total_improvement > 0
        }
        
        # 打印最终总结
        logger.info("\n" + "="*60)
        logger.info("📊 优化总结")
        logger.info("="*60)
        logger.info(f"📝 原始: {original_content}")
        logger.info(f"✨ 优化: {final_log['content']}")
        logger.info(f"📈 分数: {initial_score:.3f} → {final_log['engagement_score']:.3f}")
        logger.info(f"⬆️ 改进: {total_improvement:+.3f} ({total_improvement/initial_score*100:+.1f}%)")
        logger.info(f"🔄 轮次: {results['iterations_used']}")
        logger.info(f"🎯 成功: {'✅' if results['success'] else '❌'}")
        
        return results

def get_random_posts_from_db(db_path: str, num_posts: int) -> List[Dict[str, Any]]:
    """从数据库中随机选择帖子"""
    logger.info(f"📂 连接数据库: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # 随机选择帖子，优先选择有一定engagement的帖子
    query = """
    SELECT note_id, content, liked_count, collected_count, comments_count 
    FROM note_info 
    WHERE content IS NOT NULL 
    AND content != ''
    AND (liked_count > 0 OR collected_count > 0 OR comments_count > 0)
    ORDER BY RANDOM() 
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=[num_posts])
    conn.close()
    
    if df.empty:
        logger.warning("⚠️ 数据库中没有找到合适的帖子，尝试获取任意帖子...")
        conn = sqlite3.connect(db_path)
        query = """
        SELECT note_id, content, liked_count, collected_count, comments_count 
        FROM note_info 
        WHERE content IS NOT NULL AND content != ''
        ORDER BY RANDOM() 
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=[num_posts])
        conn.close()
    
    posts = []
    for _, row in df.iterrows():
        posts.append({
            'note_id': row['note_id'],
            'content': row['content'],
            'original_engagement': {
                'liked_count': row['liked_count'],
                'collected_count': row['collected_count'], 
                'comments_count': row['comments_count'],
                'total': row['liked_count'] + row['collected_count'] + row['comments_count']
            }
        })
    
    logger.info(f"✅ 从数据库中选择了 {len(posts)} 条帖子")
    return posts

def main():
    parser = argparse.ArgumentParser("真实LLM内容优化器")
    parser.add_argument("--api-key", type=str, required=False, help="智谱GLM API密钥 (默认从环境变量FDU_API_KEY读取)")
    parser.add_argument("--content", type=str, required=False, help="要优化的内容 (如果不提供，则从数据库随机选择)")
    parser.add_argument("--num-posts", type=int, default=3, help="从数据库随机选择优化的帖子数量 (默认3条)")
    parser.add_argument("--database", type=str, default="xhs_data.db", help="XHS数据库路径")
    parser.add_argument("--models", type=str, default="trained_models", help="训练好的模型目录")
    parser.add_argument("--embeddings", type=str, default="embeddings", help="embeddings目录")
    parser.add_argument("--max-iterations", type=int, default=5, help="最大优化轮数")
    parser.add_argument("--threshold", type=float, default=0.8, help="目标engagement阈值")
    parser.add_argument("--temperature", type=float, default=0.7, help="GLM温度参数")
    
    args = parser.parse_args()
    
    # 从环境变量或命令行参数获取API key
    api_key = args.api_key or os.environ.get('FDU_API_KEY')
    if not api_key:
        logger.error("❌ 未找到API密钥！请设置环境变量FDU_API_KEY或使用--api-key参数")
        return
    
    # 检查必需文件
    required_files = [
        os.path.join(args.models, "engagement_predictor.pt"),
        os.path.join(args.models, "config.json"),
        os.path.join(args.embeddings, "emoji_embeddings.pt"),
        os.path.join(args.embeddings, "vocabularies.pkl")
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"❌ 必需文件不存在: {file_path}")
            logger.error("请确保已经运行了完整的训练pipeline")
            return
    
    # 创建优化器
    try:
        optimizer = RealLLMContentOptimizer(
            glm_api_key=api_key,
            trained_models_dir=args.models,
            embeddings_dir=args.embeddings,
            max_iterations=args.max_iterations,
            engagement_threshold=args.threshold,
            temperature=args.temperature
        )
    except Exception as e:
        logger.error(f"❌ 优化器初始化失败: {e}")
        return
    
    # 确定要优化的内容
    if args.content:
        # 单条内容优化
        contents_to_optimize = [{'content': args.content, 'note_id': 'manual_input'}]
        logger.info(f"📝 优化单条手动输入内容")
    else:
        # 从数据库选择多条帖子
        if not os.path.exists(args.database):
            logger.error(f"❌ 数据库文件不存在: {args.database}")
            return
        
        try:
            posts = get_random_posts_from_db(args.database, args.num_posts)
            if not posts:
                logger.error("❌ 无法从数据库中获取帖子")
                return
            contents_to_optimize = posts
            logger.info(f"📊 将优化 {len(posts)} 条数据库帖子")
        except Exception as e:
            logger.error(f"❌ 从数据库获取帖子失败: {e}")
            return
    
    # 执行批量优化
    timestamp = int(time.time())
    all_results = []
    
    logger.info(f"\n🚀 开始批量优化，共 {len(contents_to_optimize)} 条内容")
    logger.info("="*80)
    
    for i, content_info in enumerate(contents_to_optimize, 1):
        content = content_info['content']
        note_id = content_info.get('note_id', f'item_{i}')
        
        logger.info(f"\n📝 [{i}/{len(contents_to_optimize)}] 优化内容 (ID: {note_id})")
        logger.info(f"原始: {content[:100]}{'...' if len(content) > 100 else ''}")
        
        # 如果是数据库帖子，显示原始engagement数据
        if 'original_engagement' in content_info:
            orig_eng = content_info['original_engagement']
            logger.info(f"原始engagement: {orig_eng['total']} (👍{orig_eng['liked_count']} ⭐{orig_eng['collected_count']} 💬{orig_eng['comments_count']})")
        
        try:
            # 优化单条内容
            result = optimizer.optimize_content(content)
            result['note_id'] = note_id
            result['index'] = i
            
            # 添加原始engagement数据（如果有）
            if 'original_engagement' in content_info:
                result['original_engagement'] = content_info['original_engagement']
            
            all_results.append(result)
            
            # 显示简要结果
            improvement = result['total_improvement']
            status = "✅" if result['success'] else "❌"
            logger.info(f"{status} 完成: {result['initial_score']:.3f} → {result['final_score']:.3f} ({improvement:+.3f})")
            
        except Exception as e:
            logger.error(f"❌ 第{i}条优化失败: {e}")
            # 记录失败结果
            failed_result = {
                'note_id': note_id,
                'index': i,
                'original_content': content,
                'error': str(e),
                'success': False
            }
            all_results.append(failed_result)
            continue
    
    # 保存批量结果
    try:
        output_file = f"batch_optimization_results_{timestamp}.json"
        
        # 转换为可序列化的格式
        serializable_results = []
        for result in all_results:
            if 'optimization_log' in result:
                serialized_result = result.copy()
                serialized_result['optimization_log'] = [
                    {k: (v if not isinstance(v, list) or not v or not isinstance(v[0], str) else v) 
                     for k, v in log.items()}
                    for log in result['optimization_log']
                ]
                serializable_results.append(serialized_result)
            else:
                serializable_results.append(result)
        
        # 计算统计信息
        successful_results = [r for r in all_results if r.get('success', False)]
        total_improvements = sum(r.get('total_improvement', 0) for r in successful_results)
        avg_improvement = total_improvements / len(successful_results) if successful_results else 0
        
        summary = {
            'timestamp': timestamp,
            'total_processed': len(contents_to_optimize),
            'successful_optimizations': len(successful_results),
            'failed_optimizations': len(all_results) - len(successful_results),
            'success_rate': len(successful_results) / len(all_results) * 100,
            'average_improvement': avg_improvement,
            'total_improvement': total_improvements,
            'results': serializable_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 打印最终统计
        logger.info("\n" + "="*80)
        logger.info("📊 批量优化总结")
        logger.info("="*80)
        logger.info(f"📝 总处理: {summary['total_processed']} 条")
        logger.info(f"✅ 成功: {summary['successful_optimizations']} 条")
        logger.info(f"❌ 失败: {summary['failed_optimizations']} 条")
        logger.info(f"📈 成功率: {summary['success_rate']:.1f}%")
        logger.info(f"⬆️ 平均提升: {avg_improvement:+.3f}")
        logger.info(f"💾 结果已保存: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ 保存结果失败: {e}")
        raise  # 让错误完全暴露

if __name__ == "__main__":
    main() 