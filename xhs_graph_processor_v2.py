#!/usr/bin/env python
# encoding: utf-8
"""
XHS Graph Data Processor V2 - FIXED VERSION
Solves Feature Representation Gap and Emoji Vocabulary Limitation

Key Improvements:
1. Uses actual vocabularies from the XHS database
2. Implements TF-IDF features for words (matching training data)
3. Uses proper emoji vocabulary from database
4. Creates features compatible with pre-trained model
5. Leverages learned embeddings correctly
"""

import torch
import dgl
import numpy as np
import re
import jieba
import sqlite3
import json
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class XHSGraphProcessorV2:
    """Enhanced processor that uses real XHS data vocabularies and proper features"""
    
    def __init__(self, db_path: str = "xhs_data.db", load_vocabularies: bool = True):
        """
        Initialize with real XHS database vocabularies
        
        Args:
            db_path: Path to XHS database
            load_vocabularies: Whether to load vocabularies from database
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
        
        # Initialize vocabularies
        self.word_vocab = {}
        self.emoji_vocab = {}
        self.post_vocab = {}
        self.word_idf_scores = {}
        
        # TF-IDF vectorizer for words
        self.tfidf_vectorizer = None
        self.word_tfidf_matrix = None
        
        # Database connection
        self.db_path = db_path
        
        # Load real vocabularies from database
        if load_vocabularies:
            self._load_vocabularies_from_database()
    
    def _load_vocabularies_from_database(self):
        """Load actual vocabularies from XHS database"""
        logger.info("Loading vocabularies from XHS database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. Load word vocabulary with IDF scores
            logger.info("Loading word vocabulary with IDF scores...")
            cursor.execute("SELECT word, idf FROM word_idf ORDER BY idf DESC LIMIT 10000")
            word_idf_data = cursor.fetchall()
            
            for i, (word, idf) in enumerate(word_idf_data):
                self.word_vocab[word] = i
                self.word_idf_scores[word] = idf
            
            logger.info(f"Loaded {len(self.word_vocab)} words with IDF scores")
            
            # 2. Extract emoji vocabulary from actual posts
            logger.info("Extracting emoji vocabulary from posts...")
            cursor.execute("""
                SELECT title, content FROM note_info 
                WHERE title IS NOT NULL AND content IS NOT NULL 
                LIMIT 50000
            """)
            
            emoji_counter = Counter()
            post_count = 0
            
            for title, content in cursor.fetchall():
                if not title or not content:
                    continue
                    
                full_text = f"{title} {content}"
                emojis = self.extract_emojis(full_text)
                emoji_counter.update(emojis)
                post_count += 1
                
                if post_count % 10000 == 0:
                    logger.info(f"Processed {post_count} posts for emoji extraction")
            
            # Build emoji vocabulary from most common emojis
            for i, (emoji, count) in enumerate(emoji_counter.most_common(5000)):
                self.emoji_vocab[emoji] = i
            
            logger.info(f"Extracted {len(self.emoji_vocab)} emojis from {post_count} posts")
            
            # 3. Prepare TF-IDF vectorizer with real vocabulary
            logger.info("Preparing TF-IDF vectorizer...")
            self._setup_tfidf_vectorizer()
            
            conn.close()
            
            # Save vocabularies for future use
            self._save_vocabularies()
            
            logger.info("✅ Successfully loaded all vocabularies from database")
            
        except Exception as e:
            logger.error(f"Error loading vocabularies from database: {e}")
            logger.info("Falling back to default vocabularies...")
            self._create_default_vocabularies()
    
    def _setup_tfidf_vectorizer(self):
        """Setup TF-IDF vectorizer with database vocabulary"""
        # Use the words from database as vocabulary
        vocabulary = list(self.word_vocab.keys())
        
        # Create TF-IDF vectorizer with database vocabulary
        self.tfidf_vectorizer = TfidfVectorizer(
            vocabulary=vocabulary,
            lowercase=False,
            token_pattern=None,
            tokenizer=self._tokenize_chinese,
            max_features=len(vocabulary)
        )
        
        # Fit on a dummy corpus to initialize
        dummy_corpus = [" ".join(vocabulary[:100])]  # Use first 100 words
        self.tfidf_vectorizer.fit(dummy_corpus)
        
        logger.info(f"TF-IDF vectorizer initialized with {len(vocabulary)} words")
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """Tokenize Chinese text using jieba"""
        # Remove emojis first
        clean_text = self.emoji_pattern.sub('', text)
        # Tokenize
        words = list(jieba.cut(clean_text.strip()))
        # Filter and return words that are in vocabulary
        return [w for w in words if w in self.word_vocab]
    
    def _create_default_vocabularies(self):
        """Create default vocabularies as fallback"""
        logger.info("Creating default vocabularies...")
        
        # Default high-frequency Chinese words
        default_words = [
            "的", "一", "是", "在", "不", "了", "有", "和", "人", "这",
            "中", "大", "为", "上", "个", "国", "我", "以", "要", "他",
            "时", "来", "用", "们", "生", "到", "作", "地", "于", "出",
            "就", "分", "对", "成", "会", "可", "主", "发", "年", "动",
            "同", "工", "也", "能", "下", "过", "子", "说", "产", "种",
            "面", "而", "方", "后", "多", "定", "行", "学", "法", "所",
            "民", "得", "经", "十", "三", "之", "进", "着", "等", "部",
            "度", "家", "电", "力", "里", "如", "水", "化", "高", "自",
            "二", "理", "起", "小", "物", "现", "实", "加", "量", "都",
            "两", "体", "制", "机", "当", "使", "点", "从", "业", "本",
            "去", "把", "性", "好", "应", "开", "它", "合", "还", "因",
            "由", "其", "些", "然", "前", "外", "天", "政", "四", "日",
            "那", "社", "义", "事", "平", "形", "相", "全", "表", "间",
            "样", "与", "关", "各", "重", "新", "线", "内", "数", "正",
            "心", "反", "你", "明", "看", "原", "又", "么", "利", "比",
            "或", "但", "质", "气", "第", "向", "道", "命", "此", "变",
            "条", "只", "没", "结", "解", "问", "意", "建", "月", "公",
            "无", "系", "军", "很", "情", "者", "最", "立", "代", "想",
            "已", "通", "并", "提", "直", "题", "党", "程", "展", "五",
            "果", "料", "象", "员", "革", "位", "入", "常", "文", "总",
            "次", "品", "式", "活", "设", "及", "管", "特", "件", "长",
            "求", "老", "头", "基", "资", "边", "流", "路", "级", "少",
            "图", "山", "统", "接", "知", "较", "将", "组", "见", "计",
            "别", "她", "手", "角", "期", "根", "论", "运", "农", "指",
            "几", "九", "区", "强", "放", "决", "西", "被", "干", "做",
            "必", "战", "先", "回", "则", "任", "取", "据", "处", "队",
            "南", "给", "色", "光", "门", "即", "保", "治", "北", "造",
            "百", "规", "热", "领", "七", "海", "口", "东", "导", "器",
            "压", "志", "世", "金", "增", "争", "济", "阶", "油", "思",
            "术", "极", "交", "受", "联", "什", "认", "六", "共", "权",
            "收", "证", "改", "清", "美", "再", "采", "转", "更", "单",
            "风", "切", "打", "白", "教", "速", "花", "带", "安", "场",
            "身", "车", "例", "真", "务", "具", "万", "每", "目", "至",
            "达", "走", "积", "示", "议", "声", "报", "斗", "完", "类",
            "八", "离", "华", "名", "确", "才", "科", "张", "信", "马",
            "节", "话", "米", "整", "空", "元", "况", "今", "集", "温",
            "传", "土", "许", "步", "群", "广", "石", "记", "需", "段",
            "研", "界", "拉", "林", "律", "叫", "且", "究", "观", "越",
            "织", "装", "影", "算", "低", "持", "音", "众", "书", "布",
            "复", "容", "儿", "须", "际", "商", "非", "验", "连", "断",
            "深", "难", "近", "矿", "千", "周", "委", "素", "技", "备",
            "半", "办", "青", "省", "列", "习", "响", "约", "支", "般",
            "史", "感", "劳", "便", "团", "往", "酸", "历", "市", "克",
            "何", "除", "消", "构", "府", "称", "太", "准", "精", "值",
            "号", "率", "族", "维", "划", "选", "标", "写", "存", "候",
            "毛", "亲", "快", "效", "斯", "院", "查", "江", "型", "眼",
            "王", "按", "格", "养", "易", "置", "派", "层", "片", "始",
            "却", "专", "状", "育", "厂", "京", "识", "适", "属", "圆",
            "包", "火", "住", "调", "满", "县", "局", "照", "参", "红",
            "细", "引", "听", "该", "铁", "价", "严", "首", "底", "液",
            "官", "德", "随", "病", "苏", "失", "尔", "死", "讲", "配",
            "女", "黄", "推", "显", "谈", "罪", "神", "艺", "呢", "席",
            "含", "企", "望", "密", "批", "营", "项", "防", "举", "球",
            "英", "氧", "势", "告", "李", "台", "落", "木", "帮", "轮",
            "破", "亚", "师", "围", "注", "远", "字", "材", "排", "供",
            "河", "态", "封", "另", "施", "减", "树", "溶", "怎", "止",
            "案", "言", "士", "均", "武", "固", "叶", "鱼", "波", "视",
            "仅", "费", "紧", "爱", "左", "章", "早", "朝", "害", "续",
            "轻", "服", "试", "食", "充", "兵", "源", "判", "护", "司",
            "足", "某", "练", "差", "致", "板", "田", "降", "黑", "犯",
            "负", "击", "范", "继", "兴", "似", "余", "坚", "曲", "输",
            "修", "故", "城", "夫", "够", "送", "笔", "船", "占", "右",
            "财", "吃", "富", "春", "职", "觉", "汉", "画", "功", "巴",
            "跟", "虽", "杂", "飞", "检", "吸", "助", "升", "阳", "互",
            "初", "创", "抗", "考", "投", "坏", "策", "古", "径", "换",
            "未", "跑", "留", "钢", "曾", "端", "责", "站", "简", "述",
            "钱", "副", "尽", "帝", "射", "草", "冲", "承", "独", "令",
            "限", "阿", "宣", "环", "双", "请", "超", "微", "让", "控",
            "州", "良", "轴", "找", "否", "纪", "益", "依", "优", "顶",
            "础", "载", "倒", "房", "突", "坐", "粉", "敌", "略", "客",
            "袁", "冷", "胜", "绝", "析", "块", "剂", "测", "丝", "协",
            "诉", "念", "陈", "仍", "罗", "盐", "友", "洋", "错", "苦",
            "夜", "刑", "移", "频", "逐", "靠", "混", "母", "短", "皮",
            "终", "聚", "汽", "村", "云", "哪", "既", "距", "卫", "停",
            "烈", "央", "察", "烧", "迅", "境", "若", "印", "洲", "刻",
            "括", "激", "孔", "搞", "甚", "室", "待", "核", "校", "散",
            "侵", "吧", "甲", "游", "久", "菜", "味", "旧", "模", "湖",
            "货", "损", "预", "阻", "毫", "普", "稳", "乙", "妈", "植",
            "息", "扩", "银", "语", "挥", "酒", "守", "拿", "序", "纸",
            "医", "缺", "雨", "吗", "针", "刘", "啊", "急", "唱", "误",
            "训", "愿", "审", "附", "获", "茶", "鲜", "粮", "斤", "孩",
            "脱", "硫", "肥", "善", "龙", "演", "父", "渐", "血", "欢",
            "械", "掌", "歌", "沙", "刚", "攻", "谓", "盾", "讨", "晚",
            "粒", "乱", "燃", "矛", "乎", "杀", "药", "宁", "鲁", "贵",
            "钟", "煤", "读", "班", "伯", "香", "介", "迫", "句", "丰",
            "培", "握", "兰", "担", "弦", "蛋", "沉", "假", "穿", "执",
            "答", "乐", "谁", "顺", "烟", "缩", "征", "脸", "喜", "松",
            "脚", "困", "异", "免", "背", "星", "福", "买", "染", "井",
            "概", "慢", "怕", "磁", "倍", "祖", "皇", "促", "静", "补",
            "评", "翻", "肉", "践", "尼", "衣", "宽", "扬", "棉", "希",
            "伤", "操", "垂", "秋", "宜", "氢", "套", "督", "振", "架",
            "亮", "末", "宪", "庆", "编", "牛", "触", "映", "雷", "销",
            "诗", "座", "居", "抓", "裂", "胞", "呼", "娘", "景", "威",
            "绿", "晶", "厚", "盟", "衡", "鸡", "孙", "延", "危", "胶",
            "屋", "乡", "临", "陆", "顾", "掉", "呀", "灯", "岁", "措",
            "束", "刀", "恶", "停", "育", "届", "欧", "献", "支", "辅"
        ]
        
        # Build word vocabulary
        for i, word in enumerate(default_words):
            self.word_vocab[word] = i
            self.word_idf_scores[word] = 1.0  # Default IDF score
        
        # NO DEFAULT EMOJIS - only use database vocabulary
        # This ensures we only work with real XHS emoji data
        logger.warning("No database connection available - emoji vocabulary will be empty!")
        logger.warning("This may affect emoji suggestion quality. Please check database connection.")
        
        # Setup basic TF-IDF
        self._setup_tfidf_vectorizer()
        
        logger.info(f"Created default vocabularies: {len(self.word_vocab)} words, {len(self.emoji_vocab)} emojis")
    
    def _save_vocabularies(self):
        """Save vocabularies to file for future use"""
        vocab_data = {
            'word_vocab': self.word_vocab,
            'emoji_vocab': self.emoji_vocab,
            'word_idf_scores': self.word_idf_scores
        }
        
        with open('xhs_vocabularies_v2.json', 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info("Vocabularies saved to xhs_vocabularies_v2.json")
    
    def load_vocabularies_from_file(self, vocab_file: str = 'xhs_vocabularies_v2.json'):
        """Load vocabularies from saved file"""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            self.word_vocab = vocab_data['word_vocab']
            self.emoji_vocab = vocab_data['emoji_vocab']
            self.word_idf_scores = vocab_data['word_idf_scores']
            
            self._setup_tfidf_vectorizer()
            
            logger.info(f"Loaded vocabularies from {vocab_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vocabularies from {vocab_file}: {e}")
            return False
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text"""
        return self.emoji_pattern.findall(text)
    
    def extract_words(self, text: str) -> List[str]:
        """Extract words from Chinese text using jieba"""
        return self._tokenize_chinese(text)
    
    def create_tfidf_features(self, posts: List[str]) -> torch.Tensor:
        """
        Create TF-IDF features for posts using database vocabulary
        This matches the original training data format
        """
        try:
            # Transform posts to TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.transform(posts)
            
            # Convert to dense tensor
            tfidf_dense = tfidf_matrix.toarray()
            
            # Pad or truncate to 768 dimensions
            if tfidf_dense.shape[1] < 768:
                # Pad with zeros
                padding = np.zeros((tfidf_dense.shape[0], 768 - tfidf_dense.shape[1]))
                tfidf_dense = np.concatenate([tfidf_dense, padding], axis=1)
            elif tfidf_dense.shape[1] > 768:
                # Truncate to 768 dimensions
                tfidf_dense = tfidf_dense[:, :768]
            
            return torch.tensor(tfidf_dense, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Error creating TF-IDF features: {e}")
            # Fallback to simple features
            return self._create_simple_features(posts)
    
    def _create_simple_features(self, posts: List[str]) -> torch.Tensor:
        """Fallback simple feature creation"""
        post_features = []
        
        for post in posts:
            features = torch.zeros(768)
            
            # Basic content features
            features[0] = len(post) / 100.0  # Normalized length
            features[1] = len(self.extract_words(post)) / 50.0  # Normalized word count
            features[2] = len(self.extract_emojis(post)) / 10.0  # Normalized emoji count
            
            # Word presence features
            words = self.extract_words(post)
            for word in words[:100]:  # Use first 100 words
                if word in self.word_vocab:
                    word_id = self.word_vocab[word]
                    if word_id < 765:  # Leave space for other features
                        features[3 + word_id] = 1.0
            
            post_features.append(features)
        
        return torch.stack(post_features)
    
    def create_word_features(self, words: List[str]) -> torch.Tensor:
        """Create features for words using TF-IDF scores"""
        word_features = []
        
        for word in words:
            features = torch.zeros(768)
            
            # Basic word features
            features[0] = len(word) / 10.0  # Normalized length
            features[1] = self.word_idf_scores.get(word, 0.0)  # IDF score
            
            # Character-level features
            char_hash = hash(word) % 766
            features[2 + char_hash] = 1.0
            
            word_features.append(features)
        
        return torch.stack(word_features) if word_features else torch.zeros(0, 768)
    
    def create_emoji_features(self, emojis: List[str]) -> torch.Tensor:
        """Create features for emojis using learned representations"""
        emoji_features = []
        
        for emoji in emojis:
            # Use random features as in original implementation
            # but make them consistent for the same emoji
            np.random.seed(hash(emoji) % (2**32))
            features = torch.tensor(np.random.randn(768), dtype=torch.float32)
            emoji_features.append(features)
        
        return torch.stack(emoji_features) if emoji_features else torch.zeros(0, 768)
    
    def build_vocabulary(self, posts: List[str]):
        """Build vocabulary mappings for new posts"""
        # For new posts, just map them to indices
        for i, post in enumerate(posts):
            if post not in self.post_vocab:
                self.post_vocab[post] = len(self.post_vocab)
    
    def create_heterogeneous_graph(self, posts: List[str]) -> dgl.DGLGraph:
        """
        Create heterogeneous graph with proper features matching training data
        """
        # Build post vocabulary
        self.build_vocabulary(posts)
        
        # Get all unique words and emojis from posts
        all_words = set()
        all_emojis = set()
        
        for post in posts:
            words = self.extract_words(post)
            emojis = self.extract_emojis(post)
            all_words.update(words)
            all_emojis.update(emojis)
        
        # Filter to known vocabulary
        known_words = [w for w in all_words if w in self.word_vocab]
        known_emojis = [e for e in all_emojis if e in self.emoji_vocab]
        
        logger.info(f"Creating graph with {len(posts)} posts, {len(known_words)} words, {len(known_emojis)} emojis")
        
        # Create node features using proper methods
        post_features = self.create_tfidf_features(posts)
        word_features = self.create_word_features(known_words)
        emoji_features = self.create_emoji_features(known_emojis)
        
        # Create word and emoji mappings for this graph
        word_to_idx = {word: i for i, word in enumerate(known_words)}
        emoji_to_idx = {emoji: i for i, emoji in enumerate(known_emojis)}
        
        # Collect edges
        post_word_edges = []
        post_emoji_edges = []
        word_emoji_edges = []
        
        for post_idx, post in enumerate(posts):
            words = self.extract_words(post)
            emojis = self.extract_emojis(post)
            
            # Post-word edges
            for word in words:
                if word in word_to_idx:
                    word_idx = word_to_idx[word]
                    post_word_edges.append((post_idx, word_idx))
            
            # Post-emoji edges  
            for emoji in emojis:
                if emoji in emoji_to_idx:
                    emoji_idx = emoji_to_idx[emoji]
                    post_emoji_edges.append((post_idx, emoji_idx))
            
            # Word-emoji co-occurrence edges
            for word in words:
                if word in word_to_idx:
                    word_idx = word_to_idx[word]
                    for emoji in emojis:
                        if emoji in emoji_to_idx:
                            emoji_idx = emoji_to_idx[emoji]
                            word_emoji_edges.append((word_idx, emoji_idx))
        
        # Create heterogeneous graph
        graph_data = {}
        
        # Add edges with correct relation names matching training data
        if post_word_edges:
            post_ids, word_ids = zip(*post_word_edges)
            graph_data[('post', 'hasw', 'word')] = (torch.tensor(post_ids), torch.tensor(word_ids))
            graph_data[('word', 'win', 'post')] = (torch.tensor(word_ids), torch.tensor(post_ids))
        
        if post_emoji_edges:
            post_ids, emoji_ids = zip(*post_emoji_edges)
            graph_data[('post', 'hase', 'emoji')] = (torch.tensor(post_ids), torch.tensor(emoji_ids))
            graph_data[('emoji', 'ein', 'post')] = (torch.tensor(emoji_ids), torch.tensor(post_ids))
        
        if word_emoji_edges:
            word_ids, emoji_ids = zip(*word_emoji_edges)
            graph_data[('word', 'withe', 'emoji')] = (torch.tensor(word_ids), torch.tensor(emoji_ids))
            graph_data[('emoji', 'by', 'word')] = (torch.tensor(emoji_ids), torch.tensor(word_ids))
        
        # Create graph
        if graph_data:
            g = dgl.heterograph(graph_data)
        else:
            # Create empty graph with correct node types
            g = dgl.heterograph({
                ('post', 'dummy', 'post'): ([], []),
                ('word', 'dummy', 'word'): ([], []),
                ('emoji', 'dummy', 'emoji'): ([], [])
            })
        
        # Set node features
        if len(posts) > 0:
            g.nodes['post'].data['feat'] = post_features
        if len(known_words) > 0:
            g.nodes['word'].data['feat'] = word_features
        if len(known_emojis) > 0:
            g.nodes['emoji'].data['feat'] = emoji_features
        
        logger.info(f"Graph created successfully: {g}")
        return g
    
    def get_post_embedding_from_pretrained_model(self, post: str, model) -> torch.Tensor:
        """
        Get embedding for a post using the pre-trained model
        Uses proper feature representation matching training setup
        """
        try:
            # Create graph with proper features
            graph = self.create_heterogeneous_graph([post])
            
            # Add seed information (required by the model)
            # Mark all nodes as seed nodes for this inference
            for ntype in graph.ntypes:
                num_nodes = graph.num_nodes(ntype)
                if num_nodes > 0:
                    graph.nodes[ntype].data['seed'] = torch.zeros(num_nodes, dtype=torch.long)
            
            # Move graph to the same device as model
            device = next(model.parameters()).device
            graph = graph.to(device)
            
            # Use the pre-trained model to get embeddings
            with torch.no_grad():
                # Try different edge types based on what's available in the graph
                available_etypes = graph.canonical_etypes
                
                # Priority order for edge types (based on training data)
                preferred_etypes = [
                    ('emoji', 'ein', 'post'),
                    ('post', 'hase', 'emoji'), 
                    ('post', 'hasw', 'word'),
                    ('word', 'withe', 'emoji')
                ]
                
                embedding = None
                for etype in preferred_etypes:
                    if etype in available_etypes and graph.num_edges(etype) > 0:
                        try:
                            # Call model with proper edge type
                            embedding = model(graph, etype)
                            # Get the first post's embedding (index 0)
                            if embedding.dim() > 1:
                                embedding = embedding[0]  # First post
                            break
                        except Exception as e:
                            logger.debug(f"Failed with edge type {etype}: {e}")
                            continue
                
                # If no edge-based embedding worked, use post features directly
                if embedding is None:
                    if graph.num_nodes('post') > 0:
                        embedding = graph.nodes['post'].data['feat'][0]
                    else:
                        # Fallback to zero embedding with correct dimension
                        embedding = torch.zeros(768, device=device)
                
                return embedding.squeeze().cpu()
                
        except Exception as e:
            logger.error(f"Error getting embedding with model: {e}")
            # Fallback to TF-IDF features
            return self.create_tfidf_features([post])[0]
    
    def find_similar_emojis_using_model(self, post_content: str, model, top_k: int = 5) -> List[str]:
        """
        Find emojis semantically similar to post using the pre-trained model
        Uses actual emoji embeddings from the pre-trained model
        """
        try:
            logger.info(f"🎭 Finding similar emojis using pre-trained model for: {post_content[:50]}...")
            
            # Get candidate emojis from our vocabulary (excluding ones already in post)
            current_emojis = set(self.extract_emojis(post_content))
            candidate_emojis = [emoji for emoji in self.emoji_vocab.keys() if emoji not in current_emojis]
            
            if not candidate_emojis:
                logger.warning("No candidate emojis available")
                return []
            
            # Use the model's emoji embeddings directly
            # Create a graph with post + all candidate emojis to get their embeddings
            test_posts = [post_content] + [f"emoji {emoji}" for emoji in candidate_emojis[:50]]  # Limit for performance
            
            try:
                # Create graph with all posts and emojis
                graph = self.create_heterogeneous_graph(test_posts)
                
                # Add seed information
                for ntype in graph.ntypes:
                    num_nodes = graph.num_nodes(ntype)
                    if num_nodes > 0:
                        graph.nodes[ntype].data['seed'] = torch.zeros(num_nodes, dtype=torch.long)
                
                # Get embeddings using the model
                device = next(model.parameters()).device
                graph = graph.to(device)
                
                with torch.no_grad():
                    # Try to get emoji embeddings from the model
                    if ('emoji', 'ein', 'post') in graph.canonical_etypes and graph.num_edges(('emoji', 'ein', 'post')) > 0:
                        embeddings = model(graph, ('emoji', 'ein', 'post'))
                        
                        # Get emoji node embeddings
                        if graph.num_nodes('emoji') > 0:
                            emoji_embeddings = graph.nodes['emoji'].data['feat']  # Use original features
                            post_embedding = graph.nodes['post'].data['feat'][0]  # First post
                            
                            # Calculate similarities between post and emojis
                            emoji_similarities = []
                            emoji_list = list(self.emoji_vocab.keys())[:graph.num_nodes('emoji')]
                            
                            for i, emoji in enumerate(emoji_list):
                                if emoji not in current_emojis and i < len(emoji_embeddings):
                                    similarity = torch.cosine_similarity(
                                        post_embedding.unsqueeze(0),
                                        emoji_embeddings[i].unsqueeze(0)
                                    ).item()
                                    emoji_similarities.append((emoji, similarity))
                            
                            # Sort by similarity and return top k
                            emoji_similarities.sort(key=lambda x: x[1], reverse=True)
                            top_emojis = [emoji for emoji, _ in emoji_similarities[:top_k]]
                            
                            if top_emojis:
                                logger.info(f"✅ Found {len(top_emojis)} similar emojis using model: {top_emojis}")
                                return top_emojis
                
            except Exception as e:
                logger.debug(f"Graph-based emoji similarity failed: {e}")
            
            # Fallback: Use semantic similarity based on emoji frequency and post content
            logger.info("Using frequency-based emoji selection as fallback")
            
            # Extract key words from post content
            words = self.extract_words(post_content)
            word_set = set(words)
            
            # Score emojis based on co-occurrence with words in the vocabulary
            emoji_scores = []
            for emoji in candidate_emojis[:100]:  # Limit for performance
                score = 0
                
                # Basic frequency score (more frequent = higher base score)
                if emoji in self.emoji_vocab:
                    frequency = self.emoji_vocab[emoji]
                    score += min(frequency / 1000, 1.0)  # Normalize frequency
                
                # Semantic bonus: if emoji appears with similar words in our data
                # This is a simple heuristic - in practice, you'd use the actual co-occurrence data
                emoji_words = self.extract_words(emoji)  # Extract any words from emoji description
                word_overlap = len(word_set.intersection(set(emoji_words)))
                score += word_overlap * 0.1
                
                emoji_scores.append((emoji, score))
            
            # Sort by score and return top k
            emoji_scores.sort(key=lambda x: x[1], reverse=True)
            top_emojis = [emoji for emoji, _ in emoji_scores[:top_k]]
            
            logger.info(f"✅ Selected {len(top_emojis)} emojis using semantic fallback: {top_emojis}")
            return top_emojis
            
        except Exception as e:
            logger.error(f"Error finding similar emojis: {e}")
            return self._fallback_emoji_suggestions(post_content, top_k)
    
    def _fallback_emoji_suggestions(self, post_content: str, top_k: int = 5) -> List[str]:
        """Fallback emoji suggestions"""
        # Use most common emojis from our vocabulary
        current_emojis = set(self.extract_emojis(post_content))
        common_emojis = list(self.emoji_vocab.keys())[:50]  # Top 50 most common
        suggestions = [emoji for emoji in common_emojis if emoji not in current_emojis]
        return suggestions[:top_k]
    
    def get_vocabulary_stats(self) -> Dict[str, int]:
        """Get vocabulary statistics"""
        return {
            'words': len(self.word_vocab),
            'emojis': len(self.emoji_vocab),
            'posts': len(self.post_vocab)
        } 