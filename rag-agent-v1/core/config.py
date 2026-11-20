import os
from dotenv import load_dotenv

# 自动加载 .env 文件中的环境变量
load_dotenv()

# 设置 HuggingFace 环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 🌟 从环境变量中读取 API Key，如果没找到则使用默认占位符
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "你的智谱API_KEY")

# 模型配置
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
LLM_MODEL_NAME = "glm-4-flash"
EMBEDDING_API_MODEL = "embedding-2"

# RAG 参数配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SIMILARITY_THRESHOLD = 0.85  # 冗余剔除阈值
RECALL_K = 15                # 粗排召回数量
RERANK_TOP_K = 3             # 精排保留数量
