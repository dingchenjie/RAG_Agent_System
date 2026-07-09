# 基于高质量数据清洗与重排的进阶 RAG 系统 (Advanced RAG)

## 🌟 项目简介
针对传统 RAG 系统在处理长文档时存在的“数据冗余高、检索噪声大、跨语言检索失效”等痛点，从 0 到 1 设计并实现了一套具备数据清洗、两阶段检索与跨语言路由能力的进阶 RAG Pipeline，并完成 FastAPI 工程化落地。

## 🚀 核心特性
1. **入库前数据清洗 (Data-centric AI)**：结合科研期间的“冗余样本筛选”经验，利用本地 Embedding 模型构建相似度矩阵，采用贪心策略剔除高相似度冗余 Chunk（实测数据压缩率达 46.4%），大幅降低存储与推理成本。
2. **两阶段检索架构 (Coarse-to-Fine Retrieval)**：采用“向量粗排召回 (Top-K) + Cross-Encoder 交叉注意力精排 (Rerank)”架构，显著提升复杂问题的召回准确率。
3. **跨语言检索路由 (Cross-lingual RAG)**：针对中英跨语言 Embedding 对齐不佳的问题，设计 Query Translation 机制，利用 LLM 实时翻译 Query，彻底解决跨语言知识库的检索断层。
4. **生产级 API 封装**：基于 FastAPI + Uvicorn 构建异步 RESTful API，集成 Swagger UI，具备直接对接前端业务的能力。

## 🛠️ 技术栈
Python, FastAPI, ChromaDB, ZhipuAI API, Sentence-Transformers (BGE), Scikit-learn

## 📦 快速开始

### 1. 环境准备
确保已安装 Python 3.9 或以上版本。强烈建议使用 Conda 创建独立的虚拟环境以避免依赖冲突：
```bash
conda create -n advanced_rag python=3.10
conda activate advanced_rag

### 2. 安装依赖
克隆本仓库并安装所需依赖（推荐使用清华源加速）：
```bash
git clone https://gitee.com/ding-chenjie/advanced_-rag_-system.git
cd advanced_-rag_-system
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

### 3. 配置环境变量 (安全规范)
本项目严格遵守企业级安全规范，绝不在代码中硬编码 API Key。 请在项目根目录下手动创建一个 .env 文件，并填入你的智谱 AI API Key：
```env
ZHIPU_API_KEY=your_actual_api_key_here
(注：获取智谱 API Key 请访问 open.bigmodel.cn)

### 4. 运行系统
方式一：启动 FastAPI 后端服务 (推荐)
```bash
python main_api.py
服务启动并构建知识库后，打开浏览器访问 http://127.0.0.1:8000/docs 即可使用自动生成的 Swagger UI 进行接口测试。

方式二：运行命令行交互模式
```bash
python main_cli.py
在终端中直接输入问题进行问答测试。