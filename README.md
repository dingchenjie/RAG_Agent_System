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
(这里写一下如何安装依赖、配置 API Key 和运行 api.py 的简单步骤)
