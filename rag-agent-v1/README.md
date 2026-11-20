# RAG Agent（v1）：数据清洗 + 交叉重排的进阶 RAG 系统

> 本目录是简历上那个 **RAG Agent** 项目的原始版本，保留可读、可查证。
> 它已经被重构进 [`hermes-agent/`](../hermes-agent/)（能力保留、实现升级），这里不再维护。

## 这个项目做了什么

一个 **ReAct 风格的 RAG Agent**：PDF 入库 → 分块去冗余 → 向量检索 + 查询翻译 + 交叉重排 → 工具循环作答 → 忠实度评测 → FastAPI 服务化。

| 模块 | 内容 |
| --- | --- |
| `core/data_pipeline.py` | `load_and_split_pdf`（PyPDF2 解析）、`filter_redundant_chunks`（相似度去冗余）、`build_vector_db`（Chroma 建库） |
| `core/retriever.py` | `translate_query_if_chinese`（中文查询翻译）、`search_and_rerank`（向量召回 + cross-encoder 重排） |
| `core/llm_service.py` | 智谱 `zhipuai` 客户端封装 |
| `agent/agent.py`、`agent/tools.py` | ReAct 循环；三个工具：`search_paper` / `calculate` / `get_current_time`；`get_tool_executor` 用 `if/elif` 分发 |
| `evaluation/eval_faithfulness.py` | RAGAS 忠实度评测，结果见 `faithfulness_result.csv` |
| `main_cli.py` / `main_api.py` / `main_agent.py` | 三种入口：CLI、FastAPI（`POST /chat`）、纯 Agent 脚本 |
| `Dockerfile`、`docker-compose.yml` | 容器化 |

## 怎么跑（历史版本，仅供参考）

```bash
pip install -r requirements.txt
# core/config.py 不入库：Key 只从环境变量读取
export ZHIPU_API_KEY=xxxx
python main_cli.py
uvicorn main_api:app --port 8000     # POST /chat
```

旧版依赖 Chroma、sklearn 全量相似矩阵与 RAGAS，安装体积较大；`chroma_db/` 等本地数据不入库。
**不建议直接演示**，它是演进证据，不是当前主项目。

## 它和 hermes-agent 的关系

| v1 的做法 | 在 hermes-agent 里的替代 |
| --- | --- |
| Chroma + sklearn 全量相似矩阵 | SQLite + FTS5/BM25 + 向量 + RRF 融合 |
| `if/elif` + `eval()` 分发工具 | 声明式 `ToolSpec` + Pydantic 校验 + 权限/审计九道关 |
| 无状态、不可恢复 | 每步 checkpoint + 跨进程 resume + 启动对账 |
| RAGAS 单一 faithfulness | 成功率 / 工具 F1 / 引用覆盖 / 忠实度 / P50-P95 / 成本 |

保留的能力：PDF/HTML/DOCX 解析、分块去冗余、语义向量、交叉重排、忠实度评测。
被丢弃的实现：`eval()`、启动即全量建库、无权限工具循环。

详细取舍见 [`../archive/README.md`](../archive/README.md)。
