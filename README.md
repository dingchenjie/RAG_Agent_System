# RAG → Agent Runtime → 业务 Agent

> 面向 AI Agent 岗位的作品集仓库：**一个主项目（hermes-agent）+ 一个业务项目（market-scout）+ 一份可查证的演进证据（rag-agent-v1）**。
> 简历上的 RAG Agent 项目仍然在仓库里，没有被删除或覆盖。

## 简历 ↔ 仓库对照

| 简历上的项目 | 仓库位置 | 当前状态 |
| --- | --- | --- |
| RAG Agent（数据清洗 + Rerank，FastAPI 服务化） | [`rag-agent-v1/`](rag-agent-v1/) | v1 原版，代码与 RAGAS 评测结果保留 |
| Hermes Agent（Agent Runtime / RAG 知识层 / 评测） | [`hermes-agent/`](hermes-agent/) | 现行主项目：v1 的能力已并入其中 |
| Market Scout（竞品调研报告 Agent） | [`market-scout/`](market-scout/) | 新增业务项目：LangGraph + Playwright + Langfuse |

演进关系：**v1 纯 RAG** → **v2 运行时骨架**（[`archive/hermes-runtime-v2/`](archive/hermes-runtime-v2/)）→ **v3 合并为 `hermes-agent/`** → **横向扩展出 `market-scout/`**。
每一版为什么改、丢了什么留下什么，见 [`archive/README.md`](archive/README.md)。

## 两个项目分别在解决什么

| | hermes-agent（主项目） | market-scout（业务项目） |
| --- | --- | --- |
| 定位 | 能托管长任务、可干预、可复盘、可评测的 Agent Runtime | 竞品调研报告 Agent，验证底座能否支撑真实业务 |
| 核心难点 | 上下文预算、工具契约与权限、状态恢复、事件与评测 | 任务形态路由、并行子任务、证据链与拒答、死循环收敛 |
| 编排 | 手写 ReAct + 调度器 + 多 Agent 主从 | LangGraph 图编排（Send 并行 / 人工审批中断） |
| 工具 | 七个内置 + MCP 双向 + 声明式连接器 | HTTP 抓取 / Playwright 真浏览器 / BM25 证据库，MCP Server |
| 观测 | 事件全序（36 种）+ 指标 + 预算熔断 | 事件全序 + JSONL + Langfuse trace 旁路导出 |
| 评测 | 22 条端到端 + 检索 recall/MRR/nDCG + 压缩专项 | 8 条用例：成功率 / 工具 F1 / 引用率 / P50-P95 / 成本 |

两者共用同一套口径：**事件是唯一事实源，工具执行走同一串护栏，评测指标定义不变**。所以它们不是两个重复项目，而是"底座 + 业务验证"。

## 60 秒验证（全部离线）

```bash
# 主项目：200+ 项测试（全离线）+ 一次带工具轨迹的问答
cd hermes-agent && pip install -e ".[dev]"
python -m pytest -q
hermes ask "LoRA 的秩 r 通常设置为多少？"

# 业务项目：固定脚本 Mock，无需 Key 和网络
cd ../market-scout && pip install -e ".[dev]"
python -m market_scout.cli demo
```

## 目录

```text
hermes-agent/     主项目：Agent Runtime + 上下文工程 + RAG 知识层 + MCP + 评测
market-scout/     业务项目：竞品调研报告 Agent（LangGraph / Playwright / Langfuse）
rag-agent-v1/     简历对应项目：v1 纯 RAG Agent（原版，只读参考）
archive/          历史版本留档：v2 运行时（只读，不参与演示）
```

## 面试口径

- 简历只写两个项目：`hermes-agent`（深度）与 `market-scout`（广度）。
- 被问到"简历上那个 RAG 项目"时：它是 `rag-agent-v1/`，能力已并入 `hermes-agent/` 的知识层，两版都在同一仓库。
- 每个项目按四问回答：**解决什么问题 → 为什么这样设计 → 怎么评估 → 失败与改进了什么**。
