# Hermes Agent —— AI Agent 岗敲门砖项目

本仓库现在只有一个主项目：[`hermes-agent/`](hermes-agent/)。

一个**可观测、可恢复、可评测**的 Agent 工程，覆盖 Agent Runtime、上下文工程、工具与 MCP、
RAG 知识层（PDF/HTML/DOCX/MD）、长期记忆、多 Agent 主从调度、权限沙箱、评测与部署。
默认离线可跑：`pip install -e .` 后 30 秒内 `pytest` 全绿；语义模型作为可选 extra 自动启用或降级。

```bash
cd hermes-agent
pip install -e ".[dev]"
python -m pytest                          # 66 项测试，全离线
hermes ask "LoRA 的秩 r 通常设置为多少？"   # 自动入库种子知识库，打印答案 + 工具轨迹
hermes eval --fail-under 0.99             # 离线评测门禁（含忠实度）
hermes serve --port 8000                  # HTTP + SSE
```

详细说明见 [hermes-agent/README.md](hermes-agent/README.md)，
架构图与时序见 [hermes-agent/docs/ARCHITECTURE.md](hermes-agent/docs/ARCHITECTURE.md)。

## 演进过程

| 版本 | 位置 | 内容 |
| --- | --- | --- |
| v1 | [`archive/rag-agent-v1/`](archive/rag-agent-v1/) | 纯 RAG Agent：ReAct + `if/elif` 工具 + Chroma + 交叉重排 + RAGAS 单指标 |
| v2 | [`archive/hermes-runtime-v2/`](archive/hermes-runtime-v2/) | Agent Runtime：上下文工程 / 工具九道关 / 记忆 / 多 Agent / 评测 / 可观测 |
| v3 | [`hermes-agent/`](hermes-agent/) | 合并版：v2 运行时骨架 + v1 真实文档链路，唯一对外项目 |

为什么这样演进、每一版丢了什么留下什么，见 [archive/README.md](archive/README.md)。

## 目录

```
hermes-agent/     主项目（唯一对外演示入口）
archive/          历史版本留档（只读，不参与演示）
LICENSE
```

> 旧版运行所需的 `core/config.py`、API Key 与 Chroma 目录不入库，`archive/` 仅作代码演进证据。
