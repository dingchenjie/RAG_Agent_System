# Hermes Runtime

一个**可观测、可恢复、可评测**的 Agent Runtime（Harness）：把"模型 + 工具"这种 Demo 级组合，做成能托管长任务、可被人干预、出问题能复盘的生产级运行时。

> 定位：不追求"多 Agent 花活"，而是把 AI Agent 工程里最容易翻车的四件事做扎实——
> **上下文预算、工具契约与权限、状态持久化与恢复、运行轨迹与自动评测。**

---

## 1. 快速开始

```bash
cd hermes-runtime
pip install -e ".[dev]"

python -m pytest                 # 45 项测试，全离线（Mock Provider）
python scripts/demo.py           # 单 Agent 工具循环 + 多 Agent 主从协作
python scripts/run_eval.py --fail-under 0.99   # 离线评测，可做 CI 门禁
python scripts/mcp_demo.py       # 自建 MCP Server 被客户端桥接进注册表

uvicorn hermes.api.app:app --reload --port 8000
```

接入真实模型（默认 Mock，不需要 Key）：

```bash
export HERMES_LLM_PROVIDER=zhipu
export ZHIPU_API_KEY=xxxx
```

---

## 2. 架构

```
                        ┌──────────────── HTTP / SSE（hermes/api/app.py）────────────────┐
                        │ POST /v1/runs   GET /events   POST /cancel /confirm /resume   │
                        │ /v1/orchestrate  /v1/knowledge/*  /v1/memory/*  /metrics      │
                        └───────────────────────────┬───────────────────────────────────┘
                                                    │
   ┌───────────────────────────────────────────────▼──────────────────────────────────────────────┐
   │ RunScheduler（优先级队列 + N worker + 背压）                                                  │
   └───────────────┬──────────────────────────────────────────────────────────────┬───────────────┘
                   │                                                              │
   ┌───────────────▼───────────────────────┐                    ┌─────────────────▼──────────────┐
   │ AgentRuntime（单 Agent ReAct 循环）    │                    │ Orchestrator（主 Agent 规划调度）│
   │  ├ ContextBuilder  上下文预算/压缩/裁剪 │                    │  ├ RulePlanner  能力域拆子任务   │
   │  ├ Tracer + UsageMeter 追踪/预算       │                    │  ├ 并行子 Agent（最小工具集）    │
   │  ├ CancelToken 可中断（LLM 与工具）     │                    │  ├ Blackboard 共享结论           │
   │  ├ checkpoint 断点续跑（跨进程）        │                    │  └ 汇总失败可降级为拼接           │
   │  └ 预演确认 → paused → confirm 恢复     │                    └─────────────────────────────────┘
   └───────────────┬───────────────────────┘
                   │
   ┌───────────────▼─────────────────────────────────────────────────────────────────────────────┐
   │ ToolRegistry：未知工具 → 权限 → Pydantic 校验 → 确认 → 幂等(run 作用域) → 超时 → 重试        │
   │              → 截断 → 审计(tool_audit)                                                      │
   │  内置：kb_search / memory_recall / memory_write / calculator / clock / python_sandbox        │
   │  MCP：MCPToolBridge 把外部 MCP Server 工具动态注入，使用同一套权限/审计/事件链路              │
   └───────────────┬─────────────────────────────────────────────────────────────────────────────┘
                   │
   ┌───────────────▼───────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
   │ Knowledge（增量索引/ACL/引用） │   │ Memory（提取/确认/纠错）  │   │ Observability            │
   │ chunk → BM25 + 向量 + RRF     │   │ propose → confirm/correct│   │ events 全序 → SSE/回放/评测│
   │ 相关性门控（防不相关强答）     │   │ → forget + TTL + 租户隔离 │   │ spans + Prometheus 指标   │
   └───────────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
                                         存储：SQLite（WAL），单文件可迁移
```

**事件驱动是全局设计核心**：所有行为（LLM 调用、工具执行、上下文压缩、记忆提议、子 Agent 委派）都写入 `events(run_id, seq)` 全序事件表。SSE 实时推送、事后回放、离线评测、故障复盘共用这一份数据源——不存在"日志和真实执行不一致"的问题。

---

## 3. JD 七条职责 → 代码落点

| JD 职责 | 本项目实现 | 关键文件 |
| --- | --- | --- |
| 1. Agent Runtime / Harness：模型调用、工具循环、任务调度、会话管理、结构化事件流、取消 | ReAct 循环 + 步数/预算限制；优先级调度器；SSE 事件流；`CancelToken` 可打断 in-flight LLM 与工具 | `runtime/runtime.py`、`runtime/scheduler.py`、`runtime/cancel.py`、`api/app.py` |
| 2. 上下文工程：预算分配、压缩、裁剪 | 分层预算（system/记忆/知识/历史）；旧工具结果头尾保留；更早对话抽取式摘要；当前轮整段保护；超预算成组丢弃且不破坏 tool_calls 配对 | `runtime/context.py` |
| 3. 工具与连接器：MCP/API/CLI、参数契约、权限、版本、异常、沙箱 | 工具声明 `args_model/version/scopes/timeout/retries/side_effect/requires_confirmation`；执行前 Pydantic 强校验；scope 鉴权；稳定错误码；AST 白名单 + 子进程隔离执行 | `tools/base.py`、`tools/registry.py`、`tools/sandbox.py`、`tools/mcp_bridge.py` |
| 4. RAG：文档处理、增量索引、检索重排、来源引用、权限过滤、版本管理 | 句边界分块 + 重叠；`doc_id+version` 版本覆盖；BM25 + 向量 + RRF；**相关性门控**；`[n]《标题》doc@v1#chunk-k` 引用；文档级 ACL | `knowledge/index.py`、`knowledge/service.py` |
| 5. 短期/长期记忆：提取确认、对象关联、召回、更新纠错、TTL、隔离 | 规则抽取 → `pending`；人工/自动确认后才 `active`；`correct` 生成新版本并 `supersedes`；`forget` 软删除；打分 = 余弦+词法+时效+置信；所有查询强制 tenant+user | `memory/service.py`、`memory/extractor.py` |
| 6. 多步可靠性：预演确认、幂等、超时重试、状态持久化、故障恢复 | 有副作用工具先返回 preview → run 进入 `paused` → `/confirm` 续跑；`X-Idempotency-Key` 去重；工具 run 作用域幂等；超时/退避/熔断；每步 checkpoint，进程重启后 `resume` 续跑 | `runtime/runtime.py`、`runtime/store.py`、`llm/resilience.py` |
| 7. 可观测与评测：轨迹、日志、用量、自动评测、回放、版本对比 | 事件全序 + span + Prometheus；token/成本预算；`EvalRunner` 指标；`replay.py` 时间线与 diff；`--fail-under` CI 门禁 | `observability/*`、`eval/*` |

---

## 4. 相对原 RAG Agent 项目，补上了什么

| 原项目的问题 | 后果 | 本项目怎么解决 |
| --- | --- | --- |
| `if/elif` 分发 + `eval()` 计算器 | 加工具要改核心循环，且存在任意代码执行风险 | 工具注册表 + 声明式契约；AST 白名单求值；沙箱工具跑在独立子进程 |
| 无权限概念 | 谁都能调所有工具、读所有文档 | scope 鉴权 + 文档级 ACL + 最小权限工具视图（子 Agent 只拿所需工具） |
| 无状态、无持久化 | 进程崩溃任务全丢，无法长任务 | 每步 checkpoint + 事件全序落库，跨进程 `resume` |
| 不能取消、不能干预 | 长任务只能等，写操作无法二次确认 | `CancelToken` 中断；`paused` + `/confirm` 预演确认 |
| 上下文"全塞进去" | 长对话必然超窗/丢关键信息 | 预算分配 + 摘要压缩 + 工具结果裁剪 + 当前轮保护 |
| 无记忆 | 每次对话从零开始，用户偏好无法沉淀 | 提取 → 确认 → 召回 → 纠错 → TTL 全生命周期，且租户隔离 |
| 仅检索 Top-K 直接拼 | 不相关问题也会硬答（幻觉源） | 检索后做相关性门控：实词覆盖率 / 向量相似度不达标就明确"没有检索到" |
| 评测只有 RAGAS faithfulness 一个分 | 无法定位"哪一步错了" | 基于事件轨迹的评测：成功率、工具 P/R/F1、引用覆盖、步数、token、P50/P95、成本 |
| 无工具生态 | 能力封闭 | MCP 双向：自建 Server 暴露知识/记忆能力，客户端把远端工具接进同一注册表 |
| 单 Agent 硬扛 | 复杂任务上下文膨胀、串行慢 | 主 Agent 规划 → 子 Agent 并行（最小工具集）→ 黑板汇总 → 单点失败隔离 |

---

## 5. 关键设计取舍（面试重点）

- **不引入 Chroma / sentence-transformers**：`HashingEmbedder` + 纯 Python BM25 + RRF，保证零重型依赖、离线可测。索引层抽象成 `KnowledgeIndex`，替换成 Milvus/Chroma 只换实现。取舍是语义泛化弱，所以加了**词法覆盖率门控**，宁可不答也不乱答。
- **确定性 Planner 而非"让模型自由规划"**：`RulePlanner` 与 `LLMPlanner` 共享同一 `plan()` 契约。企业高频任务里可解释性和稳定性优先；换成 LLM 规划不改下游代码。
- **Mock Provider 不是玩具**：它是"可回归"的前提——45 项测试和 5 条评测用例在没有网络和 Key 的情况下断言完整调用链（工具顺序、取消、恢复、熔断）。真实模型走 OpenAI 兼容 HTTP（httpx，不用 SDK）。
- **幂等键按 run 作用域**：读类工具结果随时间变化，跨 run 复用缓存会返回过期事实。这是开发中真实踩到的坑（同参数检索命中旧缓存导致答案与知识库不一致）。
- **上下文裁剪必须保配对**：丢 `assistant.tool_calls` 时必须连同其 `tool` 结果一起丢，否则 OpenAI 兼容接口会直接判请求非法。

---

## 6. 实测指标（Mock Provider，本地单机）

```
python scripts/run_eval.py --fail-under 0.99

| 指标 | 数值 |
| 任务数 | 5 |
| 任务成功率 | 100.00% |
| 工具选择 F1 | 1.0000 |
| 平均步数 | 1.0 |
| 平均 token | 1169 |
| P50 / P95 延迟 | 13 ms / 47 ms |
```

评测集覆盖：单跳 RAG、纯计算、多步（检索→计算）、ACL 拒答（应明确回答"没有"）、闲聊（不应误调工具）。延迟是本地 SQLite + Mock，**不代表真实模型时延**，工程上关心的是"轨迹可断言"而不是这个数字。

---

## 7. 已知边界（不吹）

- `python_sandbox` 是 **AST 白名单 + 独立子进程 + 超时强杀**，不是容器级隔离；生产应换成 gVisor/Firecracker 或独立执行服务。
- `HashingEmbedder` 是哈希特征，不是语义模型；同义改写召回弱，所以用 BM25 + 门控兜底。
- Planner 目前是规则版；LLM 规划器接口已留好但未接。
- SQLite 单文件适合单机与演示；多副本需换 Postgres + 消息队列（事件表结构已按 append-only 设计，迁移成本可控）。
- 分布式多 Agent 未做（当前是同进程 asyncio 并行 + 信号量限流）。

---

## 8. 目录

```
hermes/
  runtime/      runtime(ReAct循环) context(上下文工程) scheduler cancel store
  tools/        base(契约) registry(执行链) builtin sandbox mcp_bridge
  knowledge/    embedding index(BM25+向量+RRF+门控) service
  memory/       extractor service(确认/纠错/TTL/隔离)
  agents/       orchestrator(规划+并行+汇总) blackboard
  llm/          base mock zhipu resilience(超时/退避/熔断)
  observability/ events(全序事件) tracing(spans/用量) metrics(Prometheus)
  eval/         runner metrics replay
  api/          app(FastAPI + SSE)
  mcp_server/   server(自建 stdio MCP Server)
scripts/        demo.py run_eval.py mcp_demo.py
tests/          45 项，覆盖上下文/工具/知识/记忆/运行时/韧性/多Agent/API/评测/MCP
```

---

## 9. Docker

```bash
docker compose up --build      # http://localhost:8000/docs
```

## 10. 面试材料

60+ 道问答、按 JD 七条职责逐条拆解、边界问题（OpenManus / 自进化 / 分布式 / 沙箱）的标准答法，见 [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md)。
