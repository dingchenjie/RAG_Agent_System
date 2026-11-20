# market-scout：竞品调研报告 Agent

一个能"自己查资料、自己写报告、自己核对引用"的 Agent：给定竞品名称或调研问题，它会拆解子问题、并行检索公开页面（HTTP 抓取 + Playwright 真浏览器渲染）、把证据落库并编号，最后写出一份**每句结论都带 `[S<编号>]` 引用**的 Markdown 报告；证据不足时明确拒答。

> 技术栈：LangGraph 1.x（StateGraph + Send 并行 + Checkpointer + 人工审批中断）、Playwright、MCP（FastMCP）、FastAPI + SSE、SQLite FTS5、httpx、Langfuse（可选自托管）。

> 仓库位置：本目录是业务项目；同仓库主项目是 [`hermes-agent/`](../hermes-agent/)，另有 [`rag-agent-v1/`](../rag-agent-v1/)（v1 原版 RAG）。总入口见根 [README](../README.md)。

---

## 1. 场景与难点

| 难点 | 具体表现 | 本项目的解法 |
| --- | --- | --- |
| **任务形态不统一** | "Acme 定价多少"是单跳事实题，"对比 A 与 B 并给选型建议"是多步调研题，用同一套流程要么浪费要么不够 | 入口**规则路由**：短问题 + 事实型关键词 → 反应式单轮 ReAct；含"对比/调研/分析/报告"→ 深思熟虑式（先规划再并行） |
| **幻觉** | 模型会凭记忆编造不存在的价格、功能；或拿 A 公司的资料回答 B 公司的问题 | ① 所有结论必须来自工具返回的证据，Prompt 强制 `[S#]` 引用；② **实体门控**：问题里的产品名（Gamma）没在召回结果中出现，直接拒答而不是拿 Acme/Beta 冒充；③ `critique` 节点校验引用是否真实存在 |
| **死循环** | 模型反复用同样参数调同一个工具，烧钱且无进展 | 同工具同参数**第 2 次出现即拦截**（`check_loop`），写 `loop.detected` 事件，强制收敛到"基于已有证据作答"；外层还有 `max_steps` 步数上限 |
| **成本/失控** | 无限翻页、无限重试 | 每次 run 的页面预算 `SCOUT_MAX_PAGES`（默认 8）、工具超时 + 仅对 retryable 错误重试、预算 `SCOUT_MAX_RUN_COST_USD` |
| **不可复盘** | 报告错了，不知道是哪一步、哪条证据导致的 | 所有行为写入 SQLite `events`（唯一事实源，`run_id + seq` 全序）+ `data/traces.jsonl`，Web 端 SSE 实时回放；可选导出到 Langfuse 看 span 树 |
| **不确定性** | 真实模型评测贵且不稳定 | 内置确定性 `MockLLM`，离线即可跑通全链路与回归测试；真实模型走 OpenAI 兼容协议，切换只改环境变量 |

## 2. 架构

```text
                 ┌──────────────────────────── LangGraph StateGraph ────────────────────────────┐
 question ──► route ──► [reactive]  quick ─────────────────────────────────────────────┐         │
                 │                                                                     ▼         │
                 └──► [deliberative] plan ──Send×N──► collect(ReAct 子循环，并行) ──► merge ──► write ──► critique ──► revise(最多 1 次) ──► publish
                                                                                          │            │
                                                                                    （引用校验失败）──┘            │
                                                                                                     interrupt_before=["publish"] 可人工审批
```

- **route**：规则分类（`classify`），返回 `mode` 与理由，落 `route.decided` 事件。
- **plan**：真实模型让 LLM 输出 JSON 子问题，解析失败/离线时回退启发式切分。
- **collect**：每个子问题一个独立 ReAct 子循环，工具最小集（检索 + 抓取 + 浏览器），**失败互不阻塞**。
- **merge → write**：合并证据（按 `source_id` 去重）后写作，末尾附 `## 证据来源`。
- **critique → revise**：校验引用编号是否都真实存在、是否有结论无引用；不合格最多重写一次。
- **publish**：可配置 `interrupt_before=["publish"]`，人工确认后才发布（`scout ask --approve`）。

## 3. 工具定义（Function Calling / MCP 同一套注册表）

| 工具 | 作用 | 关键约束 |
| --- | --- | --- |
| `fetch_page(url)` | httpx 抓静态页，lxml 去 script/style/nav/footer 抽正文，分块入库并返回 `[S<id>]` | SSRF 防护：仅 http/https、域名白名单、DNS 解析后拒绝私网/回环/链路本地地址；计入页面预算 |
| `browse_page(url)` | **真实 Playwright**（Chromium headless）渲染 JS 页面，取 `inner_text` 并截图到 `data/shots/`，入库返回 `[S<id>]` | 25s 超时；同样受域名白名单与页面预算约束 |
| `search_evidence(query, k)` | 在当前 run 已抓取的证据里做 SQLite FTS5/BM25 检索，返回带 `[S#]` 的片段 | 中文 bigram 索引与查询同口径；**实体门控**：专有名词未命中则返回空并提示拒答 |

所有工具统一走 `ToolRegistry.execute`：未知工具 → 参数必填校验 → 页面预算 → 超时 → 仅 retryable 重试 → 结果截断 → `tool.started/finished` 审计事件。MCP Server（`python -m market_scout.mcp_server.server`）把同样三个能力以标准 MCP 工具暴露给任意客户端。

## 4. 稳定性设计

| 风险 | 机制 |
| --- | --- |
| 死循环 | `check_loop`（同工具同参数 2 次即拦截）+ `max_steps` + `recursion_limit=40` |
| 幻觉 / 张冠李戴 | 证据强制引用、`## 证据来源`、实体门控、`critique` 引用存在性校验 + 一次 revise |
| 抓取失控 | 页面预算 `SCOUT_MAX_PAGES`、工具超时、结果截断 `max_result_chars` |
| 模型/网关错误 | 错误分类：`llm_config_error` 等配置类错误不重试、不计熔断；429/5xx 才退避重试 |
| 流程中断 | LangGraph `MemorySaver` checkpoint；`interrupt_before=["publish"]` 人工审批后 `resume_run` 只补跑 publish |
| 观测缺失 | Langfuse 未配置时自动降级为 SQLite 事件 + JSONL，主流程零依赖 |

## 5. 观测与评测

**观测**：`events(run_id, seq)` 是唯一事实源，SSE 只是订阅者；`Tracer` 同时写 JSONL，并在配置了 `LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY` 时把每个事件导出为 Langfuse 观测（兼容 SDK v2–v5，同一 run 固定一条 trace，导出失败静默不影响主流程）。本地已实测：一次 run = 1 条 trace + 21 条事件观测。

**评测**（`python -m market_scout.eval.run_eval`，8 条用例：单跳事实 / 对比 / 分析 / 应拒答）：

| 指标 | 含义 |
| --- | --- |
| 任务成功率 | 关键字全覆盖 + 引用合规（应拒答用例要求 0 引用）+ 状态 succeeded |
| 工具 F1 | 期望工具集 vs 实际调用集 |
| 关键字覆盖 | 报告是否包含数据集要求的事实关键词（防"看起来像但没答到"） |
| 引用率 | 有引用的段落占比 |
| P50/P95、成本、循环拦截次数 | 性能与稳定性 |

Mock 基线（确定性、可回归）：**8/8 = 100%**，工具 F1 = 1.0，引用段落占比 0.6375，P50 = 0.19s，P95 = 0.42s，成本 $0。数字用于回归而非能力上限；换真实模型后同一套口径继续用，变差即是要发现的问题。

真实模型冒烟（agicto `gpt-4o-mini`，`调研 https://example.com`）：**31.96s / 3 页 / $0.001045**，`fetch_page + search_evidence`，页面未提供定价与功能时如实输出"证据不足"，`## 证据来源` 带真实标题与 URL。

Prompt A/B 见 [docs/prompt_iterations.md](docs/prompt_iterations.md)（含"Mock 下 A/B 为什么没有区分度"的诚实说明）。

## 6. 快速开始

```bash
# 0) 依赖
pip install -e .                      # 或 pip install langgraph langfuse mcp fastapi uvicorn httpx beautifulsoup4 lxml playwright
python -m playwright install chromium  # 只跑 fetch_page / mock 时可不装；国内网络可加 PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.npmmirror.com/binaries/playwright
# pip 慢时可换阿里云镜像：-i https://mirrors.aliyun.com/pypi/simple/

# 1) 离线跑通（零 Key、零网络；`serve` 在 mock 模式下会自动种入内置竞品资料）
python -m market_scout.cli demo
python -m market_scout.cli ask "对比 Acme CRM 与 Beta CRM 的定价与核心功能" --provider mock

# 2) 接真实模型（OpenAI 兼容网关）
$env:SCOUT_LLM_PROVIDER="openai"; $env:SCOUT_LLM_API_KEY="sk-..."; $env:SCOUT_LLM_MODEL="gpt-4o-mini"
python -m market_scout.cli ask "调研 <某竞品> 的定价与核心功能" --approve

# 3) 评测 / 服务 / 回放
python -m market_scout.eval.run_eval --provider mock --prompt v2
python -m market_scout.cli serve                 # http://127.0.0.1:8099 （输入问题 → SSE 事件流 → 报告）
python -m market_scout.cli runs                  # 最近 run
python -m market_scout.cli events <run_id>       # 全序事件回放

# 4) MCP Server（给任意 MCP 客户端用）
python -m market_scout.mcp_server.server

# 5) Langfuse 自托管（可选，观测旁路）
docker compose -f docker-compose.langfuse.yml up -d
# 打开 http://localhost:3000 ，账号 demo@example.com / demo1234
# 项目 API Key 已预置：pk-lf-market-scout / sk-lf-market-scout
$env:LANGFUSE_HOST="http://localhost:3000"; $env:LANGFUSE_PUBLIC_KEY="pk-lf-market-scout"; $env:LANGFUSE_SECRET_KEY="sk-lf-market-scout"
python -m market_scout.cli demo     # 运行后可在 Langfuse 里看到按 run 聚合的 21 条事件
# 提示：首次启动会自动建 langfuse bucket；若是旧数据卷导致没建，执行
#   docker exec market-scout-minio-1 sh -c "mc alias set local http://localhost:9000 minio miniosecret; mc mb --ignore-existing local/langfuse"

# 6) 测试
pytest -q && ruff check market_scout tests
```

> 国内网络提示：Docker Hub 直连较慢时可用镜像拉取后打回标准 tag，例如
> `docker pull docker.1ms.run/langfuse/langfuse:3 && docker tag docker.1ms.run/langfuse/langfuse:3 langfuse/langfuse:3`。
> 官方 `minio/minio` 已从 Docker Hub 下架，本地可用 `bitnamilegacy/minio` 打同名 tag（compose 里已注明）。

## 7. 目录结构

```text
market_scout/
  config.py          # 全部配置来自环境变量，默认离线可跑
  store.py           # SQLite：runs / events / sources / chunks / FTS5(BM25，中文 bigram)
  trace.py           # Tracer：事件落库 + JSONL + Langfuse span 导出（失败静默降级）
  llm.py             # OpenAI 兼容客户端（错误分类/重试/计费）+ 确定性 MockLLM
  prompts.py         # v1/v2 四类 Prompt（planner/researcher/writer/critic）
  tools/             # registry（九道关式执行链）+ web/fetch + browser(Playwright) + evidence(BM25/实体门控)
  graph/             # state / nodes（route/plan/collect/quick/write/critique/revise/publish）/ build
  engine.py          # 装配 → 建图 → 执行 → 落库/落盘；run_question / resume_run
  cli.py             # ask / demo / serve / eval / runs / events
  api/app.py         # FastAPI + SSE；api/static/index.html 单页前端
  mcp_server/        # FastMCP Server（fetch_page / browse_page / search_evidence）
  eval/              # dataset.json + metrics.py + run_eval.py
tests/               # 路由、死循环、SSRF、检索、指标、离线端到端
docker-compose.langfuse.yml   # Langfuse v3 自托管（web/worker/pg/clickhouse/redis/minio）
```

## 8. 与另一个项目（hermes-agent）的分工

| | hermes-agent | market-scout |
| --- | --- | --- |
| 定位 | Agent **运行时/Harness**：长任务、恢复、权限、评测底座 | 一个**完整业务 Agent**：调研、抓取、写报告 |
| 编排 | 手写 ReAct 循环 + 优先级调度 | LangGraph 图编排 + Send 并行 + 人工审批中断 |
| 工具 | 声明式契约 + MCP 双向 + 连接器 | 抓取/浏览器/检索三件 + 同一套执行链 |
| 观测 | events + span + Prometheus | events + JSONL + Langfuse（旁路导出） |

两者不重复造轮子：harness 解决"怎么可靠地跑"，market-scout 解决"跑什么、跑出什么结果"。

## 9. 面试讲述提纲（90 秒）

1. **场景**：竞品调研是"多步 + 多来源 + 必须可核对"的典型任务，正好覆盖 ReAct、图编排、并行子任务、引用与拒答。
2. **架构**：入口规则路由区分"单跳事实"和"深度调研"；后者拆子问题 `Send` 并行，各自独立 ReAct，失败互不阻塞；写完由 critique 校验引用再发布，可人工审批。
3. **工具**：三个工具、一套注册表、统一校验/超时/重试/截断/审计；MCP Server 复用同一实现。
4. **稳定性**：死循环靠"同工具同参数 2 次即拦截"；幻觉靠"强制引用 + 实体门控 + critique 校验"；成本靠页面预算。
5. **观测与评测**：所有行为一条事件流，SSE/回放/Langfuse 都是消费者；评测有成功率、工具 F1、关键字覆盖、引用率、P50/P95。
6. **边界**：Mock 下 A/B 没有区分度（Mock 不走 Prompt），真实模型数字才算数；浏览器只做真实 Playwright，没做 DOM 级别的自主点击规划。
