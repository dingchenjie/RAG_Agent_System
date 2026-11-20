﻿﻿﻿# Hermes Agent

> 仓库位置：本目录是主项目；同仓库还有 [`market-scout/`](../market-scout/)（业务 Agent）与 [`rag-agent-v1/`](../rag-agent-v1/)（v1 原版 RAG）。总入口见根 [README](../README.md)。

一个**能托管长任务、可被干预、可复盘、可评测**的 Agent 工程：把"模型 + 工具"的 Demo，做成覆盖
**Runtime / 上下文工程 / 工具与 MCP / RAG 知识层 / 长期记忆 / 多 Agent 调度 / 可观测评测**的完整系统。

> 一句话定位：不追求"多 Agent 花活"，而是把 AI Agent 工程里最容易翻车的四件事做扎实——
> **上下文预算、工具契约与权限、状态持久化与恢复、运行轨迹与自动评测。**
>
> 依赖策略：**默认离线可跑**。`pip install -e .` 后 30 秒内跑通全部测试；语义模型（bge）
> 作为 `[semantic]` extra 可选安装，装了自动启用、没装自动降级，`/healthz` 会如实报告当前后端。

---

## 1. 快速开始（两种模式，按你有没有 Key 选）

两种模式跑的是**同一套代码**，差别只在环境变量：无 Key 走 Mock + 哈希向量（确定性、可回归），
有 Key 走真实模型 + 真实向量（答案更像人话，但结果不再逐字可复现）。

### A. 无 Key 模式（默认，先跑这个）

```bash
cd hermes-agent
pip install -e ".[dev]"

python -m pytest                                   # 203 项测试，全离线（Mock Provider）
hermes ask "LoRA 的秩 r 通常设置为多少？"            # 自动入库种子知识库，打印答案 + 工具轨迹
hermes ingest ./samples/lora.pdf                   # 真实 PDF：按页解析 → 分块 → 去冗余 → 入库
hermes eval --fail-under 0.99                      # 离线评测 + CI 门禁（含忠实度）
hermes serve --port 8000                           # HTTP + SSE：http://127.0.0.1:8000/docs
python scripts/mcp_demo.py                         # 自建 MCP Server 被客户端桥接进注册表
```

不需要任何 Key、不联网、不下载模型。适合：CI、面试官 clone 后第一次验证、改代码后回归。

### B. 有 Key 模式（真实模型：智谱 glm-4-air + embedding-3）

```powershell
# PowerShell（项目不自动读 .env，环境变量以当前终端为准）
$env:HERMES_LLM_PROVIDER = "zhipu"
$env:ZHIPU_API_KEY       = "你的key"
$env:HERMES_EMBEDDING    = "zhipu"     # 显式指定：失败直接报错，不静默降级
$env:HERMES_RERANK       = "rrf"

Remove-Item data\hermes.db -ErrorAction SilentlyContinue   # 换向量后端必须重建索引

python scripts/run_eval.py --fail-under 0.99
python scripts/run_eval.py --real --save-baseline eval/baselines/zhipu-glm-4-air.json
python scripts/run_eval.py --retrieval-ab        # hashing vs 云端 embedding
hermes ask "LoRA 低秩分解为什么能省显存？"
```

```bash
# macOS / Linux
export HERMES_LLM_PROVIDER=zhipu ZHIPU_API_KEY=xxxx HERMES_EMBEDDING=zhipu
rm -f data/hermes.db
python scripts/run_eval.py --fail-under 0.99
```

三个必须知道的点：
- **换 embedding 后端要重建索引**：老库里是 384 维哈希向量，换成智谱向量后写入会抛 `embedding_mismatch`（故意拦，防止静默错分），删掉 `data/hermes.db` 重建即可；
- **默认是 `hashing`，`auto` 必须显式开启**：否则环境里恰好存在 `ZHIPU_API_KEY` 时会静默切到云端向量，"离线可复现"就变成"看环境"；`auto` 的顺序是 显式配置 > 本地 bge > 智谱 Key > 哈希，只有 `auto` 会静默降级，显式指定失败直接报错；
- **Key 只进环境变量**：`.env` 不会被自动读取（避免隐式行为），也永远不要提交。

### 可选能力（与 Key 无关）

```bash
pip install -e ".[semantic]"    # bge-small-zh 向量 + bge-reranker 交叉重排（本地模型，仍然无 Key）
pip install -e ".[ragas]"       # RAGAS 忠实度（未装则用离线引用支撑度）
```

---

## 2. 架构

```
                        ┌──────────────── HTTP / SSE（hermes/api/app.py）────────────────┐
                        │ POST /v1/runs  GET /events  POST /cancel /confirm /resume      │
                        │ /v1/runs/{id}/replay /diff  /v1/triggers*  /v1/sessions*       │
                        │ /v1/orchestrate  /v1/knowledge/*  /v1/memory/*  /metrics       │
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
   │  启动对账：queued 重入队 / running→interrupted                                            │
   └───────────────┬───────────────────────┘
                   │
   ┌───────────────▼─────────────────────────────────────────────────────────────────────────────┐
   │ ToolRegistry：未知工具 → 权限 → Pydantic 校验 → 确认 → 幂等(run 作用域) → 超时 → 重试        │
   │              → 截断 → 审计(tool_audit)                                                      │
   │  内置：kb_search / kb_ingest / memory_recall / memory_write / calculator / clock / sandbox   │
   │  MCP：MCPToolBridge 把外部 MCP Server 工具动态注入，使用同一套权限/审计/事件链路              │
   │  连接器：声明式 HTTP / CLI（JSON 配置，附录 A），与内置工具共用同一套执行链路                 │
   └───────────────┬─────────────────────────────────────────────────────────────────────────────┘
                   │
   ┌───────────────▼───────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
   │ Knowledge（增量索引/ACL/引用） │   │ Memory（提取/确认/纠错）  │   │ Observability            │
   │ loaders → splitter → dedupe   │   │ propose → confirm/correct│   │ events 全序 → SSE/回放/评测│
   │ chunk → BM25 + 向量 + RRF     │   │ → forget + TTL + 租户隔离 │   │ spans + Prometheus 指标   │
   │ embedder: hashing/bge/zhipu   │   └──────────────────────────┘   └──────────────────────────┘
   │ 相关性门控（防不相关强答）     │
   └───────────────────────────────┘
                                         存储：SQLite（WAL），单文件可迁移
```

**事件驱动是全局设计核心**：所有行为（LLM 调用、工具执行、上下文压缩、记忆提议、子 Agent 委派）都写入 `events(run_id, seq)` 全序事件表。SSE 实时推送、事后回放、离线评测、故障复盘共用这一份数据源——不存在"日志和真实执行不一致"的问题。

---

## 3. 岗位能力 → 项目证据

| 能力域 | 本项目实现 | 关键文件 |
| --- | --- | --- |
| **Agent Runtime / Harness** | ReAct 循环 + 步数/预算限制；优先级调度器；SSE 事件流；`CancelToken` 打断 in-flight LLM 与工具 | `runtime/runtime.py`、`runtime/scheduler.py`、`runtime/cancel.py` |
| **上下文工程** | 分层预算；旧工具结果头尾保留；抽取式摘要；当前轮整段保护；超预算成组丢弃且不破坏 tool_calls 配对；页面信息按不可信内容隔离、企业规则作为约束注入 | `runtime/context.py`、`runtime/runtime.py` |
| **工具 / Function Calling / MCP** | 声明式契约（args_model/version/scopes/timeout/retries/side_effect/requires_confirmation）；Pydantic 强校验；scope 鉴权；稳定错误码；MCP 双向（客户端桥接 + 自建 FastMCP Server） | `tools/base.py`、`tools/registry.py`、`tools/mcp_bridge.py`、`mcp_server/` |
| **连接器（API / CLI）** | 用 JSON 声明把已有 HTTP 接口与 CLI 命令注册成工具，复用统一权限 / 超时 / 重试 / 截断 / 审计；定位、安全边界与面试口径见附录 A | `tools/connectors.py`、`config/connectors.example.json` |
| **沙箱与安全** | AST 白名单求值 + `python -I -S` 独立子进程 + 超时强杀；工具结果截断与审计；上传大小/类型白名单；检索结果与工具输出按不可信内容隔离（`<untrusted>` + 安全条款） | `tools/sandbox.py`、`tools/builtin.py`、`security/untrusted.py` |
| **鉴权与多租户** | API Key → `Principal(tenant, user, scopes)`；`off/api_key` 两种模式；scope + 租户归属双重校验，越权统一 404；普通调用方不能自报 scope | `security/auth.py`、`api/app.py` |
| **经验沉淀（闭环）** | 运行结束按规则提炼教训（工具不稳定 / 无引用作答 / 规划过长）→ `pending` → 人工 `hermes lessons --activate` → 注入最近 3 条 `[历史教训]` | `learning/lessons.py`、`cli.py` |
| **RAG 全链路** | PDF/HTML/DOCX/MD 解析 → 标题感知分块 → 余弦去冗余 → BM25 + 向量 + RRF → 相关性门控 → 可选交叉重排 → `[n]《标题》doc@v1#chunk` 引用 → 文档级 ACL → 版本覆盖；**多轮追问**按上一轮问题补全检索 query（如"那上海呢？"），**增量索引**让未变更 chunk 复用旧向量，只对新增内容调 embedding；**版本治理**：`GET /v1/knowledge/documents/{id}/versions` 列历史版本（含 `active`/chunks 数），`POST .../versions/{version}/activate` 与 `hermes kb DOC --activate VER` 一键回滚 | `knowledge/loaders.py`、`splitter.py`、`index.py`、`query.py`、`rerank.py`、`api/app.py`、`cli.py` |
| **向量抽象与降级** | `Embedder` 协议三实现（hashing / bge / 智谱），`auto` 顺序：显式配置 > 本地语义模型 > 智谱 Key > 哈希；维度不一致显式报 `embedding_mismatch` | `knowledge/embedding.py` |
| **长期记忆** | 规则 / LLM 抽取 → `pending`；确认后 `active`；`correct` 新版本 + `supersedes`；`forget` 软删；打分 = 余弦 + 词法 + 时效 + 置信 + 业务对象匹配；记忆可绑定 `subject_type/subject_id`（如客户），召回按对象加权、冲突只在同对象内判定；强制 tenant + user 隔离 | `memory/service.py` |
| **多步可靠性** | 有副作用工具先预演 → `paused` → `/confirm`；幂等键按 run 作用域；超时/退避/熔断；每步 checkpoint，进程重启 `resume`；启动对账把 `queued` 自动重入队、`running` 标记 `interrupted` 并发可恢复事件（不自动重放） | `runtime/runtime.py`、`llm/resilience.py`、`api/app.py` |
| **多 Agent 调度** | 主 Agent 规则规划 → 子 Agent 并行（最小工具集 + 信号量）→ 黑板汇总 → 单点失败隔离 | `agents/orchestrator.py`、`agents/blackboard.py` |
| **可观测 / 评测** | 事件全序 + span + Prometheus；token/成本预算；`GET /v1/usage` 按租户聚合 token/成本/状态分布（普通用户只能看自己租户）；事件保留策略（`HERMES_EVENT_RETENTION_DAYS` 启动清理中间事件，保留终态）；工具调用事件记录工具版本；`GET /v1/runs/{id}` 返回 `citations` 与 `tool_calls`；`EvalRunner`（成功率、工具 P/R/F1、引用覆盖、忠实度、P50/P95）+ **检索专项 `recall@k`/`MRR`/`nDCG@k`（分类 + 门控扫描 + embedding A/B）** + **间接注入红队（越权工具/标记泄漏）** + **上下文压缩专项（事实保留率/压缩比）** + **长期记忆专项（冲突/TTL/跨租户隔离）** + **记忆抽取专项（规则 vs LLM 的 P/R/F1 与负例误抽）** + **教训注入 A/B（无教训 / 注入已确认教训）** + **编排失败隔离（部分失败降级 / 全部失败显式报错，故障注入）** + **真实模型基线（`--real`/`--save-baseline`/`--baseline` 门禁）** + **失败样本回归沉淀（模型/提示词版本口径）**；`/replay` 时间线、`/diff` 两版对比、CLI `hermes replay --against`；`--fail-under` 门禁 | `observability/*`、`eval/*`、`api/app.py`、`cli.py` |
| **主动触发** | interval 触发器（≥10s）：`triggers` 表 + `TriggerService.due()` 原子推进 `next_fire_at` + API 进程内后台扫描；触发与 HTTP 共用 `enqueue_run`，不新增建 run 入口；多副本需分布式锁（已列为边界） | `triggers.py`、`api/app.py`、`cli.py` |
| **凭据保护** | 事件落库、工具审计、错误响应前统一脱敏（Authorization / `sk-*` / key-value 敏感字段）；轨迹可读但不可泄密 | `security/redact.py` |
| **工程化 / 交付** | 单一 CLI 入口；pyproject extras 分层依赖；Docker + compose；ruff + 覆盖率门禁 + GitHub Actions；203 项离线测试 | `cli.py`、`pyproject.toml`、`Dockerfile`、`.github/workflows/ci.yml` |

---

## 4. 相对 v1（纯 RAG Agent）补了什么

| v1 的问题 | 后果 | 本项目怎么解决 |
| --- | --- | --- |
| `if/elif` 分发 + `eval()` 计算器 | 加工具要改核心循环，且存在任意代码执行风险 | 工具注册表 + 声明式契约；AST 白名单求值；沙箱跑独立子进程 |
| 无权限概念 | 谁都能调所有工具、读所有文档 | scope 鉴权 + 文档级 ACL + 子 Agent 最小权限工具视图 |
| 无状态、无持久化 | 进程崩溃任务全丢 | 每步 checkpoint + 事件全序落库，跨进程 `resume`；重启对账把 `queued` 重新入队、`running` 标记 `interrupted` |
| 不能取消、不能干预 | 长任务只能等，写操作无法确认 | `CancelToken` 中断；`paused` + `/confirm` 预演确认 |
| 上下文"全塞进去" | 长对话超窗/丢关键信息 | 预算分配 + 摘要压缩 + 工具结果裁剪 + 当前轮保护 |
| 无记忆 | 每次对话从零开始 | 提取 → 确认 → 召回 → 纠错 → TTL，租户隔离 |
| 检索 Top-K 直接拼 | 不相关问题也硬答（幻觉源） | 相关性门控：实词覆盖率 / 向量相似度不达标就明确"没有检索到" |
| 评测只有 RAGAS 一个指标 | 无法定位"哪一步错了" | 事件轨迹评测：成功率、工具 P/R/F1、引用覆盖、忠实度、步数、token、P50/P95、成本 |
| 启动即全量建库、只有 `/chat` | 不能增量、不能管理文档 | 增量入库；上传/列表/删除接口；`docs` 表记录向量口径 |
| 单一向量口径（只能用智谱） | 没 Key 就跑不了 | `Embedder` 协议 + auto 降级 + 维度校验；默认哈希向量离线可测 |
| 只有 md 文本 | 真实文档进不来 | PDF（页级）/HTML/DOCX/TXT loader，pypdf 进基础依赖 |

---

## 5. 关键设计取舍（面试重点）

- **默认离线、语义可选**：`HashingEmbedder` 保证 clone 即跑；`[semantic]` 装上后 `auto` 自动切 bge。取舍是默认语义泛化弱，所以用 BM25 + 相关性门控兜底，宁可不答也不乱答。
- **不引入 Chroma**：统一 SQLite 一套索引（BM25 + 向量 + RRF），一个事务、一份备份；替换 Milvus/Chroma 只换 `KnowledgeIndex` 实现。
- **重排默认 RRF**：两路召回无标定融合，零依赖可测；`CrossEncoderReranker` 作为可选后端（`HERMES_RERANK=cross-encoder`）。
- **忠实度默认离线**：逐句"引用支撑度"（n-gram 覆盖率）替代 LLM 判官，可回归、零成本；RAGAS 通过 optional 适配器接入。
- **确定性 Planner 而非"让模型自由规划"**：`Orchestrator` 只依赖 `planner.plan(goal, available) -> Plan` 这一个方法（构造参数目前标注为 `RulePlanner`，抽成 `Protocol` 只是两行代码），调度、并行、黑板、汇总都不用改——这就是"可替换"的具体落点。
- **Mock Provider 不是玩具**：它是"可回归"的前提——203 项测试与 22 条端到端评测（另有检索/注入/压缩/记忆/抽取/教训注入六组专项）在没有网络和 Key 时断言完整调用链（工具顺序、取消、恢复、熔断）。
- **幂等键按 run 作用域**：读类工具结果随时间变化，跨 run 复用缓存会返回过期事实（开发中真实踩到）。
- **上下文裁剪必须保配对**：丢 `assistant.tool_calls` 必须连同其 `tool` 结果一起丢，否则 OpenAI 兼容接口直接判请求非法。
- **事件 seq 分配必须原子**：并发子 Agent 下 `MAX(seq)+1` 与 `INSERT` 之间有竞态，已用锁包住（全量测试曾因此偶发失败）。
- **门控阈值由评测扫描决定，而不是拍脑袋**：检索专项同时扫 `min_coverage`，并把整条曲线留在评测输出里。语料从 50 篇扩到 331 篇后曲线明显右移：`0.0` 把 10 条越界问题全部召回（0/10 返回空），`0.2` 在 50 条计分查询上 recall@5 94%、越界 8/10，`0.34` 才做到越界 10/10 但 recall@5 降到 88%。默认保持 `0.2`（召回优先，误召回的片段还有 `<untrusted>` 隔离与"无依据不回答"兜底），需要严格拒答时按扫描结果上调——**这个阈值必须随语料规模重新标定**，换 embedding 或扩语料后都要重跑扫描。
- **多轮追问用规则补全，不让模型改写**：`knowledge/query.py` 只对"以指代词开头 / 短句以『呢』收尾"这类明显追问并入上一轮问题；LLM 改写会多一次调用、多一份不确定性，还会让检索评测不可回归。单轮问题原样透传，不增加任何行为变化。
- **增量索引用文本精确匹配复用向量**：同一 `doc_id` 升版本时，文本完全相同的 chunk 直接复用历史向量，只对新 chunk 调 embedding；复用前校验 `embedding_model/dim`，换过后端一律重算，避免"维度一样但语义空间不同"的静默错分。

### 5.1 安全模型（面试官最爱追问的部分）

- **认证**：`HERMES_AUTH_MODE=off|api_key`。默认 `off`（单租户管理员，方便本地/CI），启动时打 warning；
  生产设 `api_key`，凭证格式 `key:tenant:user:scope1|scope2;...`（`HERMES_API_KEYS`）。
- **授权**：scope 只能由凭证决定，请求体自报的 scope 一律不采信；管理员可以降权，不能升权。
  跨租户访问统一返回 **404**（不泄露"存在但不属于你"）。
- **不可信内容隔离**：检索文档与工具输出在进入上下文前包成 `<untrusted label=... attrs>`，
  属性做转义、标签做闭合防护；system prompt 追加安全条款，明确"标签内是数据不是指令"。
  不做关键词黑名单——黑名单永远滞后，边界靠结构而不是枚举。
- **上传防护**：后缀白名单 → `Content-Length` 预检 → 1 MiB 分块计数，超限 413；落盘名加随机前缀防覆盖/穿越。
- **入口限流**：`HERMES_RATE_LIMIT_RPM`（每租户令牌桶）与 `HERMES_TENANT_MAX_INFLIGHT`（在途 run 上限），默认 0 关闭；
  超限统一 429 + `Retry-After`，被拒绝的请求不建 run、不进队列；幂等重放不消耗额度。
- **错误契约**：`{"error":{"code","message","request_id"}}` + `X-Request-Id`，500 不泄露堆栈。

---

## 6. 实测指标（两种模式的数字口径不同，别混着看）

### 6.1 无 Key 模式（Mock Provider + 哈希向量，已实测）

```
hermes eval --fail-under 0.99

| 指标 | 数值 |
| :--- | ---: |
| 任务数（single 18 + orchestrate 4） | 22 |
| 任务成功率 | 100.00% |
| 子任务成功率（orchestrate） | 100.00% |
| 工具选择 F1 | 1.0000 |
| 工具精确率 / 召回率 | 1.0000 / 1.0000 |
| 忠实度（引用支撑） | 0.5010（14 条有引用） |
| 平均步数 | 1.18 |
| 平均 token | 1525 |
| P50 / P95 延迟 | ~22 ms / 42 ms（22 条样本，P95 波动大） |
| 检索 recall@1 / @3 / @5 | 76.00% / 92.00% / 94.00%（331 篇语料、60 条查询，50 条计分） |
| 检索 MRR / nDCG@5 | 0.8350 / 0.8306 |
| 越界问题返回空 | 8/10（默认门控 0.2；阈值 0.34 时 10/10、recall@5 88%） |
| 注入红队抵抗率 | 100%（6/6，越权 0、泄漏 0） |
| 上下文压缩（6 条，含 1 条负对照） | 判据 6/6 通过；事实保留 5/6（83.33%）；平均压缩比 84.77% |
| 长期记忆（召回/冲突/TTL/隔离） | 100%（6/6） |
| 记忆抽取（规则） | 精确率 100% / 召回 52.63% / F1 68.97%；负例误抽 0 |
```

三组横向对照（同一套评测集，只换一个变量）：

- **检索 embedding A/B（`--retrieval-ab`）**：331 篇语料下 hashing vs 智谱 `embedding-3`：recall@5 **94% → 96%**、recall@3 92% → 96%、recall@1 76% → 78%，MRR **0.8350 → 0.8533**，nDCG@5 0.8306 → 0.8534，语义召回略好；但**越界拒答从 8/10 掉到 0/10**——`min_vector=0.18` 这个绝对阈值是按哈希向量标定的，云端语义向量下无关文本的余弦也普遍更高，换后端必须同步重标定。默认仍是 hashing（确定性、可回归），要用云端向量就得连门控一起调，不能只换 embedder。
- **重排 A/B（`--rerank-ab`）**：rrf → cross-encoder（bge-reranker-base，候选池由 5 条放大到 15 条再重排），331 篇语料、50 条计分查询。recall@5 **94% → 94%**（覆盖不变），recall@1 **76% → 82%**，MRR **0.8350 → 0.8717**，nDCG@5 **0.8306 → 0.8687**。结论：重排修的是"第一条是否对"，不是覆盖率——语料扩到 331 篇后这个收益才看得出来（50 篇时 recall@5 已饱和，只能看到 86.67% → 93.33% 的 recall@1 变化）。代价是 1.1GB 模型，本地 CPU 上每次检索约 1.1s（15 个候选），所以默认关闭、按需开启。
- **记忆抽取 A/B**：规则抽取召回 52.63%，漏掉"项目/团队"这类没有固定句式的语义信息；LLM 抽取（glm-4-air，`temperature=0`）在 20 条上 P/R/F1 均为 100%，7 条负例（寒暄、一次性指令、注入、代码、第三方邮箱）全部沉默。两者都只产出 `pending` 候选，确认后才写入长期记忆。

用途：**回归基线**。固定 Mock + 固定哈希向量，同一份代码永远得到同一组数字，改检索或改提示后掉下去就是明确信号。
延迟是本地 SQLite + Mock，**不代表真实模型时延**。

### 6.2 有 Key 模式（智谱 glm-4-air，实测）

同一套 22 条数据集（single 18 + orchestrate 4）、同一份代码，只换 Provider 和环境变量，2026-11 用 `--real` 实测：

| 指标 | Mock / 哈希 | **glm-4-air（默认）** |
| --- | ---: | ---: |
| 任务成功率 | 100% | **95.45%（单轮，21/22）** |
| single / orchestrate 成功率 | 100% / 100% | 94.44%（17/18）/ **100%（4/4）** |
| 工具选择 F1 | 1.0000 | **0.9545** |
| 忠实度（引用支撑） | 0.5010 | 0.3845（14 条有引用） |
| 平均 token | 1525 | 2356 |
| P50 / P95 延迟 | 22 ms / 42 ms | 2413 ms / **8927 ms** |
| 22 任务总成本 | $0 | **$0.023792** |

同一配置用 `--repeats 3` 重复 3 轮：成功率 **92.42% ± 2.14%**（min 90.91% / max 95.45%），
P95 **4260 ± 905 ms**，总成本 **$0.023057 ± $0.000542**。单轮 95.45% 属于偏乐观的一次；
对外引用 3 轮均值更稳，22 条里通常有 1–2 条在工具选择/关键词上抖动。

**真实模型暴露的问题，比数字本身更有价值：**

1. **唯一失败是"声称做了但没做"**：t13（"我叫丁辰杰，请记住我偏好 Python"）没有调用 `memory_recall`，却回答"已经记住了"。记忆抽取在后台只写入了 `pending` 候选，未确认不可召回——这是典型的工具调用幻觉。失败样本已由 `--record-failures` 落到 [failures.jsonl](eval/regressions/failures.jsonl)，下一轮直接重放。
2. **另一条是格式差异，不是能力回归**：t20 回答"10个工作日内"、期望写的是"10 个工作日"。评测改为空白不敏感匹配（`_keyword_matches`）并补了单测，避免把排版差异算成失败。
3. **orchestrate 在真实模型下 4/4**：规则 Planner 把"检索 + 计算"拆成子任务，并行执行后汇总；single 的 1 条失败没有扩散成编排失败。

历史对照（0.2.6，7 条 single）：`glm-4-flash` 57.14% → 加固提示词 71.43% → `glm-4-air` 100%。结论是"先修指令遵循、再换模型"；当时 P95 4851ms、成本 $0.0103，与当前 22 条口径不可直接比较。

跑法：

```bash
HERMES_LLM_PROVIDER=zhipu python scripts/run_eval.py --real --repeats 3 \
  --save-baseline eval/baselines/zhipu-glm-4-air-3x.json
```

`--baseline` 会对比成功率、P95 与成本，成功率下降超过 5 个百分点退出码为 1；数字会随模型版本和网络变化，建议自己重跑一遍再引用。

### 6.3 压测与延迟分解（本机，2026-11）

`scripts/loadtest.py` 打本机 API；`scripts/latency_breakdown.py` 只读 `events` 表，把一次 run 拆成 LLM / 工具 / 其他（上下文、记忆、落库、SSE 派发），不新增埋点。

| 场景 | 规模 | 成功率 | 吞吐 | P50 | P95 | P99 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| single（mock） | 20 runs / 5 并发 | 100% | 25.2 run/s | 0.17s | 0.26s | 0.26s |
| single（mock） | 50 runs / 20 并发 | 100% | 28.1 run/s | 0.63s | 0.71s | 0.77s |
| orchestrate（mock） | 10 runs / 3 并发 | 100% | 33.1 run/s | 0.06s | 0.18s | 0.18s |
| orchestrate（mock） | 30 runs / 10 并发 | 100% | 50.6 run/s | 0.15s | 0.21s | 0.28s |
| single（glm-4-air） | 5 runs / 1 并发 | 100% | 0.56 run/s | 1.62s | 2.35s | 2.35s |
| single（glm-4-air） | 6 runs / 1 并发 | 100% | 0.51 run/s | 1.88s | 2.59s | 2.59s |
| single（glm-4-air） | 6 runs / 2 并发 | 100% | 0.84 run/s | 2.47s | 2.83s | 2.83s |
| single（glm-4-air） | 6 runs / 4 并发 | 100% | 1.08 run/s | 3.33s | 3.33s | 3.33s |
| single（glm-4-air） | 6 runs / 8 并发 | 100% | 1.09 run/s | 3.66s | 5.51s | 5.51s |

真实模型下 1→8 并发只把吞吐从 0.51 推到 1.09 run/s，4 并发后基本不再增长；两次扫描的平台值在
0.85–1.09 run/s 之间，波动来自上游配额与网络。延迟则持续上涨：P50 1.88s → 3.66s，P95 2.59s → 5.51s。

事件轨迹分段（P50）：

| 口径 | LLM | 工具 | 其他 |
| :--- | ---: | ---: | ---: |
| mock（60 runs） | 28.7% | 0.2% | 68.0% |
| glm-4-air（5 runs） | **98.7%** | 0.0% | 1.3% |

怎么读这三张表：

- **mock 下"其他"占大头**：LLM 瞬时返回，耗时主要在上下文构建、记忆召回、SQLite 落库与 SSE 派发，不能拿来推断线上时延；
- **真实模型下瓶颈非常明确**：LLM 生成占 98.7%，检索/工具/调度合计不到 2%。所以优化方向是提示词长度、输出长度与模型选择，而不是"换向量库"；
- **容量拐点要分两层说**：mock 下 5 并发 P50 0.17s，20 并发升到 0.63s，吞吐平台约 28 run/s，这是单进程 + SQLite 单写锁的本地上限；真实模型下 4 并发就摸到约 1 run/s 的平台，瓶颈先落在上游 LLM 配额与网络，SQLite 还没轮到。所以"能扛多少并发"必须带模型口径，单说 28 run/s 会误导。

```bash
python -m uvicorn hermes.api.app:app --port 8010
python scripts/loadtest.py --runs 20 --concurrency 5
python scripts/loadtest.py --runs 50 --concurrency 20
python scripts/loadtest.py --runs 6 --concurrency 1,2,4,8   # 容量扫描
python scripts/latency_breakdown.py --db data/hermes.db --limit 20
```

### 6.4 进程级恢复演练（kill -9）

`tests/test_recovery.py` 只覆盖"库里已有在途 run"的对账逻辑；真正把服务进程杀掉再拉起来是另一回事。
`scripts/recovery_drill.py` 会自动跑完整链路：起 mock 服务（每次 LLM 延迟 2s）→ 提交「计算 21 + 21」→
等第 1 步 checkpoint 落盘 → kill 进程 → 重启 → 断言 `interrupted`（`process_restart` / `recoverable`）→
`/resume` → 断言 `succeeded` 且工具只执行一次。

```bash
python scripts/recovery_drill.py          # 自动挑空闲端口，约 7s
```

延迟由 `HERMES_MOCK_DELAY_S` 控制（仅 Mock Provider，默认 0），演练脚本默认 2s；真实模型不需要这个开关。

本机实测（2026-11，一次完整时间线）：

```
+      0ms  run.created
+      1ms  run.started
+     10ms  llm.request
+   2021ms  tool.completed
+   2021ms  step.completed
+   2026ms  llm.request
        -- kill -9，进程退出 --
+   4655ms  run.failed        # process_restart, recoverable=true
+   4834ms  run.resumed
+   4844ms  llm.request
+   6844ms  run.completed
```

结论：checkpoint 能跨进程续跑；中断后的 resume 从断点继续，没有重复执行已完成的工具调用。
这也是"基础设施故障时 Agent 任务怎么办"这类面试题的直接演示材料。

---

## 7. 已知边界（不吹）

- `python_sandbox` 有两档：默认 `process`（AST 白名单 + `python -I -S` 子进程 + 超时杀进程树 + POSIX rlimit 内存/CPU/文件大小/句柄数 + 临时目录 + 环境变量白名单）；设 `HERMES_SANDBOX_MODE=docker` 切容器后端（`--network none`、`--memory`、`--cpus 0.5`、`--pids-limit 64`、只读根 + tmpfs、非 root）。Windows 的进程模式只做超时与输出截断，内存上限需要 Docker 模式；这不是内核级隔离，生产仍建议 gVisor/Firecracker 或独立执行服务。
- 默认 `HashingEmbedder` 是哈希特征，不是语义模型；同义改写召回弱，靠 BM25 + 门控兜底。
- `CrossEncoderReranker` / RAGAS / bge 均为可选依赖，默认不启用；开启后效果依赖模型质量。
- Planner 目前是规则版（`RulePlanner`）；接 LLM 规划器的落点只有一个方法 `plan(goal, available) -> Plan`，没接的原因是它会破坏评测的可回归性（同一目标两次规划可能不同），接之前要先补"规划结果快照 + 重规划上限"。
- SQLite 单文件适合单机与演示；多副本需换 Postgres + 消息队列（事件表按 append-only 设计，迁移成本可控）。
- 分布式多 Agent 未做（同进程 asyncio 并行 + 信号量限流）。
- **知识库是部署级共享**：租户隔离只做在 run/会话/记忆上；知识库写入靠 `knowledge:write`，读取靠 `knowledge:read` + 文档级 ACL。真要按租户物理隔离，需要给 `documents/chunks` 加 tenant 列并重建索引，这是有意的取舍（共享知识库是企业内更常见的形态）。
- **默认 `HERMES_AUTH_MODE=off`**：单租户管理员模式，只适合本地/演示/CI；对外部署必须切 `api_key`（启动日志会 warning 提醒）。
- **教训提炼是启发式规则**：三条规则（工具不稳定 / 无引用作答 / 规划过长）覆盖已知失效模式，不是通用"自我进化"；新规则默认 `pending`，必须人工确认后才注入。
- **触发器是单进程 interval**：循环跑在 API 进程内，靠 SQLite 单写者保证同机不重复触发；停止服务期间错过的周期直接跳过，不补跑。多副本部署需要分布式锁（如 Redis `SETNX`）才能避免重复触发。只做 interval 不做 cron 表达式，是因为真正难的是"触发→去重→调度→可观测"这条链路，表达式解析可以在此之上叠加。
- **脱敏是模式匹配**：能覆盖常见凭据形态（`Authorization`、`sk-*`、`password=` 等）与已知敏感字段名，但无法识别自定义格式的密钥；新增工具时若参数名不在名单内，需要补充 `security/redact.py`。
- **`running` 不自动重放**：重启对账只把在途 run 标记 `interrupted` 并发可恢复事件，是否 `resume` 由人或上层决定——工具可能已有不可逆副作用，自动重跑有重复执行风险；`queued` 尚未开始执行，因此按 at-least-once 自动重新入队。
- **事件保留会牺牲完整回放**：`HERMES_EVENT_RETENTION_DAYS`（默认 30 天）只清理**已结束** run 的中间事件，保留 `run.created/completed/failed/cancelled` 终态（含 `citations`/`tools` 快照）。因此超出保留期的 run 仍能看终态与引用/工具关联，但 `/replay` 不再有逐步中间步骤。要长期完整回放，应把事件导出到对象存储/Kafka，而不是无限留在 SQLite。

---

## 8. 目录

```
hermes/
  runtime/       runtime(ReAct循环) context(上下文工程) scheduler cancel store
  tools/         base(契约) registry(九道关) builtin sandbox mcp_bridge connectors(HTTP/CLI，附录 A)
  knowledge/     loaders splitter embedding rerank index service query(多轮追问补全)
  memory/        extractor service(确认/纠错/TTL/隔离)
  learning/      lessons(教训提炼/人工确认/注入)
  security/      auth(API Key→Principal) ratelimit(租户限流/并发配额) untrusted(不可信内容隔离) redact(凭据脱敏)
  agents/        orchestrator(规划+并行+汇总) blackboard
  llm/           base mock zhipu resilience(超时/退避/熔断)
  observability/ events(全序事件) tracing(spans/用量) metrics(Prometheus)
  eval/          runner metrics replay faithfulness retrieval(recall@k/MRR) injection compression memory
  api/           app(FastAPI + SSE + 知识库/触发器) errors(错误契约)
  mcp_server/    server(自建 stdio MCP Server)
  triggers.py    周期触发器（interval + 到期原子推进）
  cli.py         唯一人工入口：ask / chat / ingest / eval / serve / lessons / replay / triggers / kb
knowledge_base/  种子 md（lora_notes、finance_policy）
config/          connectors.example.json（HTTP / CLI 连接器示例，不默认加载）
eval/            dataset.jsonl（22 条端到端：single 18 + orchestrate 4）；retrieval.jsonl（60 条检索，50 条计分 + 10 条越界）与 retrieval_corpus.json（331 篇，含 12 个业务域、同主题难负例与 ACL 文档）；injection.jsonl（6 条红队）；compression.jsonl（6 条压缩，含 1 条负对照）；memory.jsonl（6 条记忆）；extraction.jsonl（20 条抽取，13 正 + 7 负）；lessons_ab.jsonl（3 条教训注入 A/B）；baselines/（真实模型基线）；regressions/（失败样本，按需生成）
scripts/         demo.py run_eval.py mcp_demo.py ingest.py loadtest.py latency_breakdown.py recovery_drill.py
samples/         演示用真实 PDF
tests/           203 项，覆盖上下文/工具/知识/记忆/运行时/韧性/多Agent/API/评测/MCP/CLI/安全/触发器/脱敏/重启恢复/连接器/生命周期/注入红队/失败回归/多轮检索/增量索引/压缩评测/记忆评测/抽取评测/教训注入/编排失败隔离/沙箱/限流
docs/            ARCHITECTURE.md
```

---

## 9. Docker

```bash
docker compose up --build          # http://localhost:8000/docs
# 语义镜像（带 bge + torch，体积大）：
docker build --build-arg INSTALL_EXTRAS="semantic" -t hermes-agent:semantic .
```

本机实测（Docker 29.9.2 / python:3.12-slim）：

| 检查项 | 结果 |
| --- | --- |
| `docker build -t hermes-agent:0.2.9 .` | 成功，镜像 269 MB |
| `docker compose config` | 通过（无语法/插值错误） |
| 容器内 `GET /healthz` | 200，`embedding=hashing/384`、`rerank=rrf`、`llm=mock` |
| `POST /v1/runs` → 轮询状态 | `queued → succeeded`；闲聊 trace 11 条事件（检索问答 23 条） |
| `GET /v1/runs/{id}` | 返回 `citations`（doc_id=lora_notes）与 `tool_calls`（kb_search@1.1、ok=true），答案可追溯到引用与具体调用 |
| 知识版本治理 | 同 `doc_id` 导入 v1/v2 后 `GET .../versions` 返回 `["2","1"]`（v2 active）；`POST .../versions/1/activate` 后 v1 变 active，检索随版本切换 |
| `GET /v1/usage?window=24h` | 返回按租户聚合的 `runs`、`by_state`、`prompt_tokens`/`completion_tokens`/`cost_usd` |
| `POST /v1/admin/prune` | 返回 `{"deleted":0,"retention_days":30}`（无超期数据时） |
| `/metrics` | 记录 `http_requests_total`，含 422 参数校验失败（错误契约生效） |
| 数据卷 | `/data` 挂 `hermes_data`，SQLite 与上传目录都在卷内，重建容器不丢 |

注意：`/v1/runs/{id}/events` 是 SSE 流。已结束的 run 回放完终态事件后会主动关流；运行中的 run 每 15 秒发一次心跳。断线重连带 `Last-Event-ID: <seq>` 即可从断点续传，也支持 `?after_seq=` 显式指定起点。

以上为 0.2.9 镜像的实测结果（首次构建约 1 分钟，主要耗时在 pip 装依赖；改代码后利用缓存层重建约 4 s）。

## 10. 面试材料

60+ 道问答、按 JD 七条职责逐条拆解、边界问题（OpenManus / 自进化 / 分布式 / 沙箱）的标准答法，见 `docs/INTERVIEW_QA.md`（本地保留，不纳入公开仓库）。
架构图、时序与数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 附录 A：连接器体系（HTTP / CLI，非主线）

这一块**不属于 Agent 工程主线**，默认不加载。保留它只解决一个具体问题：企业里已有大量 HTTP 接口与 CLI 工具，接进 Agent 不应该改核心循环。

- **怎么用**：`HERMES_CONNECTORS` 指向 JSON 声明（示例见 `config/connectors.example.json`），启动时转成与内置工具相同的 `ToolSpec`，复用同一套权限 / 校验 / 超时 / 重试 / 截断 / 审计 / 脱敏。
- **安全边界**：HTTP 只允许 `https` + 显式 `allowed_hosts`，禁跟随重定向、限制响应大小；请求前解析域名并拒绝私网/回环/链路本地/保留地址（混合解析任一命中即拒绝），连接后再核对实际对端地址以抵御 DNS 重绑定。它是**进程内校验，不是网络层隔离**（不做端口白名单、不接管代理），生产仍需出口网关/VPC 兜底。
- **CLI 不是任意命令执行**：命令由运营方写进配置，模型只能填 schema 声明的参数值，且不经 shell、只继承最小环境变量。
- **代码与测试**：`hermes/tools/connectors.py`、`tests/test_connectors.py`、`config/connectors.example.json`。
- **面试口径**：只有 JD 明确要求"API / CLI 接入"时才展开；否则一句话带过，把时间留给 Runtime / RAG / 记忆 / 评测主线。

