﻿# Hermes Agent

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

python -m pytest                                   # 109 项测试，全离线（Mock Provider）
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
| **沙箱与安全** | AST 白名单求值 + `python -I -S` 独立子进程 + 超时强杀；工具结果截断与审计；上传大小/类型白名单；检索结果与工具输出按不可信内容隔离（`<untrusted>` + 安全条款） | `tools/sandbox.py`、`tools/builtin.py`、`security/untrusted.py` |
| **鉴权与多租户** | API Key → `Principal(tenant, user, scopes)`；`off/api_key` 两种模式；scope + 租户归属双重校验，越权统一 404；普通调用方不能自报 scope | `security/auth.py`、`api/app.py` |
| **经验沉淀（闭环）** | 运行结束按规则提炼教训（工具不稳定 / 无引用作答 / 规划过长）→ `pending` → 人工 `hermes lessons --activate` → 注入最近 3 条 `[历史教训]` | `learning/lessons.py`、`cli.py` |
| **RAG 全链路** | PDF/HTML/DOCX/MD 解析 → 标题感知分块 → 余弦去冗余 → BM25 + 向量 + RRF → 相关性门控 → 可选交叉重排 → `[n]《标题》doc@v1#chunk` 引用 → 文档级 ACL → 版本覆盖 | `knowledge/loaders.py`、`splitter.py`、`index.py`、`rerank.py` |
| **向量抽象与降级** | `Embedder` 协议三实现（hashing / bge / 智谱），`auto` 顺序：显式配置 > 本地语义模型 > 智谱 Key > 哈希；维度不一致显式报 `embedding_mismatch` | `knowledge/embedding.py` |
| **长期记忆** | 规则抽取 → `pending`；确认后 `active`；`correct` 新版本 + `supersedes`；`forget` 软删；打分 = 余弦 + 词法 + 时效 + 置信 + 业务对象匹配；记忆可绑定 `subject_type/subject_id`（如客户），召回按对象加权、冲突只在同对象内判定；强制 tenant + user 隔离 | `memory/service.py` |
| **多步可靠性** | 有副作用工具先预演 → `paused` → `/confirm`；幂等键按 run 作用域；超时/退避/熔断；每步 checkpoint，进程重启 `resume`；启动对账把 `queued` 自动重入队、`running` 标记 `interrupted` 并发可恢复事件（不自动重放） | `runtime/runtime.py`、`llm/resilience.py`、`api/app.py` |
| **多 Agent 调度** | 主 Agent 规则规划 → 子 Agent 并行（最小工具集 + 信号量）→ 黑板汇总 → 单点失败隔离 | `agents/orchestrator.py`、`agents/blackboard.py` |
| **可观测 / 评测** | 事件全序 + span + Prometheus；token/成本预算；工具调用事件记录工具版本；`EvalRunner`（成功率、工具 P/R/F1、引用覆盖、忠实度、P50/P95）；`/replay` 时间线、`/diff` 两版对比、CLI `hermes replay --against`；`--fail-under` 门禁 | `observability/*`、`eval/*`、`cli.py` |
| **主动触发** | interval 触发器（≥10s）：`triggers` 表 + `TriggerService.due()` 原子推进 `next_fire_at` + API 进程内后台扫描；触发与 HTTP 共用 `enqueue_run`，不新增建 run 入口；多副本需分布式锁（已列为边界） | `triggers.py`、`api/app.py`、`cli.py` |
| **凭据保护** | 事件落库、工具审计、错误响应前统一脱敏（Authorization / `sk-*` / key-value 敏感字段）；轨迹可读但不可泄密 | `security/redact.py` |
| **工程化 / 交付** | 单一 CLI 入口；pyproject extras 分层依赖；Docker + compose；ruff + 覆盖率门禁 + GitHub Actions；116 项离线测试 | `cli.py`、`pyproject.toml`、`Dockerfile`、`.github/workflows/ci.yml` |

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
- **Mock Provider 不是玩具**：它是"可回归"的前提——116 项测试与 9 条评测在没有网络和 Key 时断言完整调用链（工具顺序、取消、恢复、熔断）。
- **幂等键按 run 作用域**：读类工具结果随时间变化，跨 run 复用缓存会返回过期事实（开发中真实踩到）。
- **上下文裁剪必须保配对**：丢 `assistant.tool_calls` 必须连同其 `tool` 结果一起丢，否则 OpenAI 兼容接口直接判请求非法。
- **事件 seq 分配必须原子**：并发子 Agent 下 `MAX(seq)+1` 与 `INSERT` 之间有竞态，已用锁包住（全量测试曾因此偶发失败）。

### 5.1 安全模型（面试官最爱追问的部分）

- **认证**：`HERMES_AUTH_MODE=off|api_key`。默认 `off`（单租户管理员，方便本地/CI），启动时打 warning；
  生产设 `api_key`，凭证格式 `key:tenant:user:scope1|scope2;...`（`HERMES_API_KEYS`）。
- **授权**：scope 只能由凭证决定，请求体自报的 scope 一律不采信；管理员可以降权，不能升权。
  跨租户访问统一返回 **404**（不泄露"存在但不属于你"）。
- **不可信内容隔离**：检索文档与工具输出在进入上下文前包成 `<untrusted label=... attrs>`，
  属性做转义、标签做闭合防护；system prompt 追加安全条款，明确"标签内是数据不是指令"。
  不做关键词黑名单——黑名单永远滞后，边界靠结构而不是枚举。
- **上传防护**：后缀白名单 → `Content-Length` 预检 → 1 MiB 分块计数，超限 413；落盘名加随机前缀防覆盖/穿越。
- **错误契约**：`{"error":{"code","message","request_id"}}` + `X-Request-Id`，500 不泄露堆栈。

---

## 6. 实测指标（两种模式的数字口径不同，别混着看）

### 6.1 无 Key 模式（Mock Provider + 哈希向量，已实测）

```
hermes eval --fail-under 0.99

| 指标 | 数值 |
| :--- | ---: |
| 任务数（single 7 + orchestrate 2） | 9 |
| 任务成功率 | 100.00% |
| 子任务成功率（orchestrate） | 100.00% |
| 工具选择 F1 | 1.0000 |
| 工具精确率 / 召回率 | 1.0000 / 1.0000 |
| 忠实度（引用支撑） | 0.5431 |
| 平均步数 | 1.22 |
| 平均 token | 1490 |
| P50 / P95 延迟 | ~30 ms / 60–105 ms（9 条样本，P95 波动大） |
```

用途：**回归基线**。固定 Mock + 固定哈希向量，同一份代码永远得到同一组数字，改检索或改提示后掉下去就是明确信号。
延迟是本地 SQLite + Mock，**不代表真实模型时延**。

### 6.2 有 Key 模式（智谱 glm-4-air + embedding-3，实测）

同一套评测集（当时 7 条 single 任务）、同一份代码，只换 Provider 和环境变量，2026-11 实测：

| 指标 | Mock / 哈希 | glm-4-flash | **glm-4-air（默认）** |
| --- | ---: | ---: | ---: |
| 任务成功率 | 100% | 57.14% | **100%** |
| 工具选择 F1 | 1.0000 | 0.8095 | **1.0000** |
| 忠实度（引用支撑） | 0.7168 | 0.3571 | 0.3952 |
| 平均 token | ~1049 | 2666 | 2956 |
| P95 延迟 | 24 ms | 9151 ms | **4851 ms** |
| 7 任务总成本 | $0 | $0.0019 | $0.0103 |

**两次真实模型跑出来的问题，比数字本身更有价值：**

1. `glm-4-flash` 只有 57%：该检索的不检索（凭记忆直接答 "LoRA 的 r 通常是 8"），多步题跳过检索只做计算，同一工具还会重复调用。**架构没错，是小模型指令遵循不够**——这正好说明为什么"能跑起来"和"能稳定跑对"是两件事。
2. 加固提示词（"涉及具体事实第一轮必须 kb_search，禁止凭记忆回答；同工具同参数不重复调用；查资料+算数两步都要做"）后 flash 升到 71.4%，换 `glm-4-air` 到 100%。所以默认模型定为 air，不是拍脑袋选的。
3. 真实口径的忠实度（0.3952）明显低于 Mock（当时 0.7168），原因是拒答类题目（t04/t07）和闲聊（t05）本来就没有引用，指标按 0 计入。**这是指标口径问题，不是答案变差了**——看忠实度要按类别分，别只看总分。

跑法见 §1.B。数字会随模型版本和网络变化，建议自己重跑一遍再引用。

评测集后来新增了 2 条 `mode: "orchestrate"` 的任务（t08/t09），真实模型口径尚未重跑，
所以上表只覆盖 single 模式；orchestrate 的回归基线见 §6.1（Mock + 规则规划）。

---

## 7. 已知边界（不吹）

- `python_sandbox` 是 **AST 白名单 + 独立子进程 + 超时强杀**，不是容器级隔离；生产应换 gVisor/Firecracker 或独立执行服务。
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

---

## 8. 目录

```
hermes/
  runtime/       runtime(ReAct循环) context(上下文工程) scheduler cancel store
  tools/         base(契约) registry(九道关) builtin sandbox mcp_bridge
  knowledge/     loaders splitter embedding rerank index service
  memory/        extractor service(确认/纠错/TTL/隔离)
  learning/      lessons(教训提炼/人工确认/注入)
  security/      auth(API Key→Principal) untrusted(不可信内容隔离) redact(凭据脱敏)
  agents/        orchestrator(规划+并行+汇总) blackboard
  llm/           base mock zhipu resilience(超时/退避/熔断)
  observability/ events(全序事件) tracing(spans/用量) metrics(Prometheus)
  eval/          runner metrics replay faithfulness
  api/           app(FastAPI + SSE + 知识库/触发器) errors(错误契约)
  mcp_server/    server(自建 stdio MCP Server)
  triggers.py    周期触发器（interval + 到期原子推进）
  cli.py         唯一人工入口：ask / chat / ingest / eval / serve / lessons / replay / triggers
knowledge_base/  种子 md（lora_notes、finance_policy）
eval/            dataset.jsonl（9 条：single 7 + orchestrate 2）
scripts/         demo.py run_eval.py mcp_demo.py ingest.py loadtest.py
samples/         演示用真实 PDF
tests/           116 项，覆盖上下文/工具/知识/记忆/运行时/韧性/多Agent/API/评测/MCP/CLI/安全/触发器/脱敏/重启恢复
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
| `docker build -t hermes-agent:0.2.3 .` | 成功，镜像 268 MB |
| `docker compose config` | 通过（无语法/插值错误） |
| 容器内 `GET /healthz` | 200，`embedding=hashing/384`、`rerank=rrf`、`llm=mock` |
| `POST /v1/runs` → 轮询状态 | `queued → succeeded`；闲聊 trace 11 条事件（检索问答 23 条） |
| `/metrics` | 记录 `http_requests_total`，含 422 参数校验失败（错误契约生效） |
| 数据卷 | `/data` 挂 `hermes_data`，SQLite 与上传目录都在卷内，重建容器不丢 |

注意：`/v1/runs/{id}/events` 是 SSE 流。已结束的 run 回放完终态事件后会主动关流；运行中的 run 每 15 秒发一次心跳。断线重连带 `Last-Event-ID: <seq>` 即可从断点续传，也支持 `?after_seq=` 显式指定起点。

以上为 0.2.3 镜像的实测结果（`docker build` 用时约 3 s，含缓存层）。

## 10. 面试材料

60+ 道问答、按 JD 七条职责逐条拆解、边界问题（OpenManus / 自进化 / 分布式 / 沙箱）的标准答法，见 `docs/INTERVIEW_QA.md`（本地保留，不纳入公开仓库）。
架构图、时序与数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

