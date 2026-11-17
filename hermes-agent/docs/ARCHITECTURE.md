# Hermes Agent 架构说明

> 面向"读代码前先建立地图"的读者。所有图都对应真实代码路径，便于边看边定位。

---

## 1. 分层

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 入口层：hermes/cli.py（ask/chat/ingest/eval/serve/lessons）· api/app.py    │
├──────────────────────────────────────────────────────────────────────────┤
│ 边界层：security（API Key→Principal/租户/scope）· api/errors（统一错误契约）│
├──────────────────────────────────────────────────────────────────────────┤
│ 编排层：runtime（单 Agent 循环）· scheduler（并发与背压）· agents（主从）   │
├──────────────────────────────────────────────────────────────────────────┤
│ 能力层：tools（契约/权限/沙箱/MCP）· knowledge（文档→检索）· memory（生命周期）│
│         learning（失败轨迹→教训→人工确认→注入）                            │
├──────────────────────────────────────────────────────────────────────────┤
│ 模型层：llm（Mock / 智谱 OpenAI 兼容 / 重试·退避·熔断）                     │
├──────────────────────────────────────────────────────────────────────────┤
│ 基础层：observability（事件/span/指标）· storage（SQLite WAL）· config      │
└──────────────────────────────────────────────────────────────────────────┘
```

依赖方向自上而下；`knowledge`/`memory`/`tools` 之间只通过构造函数注入协作，不互相 import 业务逻辑。

---

## 2. 一次 `run` 的时序

```
客户端            API              Scheduler        Runtime              ToolRegistry      EventStore
  │ POST /v1/runs   │                  │               │                      │               │
  ├────────────────>│ 建 run 记录       │               │                      │               │
  │<── run_id ──────┤                  │               │                      │               │
  │ GET /events(SSE)│                  │               │                      │               │
  ├────────────────>│                  ├── 入队 ──────>│                      │               │
  │                 │                  │               ├─ 检索知识（ACL）─────>│               │
  │                 │                  │               ├─ 组装上下文（预算）   │               │
  │                 │                  │               ├─ LLM 调用             │               │
  │                 │                  │               ├─ 工具调用 ───────────>│ 九道关        │
  │                 │                  │               │<──── ToolResult ─────┤               │
  │                 │                  │               ├─ 每步 checkpoint ────>│               │
  │                 │                  │               ├─ 事件（step/tool/…）─>│──────────────>│
  │<── SSE 事件 ────┤<─────────────────┤               │                      │               │
  │                 │                  │               └─ 结束/暂停/失败       │               │
```

要点：
- **提交式接口**：先返回 `run_id`，执行与流式输出解耦，前端断线可重连；
- **每步 checkpoint**：`runs` 表保存 messages 与 step，进程重启后可 `resume`；
- **有副作用工具**：Registry 返回 `needs_confirmation`，run 停在 `paused`，`POST /confirm` 续跑。

---

## 3. 知识入库与检索数据流

```
文件 ──loaders.load_document──> Document(text, source, title, meta{page})
                                     │
                    splitter.split_text（markdown 标题优先 + 句边界装箱）
                                     │
                    splitter.dedupe（余弦贪心去冗余，阈值 HERMES_DEDUPE_THRESHOLD）
                                     │
             KnowledgeIndex.ingest_chunks（embedder 批量向量化 + 维度校验）
                                     │
                     SQLite：documents(doc_id, version, acl, embedding_model, embedding_dim)
                            chunks(chunk_id, doc_id, version, ord, text, embedding)
                                     │
查询 ──分词──> BM25 召回 ─┐
      └────> 向量召回 ────┴─> RRF 融合 ──> 相关性门控 ──> rerank（RRF 截断 / 交叉重排）──> Hit[]
                                            │
                                  覆盖率或相似度不达标 → 返回空 → "没有检索到相关内容"
```

- **版本管理**：同一 `doc_id` 新版本入库时旧版本 `active=0`，查询只读 active，引用里带版本号；
- **权限过滤**：文档 `acl` 为分组集合，查询时按调用方 scopes 过滤；
- **向量口径**：写入时记录 `embedding_model/dim`，换 embedder 后继续写入会抛 `embedding_mismatch`，避免静默错分。

---

## 4. 工具执行九道关

```
① 未知工具？            → code=unknown_tool
② scope 鉴权            → code=permission_denied
③ Pydantic 参数校验      → code=invalid_arguments
④ 需要确认且未确认       → needs_confirmation（不执行，返回预览）
⑤ 幂等命中（run 作用域）  → 直接返回缓存结果
⑥ 执行（阻塞调用进线程池） → 超时 → code=timeout
⑦ 可重试错误指数退避      → 超过次数 → 原错误
⑧ 结果截断（max_result_chars）
⑨ 审计落 tool_audit（参数/结果摘要/耗时/重试次数/错误码）
```

所有分支都返回结构化 `ToolResult(ok, content, error, code, latency_ms, attempts, meta)`，
上层不需要解析字符串判断成败。

---

## 5. 记忆状态机

```
规则抽取 → pending ──confirm──> active ──archive/forget──> archived / deleted
              │                   │
              └──reject──> rejected └──correct──> 新记录 active（supersedes_id 指向旧记录）
```

召回打分：`0.45*cosine + 0.30*lexical + 0.15*recency + 0.10*confidence`，
所有查询强制 `tenant_id + user_id` 过滤，避免跨租户泄漏。

---

## 6. 事件模型

- 表：`events(run_id, seq, ts, type, span_id, parent_span_id, payload)`，`PRIMARY KEY(run_id, seq)`；
- `seq` 在写入时原子分配（锁包住 `MAX(seq)+1` 与 `INSERT`），保证并发子 Agent 下的全序；
- 用途：SSE 推送、轨迹回放、离线评测、"哪一步慢/贵/错"的定位；
- 指标：`/metrics` 输出 Prometheus 文本（run 数、工具耗时、token、错误码分布）。

---

## 7. 评测口径

`EvalRunner` 逐条执行 `eval/dataset.jsonl`（single 直连 Runtime，orchestrate 走主 Agent
并把父子 run 的事件流合并），从事件轨迹复算：

| 指标 | 计算方式 |
| --- | --- |
| 成功率 | 期望工具全部被调用 + `expected_contains` 全部命中 + 无错误 |
| 工具 P / R / F1 | 期望工具集 vs `tool.started` 实际集合 |
| 引用覆盖 | 有引用的工具调用 / 工具调用总数 |
| 忠实度 | 答案逐句与检索上下文的 n-gram 覆盖率 ≥ 0.6 记作被支撑 |
| 延迟 / token / 成本 | 基于 `llm.request` 与 `run.completed` 事件 |

门禁：`hermes eval --fail-under 0.99`，成功率不达标退出码非 0，可直接用于 CI。

---

## 8. 部署形态

- 单机：`uvicorn hermes.api.app:app` + SQLite 文件卷；
- 容器：默认 slim 镜像（无 torch）；`--build-arg INSTALL_EXTRAS=semantic` 构建语义镜像；
- 扩展点：`KnowledgeIndex` 换向量库、`EventStore` 换消息队列、`RunScheduler` 换分布式队列；
- 已知非目标：容器级沙箱、跨机调度、多租户配额计费。
