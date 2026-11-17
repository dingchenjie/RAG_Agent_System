"""应用状态与依赖构建。

``AppState`` 是单进程内的服务容器，``build_state`` 负责把各业务服务组装起来。
这里刻意不依赖 FastAPI：CLI、脚本和测试可以直接构建状态，不必启动 HTTP 服务。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..agents.orchestrator import Orchestrator
from ..config import Settings
from ..knowledge.embedding import build_embedder
from ..knowledge.index import KnowledgeIndex
from ..knowledge.rerank import build_reranker
from ..knowledge.service import KnowledgeService
from ..learning.lessons import LessonService
from ..llm.base import LLMProvider
from ..llm.mock import MockLLM
from ..llm.resilience import CircuitBreaker, ResilientLLM
from ..llm.zhipu import ZhipuLLM
from ..memory.extractor import build_memory_extractor
from ..memory.service import MemoryService
from ..observability.events import EventBus, EventStore
from ..observability.metrics import Metrics
from ..runtime.cancel import CancelToken
from ..runtime.context import ContextBuilder
from ..runtime.runtime import AgentRuntime
from ..runtime.scheduler import RunScheduler
from ..runtime.store import RunStore, SessionStore
from ..security.ratelimit import TenantRateLimiter
from ..storage.db import Database
from ..tools.builtin import build_default_tools
from ..tools.connectors import register_connectors
from ..tools.registry import ToolRegistry
from ..tools.sandbox import SandboxRunner
from ..triggers import TriggerService


@dataclass
class AppState:
    settings: Settings
    db: Database
    bus: EventBus
    event_store: EventStore
    sessions: SessionStore
    runs: RunStore
    metrics: Metrics
    registry: ToolRegistry
    knowledge: KnowledgeService
    memory: MemoryService
    lessons: LessonService
    triggers: TriggerService
    runtime: AgentRuntime
    orchestrator: Orchestrator
    scheduler: RunScheduler
    limiter: TenantRateLimiter
    cancel_tokens: dict[str, CancelToken] = field(default_factory=dict)
    scheduler_futures: dict[str, asyncio.Future] = field(default_factory=dict)


def build_llm(settings: Settings, override: LLMProvider | None = None) -> LLMProvider:
    if override is not None:
        return override
    if settings.llm_provider == "zhipu":
        return ZhipuLLM(api_key=settings.llm_api_key, model=settings.llm_model,
                        base_url=settings.llm_base_url, timeout_s=settings.llm_timeout_s)
    return MockLLM(delay_s=settings.mock_delay_s)


def build_state(settings: Settings, llm_provider: LLMProvider | None = None) -> AppState:
    db = Database(settings.db_path)
    metrics = Metrics()
    event_store = EventStore(db)
    bus = EventBus(event_store)
    sessions = SessionStore(db)
    runs = RunStore(db)

    embedder = build_embedder(
        settings.embedding_backend, dim=settings.embedding_dim,
        api_key=settings.llm_api_key, model=settings.embedding_model,
        api_model=settings.embedding_api_model,
    )
    # 记忆抽取可能用同一个 LLM，因此在 MemoryService 之前构建（带重试/熔断）
    base_llm = build_llm(settings, llm_provider)
    resilient = ResilientLLM(
        base_llm, timeout_s=settings.llm_timeout_s, max_retries=settings.llm_max_retries,
        breaker=CircuitBreaker(settings.llm_breaker_threshold, settings.llm_breaker_reset_s),
        bus=bus,
    )
    knowledge = KnowledgeService(
        KnowledgeIndex(db, embedder),
        default_top_k=settings.rerank_top_k,
        reranker=build_reranker(settings.rerank_backend, model=settings.rerank_model),
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        dedupe_threshold=settings.dedupe_threshold,
    )
    memory = MemoryService(db, embedder=embedder, ttl_days=settings.memory_ttl_days,
                           min_confidence=settings.memory_min_confidence,
                           extractor=build_memory_extractor(
                               settings.memory_extractor, resilient,
                               min_confidence=settings.memory_min_confidence))
    lessons = LessonService(db)
    triggers = TriggerService(db)

    registry = ToolRegistry(db=db, metrics=metrics,
                            default_timeout_s=settings.tool_timeout_s,
                            default_retries=settings.tool_max_retries)
    registry.register_many(build_default_tools(
        knowledge=knowledge, memory=memory,
        sandbox=SandboxRunner(mode=settings.sandbox_mode, docker_image=settings.sandbox_image),
        tool_timeout_s=settings.tool_timeout_s,
    ))
    # 连接器与内置工具走同一个注册表：权限、超时、重试、截断、审计全部复用
    register_connectors(registry, settings)

    runtime = AgentRuntime(
        settings=settings, sessions=sessions, runs=runs, bus=bus, registry=registry,
        llm=resilient, metrics=metrics, memory=memory, knowledge=knowledge,
        builder=ContextBuilder(settings.context), lessons=lessons,
    )
    orchestrator = Orchestrator(runtime=runtime, llm=resilient, bus=bus)
    scheduler = RunScheduler(settings.max_concurrency)
    limiter = TenantRateLimiter(rpm=settings.rate_limit_rpm,
                                max_inflight=settings.tenant_max_inflight)

    return AppState(
        settings=settings, db=db, bus=bus, event_store=event_store, sessions=sessions, runs=runs,
        metrics=metrics, registry=registry, knowledge=knowledge, memory=memory,
        lessons=lessons, triggers=triggers, runtime=runtime,
        orchestrator=orchestrator, scheduler=scheduler, limiter=limiter,
    )
