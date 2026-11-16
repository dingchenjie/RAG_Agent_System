"""集中配置：全部来自环境变量，带安全默认值（默认 mock 模型，离线可跑）。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


@dataclass
class ContextConfig:
    """上下文预算配置。"""

    total_tokens: int = 4096
    reserve_output_tokens: int = 512
    recent_turns_keep: int = 4          # 最近 N 轮原文保留
    tool_result_max_chars: int = 1200   # 单条工具结果入库前的硬截断
    old_tool_result_chars: int = 300    # 历史工具结果在上下文中的保留长度
    summary_max_tokens: int = 320       # 压缩摘要上限


@dataclass
class Settings:
    # 运行时
    db_path: str = "data/hermes.db"
    max_steps: int = 12
    step_timeout_s: float = 60.0
    max_concurrency: int = 4
    max_run_tokens: int = 60_000
    max_run_cost_usd: float = 0.5

    # LLM
    llm_provider: str = "mock"           # mock | zhipu
    llm_model: str = "glm-4-air"
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_api_key: str = ""
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.1
    llm_breaker_threshold: int = 5       # 连续失败 N 次熔断
    llm_breaker_reset_s: float = 30.0

    # 工具
    tool_timeout_s: float = 15.0
    tool_max_retries: int = 2

    # 知识库
    chunk_size: int = 400
    chunk_overlap: int = 60
    recall_k: int = 12
    rerank_top_k: int = 4
    embedding_dim: int = 384
    embedding_backend: str = "hashing"    # hashing | zhipu | sentence-transformers | auto（auto 需显式开启）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"   # 本地语义模型（sentence-transformers）
    embedding_api_model: str = "embedding-3"          # 云端 embedding 的模型名（智谱）
    rerank_backend: str = "auto"          # rrf | cross-encoder | auto
    rerank_model: str = "BAAI/bge-reranker-base"
    dedupe_threshold: float = 0.92        # 分块去冗余的余弦阈值，<=0 关闭
    faithfulness_min_support: float = 0.6  # 单句被引用支撑的最低 n-gram 覆盖率
    upload_dir: str = "data/uploads"

    # 记忆
    memory_top_k: int = 4
    memory_ttl_days: int = 90
    memory_min_confidence: float = 0.5

    # MCP
    mcp_command: str = "python"
    mcp_args: list[str] = field(default_factory=lambda: ["-m", "hermes.mcp_server.server"])

    context: ContextConfig = field(default_factory=ContextConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        ctx = ContextConfig(
            total_tokens=_env_int("HERMES_CONTEXT_TOKENS", 4096),
            reserve_output_tokens=_env_int("HERMES_RESERVE_OUTPUT_TOKENS", 512),
            recent_turns_keep=_env_int("HERMES_RECENT_TURNS", 4),
            tool_result_max_chars=_env_int("HERMES_TOOL_RESULT_MAX_CHARS", 1200),
        )
        args = _env_str("HERMES_MCP_ARGS", "-m,hermes.mcp_server.server")
        return cls(
            db_path=_env_str("HERMES_DB_PATH", "data/hermes.db"),
            max_steps=_env_int("HERMES_MAX_STEPS", 12),
            step_timeout_s=_env_float("HERMES_STEP_TIMEOUT_S", 60.0),
            max_concurrency=_env_int("HERMES_MAX_CONCURRENCY", 4),
            max_run_tokens=_env_int("HERMES_MAX_RUN_TOKENS", 60_000),
            max_run_cost_usd=_env_float("HERMES_MAX_RUN_COST_USD", 0.5),
            llm_provider=_env_str("HERMES_LLM_PROVIDER", "mock"),
            llm_model=_env_str("HERMES_LLM_MODEL", "glm-4-air"),
            llm_base_url=_env_str("HERMES_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            llm_api_key=_env_str("ZHIPU_API_KEY", _env_str("HERMES_LLM_API_KEY", "")),
            llm_timeout_s=_env_float("HERMES_LLM_TIMEOUT_S", 60.0),
            llm_max_retries=_env_int("HERMES_LLM_MAX_RETRIES", 3),
            tool_timeout_s=_env_float("HERMES_TOOL_TIMEOUT_S", 15.0),
            tool_max_retries=_env_int("HERMES_TOOL_MAX_RETRIES", 2),
            chunk_size=_env_int("HERMES_CHUNK_SIZE", 400),
            chunk_overlap=_env_int("HERMES_CHUNK_OVERLAP", 60),
            recall_k=_env_int("HERMES_RECALL_K", 12),
            rerank_top_k=_env_int("HERMES_RERANK_TOP_K", 4),
            embedding_dim=_env_int("HERMES_EMBEDDING_DIM", 384),
            embedding_backend=_env_str("HERMES_EMBEDDING", "hashing"),
            embedding_model=_env_str("HERMES_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            embedding_api_model=_env_str("HERMES_EMBEDDING_API_MODEL", "embedding-3"),
            rerank_backend=_env_str("HERMES_RERANK", "auto"),
            rerank_model=_env_str("HERMES_RERANK_MODEL", "BAAI/bge-reranker-base"),
            dedupe_threshold=_env_float("HERMES_DEDUPE_THRESHOLD", 0.92),
            faithfulness_min_support=_env_float("HERMES_FAITHFULNESS_MIN_SUPPORT", 0.6),
            upload_dir=_env_str("HERMES_UPLOAD_DIR", "data/uploads"),
            memory_top_k=_env_int("HERMES_MEMORY_TOP_K", 4),
            memory_ttl_days=_env_int("HERMES_MEMORY_TTL_DAYS", 90),
            memory_min_confidence=_env_float("HERMES_MEMORY_MIN_CONFIDENCE", 0.5),
            mcp_command=_env_str("HERMES_MCP_COMMAND", "python"),
            mcp_args=[part for part in args.split(",") if part],
            context=ctx,
        )
