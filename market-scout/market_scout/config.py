"""集中配置：全部来自环境变量，默认离线可跑（mock 模型 + 本地证据库）。"""
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class Settings:
    # 运行时
    db_path: str = "data/scout.db"
    output_dir: str = "data/reports"
    max_steps: int = 10              # 单个 ReAct 收集器的步数上限（防死循环的第一道闸）
    max_sections: int = 3            # 深思虑模式最多并行几个调研子任务
    step_timeout_s: float = 60.0
    tool_timeout_s: float = 20.0
    tool_max_retries: int = 2
    max_pages_per_run: int = 8       # 单次调研最多抓多少个页面（成本闸）
    max_run_cost_usd: float = 0.05

    # LLM（OpenAI 兼容协议，任何网关都能接）
    llm_provider: str = "mock"       # mock | openai
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.agicto.cn/v1"
    llm_api_key: str = ""
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.2
    prompt_version: str = "v2"       # v1 | v2，评测里做 A/B

    # 浏览器（Browser Use）
    browser_headless: bool = True
    browser_timeout_ms: int = 25000
    browser_allow_domains: str = ""  # 逗号分隔；空表示不限制
    browser_max_chars: int = 6000

    # 观测
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    trace_jsonl: str = "data/traces.jsonl"

    api_host: str = "127.0.0.1"
    api_port: int = 8099

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_path=_env_str("SCOUT_DB_PATH", "data/scout.db"),
            output_dir=_env_str("SCOUT_OUTPUT_DIR", "data/reports"),
            max_steps=_env_int("SCOUT_MAX_STEPS", 10),
            max_sections=_env_int("SCOUT_MAX_SECTIONS", 3),
            step_timeout_s=_env_float("SCOUT_STEP_TIMEOUT_S", 60.0),
            tool_timeout_s=_env_float("SCOUT_TOOL_TIMEOUT_S", 20.0),
            tool_max_retries=_env_int("SCOUT_TOOL_MAX_RETRIES", 2),
            max_pages_per_run=_env_int("SCOUT_MAX_PAGES", 8),
            max_run_cost_usd=_env_float("SCOUT_MAX_RUN_COST_USD", 0.05),
            llm_provider=_env_str("SCOUT_LLM_PROVIDER", "mock"),
            llm_model=_env_str("SCOUT_LLM_MODEL", "gpt-4o-mini"),
            llm_base_url=_env_str("SCOUT_LLM_BASE_URL", "https://api.agicto.cn/v1"),
            llm_api_key=_env_str("SCOUT_LLM_API_KEY",
                                 _env_str("OPENAI_API_KEY", _env_str("AGICTO_API_KEY", ""))),
            llm_timeout_s=_env_float("SCOUT_LLM_TIMEOUT_S", 60.0),
            llm_max_retries=_env_int("SCOUT_LLM_MAX_RETRIES", 3),
            llm_temperature=_env_float("SCOUT_LLM_TEMPERATURE", 0.2),
            prompt_version=_env_str("SCOUT_PROMPT_VERSION", "v2"),
            browser_headless=_env_bool("SCOUT_BROWSER_HEADLESS", True),
            browser_timeout_ms=_env_int("SCOUT_BROWSER_TIMEOUT_MS", 25000),
            browser_allow_domains=_env_str("SCOUT_BROWSER_ALLOW_DOMAINS", ""),
            browser_max_chars=_env_int("SCOUT_BROWSER_MAX_CHARS", 6000),
            langfuse_host=_env_str("LANGFUSE_HOST", ""),
            langfuse_public_key=_env_str("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=_env_str("LANGFUSE_SECRET_KEY", ""),
            trace_jsonl=_env_str("SCOUT_TRACE_JSONL", "data/traces.jsonl"),
            api_host=_env_str("SCOUT_API_HOST", "127.0.0.1"),
            api_port=_env_int("SCOUT_API_PORT", 8099),
        )


@dataclass
class Pricing:
    """按 1K token 计价的估算表（只用于预算熔断与成本对比，不代表账单）。"""

    table: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4o": (0.0025, 0.01),
        "deepseek-flash": (0.0001, 0.0004),
        "deepseek-v4-pro": (0.0003, 0.0012),
    })

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        # 表里的价格是"每 1K token"，必须除以 1000，否则预算会被放大 1000 倍
        price_in, price_out = self.table.get(model, (0.0, 0.0))
        return (prompt_tokens * price_in + completion_tokens * price_out) / 1000
