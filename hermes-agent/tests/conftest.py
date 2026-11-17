from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes.api.app import build_state  # noqa: E402
from hermes.config import ContextConfig, Settings  # noqa: E402
from hermes.llm.mock import MockLLM  # noqa: E402


def make_settings(tmp_path: Path, **overrides) -> Settings:
    settings = Settings(
        db_path=str(tmp_path / "hermes.db"),
        llm_provider="mock",
        max_steps=6,
        max_concurrency=2,
        llm_timeout_s=5.0,
        tool_timeout_s=3.0,
        # 测试固定走离线路径，避免机器上装了 sentence-transformers 时行为漂移
        embedding_backend="hashing",
        rerank_backend="rrf",
        # 测试不启动触发器后台循环，触发器行为由 test_triggers.py 直接验证服务层
        trigger_tick_s=0.0,
        upload_dir=str(tmp_path / "uploads"),
        context=ContextConfig(
            total_tokens=3000,
            reserve_output_tokens=300,
            recent_turns_keep=4,
            tool_result_max_chars=1200,
            old_tool_result_chars=300,
            summary_max_tokens=200,
        ),
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def state(settings):
    return build_state(settings, MockLLM())


def ingest_seed_docs(state) -> None:
    root = Path(__file__).resolve().parents[1] / "knowledge_base"
    state.knowledge.ingest_file(root / "lora_notes.md", doc_id="lora_notes", version="1")
    state.knowledge.ingest_file(root / "finance_policy.md", doc_id="finance_policy", version="1",
                                acl=["finance"])
