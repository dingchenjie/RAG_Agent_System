import asyncio
import os

import pytest

from market_scout.config import Settings
from market_scout.demo import seed_demo_sources
from market_scout.engine import create_engine, run_question
from market_scout.eval.metrics import citation_rate, keyword_coverage, tool_f1
from market_scout.graph import classify
from market_scout.store import Store
from market_scout.tools.registry import check_loop
from market_scout.tools.web import assert_safe_url


def _settings(tmp: str) -> Settings:
    settings = Settings()
    settings.llm_provider = "mock"
    settings.db_path = os.path.join(tmp, "test.db")
    settings.trace_jsonl = os.path.join(tmp, "trace.jsonl")
    settings.output_dir = os.path.join(tmp, "reports")
    return settings


def test_router_picks_reactive_and_deliberative():
    assert classify("Acme CRM 的定价是多少？")[0] == "reactive"
    assert classify("对比 Acme CRM 与 Beta CRM 的定价与核心功能，给出选型建议")[0] == "deliberative"


def test_loop_detection_triggers_on_repeated_signature():
    history: list[tuple[str, str]] = []
    assert check_loop(history, "search_evidence", {"query": "x"}, threshold=2) is False
    assert check_loop(history, "search_evidence", {"query": "x"}, threshold=2) is True


def test_ssrf_and_domain_allowlist():
    with pytest.raises(ValueError):
        assert_safe_url("http://127.0.0.1/secret", "")
    with pytest.raises(ValueError):
        assert_safe_url("file:///etc/passwd", "")
    with pytest.raises(ValueError):
        assert_safe_url("https://example.com/pricing", "example.org")


def test_store_search_returns_source_id(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    saved = store.add_source("r1", "https://example.com/a", "定价页", "Acme 每席位每月 12 美元")
    hits = store.search("定价", run_id="r1")
    assert hits and hits[0]["source_id"] == saved["source_id"]
    assert store.search("定价", run_id="other") == []
    store.close()


def test_metrics_basics():
    assert tool_f1(["a"], ["a"]) == (1.0, 1.0, 1.0)
    assert tool_f1(["a"], ["b"]) == (0.0, 0.0, 0.0)
    assert keyword_coverage(["x", "y"], "只有 x") == 0.5
    assert citation_rate("## 标题\n事实 [S1]\n无引用段落") == 0.5


def test_truncation_keeps_structured_fields():
    """结果超长时只能裁正文，source_id/title 必须保留，否则证据链断掉。"""
    from market_scout.tools.registry import _shrink

    shrunk = _shrink({"ok": True, "source_id": 7, "url": "https://e.com", "title": "定价页",
                      "text": "x" * 5000}, 800)
    assert shrunk["source_id"] == 7 and shrunk["title"] == "定价页"
    assert shrunk["truncated"] is True and len(shrunk["text"]) <= 800


def test_pricing_is_per_1k_tokens():
    from market_scout.config import Pricing

    assert Pricing().cost("gpt-4o-mini", 1000, 0) == pytest.approx(0.00015)


def test_offline_run_end_to_end(tmp_path):
    engine = create_engine(_settings(str(tmp_path)))
    seed_demo_sources(engine.store, engine.run_id)
    result = asyncio.run(run_question(engine, "对比 Acme CRM 与 Beta CRM 的定价与核心功能，给出选型建议"))
    assert result["status"] == "succeeded"
    assert "[S" in result["report"]
    assert "## 证据来源" in result["report"]
    assert result["meta"]["citations"]
    assert result["meta"]["loop_detected"] is False
    events = engine.store.list_events(engine.run_id)
    assert any(e["type"] == "route.decided" for e in events)
    assert any(e["type"] == "report.published" for e in events)
    engine.store.close()
