"""记忆抽取：JSON 解析校验、LLM 抽取降级、异步写入链路与评测指标。"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hermes.eval.extraction import evaluate_extraction, load_extraction_tasks
from hermes.llm.base import LLMProvider, LLMResponse
from hermes.memory import (
    LLMMemoryExtractor,
    MemoryCandidate,
    RuleMemoryExtractor,
    build_memory_extractor,
    parse_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_DATASET = ROOT / "eval" / "extraction.jsonl"


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, content: str | Exception):
        self.content = content
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, *, temperature: float = 0.1,
                   max_tokens: int | None = None) -> LLMResponse:
        self.calls.append({
            "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "tools": tools,
        })
        if isinstance(self.content, Exception):
            raise self.content
        return LLMResponse(content=self.content, model="fake")


def test_parse_candidates_accepts_fenced_json_and_keeps_best_per_key():
    raw = """```json
    [
      {"key": "name", "value": "丁辰杰", "type": "profile", "confidence": 0.9},
      {"key": "name", "value": "丁辰杰", "type": "profile", "confidence": 0.95},
      {"key": "preference.language", "value": "Python", "type": "unknown", "confidence": "0.8"}
    ]
    ```"""
    candidates = parse_candidates(raw)
    assert [c.mkey for c in candidates] == ["name", "preference.language"]
    assert candidates[0].confidence == pytest.approx(0.95)
    assert candidates[0].mtype == "profile"
    # 非白名单 type 归为 fact，而不是整条丢弃
    assert candidates[1].mtype == "fact"
    assert candidates[1].confidence == pytest.approx(0.8)


def test_parse_candidates_drops_invalid_entries():
    raw = ('[{"key": "中文key", "value": "x", "confidence": 0.9},'
           ' {"key": "ok", "value": "", "confidence": 0.9},'
           ' {"key": "bad_conf", "value": "x", "confidence": "abc"},'
           ' {"key": "low", "value": "x", "confidence": 0.1},'
           ' {"key": "toolong", "value": "%s", "confidence": 0.9},'
           ' "not-an-object"]' % ("x" * 201))
    assert parse_candidates(raw, min_confidence=0.5) == []


@pytest.mark.parametrize("raw", ["", "没有 JSON", "{}", '{"key": "name"}', "[1, 2, 3]"])
def test_parse_candidates_returns_empty_for_broken_output(raw):
    assert parse_candidates(raw) == []


def test_llm_extractor_returns_empty_when_llm_fails():
    llm = FakeLLM(RuntimeError("timeout"))
    extractor = LLMMemoryExtractor(llm)
    assert asyncio.run(extractor.extract_async("我叫丁辰杰")) == []
    assert len(llm.calls) == 1


def test_llm_extractor_builds_prompt_and_truncates_input():
    llm = FakeLLM('[{"key": "name", "value": "丁辰杰", "type": "profile", "confidence": 0.9}]')
    extractor = LLMMemoryExtractor(llm, max_input_chars=100)
    candidates = asyncio.run(extractor.extract_async("我叫丁辰杰" + "很长的输入" * 100))
    assert [c.mkey for c in candidates] == ["name"]
    messages = llm.calls[0]["messages"]
    assert messages[0].role == "system" and "JSON" in (messages[0].content or "")
    assert len(messages[1].content or "") == 100
    assert llm.calls[0]["temperature"] == 0.0


def test_llm_extractor_sync_entry_point_is_rejected():
    with pytest.raises(RuntimeError):
        LLMMemoryExtractor(FakeLLM("[]")).extract("我叫丁辰杰")


def test_build_memory_extractor_switches_backend():
    llm = FakeLLM("[]")
    assert isinstance(build_memory_extractor("rule", llm), RuleMemoryExtractor)
    assert isinstance(build_memory_extractor("llm", llm), LLMMemoryExtractor)
    with pytest.raises(ValueError):
        build_memory_extractor("unknown", llm)


def test_propose_from_text_async_uses_rule_extractor(state):
    created = asyncio.run(state.memory.propose_from_text_async(
        tenant_id="t1", user_id="u1", text="我叫丁辰杰，我更喜欢 Python。"))
    assert {record.mkey for record in created} == {"name", "preference"}
    assert all(record.status == "pending" for record in created)


def test_propose_from_text_async_uses_llm_extractor(state):
    state.memory.extractor = LLMMemoryExtractor(FakeLLM(
        '[{"key": "project.codename", "value": "Apollo", "type": "fact", "confidence": 0.9}]'))
    created = asyncio.run(state.memory.propose_from_text_async(
        tenant_id="t1", user_id="u1", text="我最近在负责 Apollo 项目。"))
    assert [(record.mkey, record.value) for record in created] == [("project.codename", "Apollo")]


def test_rule_extractor_baseline_on_dataset():
    tasks = load_extraction_tasks(EXTRACTION_DATASET)
    result = asyncio.run(evaluate_extraction(RuleMemoryExtractor().extract_async, tasks))
    assert result["cases"] == 20
    # 规则覆盖画像字段，但拿不到项目/团队这类语义信息：召回有缺口、精确率不掉
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(10 / 19)
    assert result["false_positive_rate"] == 0.0
    assert result["abstain_accuracy"] == pytest.approx(1.0)
    assert result["by_category"]["negative"]["cases"] == 7


def test_evaluate_extraction_counts_extra_candidates_as_false_positives():
    async def fake_extract(text: str) -> list[MemoryCandidate]:
        return [MemoryCandidate(mkey="name", value="x")]

    tasks = [
        {"id": "pos", "text": "a", "expect": [{"any_keys": ["name"], "value_contains": "x"}]},
        {"id": "neg", "text": "b", "expect_empty": True},
    ]
    result = asyncio.run(evaluate_extraction(fake_extract, tasks))
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (1, 1, 0)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == pytest.approx(1.0)
    assert result["abstain_accuracy"] == pytest.approx(0.0)
