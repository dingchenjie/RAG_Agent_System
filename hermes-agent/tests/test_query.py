"""多轮检索 query 补全：只对明显的追问启用，并且不破坏单轮问题。"""
from __future__ import annotations

import pytest

from hermes.knowledge.query import build_retrieval_query, is_follow_up, previous_user_message
from hermes.types import Message


@pytest.mark.parametrize("query", [
    "那上海呢？", "它的默认超时呢？", "换成 16 呢？", "上海呢？", "还有别的吗？", "另外呢",
])
def test_follow_up_is_detected(query):
    assert is_follow_up(query)


@pytest.mark.parametrize("query", [
    "LoRA 的秩 r 通常取多少？", "差旅住宿一线城市的上限是多少？", "你好", "",
])
def test_standalone_question_is_kept_as_is(query):
    assert not is_follow_up(query)


def test_build_merges_previous_question():
    merged = build_retrieval_query("那上海呢？", previous="差旅住宿一线城市的上限是多少？")
    assert merged == "差旅住宿一线城市的上限是多少？ 那上海呢？"


def test_build_without_previous_is_noop():
    assert build_retrieval_query("那上海呢？", previous=None) == "那上海呢？"


def test_build_truncates_previous_from_head():
    merged = build_retrieval_query("那上海呢？", previous="上" * 300, max_chars=40)
    assert len(merged) <= 40
    assert merged.endswith(" 那上海呢？")


def test_previous_user_message_skips_current_input():
    history = [
        Message(role="user", content="上一轮问题"),
        Message(role="assistant", content="回答"),
        Message(role="user", content="那上海呢？"),
    ]
    assert previous_user_message(history, "那上海呢？") == "上一轮问题"
    assert previous_user_message([Message(role="user", content="只有当前")], "只有当前") is None


def test_orchestrator_task_prefix_is_stripped():
    """子 Agent 的 goal 带指令前缀，检索时只保留真正的用户问题。"""
    goal = "检索并总结与问题相关的资料：LoRA 的 dropout 常用范围是多少？另外计算 10/2 的结果"
    assert build_retrieval_query(goal, previous=None) == "LoRA 的 dropout 常用范围是多少？另外计算 10/2 的结果"
