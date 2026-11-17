"""凭据脱敏：文本模式、结构化字段、事件落库入口。"""
from __future__ import annotations

from hermes.observability.events import Event, EventStore, EventType
from hermes.security.redact import redact, redact_obj
from hermes.storage.db import Database


def test_redact_authorization_header():
    assert redact("Authorization: Bearer abc123def456") == "Authorization=***"
    assert redact("authorization=Basic dXNlcjpwYXNz") == "Authorization=***"


def test_redact_known_token_shapes():
    assert redact("key is sk-abcdef1234567890") == "key is [REDACTED]"
    assert redact("use ghp_ABCDEFGHIJKLMNOPQRSTUV") == "use [REDACTED]"


def test_redact_key_value_pairs():
    assert redact("password=hunter2") == "password=***"
    assert redact('{"api_key": "abc123456789"}') == '{"api_key": "***"}'
    assert redact("access_token: tok_123456") == "access_token: ***"


def test_redact_keeps_normal_text():
    text = "我的邮箱是 dcj@example.com，请帮我总结 LoRA 的秩 r"
    assert redact(text) == text


def test_redact_obj_recurses_and_blocks_secret_keys():
    payload = {
        "tool": "http_get",
        "arguments": {"url": "https://example.com", "api_key": "plain-value"},
        "headers": [{"authorization": "Bearer token"}, {"accept": "application/json"}],
        "note": "password=hunter2",
    }
    cleaned = redact_obj(payload)
    assert cleaned["arguments"]["url"] == "https://example.com"
    assert cleaned["arguments"]["api_key"] == "***"
    assert cleaned["headers"][0]["authorization"] == "***"
    assert cleaned["headers"][1]["accept"] == "application/json"
    assert cleaned["note"] == "password=***"


def test_redact_is_idempotent():
    once = redact("Authorization: Bearer abc123 api_key=sk-abcdef1234567890 password=x")
    assert redact(once) == once


def test_event_store_redacts_before_persisting(tmp_path):
    store = EventStore(Database(tmp_path / "events.db"))
    event = Event(
        run_id="run_1", type=EventType.TOOL_CALL_STARTED,
        payload={"tool": "http_get", "arguments": {"api_key": "sk-abcdef1234567890"}},
    )
    store.append(event)

    # 实时订阅者与落库内容都应已脱敏
    assert event.payload["arguments"]["api_key"] == "***"
    stored = store.history("run_1")[0]
    assert stored.payload["arguments"]["api_key"] == "***"
