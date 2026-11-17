"""租户级限流与并发配额：令牌桶、在途上限、429 契约与指标。"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from hermes.api.app import create_app
from hermes.llm.mock import MockLLM
from hermes.security.ratelimit import TenantRateLimiter


def test_token_bucket_refills_over_time():
    now = [0.0]
    limiter = TenantRateLimiter(rpm=2, clock=lambda: now[0])

    assert limiter.acquire("t1").allowed is True
    assert limiter.acquire("t1").allowed is True
    denied = limiter.acquire("t1")
    assert denied.allowed is False
    assert denied.reason == "rate_limited"
    assert denied.retry_after_s == 30

    # 其他租户的桶互不影响
    assert limiter.acquire("t2").allowed is True

    now[0] = 30.0
    assert limiter.acquire("t1").allowed is True


def test_inflight_quota_blocks_and_recovers():
    limiter = TenantRateLimiter(max_inflight=1)

    assert limiter.acquire("t1").allowed is True
    denied = limiter.acquire("t1")
    assert denied.allowed is False
    assert denied.reason == "inflight_limited"

    limiter.release("t1")
    assert limiter.acquire("t1").allowed is True


def test_disabled_limiter_never_tracks():
    limiter = TenantRateLimiter()
    assert limiter.enabled is False
    for _ in range(50):
        assert limiter.acquire("t1").allowed is True
    assert limiter.snapshot()["inflight"] == {}


def test_idempotent_replay_does_not_consume_quota(settings):
    settings.rate_limit_rpm = 2
    app = create_app(settings, MockLLM())
    with TestClient(app) as client:
        headers = {"X-Idempotency-Key": "k-1"}
        first = client.post("/v1/runs", json={"input": "你好"}, headers=headers)
        replay = client.post("/v1/runs", json={"input": "你好"}, headers=headers)
        assert first.status_code == 202
        assert replay.status_code == 202
        assert replay.json()["deduplicated"] is True

        # 重放没有消耗令牌，第二个新提交仍能通过
        second = client.post("/v1/runs", json={"input": "你好"})
        assert second.status_code == 202
        third = client.post("/v1/runs", json={"input": "你好"})
        assert third.status_code == 429


def test_api_returns_429_with_retry_after(settings):
    settings.rate_limit_rpm = 2
    settings.tenant_max_inflight = 0
    app = create_app(settings, MockLLM())
    with TestClient(app) as client:
        first = client.post("/v1/runs", json={"input": "你好"})
        second = client.post("/v1/runs", json={"input": "你好"})
        third = client.post("/v1/runs", json={"input": "你好"})

        assert first.status_code == 202
        assert second.status_code == 202
        assert third.status_code == 429
        body = third.json()["error"]
        assert body["code"] == "rate_limited"
        assert body["details"]["retry_after_s"] >= 1
        assert int(third.headers["Retry-After"]) >= 1

        counters = client.get("/v1/metrics/json").json()["metrics"]["counters"]
        assert any("run_rejected_total" in key for key in counters)

        run_ids = [first.json()["run_id"], second.json()["run_id"]]
        deadline = time.time() + 5
        while time.time() < deadline:
            states = [client.get(f"/v1/runs/{rid}").json()["state"] for rid in run_ids]
            if all(state in {"succeeded", "failed", "cancelled"} for state in states):
                break
            time.sleep(0.05)
        # run 结束后在途配额已释放，不会残留
        assert app.state.hermes.limiter.snapshot()["inflight"].get("default", 0) == 0
