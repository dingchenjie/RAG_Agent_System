"""指标：计数器/直方图渲染，以及 API 是否暴露 HTTP 维度。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from hermes.api.app import create_app
from hermes.llm.mock import MockLLM
from hermes.observability.metrics import Metrics
from tests.conftest import make_settings


def test_render_prometheus_contains_quantiles_and_up():
    metrics = Metrics()
    metrics.inc("hermes_runs_total", state="succeeded")
    metrics.observe("hermes_run_steps", 3)
    text = metrics.render_prometheus()
    assert "# TYPE hermes_up gauge" in text
    assert "hermes_up 1" in text
    assert 'hermes_runs_total{state="succeeded"} 1' in text
    assert 'quantile="0.95"' in text


def test_api_metrics_record_http_dimension(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings, MockLLM())) as client:
        client.get("/healthz")
        text = client.get("/metrics").text
    assert "http_requests_total" in text
    assert 'path="/healthz"' in text
