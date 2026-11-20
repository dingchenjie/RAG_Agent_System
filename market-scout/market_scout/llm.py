"""LLM 客户端：OpenAI 兼容协议（任何网关）+ 确定性 Mock（离线/测试）。"""
from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

import httpx

from .config import Pricing, Settings


class LLMError(RuntimeError):
    def __init__(self, message: str, *, code: str = "llm_error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {"raw": str(raw)}


class LLMClient:
    """OpenAI 兼容客户端：重试在同一处做，配置类错误不重试（避免把熔断打穿）。"""

    def __init__(self, settings: Settings, pricing: Pricing | None = None):
        if not settings.llm_api_key:
            raise LLMError("未配置 SCOUT_LLM_API_KEY / AGICTO_API_KEY", code="missing_credential")
        self.settings = settings
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.pricing = pricing or Pricing()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def cost_usd(self) -> float:
        return self.pricing.cost(self.model, self.prompt_tokens, self.completion_tokens)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   *, temperature: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}

        last: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
                    resp = await client.post(f"{self.base_url}/chat/completions",
                                             json=payload, headers=headers)
                if resp.status_code >= 400:
                    body = resp.text[:300]
                    hint = any(k in body.lower() for k in (
                        "not supported", "invalid model", "unknown model", "model names are",
                        "not found", "未开通", "不存在"))
                    if hint or resp.status_code in (400, 401, 403, 404):
                        raise LLMError(f"模型调用被拒绝（配置类，不重试）：{body}",
                                       code="llm_config_error")
                    raise LLMError(f"模型返回 {resp.status_code}: {body}",
                                   retryable=resp.status_code == 429 or resp.status_code >= 500)
                return self._parse(resp.json())
            except LLMError as exc:
                last = exc
                if not exc.retryable or attempt >= self.settings.llm_max_retries:
                    raise
                await asyncio.sleep(min(8.0, 1.5 * (2 ** attempt)))
            except httpx.HTTPError as exc:
                last = exc
                if attempt >= self.settings.llm_max_retries:
                    raise LLMError(f"网络错误: {exc}", retryable=True) from exc
                await asyncio.sleep(min(8.0, 1.5 * (2 ** attempt)))
        raise LLMError(f"模型调用失败: {last}", retryable=True)

    def _parse(self, data: dict[str, Any]) -> dict[str, Any]:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        calls = [{"id": tc.get("id") or f"call_{i}",
                  "name": (tc.get("function") or {}).get("name", ""),
                  "arguments": _parse_args((tc.get("function") or {}).get("arguments"))}
                 for i, tc in enumerate(message.get("tool_calls") or [])]
        return {"content": message.get("content") or "", "tool_calls": calls,
                "model": data.get("model", self.model), "usage": usage}


class MockLLM:
    """确定性 Mock：先检索证据，再基于证据写结论；用于离线演示与回归测试。"""

    def __init__(self, settings: Settings | None = None):
        self.model = "mock"
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    @property
    def cost_usd(self) -> float:
        return 0.0

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   *, temperature: float | None = None) -> dict[str, Any]:
        self.calls += 1
        names = {t["function"]["name"] for t in (tools or [])}
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if "search_evidence" in names and not tool_messages:
            query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            if "本节聚焦：" in query:                    # 只查本节主题，避免三节命中同一批页面
                head, section = query.split("本节聚焦：")
                # 子问题必须带上目标实体，否则"Gamma 定价"会被 Acme/Beta 的资料冒充
                entities = list(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", head)))
                query = " ".join([*entities, section])
            return {"content": "", "tool_calls": [{"id": f"mock_{self.calls}",
                                                   "name": "search_evidence",
                                                   "arguments": {"query": query[:120]}}],
                    "model": self.model, "usage": {}}
        if tool_messages:
            raw = tool_messages[-1].get("content", "")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
            hits = data.get("hits") or []
            if hits:
                lines = [f"- {h['text'][:140]} [S{h['source_id']}]" for h in hits[:3]]
                return {"content": "关键事实：\n" + "\n".join(lines),
                        "tool_calls": [], "model": self.model, "usage": {}}
            return {"content": "未找到与该子问题直接相关的公开信息。",
                    "tool_calls": [], "model": self.model, "usage": {}}
        jitter = random.Random(len(messages)).random() * 0.01
        return {"content": f"离线模式回答（{jitter:.4f}）：缺少可用工具，无法进一步调研。",
                "tool_calls": [], "model": self.model, "usage": {}}


def build_llm(settings: Settings) -> Any:
    if settings.llm_provider == "openai":
        return LLMClient(settings)
    return MockLLM(settings)
