"""智谱开放平台适配（OpenAI 兼容协议，直接用 httpx，不引入厂商 SDK）。"""
from __future__ import annotations

import httpx

from ..errors import LLMError, LLMTimeout
from ..types import Message, ToolCall, Usage
from .base import LLMProvider, LLMResponse, parse_arguments


class ZhipuLLM(LLMProvider):
    name = "zhipu"

    def __init__(self, *, api_key: str, model: str, base_url: str, timeout_s: float = 60.0,
                 embed_model: str = "embedding-3"):
        if not api_key:
            raise LLMError("未配置 ZHIPU_API_KEY，无法使用真实模型；可设置 HERMES_LLM_PROVIDER=mock")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.embed_model = embed_model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"调用模型超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"网络错误: {exc}", retryable=True) from exc

        if resp.status_code >= 400:
            retryable = resp.status_code == 429 or resp.status_code >= 500
            raise LLMError(f"模型返回 {resp.status_code}: {resp.text[:300]}", retryable=retryable)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{idx}",
                name=(tc.get("function") or {}).get("name", ""),
                arguments=parse_arguments((tc.get("function") or {}).get("arguments")),
            )
            for idx, tc in enumerate(message.get("tool_calls") or [])
        ]
        usage_raw = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
            completion_tokens=int(usage_raw.get("completion_tokens", 0)),
        )
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            usage=usage,
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.embed_model, "input": texts},
                    headers=self._headers(),
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"向量化超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"向量化网络错误: {exc}", retryable=True) from exc
        if resp.status_code >= 400:
            raise LLMError(f"向量化返回 {resp.status_code}: {resp.text[:300]}", retryable=resp.status_code >= 500)
        return [item["embedding"] for item in resp.json().get("data", [])]
