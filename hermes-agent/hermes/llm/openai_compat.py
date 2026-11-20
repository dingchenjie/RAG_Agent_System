"""OpenAI 兼容协议适配：任何提供 /chat/completions 与 /embeddings 的服务都能接。

不绑定具体厂商：base_url 与 model 全部来自配置（智谱、agicto、OpenAI、DeepSeek、
vLLM、OneAPI 等只是不同的 base_url），直接用 httpx，不引入厂商 SDK。
"""
from __future__ import annotations

import httpx

from ..errors import LLMError, LLMTimeout
from ..types import Message, ToolCall, Usage
from .base import LLMProvider, LLMResponse, parse_arguments


class OpenAICompatLLM(LLMProvider):
    name = "openai"

    def __init__(self, *, api_key: str, model: str, base_url: str, timeout_s: float = 60.0,
                 embed_model: str = "embedding-3"):
        if not api_key:
            raise LLMError("未配置 API Key（HERMES_LLM_API_KEY 或 OPENAI_API_KEY），"
                           "无法使用真实模型；可设置 HERMES_LLM_PROVIDER=mock")
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
            body = resp.text[:300]
            # 网关常把"模型名不存在/未开通"报成 500，按可重试处理会白白重试并触发熔断
            config_hint = any(k in resp.text.lower() for k in (
                "not supported", "not found", "does not exist", "invalid model", "unknown model",
                "model names are", "supported api model", "model_not_found",
                "不支持", "不存在", "未开通"))
            if config_hint or resp.status_code in (400, 401, 403, 404):
                raise LLMError(f"模型调用被拒绝（配置类错误，不重试）：{body}",
                               code="llm_config_error", retryable=False)
            raise LLMError(f"模型返回 {resp.status_code}: {body}",
                           retryable=resp.status_code == 429 or resp.status_code >= 500)

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
