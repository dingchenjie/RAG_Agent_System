from .base import BaseLLMProvider, LLMResponse, ToolCall
from .zhipu_provider import ZhipuProvider

def create_provider(provider_name: str, api_key: str, model: str) -> BaseLLMProvider:
    if provider_name == "zhipu":
        return ZhipuProvider(api_key, model)
    else:
        raise ValueError(f"不支持的 Provider: {provider_name}")