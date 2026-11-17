"""凭据脱敏：事件、工具审计、错误信息落库或返回前统一清洗。

原则：只屏蔽"看起来像凭据"的片段，保留字段名与上下文。
整段打码会让轨迹失去排障价值，而排障恰恰是轨迹存在的理由。
"""
from __future__ import annotations

import re
from typing import Any

# Authorization 整个值屏蔽：可能是 Bearer / Basic / 自定义方案
_AUTH_HEADER = re.compile(r"(?i)\bauthorization\b\s*[:=]\s*[^\n,;}\]]+")

# 常见厂商密钥形态：OpenAI/智谱 sk-xxx、GitHub ghp_/gho_、GitLab glpat-、Slack xox*、阿里 LTAI
_TOKEN_SHAPES = re.compile(r"\b(?:sk|ghp|gho|glpat|xox[baprs]|LTAI)[-_A-Za-z0-9]{10,}\b")

# key=value / key: value 形式的敏感字段（保留引号，只替换值）
_SECRET_FIELDS = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|client[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|password|passwd|token)\b"
    r"(\"?\s*[:=]\s*)([\"']?)([^\s,;&\"'}\]]+)([\"']?)"
)

_SECRET_KEYS = {
    "api_key", "apikey", "access_key", "secret_key", "client_secret",
    "access_token", "refresh_token", "token", "secret", "password", "passwd",
    "authorization",
}


def redact(text: str) -> str:
    """把文本中形似凭据的片段替换为 ``***`` / ``[REDACTED]``。"""
    if not text:
        return text
    text = _AUTH_HEADER.sub("Authorization=***", text)
    text = _TOKEN_SHAPES.sub("[REDACTED]", text)
    return _SECRET_FIELDS.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}***{m.group(5)}", text)


def redact_obj(value: Any) -> Any:
    """递归清洗结构化数据；敏感字段名直接整值屏蔽，不依赖值的形态。"""
    if isinstance(value, dict):
        return {
            key: "***" if str(key).lower() in _SECRET_KEYS else redact_obj(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_obj(item) for item in value]
    if isinstance(value, str):
        return redact(value)
    return value
