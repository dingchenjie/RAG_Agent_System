"""统一错误契约。

所有对外可见的失败都带 ``code``（稳定、可被前端/评测消费）与 ``retryable``
（决定上层是否重试），避免把 Python 异常字符串直接泄露给模型或用户。
"""
from __future__ import annotations


class HermesError(Exception):
    code = "hermes_error"
    retryable = False

    def __init__(self, message: str, *, code: str | None = None, retryable: bool | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


class LLMError(HermesError):
    code = "llm_error"


class LLMTimeout(LLMError):
    code = "llm_timeout"
    retryable = True


class ToolError(HermesError):
    code = "tool_error"

    def __init__(self, message: str, *, code: str | None = None, retryable: bool | None = None,
                 details: dict | None = None):
        super().__init__(message, code=code, retryable=retryable)
        self.details = details or {}


class PermissionDenied(ToolError):
    code = "permission_denied"


class ArgumentInvalid(ToolError):
    code = "invalid_arguments"


class CancelledByUser(HermesError):
    code = "cancelled"


class BudgetExceeded(HermesError):
    code = "budget_exceeded"


class NotFound(HermesError):
    code = "not_found"


class ApiError(HermesError):
    """对外 HTTP 错误：``status`` 决定状态码，``code`` 保持稳定可断言。"""

    code = "api_error"

    def __init__(self, message: str, *, code: str | None = None, status: int = 400,
                 details: dict | None = None, headers: dict[str, str] | None = None):
        super().__init__(message, code=code)
        self.status = status
        self.details = details or {}
        self.headers = headers or {}

