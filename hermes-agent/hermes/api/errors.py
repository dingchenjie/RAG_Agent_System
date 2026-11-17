"""HTTP 错误契约与请求追踪。

对外的失败形状固定为 ``{"error": {"code", "message", "request_id"}}``：
- ``code`` 稳定，可直接被脚本/评测断言；
- 500 不返回栈、文件路径或内部类名，只留服务端日志；
- 每个响应都带 ``X-Request-Id``，客户端传入则沿用，便于跨日志排查。
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..errors import ApiError
from ..security.redact import redact, redact_obj

logger = logging.getLogger("hermes.api")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or uuid.uuid4().hex


def _error_response(request: Request, *, status: int, code: str, message: str,
                    details: dict | None = None) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message, "request_id": _request_id(request)}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body,
                        headers={"X-Request-Id": _request_id(request)})


def install_error_handlers(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        logger.warning("[%s] %s %s -> %s %s", _request_id(request), request.method,
                       request.url.path, exc.status, exc.code)
        return _error_response(request, status=exc.status, code=exc.code,
                               message=redact(exc.message), details=redact_obj(exc.details))

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 校验错误会回显用户输入，可能带凭据，返回前统一清洗
        return _error_response(request, status=422, code="invalid_argument",
                               message="请求参数校验失败",
                               details=redact_obj({"errors": exc.errors()[:5]}))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error("[%s] unhandled error on %s %s: %s", _request_id(request),
                     request.method, request.url.path, redact(str(exc)), exc_info=exc)
        return _error_response(request, status=500, code="internal_error",
                               message="服务内部错误，请凭 request_id 反馈")
