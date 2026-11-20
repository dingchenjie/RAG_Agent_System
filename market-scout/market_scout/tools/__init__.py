"""工具装配：三个工具，全部走同一个注册表（校验/超时/重试/截断/审计）。"""
from __future__ import annotations

from ..config import Settings
from .browser import build_browser_tool
from .evidence import build_search_tool
from .registry import ToolContext, ToolRegistry, ToolSpec, check_loop
from .web import build_fetch_tool

__all__ = ["ToolContext", "ToolRegistry", "ToolSpec", "check_loop", "build_registry"]


def build_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry(settings)
    registry.register(build_fetch_tool(settings))
    registry.register(build_browser_tool(settings))
    registry.register(build_search_tool(settings))
    return registry
