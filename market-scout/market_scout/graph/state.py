"""图状态：所有跨节点数据都在这里，事件另存 SQLite（状态与轨迹分离）。"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ScoutState(TypedDict, total=False):
    question: str
    mode: str                              # reactive | deliberative
    route_reason: str
    sections: Annotated[list[str], operator.add]
    findings: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    tools_used: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    pages_used: Annotated[int, operator.add]
    loop_detected: Annotated[bool, operator.or_]
    draft: str
    critique: list[str]
    revisions: int
    report: str
    usage: dict[str, Any]
