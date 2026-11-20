"""图装配：LangGraph 的 StateGraph + 条件边 + Send 并行 + 可选人工审批。"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import Nodes, Runtime
from .state import ScoutState


def build_graph(rt: Runtime, *, require_approval: bool = False) -> Any:
    nodes = Nodes(rt)
    graph: StateGraph = StateGraph(ScoutState)

    graph.add_node("route", nodes.route)
    graph.add_node("quick", nodes.quick)
    graph.add_node("plan", nodes.plan)
    graph.add_node("collect", nodes.collect)
    graph.add_node("merge", nodes.merge)
    graph.add_node("write", nodes.write)
    graph.add_node("critique", nodes.critique)
    graph.add_node("revise", nodes.revise)
    graph.add_node("publish", nodes.publish)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", nodes.dispatch_route,
                                {"quick": "quick", "plan": "plan"})
    graph.add_edge("quick", END)

    graph.add_conditional_edges("plan", nodes.dispatch_sections, ["collect"])
    graph.add_edge("collect", "merge")
    graph.add_edge("merge", "write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges(
        "critique",
        lambda s: "revise" if s.get("critique") and s.get("revisions", 0) < 1 else "publish",
        {"revise": "revise", "publish": "publish"})
    graph.add_edge("revise", "publish")
    graph.add_edge("publish", END)

    interrupt = ["publish"] if require_approval else None
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=interrupt)
