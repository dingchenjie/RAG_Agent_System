"""评测指标：只度量能被轨迹证伪的东西。

任务成功率、工具选择质量、步数、P50/P95 延迟、token/成本。
忠实度（faithfulness）在 Agent 场景下退化为"回答是否引用了检索来源"，
由 knowledge 的 citations 覆盖率近似，避免引入 RAGAS 这类重依赖。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class TaskResult:
    task_id: str
    category: str
    ok: bool
    answer: str
    expected_contains: list[str]
    expected_tools: list[str]
    called_tools: list[str]
    latency_ms: float
    steps: int
    tokens: int
    cost_usd: float
    citations: int = 0
    error: str | None = None

    @property
    def missing_keywords(self) -> list[str]:
        return [kw for kw in self.expected_contains if kw not in self.answer]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "category": self.category, "ok": self.ok,
            "answer": self.answer, "missing_keywords": self.missing_keywords,
            "expected_tools": self.expected_tools, "called_tools": self.called_tools,
            "latency_ms": round(self.latency_ms, 2), "steps": self.steps,
            "tokens": self.tokens, "cost_usd": round(self.cost_usd, 6),
            "citations": self.citations, "error": self.error,
        }


@dataclass
class EvalReport:
    results: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "tasks": [r.to_dict() for r in self.results]}

    def summary(self) -> dict:
        if not self.results:
            return {}
        total = len(self.results)
        latencies = sorted(r.latency_ms for r in self.results)
        return {
            "tasks": total,
            "success_rate": round(sum(1 for r in self.results if r.ok) / total, 4),
            "tool_precision": round(_mean(_precision(r) for r in self.results), 4),
            "tool_recall": round(_mean(_recall(r) for r in self.results), 4),
            "tool_f1": round(_mean(_f1(r) for r in self.results), 4),
            "citation_coverage": round(_mean(
                min(1.0, r.citations / max(1, len(r.called_tools))) for r in self.results), 4),
            "avg_steps": round(_mean(r.steps for r in self.results), 2),
            "avg_tokens": int(_mean(r.tokens for r in self.results)),
            "total_cost_usd": round(sum(r.cost_usd for r in self.results), 6),
            "latency_p50_ms": round(_percentile(latencies, 50), 2),
            "latency_p95_ms": round(_percentile(latencies, 95), 2),
        }

    def markdown(self) -> str:
        s = self.summary()
        lines = [
            "| 指标 | 数值 |",
            "| :--- | ---: |",
            f"| 任务数 | {s.get('tasks', 0)} |",
            f"| 任务成功率 | {s.get('success_rate', 0):.2%} |",
            f"| 工具选择 F1 | {s.get('tool_f1', 0):.4f} |",
            f"| 工具精确率 / 召回率 | {s.get('tool_precision', 0):.4f} / {s.get('tool_recall', 0):.4f} |",
            f"| 平均步数 | {s.get('avg_steps', 0)} |",
            f"| 平均 token | {s.get('avg_tokens', 0)} |",
            f"| P50 / P95 延迟 | {s.get('latency_p50_ms', 0)} ms / {s.get('latency_p95_ms', 0)} ms |",
            "",
            "| 任务 | 类别 | 结果 | 期望工具 | 实际工具 | 延迟(ms) |",
            "| :--- | :--- | :--- | :--- | :--- | ---: |",
        ]
        for r in self.results:
            lines.append(
                f"| {r.task_id} | {r.category} | {'✅' if r.ok else '❌'} | "
                f"{','.join(r.expected_tools) or '-'} | {','.join(r.called_tools) or '-'} | "
                f"{r.latency_ms:.0f} |"
            )
        return "\n".join(lines)


def _precision(r: TaskResult) -> float:
    if not r.called_tools:
        return 1.0 if not r.expected_tools else 0.0
    if not r.expected_tools:
        return 1.0
    return len(set(r.called_tools) & set(r.expected_tools)) / len(set(r.called_tools))


def _recall(r: TaskResult) -> float:
    if not r.expected_tools:
        return 1.0
    return len(set(r.called_tools) & set(r.expected_tools)) / len(set(r.expected_tools))


def _f1(r: TaskResult) -> float:
    p, rc = _precision(r), _recall(r)
    return 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)


def _mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct / 100))
    return sorted_values[idx]
