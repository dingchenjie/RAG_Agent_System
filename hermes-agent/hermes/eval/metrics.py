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
    faithfulness: float | None = None
    error: str | None = None
    mode: str = "single"
    subtasks: int = 0
    subtasks_ok: int = 0

    @property
    def missing_keywords(self) -> list[str]:
        return [kw for kw in self.expected_contains if kw not in self.answer]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "category": self.category, "mode": self.mode, "ok": self.ok,
            "answer": self.answer, "missing_keywords": self.missing_keywords,
            "expected_tools": self.expected_tools, "called_tools": self.called_tools,
            "latency_ms": round(self.latency_ms, 2), "steps": self.steps,
            "tokens": self.tokens, "cost_usd": round(self.cost_usd, 6),
            "citations": self.citations, "faithfulness": self.faithfulness, "error": self.error,
            "subtasks": self.subtasks, "subtasks_ok": self.subtasks_ok,
        }


@dataclass
class EvalReport:
    results: list[TaskResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"meta": self.meta, "summary": self.summary(),
                "tasks": [r.to_dict() for r in self.results]}

    def summary(self) -> dict:
        if not self.results:
            return {}
        total = len(self.results)
        latencies = sorted(r.latency_ms for r in self.results)
        scored = [r.faithfulness for r in self.results if r.faithfulness is not None]
        subtask_total = sum(r.subtasks for r in self.results)
        return {
            "tasks": total,
            "success_rate": round(sum(1 for r in self.results if r.ok) / total, 4),
            "subtask_success_rate": (
                round(sum(r.subtasks_ok for r in self.results) / subtask_total, 4)
                if subtask_total else None
            ),
            "by_mode": self._by_mode(),
            "tool_precision": round(_mean(_precision(r) for r in self.results), 4),
            "tool_recall": round(_mean(_recall(r) for r in self.results), 4),
            "tool_f1": round(_mean(_f1(r) for r in self.results), 4),
            "citation_coverage": round(_mean(
                min(1.0, r.citations / max(1, len(r.called_tools))) for r in self.results), 4),
            "faithfulness_avg": round(_mean(scored), 4) if scored else None,
            "faithfulness_tasks": len(scored),
            "avg_steps": round(_mean(r.steps for r in self.results), 2),
            "avg_tokens": int(_mean(r.tokens for r in self.results)),
            "total_cost_usd": round(sum(r.cost_usd for r in self.results), 6),
            "latency_p50_ms": round(_percentile(latencies, 50), 2),
            "latency_p95_ms": round(_percentile(latencies, 95), 2),
        }

    def _by_mode(self) -> dict:
        groups: dict[str, list[TaskResult]] = {}
        for result in self.results:
            groups.setdefault(result.mode, []).append(result)
        by_mode = {}
        for mode, items in sorted(groups.items()):
            subtask_total = sum(r.subtasks for r in items)
            by_mode[mode] = {
                "tasks": len(items),
                "success_rate": round(sum(1 for r in items if r.ok) / len(items), 4),
                "subtask_success_rate": (
                    round(sum(r.subtasks_ok for r in items) / subtask_total, 4)
                    if subtask_total else None
                ),
                "avg_latency_ms": round(_mean(r.latency_ms for r in items), 2),
            }
        return by_mode

    def markdown(self) -> str:
        s = self.summary()
        lines = [
            "| 指标 | 数值 |",
            "| :--- | ---: |",
            f"| 任务数 | {s.get('tasks', 0)} |",
            f"| 任务成功率 | {s.get('success_rate', 0):.2%} |",
        ]
        if s.get("subtask_success_rate") is not None:
            lines.append(f"| 子任务成功率 | {s['subtask_success_rate']:.2%} |")
        lines += [
            f"| 工具选择 F1 | {s.get('tool_f1', 0):.4f} |",
            f"| 工具精确率 / 召回率 | {s.get('tool_precision', 0):.4f} / {s.get('tool_recall', 0):.4f} |",
            f"| 忠实度（引用支撑） | {_fmt_faithfulness(s.get('faithfulness_avg'))} |",
            f"| 平均步数 | {s.get('avg_steps', 0)} |",
            f"| 平均 token | {s.get('avg_tokens', 0)} |",
            f"| P50 / P95 延迟 | {s.get('latency_p50_ms', 0)} ms / {s.get('latency_p95_ms', 0)} ms |",
            "",
        ]
        if self.meta:
            lines += [
                "| 运行环境 | 取值 |",
                "| :--- | :--- |",
                f"| 数据集 | {self.meta.get('dataset', '-')}（{self.meta.get('dataset_size', 0)} 条） |",
                f"| 模式 | {', '.join(self.meta.get('modes', [])) or '-'} |",
                f"| LLM | {self.meta.get('llm_provider', '-')} / {self.meta.get('llm_model', '-')} |",
                f"| Prompt 版本 | {self.meta.get('prompt_version', '-')} |",
                f"| Embedding / Rerank | {self.meta.get('embedding', '-')} / {self.meta.get('rerank', '-')} |",
                "",
            ]
        if len(s.get("by_mode", {})) > 1:
            lines += ["| 模式 | 任务数 | 成功率 | 子任务成功率 | 平均延迟(ms) |",
                      "| :--- | ---: | ---: | ---: | ---: |"]
            for mode, item in s["by_mode"].items():
                sub = item.get("subtask_success_rate")
                lines.append(
                    f"| {mode} | {item['tasks']} | {item['success_rate']:.2%} | "
                    f"{'-' if sub is None else f'{sub:.2%}'} | {item['avg_latency_ms']:.0f} |"
                )
            lines.append("")
        lines += [
            "| 任务 | 类别 | 模式 | 结果 | 期望工具 | 实际工具 | 忠实度 | 延迟(ms) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: |",
        ]
        for r in self.results:
            lines.append(
                f"| {r.task_id} | {r.category} | {r.mode} | {'✅' if r.ok else '❌'} | "
                f"{','.join(r.expected_tools) or '-'} | {','.join(r.called_tools) or '-'} | "
                f"{_fmt_faithfulness(r.faithfulness)} | {r.latency_ms:.0f} |"
            )
        return "\n".join(lines)


_AGGREGATED_KEYS = ("success_rate", "tool_f1", "latency_p95_ms", "total_cost_usd")


def aggregate_summaries(summaries: list[dict],
                        keys: tuple[str, ...] = _AGGREGATED_KEYS) -> dict:
    """把同一份数据集的多轮评测汇总成均值 ± 标准差。

    真实模型的单轮结果会抖动，只报一次成功率无法区分"模型波动"和"真的回归"；
    多轮取均值、标准差与极值，才能把结论写成可复现的区间。
    """
    if not summaries:
        return {}
    merged = dict(summaries[0])
    merged["repeats"] = len(summaries)
    for key in keys:
        values = [float(item.get(key) or 0.0) for item in summaries]
        digits = 6 if key == "total_cost_usd" else 4
        merged[key] = round(statistics.fmean(values), digits)
        merged[f"{key}_std"] = round(statistics.pstdev(values), digits)
        merged[f"{key}_min"] = round(min(values), digits)
        merged[f"{key}_max"] = round(max(values), digits)
    scored = [item["faithfulness_avg"] for item in summaries
              if item.get("faithfulness_avg") is not None]
    merged["faithfulness_avg"] = round(statistics.fmean(scored), 4) if scored else None
    return merged


def _fmt_faithfulness(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


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
