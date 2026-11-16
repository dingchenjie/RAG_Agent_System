"""极简 Prometheus 指标（不引入 prometheus_client，减少部署依赖）。"""
from __future__ import annotations

import threading
from collections import defaultdict


def _labels_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _render_labels(key: tuple[tuple[str, str], ...]) -> str:
    if not key:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in key) + "}"


class Metrics:
    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[(name, _labels_key(labels))] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._histograms[(name, _labels_key(labels))].append(value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": {f"{k[0]}{_render_labels(k[1])}": v for k, v in self._counters.items()},
                "histograms": {
                    f"{k[0]}{_render_labels(k[1])}": {
                        "count": len(v),
                        "avg": round(sum(v) / len(v), 2) if v else 0,
                        "max": round(max(v), 2) if v else 0,
                    }
                    for k, v in self._histograms.items()
                },
            }

    def render_prometheus(self) -> str:
        lines: list[str] = ["# TYPE hermes_up gauge", "hermes_up 1"]
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: list(v) for k, v in self._histograms.items()}
        emitted: set[str] = set()
        for (name, labels), value in counters.items():
            if name not in emitted:
                lines.append(f"# TYPE {name} counter")
                emitted.add(name)
            lines.append(f"{name}{_render_labels(labels)} {value}")
        for (name, labels), values in histograms.items():
            base = name.removesuffix("_ms") + "_ms"
            if base not in emitted:
                lines.append(f"# TYPE {base} histogram")
                emitted.add(base)
            sorted_values = sorted(values)
            total = len(sorted_values)
            for label, quantile in (("0.5", 0.5), ("0.95", 0.95), ("0.99", 0.99)):
                idx = min(total - 1, int(total * quantile)) if total else 0
                value = sorted_values[idx] if total else 0
                lines.append(f'{base}{_render_labels(labels + (("quantile", label),))} {value}')
            lines.append(f'{base}_sum{_render_labels(labels)} {sum(values)}')
            lines.append(f'{base}_count{_render_labels(labels)} {total}')
        return "\n".join(lines) + "\n"
