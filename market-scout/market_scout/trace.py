"""可观测：事件落库 + JSONL 落盘 + Langfuse（自托管）导出。

设计取舍：事件（SQLite）是唯一事实源，Langfuse 只是旁路导出器——
没配 Langfuse 也能完整运行与回放，配了就能在官方 UI 里看按 run 聚合的观测。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from .config import Settings
from .store import Store


class Tracer:
    def __init__(self, store: Store, run_id: str, settings: Settings):
        self.store = store
        self.run_id = run_id
        self.settings = settings
        self._started = time.time()
        self._trace_id: str | None = None
        os.makedirs(os.path.dirname(settings.trace_jsonl) or ".", exist_ok=True)
        self._lf = self._init_langfuse()

    def _init_langfuse(self) -> Any:
        if not (self.settings.langfuse_host and self.settings.langfuse_public_key
                and self.settings.langfuse_secret_key):
            return None
        try:
            from langfuse import Langfuse

            client = Langfuse(host=self.settings.langfuse_host,
                              public_key=self.settings.langfuse_public_key,
                              secret_key=self.settings.langfuse_secret_key)
            # 同一 run 的事件必须聚到同一条 trace 上：用 run_id 派生稳定 trace_id
            if hasattr(client, "create_trace_id"):
                self._trace_id = client.create_trace_id(seed=self.run_id)
            return client
        except Exception:  # 观测不能影响主流程
            return None

    def emit(self, type_: str, **data: Any) -> int:
        seq = self.store.append_event(self.run_id, type_, data)
        record = {"run_id": self.run_id, "seq": seq, "type": type_,
                  "ts": round(time.time(), 4), "data": data}
        try:
            with open(self.settings.trace_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass
        self._export(type_, data)
        return seq

    def _export(self, type_: str, data: dict[str, Any]) -> None:
        """把事件导出到 Langfuse；各 SDK 版本 API 不同，失败一律静默降级。"""
        if self._lf is None:
            return
        ctx = {"trace_id": self._trace_id} if self._trace_id else None
        payload = {"input": data.get("input"), "output": data.get("output"), "metadata": data}
        try:
            if hasattr(self._lf, "create_event"):        # v5：事件是瞬时点，语义最准确
                self._lf.create_event(trace_context=ctx, name=type_, **payload)
            elif hasattr(self._lf, "start_observation"):  # v4
                span = self._lf.start_observation(trace_context=ctx, name=type_,
                                                  as_type="span", **payload)
                span.end()
            elif hasattr(self._lf, "trace"):              # v2
                self._lf.trace(id=self.run_id, name="market-scout").span(
                    name=type_, start_time=time.time(), **payload)
            elif hasattr(self._lf, "start_span"):         # v3
                self._lf.start_span(name=type_, **payload).end()
        except Exception:
            pass

    def flush(self) -> None:
        if self._lf is not None and hasattr(self._lf, "flush"):
            try:
                self._lf.flush()
            except Exception:
                pass
