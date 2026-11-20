"""重排策略。

- ``rrf``（默认）：索引层已经用 BM25 + 向量做了 RRF 融合，这里只做截断，
  零依赖、离线可测，也是"两路召回无标定融合"的标准做法；
- ``cross-encoder``：用 bge-reranker 对 (query, chunk) 逐对打分，精度更高但需要
  ``pip install -e ".[semantic]"``，因此作为可选后端；
- ``auto``：装了 sentence-transformers 就用 cross-encoder，否则 RRF；cross-encoder 模型
  加载失败（无网、缓存缺文件）时同样降级为 RRF 并记一条 warning。
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Protocol

from ..errors import HermesError
from .index import Hit

log = logging.getLogger(__name__)

# 权重按模型名进程内复用：每次 build_state 都新建实例，不缓存就会反复加载 GB 级权重
_MODEL_CACHE: dict[str, object] = {}


class Reranker(Protocol):
    name: str
    candidate_factor: int  # 候选池放大倍数：重排只能从召回的候选里挑，池子不能等于最终 top-k

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class RRFReranker:
    name = "rrf"
    candidate_factor = 1  # 索引层已融合，无需扩大候选

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class CrossEncoderReranker:
    name = "cross-encoder"
    candidate_factor = 3

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, fallback: bool = False):
        self.model_name = model_name
        self.fallback = fallback
        self._model: object | None | bool = None

    def _load(self):
        if self._model is False:  # 已降级，不再尝试
            return None
        if self._model is not None:
            return self._model
        cached = _MODEL_CACHE.get(self.model_name)
        if cached is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise HermesError(
                    "未安装 sentence-transformers，请执行 pip install -e \".[semantic]\"",
                    code="missing_dependency") from exc
            try:
                cached = CrossEncoder(self.model_name)
            except Exception as exc:  # noqa: BLE001 模型不可用（无网/缓存缺文件）时按配置降级
                if not self.fallback:
                    raise
                log.warning("重排模型 %s 加载失败，本次进程内降级为 RRF：%s", self.model_name, exc)
                self._model = False
                return None
            _MODEL_CACHE[self.model_name] = cached
        self._model = cached
        return cached

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        model = self._load()
        if model is None:
            return hits[:top_k]
        scores = model.predict([[query, hit.text] for hit in hits])
        ordered = sorted(zip(hits, scores, strict=False), key=lambda pair: float(pair[1]), reverse=True)
        return [hit for hit, _ in ordered[:top_k]]


def build_reranker(kind: str = "auto", *, model: str = "") -> Reranker:
    kind = (kind or "auto").strip().lower()
    auto = kind == "auto"
    if auto:
        kind = "cross-encoder" if importlib.util.find_spec("sentence_transformers") else "rrf"
    if kind == "rrf":
        return RRFReranker()
    if kind in ("cross-encoder", "cross_encoder"):
        # auto 模式下模型加载失败要退回 RRF；显式指定 cross-encoder 时则如实报错
        return CrossEncoderReranker(model or "BAAI/bge-reranker-base", fallback=auto)
    raise HermesError(f"未知的重排后端：{kind}", code="invalid_config")
