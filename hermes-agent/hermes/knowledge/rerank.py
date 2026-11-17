"""重排策略。

- ``rrf``（默认）：索引层已经用 BM25 + 向量做了 RRF 融合，这里只做截断，
  零依赖、离线可测，也是"两路召回无标定融合"的标准做法；
- ``cross-encoder``：用 bge-reranker 对 (query, chunk) 逐对打分，精度更高但需要
  ``pip install -e ".[semantic]"``，因此作为可选后端；
- ``auto``：装了 sentence-transformers 就用 cross-encoder，否则 RRF。
"""
from __future__ import annotations

import importlib.util
from typing import Protocol

from ..errors import HermesError
from .index import Hit


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

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise HermesError(
                "未安装 sentence-transformers，请执行 pip install -e \".[semantic]\"",
                code="missing_dependency") from exc
        self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        model = self._load()
        scores = model.predict([[query, hit.text] for hit in hits])
        ordered = sorted(zip(hits, scores, strict=False), key=lambda pair: float(pair[1]), reverse=True)
        return [hit for hit, _ in ordered[:top_k]]


def build_reranker(kind: str = "auto", *, model: str = "") -> Reranker:
    kind = (kind or "auto").strip().lower()
    if kind == "auto":
        kind = "cross-encoder" if importlib.util.find_spec("sentence_transformers") else "rrf"
    if kind == "rrf":
        return RRFReranker()
    if kind in ("cross-encoder", "cross_encoder"):
        return CrossEncoderReranker(model or "BAAI/bge-reranker-base")
    raise HermesError(f"未知的重排后端：{kind}", code="invalid_config")
