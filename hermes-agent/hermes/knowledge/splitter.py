"""分块与去冗余。

两件事分开做：
1. ``split_text``：先按 markdown 标题切段，再在段内按句子边界装箱（不切断句子）；
2. ``dedupe``：用余弦相似度做贪心去重，替代旧项目里 sklearn 全量相似矩阵的实现
   （O(n²) 内存 → O(n·k)，且复用同一个 embedder，避免两套向量口径）。
"""
from __future__ import annotations

import re

from .embedding import Embedder, cosine
from .index import chunk_text

_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+.*$")


def _sections(text: str) -> list[str]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [text]
    blocks: list[str] = []
    if matches[0].start() > 0:
        blocks.append(text[:matches[0].start()])
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[match.start():end])
    return [block for block in blocks if block.strip()]


def split_text(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    """按标题优先、句边界装箱的方式分块。"""
    if not text or not text.strip():
        return []
    chunks: list[str] = []
    for section in _sections(text):
        chunks.extend(chunk_text(section, size, overlap))
    return chunks


def dedupe(chunks: list[str], embedder: Embedder, threshold: float = 0.92) -> list[str]:
    """丢弃与已保留块余弦相似度超过阈值的重复块（阈值 <=0 表示关闭）。"""
    if threshold <= 0 or len(chunks) <= 1:
        return list(chunks)
    kept: list[str] = []
    kept_vectors: list[list[float]] = []
    for chunk, vector in zip(chunks, embedder.embed(chunks), strict=False):
        if any(cosine(vector, kept_vector) > threshold for kept_vector in kept_vectors):
            continue
        kept.append(chunk)
        kept_vectors.append(vector)
    return kept
