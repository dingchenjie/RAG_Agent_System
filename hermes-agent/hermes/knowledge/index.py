"""知识索引：分块、混合检索（BM25 + 向量）、RRF 融合、ACL 过滤、增量版本。

为什么不用 Chroma：
- 本项目要保证"零重型依赖可跑 + 测试可复现"，向量库会引入 ~500MB 依赖；
- 索引层被抽象成 ``KnowledgeIndex``，真实生产替换成 Chroma/Milvus 只需换实现；
- 面试时能讲清楚"为什么这么选"，比堆组件更有价值。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..errors import HermesError
from ..storage.db import Database
from .embedding import Embedder, cosine, tokenize

_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.\n])")

# 命中门控用的停用词：只做词法覆盖率判断，避免"的/是/多少"这类高频词制造假命中
_STOP_TOKENS = {
    "的", "是", "了", "在", "和", "与", "有", "吗", "呢", "啊", "我", "你", "他", "它", "们",
    "这", "那", "个", "请", "问", "多", "少", "多少", "上", "下", "如何", "怎么", "什么", "为",
}


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    version: str
    ord: int
    text: str
    title: str | None = None
    score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    citation: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "version": self.version,
            "ord": self.ord,
            "text": self.text,
            "title": self.title,
            "score": round(self.score, 6),
            "vector_score": round(self.vector_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "citation": self.citation,
        }


@dataclass
class IngestReport:
    doc_id: str
    version: str
    chunks: int
    replaced_versions: int
    chars: int
    duration_ms: float
    doc_ids: list[str] = field(default_factory=list)
    # 增量索引口径：只对文本发生变化的 chunk 调 embedding，未变动的复用旧向量
    embedded_chunks: int = 0
    reused_chunks: int = 0


def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    """按句子边界装箱，避免在句子中间切断；相邻块保留 overlap 字符。"""
    text = re.sub(r"[ \t]+", " ", text.strip())
    if not text:
        return []
    sentences = [s for s in _SENT_SPLIT.split(text) if s and s.strip()]
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        if len(buf) + len(sentence) <= size or not buf:
            buf += sentence
            continue
        chunks.append(buf.strip())
        buf = (buf[-overlap:] if overlap > 0 else "") + sentence
        while len(buf) > size:  # 超长句强制切
            chunks.append(buf[:size])
            buf = buf[size - overlap:]
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


class KnowledgeIndex:
    def __init__(self, db: Database, embedder: Embedder):
        self.db = db
        self.embedder = embedder

    # ---------- 写入 ----------
    def ingest(
        self,
        *,
        doc_id: str,
        text: str,
        version: str = "1",
        title: str | None = None,
        acl: Iterable[str] = (),
        chunk_size: int = 400,
        overlap: int = 60,
        activate: bool = True,
    ) -> IngestReport:
        chunks = chunk_text(text, chunk_size, overlap)
        return self.ingest_chunks(
            doc_id=doc_id, chunks=chunks, version=version, title=title, acl=acl,
            activate=activate, raw_chars=len(text),
        )

    def ingest_chunks(
        self,
        *,
        doc_id: str,
        chunks: list[str],
        version: str = "1",
        title: str | None = None,
        acl: Iterable[str] = (),
        activate: bool = True,
        raw_chars: int | None = None,
    ) -> IngestReport:
        """写入已分块（并已去冗余）的文本，供 loader/splitter 链路复用，避免二次切分。

        同一文档的新版本里，文本完全相同的 chunk 直接复用历史向量，只对新增/变更的
        chunk 调用 embedding；换过 embedding 后端（model/dim 不同）时不复用。
        """
        started = time.perf_counter()
        self._assert_embedding_compatible()
        reusable = self._reusable_embeddings(doc_id, chunks)
        pending = list(dict.fromkeys(chunk for chunk in chunks if chunk not in reusable))
        fresh = dict(zip(pending, self.embedder.embed(pending), strict=False)) if pending else {}
        embeddings = [reusable[chunk] if chunk in reusable else fresh[chunk] for chunk in chunks]
        now = time.time()

        with self.db.transaction() as conn:
            old = conn.execute(
                "SELECT version FROM documents WHERE doc_id=? AND active=1", (doc_id,)
            ).fetchall()
            replaced = len(old)
            conn.execute("UPDATE documents SET active=0 WHERE doc_id=?", (doc_id,))
            conn.execute("UPDATE chunks SET active=0 WHERE doc_id=?", (doc_id,))
            conn.execute(
                "INSERT OR REPLACE INTO documents(doc_id, version, title, acl, active, "
                "embedding_model, embedding_dim, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, version, title, json.dumps(sorted(set(acl)), ensure_ascii=False),
                 1 if activate else 0, getattr(self.embedder, "name", "unknown"),
                 getattr(self.embedder, "dim", None), now),
            )
            for idx, (chunk, vec) in enumerate(zip(chunks, embeddings, strict=False)):
                chunk_id = hashlib.sha1(f"{doc_id}|{version}|{idx}|{chunk}".encode()).hexdigest()[:24]
                conn.execute(
                    "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, version, ord, text, embedding, active, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (chunk_id, doc_id, version, idx, chunk, json.dumps(vec), 1 if activate else 0, now),
                )
        return IngestReport(
            doc_id=doc_id,
            version=version,
            chunks=len(chunks),
            replaced_versions=replaced,
            chars=raw_chars if raw_chars is not None else sum(len(c) for c in chunks),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            embedded_chunks=len(pending),
            reused_chunks=sum(1 for chunk in chunks if chunk in reusable),
        )

    def _reusable_embeddings(self, doc_id: str, chunks: list[str]) -> dict[str, list[float]]:
        """从同一文档的历史版本里找文本完全一致的 chunk 向量。"""
        wanted = set(chunks)
        if not wanted:
            return {}
        model = getattr(self.embedder, "name", "unknown")
        dim = getattr(self.embedder, "dim", None)
        rows = self.db.query(
            "SELECT c.text, c.embedding FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id AND d.version = c.version "
            "WHERE c.doc_id = ? AND d.embedding_model IS ? AND d.embedding_dim IS ?",
            (doc_id, model, dim),
        )
        reusable: dict[str, list[float]] = {}
        for row in rows:
            text = row["text"]
            if text not in wanted or text in reusable:
                continue
            try:
                vector = json.loads(row["embedding"])
            except (TypeError, json.JSONDecodeError):
                continue
            if dim and len(vector) != dim:
                continue
            reusable[text] = vector
        return reusable

    def _assert_embedding_compatible(self) -> None:
        """维度不一致会导致余弦相似度静默错分，这里显式失败并要求重建。"""
        dim = getattr(self.embedder, "dim", None)
        if not dim:
            return
        row = self.db.query_one(
            "SELECT doc_id, embedding_dim FROM documents "
            "WHERE active=1 AND embedding_dim IS NOT NULL AND embedding_dim<>? LIMIT 1",
            (dim,),
        )
        if row is not None:
            raise HermesError(
                f"已有索引使用 {row['embedding_dim']} 维向量（文档 {row['doc_id']}），"
                f"当前 embedder 为 {dim} 维；请清空索引后重建",
                code="embedding_mismatch",
            )

    def delete(self, doc_id: str) -> int:
        with self.db.transaction() as conn:
            conn.execute("UPDATE documents SET active=0 WHERE doc_id=?", (doc_id,))
            cur = conn.execute("UPDATE chunks SET active=0 WHERE doc_id=?", (doc_id,))
            return cur.rowcount

    def documents(self) -> list[dict]:
        rows = self.db.query(
            "SELECT doc_id, version, title, acl, embedding_model, embedding_dim, updated_at "
            "FROM documents WHERE active=1 ORDER BY updated_at DESC"
        )
        return [
            {
                "doc_id": row["doc_id"], "version": row["version"], "title": row["title"],
                "acl": json.loads(row["acl"] or "[]"),
                "embedding_model": row["embedding_model"],
                "embedding_dim": row["embedding_dim"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def versions(self, doc_id: str) -> list[dict]:
        """列出某个文档的全部历史版本（含已下线的），用于回滚前确认。"""
        rows = self.db.query(
            "SELECT d.version, d.title, d.acl, d.active, d.updated_at, "
            " (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.doc_id AND c.version=d.version) AS chunks "
            "FROM documents d WHERE d.doc_id=? ORDER BY d.updated_at DESC, d.version DESC",
            (doc_id,),
        )
        return [
            {
                "version": row["version"], "title": row["title"],
                "acl": json.loads(row["acl"] or "[]"), "active": bool(row["active"]),
                "chunks": int(row["chunks"]), "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def activate(self, doc_id: str, version: str) -> bool:
        """把指定历史版本切回线上；旧数据的 active 标志就是版本开关，不需要重建索引。"""
        with self.db.transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM documents WHERE doc_id=? AND version=?", (doc_id, version)
            ).fetchone()
            if not exists:
                return False
            conn.execute("UPDATE documents SET active=0 WHERE doc_id=?", (doc_id,))
            conn.execute("UPDATE chunks SET active=0 WHERE doc_id=?", (doc_id,))
            conn.execute("UPDATE documents SET active=1 WHERE doc_id=? AND version=?",
                         (doc_id, version))
            conn.execute("UPDATE chunks SET active=1 WHERE doc_id=? AND version=?",
                         (doc_id, version))
        return True

    # ---------- 读取 ----------
    def _active_chunks(self, acl_groups: set[str] | None) -> list[dict]:
        rows = self.db.query(
            "SELECT c.chunk_id, c.doc_id, c.version, c.ord, c.text, c.embedding, d.title, d.acl "
            "FROM chunks c JOIN documents d ON d.doc_id=c.doc_id AND d.version=c.version "
            "WHERE c.active=1 AND d.active=1"
        )
        result = []
        for row in rows:
            acl = set(json.loads(row["acl"] or "[]"))
            # 空 ACL = 公开；否则必须与调用方分组有交集（最小权限）
            if acl and acl_groups is not None and not (acl & acl_groups):
                continue
            if acl and acl_groups is None:
                continue
            result.append(dict(row))
        return result

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        acl_groups: set[str] | None = None,
        rerank_pool: int = 20,
        min_coverage: float = 0.2,
        min_vector: float = 0.18,
    ) -> list[Hit]:
        """混合检索：BM25 + 向量 + RRF 融合，并做相关性门控。

        门控是必要的一步：RRF 只负责排序，不负责判断"是否相关"。
        没有门控时，任何问题都会捞回若干片段，模型被迫用不相关材料作答（幻觉的主要来源）。
        规则：查询实词在片段中的覆盖率 ≥ min_coverage，或向量相似度 ≥ min_vector，才认为相关。
        """
        chunks = self._active_chunks(acl_groups)
        if not chunks:
            return []

        # 查询向量与索引向量维度必须一致：不校验的话 zip 会静默截断，返回"看起来有结果"的垃圾。
        self._assert_embedding_compatible()

        q_tokens = tokenize(query)
        content_tokens = {t for t in q_tokens if t not in _STOP_TOKENS}
        q_vec = self.embedder.embed([query])[0]

        # BM25
        df: dict[str, int] = {}
        doc_tokens: list[list[str]] = []
        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            doc_tokens.append(tokens)
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        n = len(chunks)
        avg_len = sum(len(t) for t in doc_tokens) / max(1, n)
        k1, b = 1.5, 0.75
        bm25_scores = []
        for tokens in doc_tokens:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            score = 0.0
            for token in q_tokens:
                if token not in tf:
                    continue
                idf = math.log(1 + (n - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
                score += idf * (tf[token] * (k1 + 1)) / (tf[token] + k1 * (1 - b + b * len(tokens) / max(1, avg_len)))
            bm25_scores.append(score)

        vec_scores = [cosine(q_vec, json.loads(c["embedding"] or "[]")) for c in chunks]

        bm25_rank = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:rerank_pool]
        vec_rank = sorted(range(n), key=lambda i: vec_scores[i], reverse=True)[:rerank_pool]

        # Reciprocal Rank Fusion：对两个异构排序做无标定融合
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(bm25_rank, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(vec_rank, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

        hits = []
        for idx, score in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True):
            if content_tokens:
                coverage = len(content_tokens & set(doc_tokens[idx])) / len(content_tokens)
                if coverage < min_coverage and vec_scores[idx] < min_vector:
                    continue
            chunk = chunks[idx]
            citation = f"{chunk['doc_id']}@v{chunk['version']}#chunk-{chunk['ord']}"
            hits.append(Hit(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                version=chunk["version"],
                ord=chunk["ord"],
                text=chunk["text"],
                title=chunk["title"],
                score=score,
                vector_score=vec_scores[idx],
                bm25_score=bm25_scores[idx],
                citation=citation,
            ))
            if len(hits) >= top_k:
                break
        return hits

    def stats(self) -> dict:
        return {
            "documents": int(self.db.scalar("SELECT COUNT(DISTINCT doc_id) FROM documents WHERE active=1") or 0),
            "versions": int(self.db.scalar("SELECT COUNT(*) FROM documents WHERE active=1") or 0),
            "chunks": int(self.db.scalar("SELECT COUNT(*) FROM chunks WHERE active=1") or 0),
        }
