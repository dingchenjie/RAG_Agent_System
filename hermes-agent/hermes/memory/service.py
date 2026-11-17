"""长期记忆服务。

写入链路：抽取 -> 提议(pending) -> 确认(active) -> 可被召回
治理能力：TTL 过期归档、同 key 冲突检测、纠错(supersedes)、按租户/用户硬隔离。
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

from ..errors import NotFound
from ..ids import new_memory_id
from ..knowledge.embedding import Embedder, cosine, tokenize
from ..storage.db import Database
from .extractor import MemoryCandidate, MemoryExtractor, RuleMemoryExtractor


@dataclass
class MemoryRecord:
    id: str
    tenant_id: str
    user_id: str
    mtype: str
    mkey: str
    value: str
    confidence: float
    status: str
    source_run_id: str | None
    supersedes_id: str | None
    created_at: float
    updated_at: float
    expires_at: float | None
    subject_type: str | None = None
    subject_id: str | None = None

    @classmethod
    def from_row(cls, row) -> MemoryRecord:
        return cls(
            id=row["id"], tenant_id=row["tenant_id"], user_id=row["user_id"], mtype=row["mtype"],
            mkey=row["mkey"], value=row["value"], confidence=row["confidence"], status=row["status"],
            source_run_id=row["source_run_id"], supersedes_id=row["supersedes_id"],
            created_at=row["created_at"], updated_at=row["updated_at"], expires_at=row["expires_at"],
            subject_type=row["subject_type"] if "subject_type" in row.keys() else None,
            subject_id=row["subject_id"] if "subject_id" in row.keys() else None,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "mtype": self.mtype, "key": self.mkey, "value": self.value,
            "subject_type": self.subject_type, "subject_id": self.subject_id,
            "confidence": round(self.confidence, 3), "status": self.status,
            "source_run_id": self.source_run_id, "supersedes_id": self.supersedes_id,
            "created_at": self.created_at, "updated_at": self.updated_at, "expires_at": self.expires_at,
        }


@dataclass
class ScoredMemory:
    record: MemoryRecord
    score: float

    def to_dict(self) -> dict:
        data = self.record.to_dict()
        data["score"] = round(self.score, 4)
        return data


class MemoryService:
    def __init__(
        self,
        db: Database,
        *,
        embedder: Embedder | None = None,
        extractor: MemoryExtractor | None = None,
        ttl_days: int = 90,
        min_confidence: float = 0.5,
    ):
        self.db = db
        self.embedder = embedder
        self.extractor = extractor or RuleMemoryExtractor()
        self.ttl_days = ttl_days
        self.min_confidence = min_confidence
        self._vec_cache: dict[str, tuple[float, list[float]]] = {}

    # ---------- 写入 ----------
    def propose(self, *, tenant_id: str, user_id: str, candidates: Iterable[MemoryCandidate],
                source_run_id: str | None = None, ttl_days: int | None = None,
                subject_type: str | None = None, subject_id: str | None = None) -> list[MemoryRecord]:
        now = time.time()
        created: list[MemoryRecord] = []
        for candidate in candidates:
            if candidate.confidence < self.min_confidence:
                continue
            exists = self.db.query_one(
                "SELECT * FROM memories WHERE tenant_id=? AND user_id=? AND mkey=? AND value=? "
                "AND COALESCE(subject_type,'')=? AND COALESCE(subject_id,'')=? "
                "AND status IN ('pending','active')",
                (tenant_id, user_id, candidate.mkey, candidate.value,
                 subject_type or "", subject_id or ""),
            )
            if exists:  # 同一业务对象下的重复记忆不重复写入
                continue
            expires = now + (ttl_days if ttl_days is not None else self.ttl_days) * 86400
            record = MemoryRecord(
                id=new_memory_id(), tenant_id=tenant_id, user_id=user_id, mtype=candidate.mtype,
                mkey=candidate.mkey, value=candidate.value, confidence=candidate.confidence,
                status="pending", source_run_id=source_run_id, supersedes_id=None,
                created_at=now, updated_at=now, expires_at=expires,
                subject_type=subject_type, subject_id=subject_id,
            )
            self.db.execute(
                "INSERT INTO memories(id, tenant_id, user_id, mtype, mkey, value, subject_type, "
                "subject_id, confidence, status, source_run_id, supersedes_id, created_at, updated_at, "
                "expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.id, tenant_id, user_id, record.mtype, record.mkey, record.value,
                 record.subject_type, record.subject_id, record.confidence, record.status,
                 record.source_run_id, None, now, now, expires),
            )
            created.append(record)
        return created

    def propose_from_text(self, *, tenant_id: str, user_id: str, text: str,
                          source_run_id: str | None = None,
                          subject_type: str | None = None,
                          subject_id: str | None = None) -> list[MemoryRecord]:
        return self.propose(tenant_id=tenant_id, user_id=user_id,
                            candidates=self.extractor.extract(text), source_run_id=source_run_id,
                            subject_type=subject_type, subject_id=subject_id)

    def confirm(self, memory_id: str, *, tenant_id: str, user_id: str) -> MemoryRecord:
        record = self._get_owned(memory_id, tenant_id, user_id)
        # 冲突只在同一业务对象内成立：客户 A 的联系偏好不应归档客户 B 的
        conflicts = self.db.query(
            "SELECT * FROM memories WHERE tenant_id=? AND user_id=? AND mkey=? AND status='active' AND id<>? "
            "AND COALESCE(subject_type,'')=? AND COALESCE(subject_id,'')=?",
            (tenant_id, user_id, record.mkey, record.id,
             record.subject_type or "", record.subject_id or ""),
        )
        now = time.time()
        with self.db.transaction() as conn:
            for row in conflicts:
                # 同 key 的旧值归档，新的胜出（可在 API 层返回冲突提示给用户确认）
                conn.execute("UPDATE memories SET status='archived', updated_at=? WHERE id=?", (now, row["id"]))
            conn.execute("UPDATE memories SET status='active', updated_at=? WHERE id=?", (now, record.id))
        return self._get_owned(memory_id, tenant_id, user_id)

    def reject(self, memory_id: str, *, tenant_id: str, user_id: str) -> MemoryRecord:
        self._get_owned(memory_id, tenant_id, user_id)
        self.db.execute("UPDATE memories SET status='rejected', updated_at=? WHERE id=?",
                        (time.time(), memory_id))
        return self._get_owned(memory_id, tenant_id, user_id)

    def correct(self, memory_id: str, *, tenant_id: str, user_id: str, value: str,
                reason: str | None = None) -> MemoryRecord:
        old = self._get_owned(memory_id, tenant_id, user_id)
        now = time.time()
        new_record = MemoryRecord(
            id=new_memory_id(), tenant_id=tenant_id, user_id=user_id, mtype=old.mtype, mkey=old.mkey,
            value=value, confidence=max(old.confidence, 0.9), status="active",
            source_run_id=old.source_run_id, supersedes_id=old.id, created_at=now, updated_at=now,
            expires_at=old.expires_at, subject_type=old.subject_type, subject_id=old.subject_id,
        )
        with self.db.transaction() as conn:
            conn.execute("UPDATE memories SET status='archived', updated_at=? WHERE id=?", (now, old.id))
            conn.execute(
                "INSERT INTO memories(id, tenant_id, user_id, mtype, mkey, value, subject_type, "
                "subject_id, confidence, status, source_run_id, supersedes_id, created_at, updated_at, "
                "expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_record.id, tenant_id, user_id, new_record.mtype, new_record.mkey, new_record.value,
                 new_record.subject_type, new_record.subject_id, new_record.confidence,
                 new_record.status, new_record.source_run_id, old.id, now, now,
                 new_record.expires_at),
            )
        return new_record

    def forget(self, memory_id: str, *, tenant_id: str, user_id: str) -> None:
        self._get_owned(memory_id, tenant_id, user_id)
        self.db.execute("UPDATE memories SET status='deleted', updated_at=? WHERE id=?",
                        (time.time(), memory_id))

    # ---------- 读取 ----------
    def recall(self, *, tenant_id: str, user_id: str, query: str, top_k: int = 4,
               now: float | None = None, subject_type: str | None = None,
               subject_id: str | None = None) -> list[ScoredMemory]:
        now = now or time.time()
        rows = self.db.query(
            "SELECT * FROM memories WHERE tenant_id=? AND user_id=? AND status='active' "
            "AND (expires_at IS NULL OR expires_at > ?)",
            (tenant_id, user_id, now),
        )
        if not rows:
            return []
        records = [MemoryRecord.from_row(r) for r in rows]
        q_vec = self.embedder.embed([query])[0] if self.embedder else None
        q_tokens = set(tokenize(query))
        scored: list[ScoredMemory] = []
        for record in records:
            text = f"{record.mkey} {record.value}"
            if q_vec is not None:
                vec = self._embed_cached(record)
                semantic = cosine(q_vec, vec)
            else:
                semantic = 0.0
            m_tokens = set(tokenize(text))
            lexical = len(q_tokens & m_tokens) / max(1, len(q_tokens))
            age_days = max(0.0, (now - record.created_at) / 86400)
            recency = 1.0 / (1.0 + age_days / 30.0)
            subject_match = (
                1.0 if subject_type and subject_id
                and record.subject_type == subject_type and record.subject_id == subject_id
                else 0.0
            )
            score = (0.45 * semantic + 0.3 * lexical + 0.15 * recency
                     + 0.1 * record.confidence + 0.15 * subject_match)
            scored.append(ScoredMemory(record=record, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def list_memories(self, *, tenant_id: str, user_id: str, status: str | None = None,
                      subject_type: str | None = None,
                      subject_id: str | None = None) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE tenant_id=? AND user_id=?"
        params: list = [tenant_id, user_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        if subject_type:
            sql += " AND subject_type=?"
            params.append(subject_type)
        if subject_id:
            sql += " AND subject_id=?"
            params.append(subject_id)
        sql += " ORDER BY updated_at DESC"
        return [MemoryRecord.from_row(r) for r in self.db.query(sql, params)]

    def sweep(self, *, now: float | None = None) -> int:
        """过期归档，返回归档条数。应由定时任务周期调用。"""
        now = now or time.time()
        cur = self.db.execute(
            "UPDATE memories SET status='archived', updated_at=? WHERE status='active' AND expires_at IS NOT NULL AND expires_at <= ?",
            (now, now),
        )
        return cur.rowcount

    def stats(self, *, tenant_id: str, user_id: str) -> dict:
        rows = self.db.query(
            "SELECT status, COUNT(*) AS n FROM memories WHERE tenant_id=? AND user_id=? GROUP BY status",
            (tenant_id, user_id),
        )
        return {row["status"]: row["n"] for row in rows}

    # ---------- 内部 ----------
    def _get_owned(self, memory_id: str, tenant_id: str, user_id: str) -> MemoryRecord:
        row = self.db.query_one(
            "SELECT * FROM memories WHERE id=? AND tenant_id=? AND user_id=?",
            (memory_id, tenant_id, user_id),
        )
        if row is None:
            raise NotFound(f"记忆不存在或无权访问: {memory_id}")
        return MemoryRecord.from_row(row)

    def _embed_cached(self, record: MemoryRecord) -> list[float]:
        cached = self._vec_cache.get(record.id)
        if cached and cached[0] == record.updated_at:
            return cached[1]
        vec = self.embedder.embed([f"{record.mkey} {record.value}"])[0]  # type: ignore[union-attr]
        self._vec_cache[record.id] = (record.updated_at, vec)
        return vec

    @staticmethod
    def render(recalled: list[ScoredMemory]) -> str:
        if not recalled:
            return ""
        return "\n".join(f"- {item.record.mkey}: {item.record.value}（置信度 {item.record.confidence:.2f}）"
                         for item in recalled)
