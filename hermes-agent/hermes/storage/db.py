"""SQLite 存储层。

选型说明（面试可讲）：
- Agent Runtime 的状态量小但要求强一致、可恢复、零外部依赖，SQLite(WAL) 比 Redis/Postgres 更合适；
- 所有写操作串行化（RLock），读走同一连接，避免连接池带来的复杂度；
- 通过 ``asyncio.to_thread`` 在异步路径中调用，不阻塞事件循环。
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    title       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    input           TEXT NOT NULL,
    state           TEXT NOT NULL,
    answer          TEXT,
    error           TEXT,
    steps           INTEGER NOT NULL DEFAULT 0,
    checkpoint      TEXT,
    usage           TEXT,
    idempotency_key TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_idem ON runs(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id, created_at);

CREATE TABLE IF NOT EXISTS events (
    run_id          TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    ts              REAL NOT NULL,
    type            TEXT NOT NULL,
    span_id         TEXT,
    parent_span_id  TEXT,
    payload         TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS tool_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    tool        TEXT NOT NULL,
    version     TEXT NOT NULL,
    arguments   TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    error_code  TEXT,
    latency_ms  REAL NOT NULL,
    attempts    INTEGER NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON tool_audit(run_id);

CREATE TABLE IF NOT EXISTS idempotency (
    key         TEXT NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (key, kind)
);

CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    mtype           TEXT NOT NULL,
    mkey            TEXT NOT NULL,
    value           TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.8,
    status          TEXT NOT NULL DEFAULT 'pending',
    source_run_id   TEXT,
    supersedes_id   TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    expires_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_mem_owner ON memories(tenant_id, user_id, status);
CREATE INDEX IF NOT EXISTS idx_mem_key ON memories(tenant_id, user_id, mkey);

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT NOT NULL,
    version     TEXT NOT NULL,
    title       TEXT,
    acl         TEXT NOT NULL DEFAULT '[]',
    active      INTEGER NOT NULL DEFAULT 1,
    embedding_model TEXT,
    embedding_dim   INTEGER,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (doc_id, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    version     TEXT NOT NULL,
    ord         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, version, active);

CREATE TABLE IF NOT EXISTS lessons (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    rule        TEXT NOT NULL,
    content     TEXT NOT NULL,
    evidence    TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'pending',
    hits        INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_lessons_rule ON lessons(tenant_id, rule);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """轻量向前迁移：老库缺列时补齐，避免用户必须先删库。"""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(documents)")}
        if "embedding_model" not in columns:
            self._conn.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT")
        if "embedding_dim" not in columns:
            self._conn.execute("ALTER TABLE documents ADD COLUMN embedding_dim INTEGER")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def executemany(self, sql: str, seq_params: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, seq_params)
            self._conn.commit()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.query_one(sql, params)
        return None if row is None else row[0]

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()
