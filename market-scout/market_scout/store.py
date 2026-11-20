"""SQLite 存储：run 状态、事件流、证据库（FTS5 全文索引）。"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'deliberative',
    status TEXT NOT NULL DEFAULT 'running',
    report TEXT,
    meta TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    chunk TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk, title, content='chunks', content_rowid='id');
"""


def _chunk(text: str, size: int = 450, overlap: int = 60) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    """FTS5 默认 tokenizer 不切中文，索引与查询两侧统一做 bigram 展开。

    中文按二元组切分（"定价页" → "定价" "价页"），英文数字按词；
    两侧用同一个函数，保证写入和检索的口径一致。
    """
    lowered = text.lower()
    terms: list[str] = []
    for run in _CJK_RUN.findall(lowered):
        terms.extend([run] if len(run) == 1 else
                     [run[i:i + 2] for i in range(len(run) - 1)])
    terms.extend(_WORD.findall(lowered))
    return terms


class Store:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- runs ----
    def create_run(self, run_id: str, question: str, mode: str = "deliberative") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, question, mode, status, created_at) VALUES (?,?,?,?,?)",
                (run_id, question, mode, "running", time.time()))
            self._conn.commit()

    def finish_run(self, run_id: str, status: str, report: str, meta: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status=?, report=?, meta=?, finished_at=? WHERE id=?",
                (status, report, json.dumps(meta, ensure_ascii=False), time.time(), run_id))
            self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, question, mode, status, created_at, finished_at FROM runs "
            "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---- events ----
    def append_event(self, run_id: str, type_: str, data: dict[str, Any]) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM events WHERE run_id=?", (run_id,)).fetchone()
            seq = int(row["seq"]) + 1
            self._conn.execute(
                "INSERT INTO events (run_id, seq, type, data, ts) VALUES (?,?,?,?,?)",
                (run_id, seq, type_, json.dumps(data, ensure_ascii=False), time.time()))
            self._conn.commit()
            return seq

    def list_events(self, run_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT seq, type, data, ts FROM events WHERE run_id=? AND seq>? ORDER BY seq",
            (run_id, after_seq)).fetchall()
        return [{"seq": r["seq"], "type": r["type"], "data": json.loads(r["data"]), "ts": r["ts"]}
                for r in rows]

    # ---- evidence ----
    def add_source(self, run_id: str, url: str, title: str, text: str) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sources (run_id, url, title, text, created_at) VALUES (?,?,?,?,?)",
                (run_id, url, title, text, time.time()))
            source_id = int(cur.lastrowid)
            for piece in _chunk(text):
                c = self._conn.execute(
                    "INSERT INTO chunks (source_id, run_id, url, title, chunk) VALUES (?,?,?,?,?)",
                    (source_id, run_id, url, title, piece))
                self._conn.execute(
                    "INSERT INTO chunks_fts (rowid, chunk, title) VALUES (?,?,?)",
                    (c.lastrowid, " ".join(_terms(piece)), " ".join(_terms(title))))
            self._conn.commit()
        return {"source_id": source_id, "url": url, "title": title, "chars": len(text)}

    def search(self, query: str, run_id: str | None = None, k: int = 4) -> list[dict[str, Any]]:
        terms = list(dict.fromkeys(_terms(query)))
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms[:8])
        try:
            sql = ("SELECT c.id, c.source_id, c.url, c.title, c.chunk, "
                   "bm25(chunks_fts, 1.0, 2.0) AS score "
                   "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                   "WHERE chunks_fts MATCH ?" + (" AND c.run_id = ?" if run_id else "") +
                   " ORDER BY score LIMIT ?")
            args: list[Any] = [match] + ([run_id] if run_id else []) + [k]
            rows = self._conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            rows = self._conn.execute(
                "SELECT id, source_id, url, title, chunk, 0 AS score FROM chunks "
                "WHERE (chunk LIKE ? OR title LIKE ?)" + (" AND run_id=?" if run_id else "") +
                " LIMIT ?",
                [f"%{terms[0]}%", f"%{terms[0]}%"] + ([run_id] if run_id else []) + [k]).fetchall()
        return [{"chunk_id": r["id"], "source_id": r["source_id"], "url": r["url"],
                 "title": r["title"], "text": r["chunk"], "score": round(float(r["score"]), 4)}
                for r in rows]

    def close(self) -> None:
        # CPython 用 sqlite3_close_v2 延迟关闭：连接对象析构前 Windows 上文件句柄不释放。
        # 因此这里主动丢掉引用，让句柄在 close() 返回时就确定性地释放（测试里能删临时目录）。
        conn, self._conn = getattr(self, "_conn", None), None
        if conn is not None:
            conn.close()
