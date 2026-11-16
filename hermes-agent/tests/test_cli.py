"""CLI：统一入口的 ask / ingest（固定离线后端，走 MockLLM）。"""
from __future__ import annotations

from pathlib import Path

from hermes.cli import main


def _offline_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("HERMES_EMBEDDING", "hashing")
    monkeypatch.setenv("HERMES_RERANK", "rrf")
    monkeypatch.setenv("HERMES_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("HERMES_LLM_PROVIDER", "mock")


def test_cli_ask_seeds_knowledge_and_prints_trace(tmp_path, monkeypatch, capsys):
    _offline_env(monkeypatch, tmp_path)
    code = main(["ask", "LoRA 的秩 r 通常设置为多少？"])
    output = capsys.readouterr().out
    assert code == 0
    assert "8" in output
    assert "kb_search" in output


def test_cli_ingest_file(tmp_path, monkeypatch, capsys):
    _offline_env(monkeypatch, tmp_path)
    document = tmp_path / "note.md"
    document.write_text("# 笔记\nLoRA 的秩 r 常用 8。", encoding="utf-8")
    code = main(["ingest", str(document)])
    output = capsys.readouterr().out
    assert code == 0
    assert "chunks" in output and "chars" in output
