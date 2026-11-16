"""分块与去冗余：标题边界、空输入、近重复过滤。"""
from __future__ import annotations

from hermes.knowledge.embedding import HashingEmbedder
from hermes.knowledge.splitter import dedupe, split_text


def test_split_text_prefers_markdown_headings():
    text = "# 第一节\n" + "内容甲。" * 40 + "\n# 第二节\n" + "内容乙。" * 40
    chunks = split_text(text, size=100, overlap=20)
    assert len(chunks) >= 2
    assert any(chunk.startswith("# 第一节") for chunk in chunks)
    assert any(chunk.startswith("# 第二节") for chunk in chunks)
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_split_text_handles_empty_input():
    assert split_text("", 100, 20) == []
    assert split_text("   \n  ", 100, 20) == []


def test_dedupe_drops_near_duplicates():
    chunks = [
        "LoRA 使用低秩矩阵 A 与 B 做增量更新。",
        "LoRA 使用低秩矩阵 A 与 B 做增量更新。",
        "差旅住宿一线城市上限为每晚 600 元。",
    ]
    kept = dedupe(chunks, HashingEmbedder(384), threshold=0.92)
    assert len(kept) == 2
    assert kept[0].startswith("LoRA")
    assert "600" in kept[1]


def test_dedupe_can_be_disabled():
    chunks = ["同一段内容。", "同一段内容。"]
    assert dedupe(chunks, HashingEmbedder(384), threshold=0) == chunks
