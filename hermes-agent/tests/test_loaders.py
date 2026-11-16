"""文档加载：格式分派、PDF 页级切分、缺失文件/依赖的显式报错。"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from hermes.errors import HermesError
from hermes.knowledge.loaders import load_document


def test_load_markdown_and_txt(tmp_path: Path):
    markdown = tmp_path / "note.md"
    markdown.write_text("# 标题\n正文内容", encoding="utf-8")
    docs = load_document(markdown)
    assert len(docs) == 1
    assert docs[0].title == "note.md"
    assert "正文内容" in docs[0].text

    plain = tmp_path / "plain.txt"
    plain.write_text("纯文本内容", encoding="utf-8")
    assert load_document(plain)[0].text == "纯文本内容"


def test_load_html_strips_tags_and_scripts(tmp_path: Path):
    page = tmp_path / "page.html"
    page.write_text(
        "<html><head><style>p{color:red}</style></head>"
        "<body><h1>标题</h1><script>var a=1</script><p>正文段落</p></body></html>",
        encoding="utf-8",
    )
    text = load_document(page)[0].text
    assert "正文段落" in text
    assert "var a" not in text
    assert "color:red" not in text


def test_load_pdf_splits_by_page(tmp_path: Path, monkeypatch):
    class FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        def __init__(self, path):  # noqa: ARG002
            self.pages = [FakePage("第一页内容"), FakePage(""), FakePage("第三页内容")]

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=FakeReader))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    docs = load_document(pdf)
    assert [doc.meta["page"] for doc in docs] == [1, 3]
    assert docs[0].text == "第一页内容"


def test_unsupported_type_and_missing_file(tmp_path: Path):
    unknown = tmp_path / "a.xyz"
    unknown.write_text("x", encoding="utf-8")
    with pytest.raises(HermesError) as exc:
        load_document(unknown)
    assert exc.value.code == "unsupported_file_type"

    with pytest.raises(HermesError) as missing:
        load_document(tmp_path / "nope.md")
    assert missing.value.code == "file_not_found"
