"""文档加载：把不同格式统一成 ``Document``。

设计取舍：
- PDF 按页产出 ``Document``，页号进 ``meta``，这样引用精度能到页；
- 只把解析能力做成薄封装，不做格式转换中间层（过度设计的重灾区）；
- 缺可选依赖时给出可直接复制执行的安装提示，而不是抛裸 ImportError。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from ..errors import HermesError

_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
_HTML_SUFFIXES = {".html", ".htm"}


@dataclass
class Document:
    text: str
    source: str
    title: str | None = None
    meta: dict = field(default_factory=dict)


class _HTMLTextExtractor(HTMLParser):
    _SKIP = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._parts.append(data.strip())

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


def load_document(path: str | Path) -> list[Document]:
    """按后缀分派到具体 loader，一个文件可能产出多个 ``Document``（PDF 按页）。"""
    path = Path(path)
    if not path.exists():
        raise HermesError(f"文件不存在：{path}", code="file_not_found")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in _HTML_SUFFIXES:
        return _load_html(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in _TEXT_SUFFIXES or suffix == "":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(text=text, source=str(path), title=path.name, meta={"suffix": suffix})]
    raise HermesError(f"不支持的文件类型：{suffix}", code="unsupported_file_type")


def _load_pdf(path: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise HermesError("解析 PDF 需要安装 pypdf：pip install pypdf",
                          code="missing_dependency") from exc

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(Document(
            text=text, source=str(path), title=path.name,
            meta={"page": index, "suffix": ".pdf"},
        ))
    if not documents:
        raise HermesError(f"PDF 中没有可提取的文本：{path}", code="empty_document")
    return documents


def _load_html(path: Path) -> list[Document]:
    parser = _HTMLTextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return [Document(text=parser.text, source=str(path), title=path.name,
                     meta={"suffix": path.suffix.lower()})]


def _load_docx(path: Path) -> list[Document]:
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise HermesError("解析 docx 需要安装 python-docx：pip install python-docx",
                          code="missing_dependency") from exc

    paragraphs = [p.text for p in docx.Document(str(path)).paragraphs if p.text.strip()]
    return [Document(text="\n".join(paragraphs), source=str(path), title=path.name,
                     meta={"suffix": ".docx"})]
