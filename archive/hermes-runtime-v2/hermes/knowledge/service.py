"""知识服务：检索 + 引用 + 上下文渲染。"""
from __future__ import annotations

from pathlib import Path

from .index import Hit, IngestReport, KnowledgeIndex


class KnowledgeService:
    def __init__(self, index: KnowledgeIndex, *, default_top_k: int = 4):
        self.index = index
        self.default_top_k = default_top_k

    def ingest_text(self, doc_id: str, text: str, *, version: str = "1", title: str | None = None,
                    acl: list[str] | None = None, chunk_size: int = 400, overlap: int = 60) -> IngestReport:
        return self.index.ingest(
            doc_id=doc_id, text=text, version=version, title=title,
            acl=acl or [], chunk_size=chunk_size, overlap=overlap,
        )

    def ingest_file(self, path: str | Path, *, doc_id: str | None = None, version: str = "1",
                    acl: list[str] | None = None) -> IngestReport:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(
            doc_id=doc_id or path.stem,
            text=text,
            version=version,
            title=path.name,
            acl=acl,
        )

    def search(self, query: str, *, top_k: int | None = None, acl_groups: set[str] | None = None) -> list[Hit]:
        return self.index.search(query, top_k=top_k or self.default_top_k, acl_groups=acl_groups)

    @staticmethod
    def render_context(hits: list[Hit], *, max_chars_per_hit: int = 500) -> str:
        if not hits:
            return ""
        blocks = []
        for i, hit in enumerate(hits, start=1):
            text = hit.text[:max_chars_per_hit]
            title = f"《{hit.title}》" if hit.title else ""
            blocks.append(f"[{i}] {title}{hit.citation}\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def citations(hits: list[Hit]) -> list[dict]:
        return [
            {"index": i, "doc_id": h.doc_id, "version": h.version, "chunk": h.ord,
             "citation": h.citation, "score": round(h.score, 6)}
            for i, h in enumerate(hits, start=1)
        ]

    def stats(self) -> dict:
        return self.index.stats()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # 可选依赖
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)
