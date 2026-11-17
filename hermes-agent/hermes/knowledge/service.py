"""知识服务：加载 → 分块 → 去冗余 → 入库 → 检索 → 引用渲染。"""
from __future__ import annotations

from pathlib import Path

from ..security.untrusted import wrap_untrusted
from .index import Hit, IngestReport, KnowledgeIndex
from .loaders import load_document
from .rerank import Reranker, RRFReranker
from .splitter import dedupe, split_text

_SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".text", ".html", ".htm", ".docx"}


class KnowledgeService:
    def __init__(
        self,
        index: KnowledgeIndex,
        *,
        default_top_k: int = 4,
        reranker: Reranker | None = None,
        chunk_size: int = 400,
        overlap: int = 60,
        dedupe_threshold: float = 0.92,
    ):
        self.index = index
        self.default_top_k = default_top_k
        self.reranker = reranker or RRFReranker()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.dedupe_threshold = dedupe_threshold

    # ---------- 写入 ----------
    def ingest_text(self, doc_id: str, text: str, *, version: str = "1", title: str | None = None,
                    acl: list[str] | None = None, chunk_size: int | None = None,
                    overlap: int | None = None) -> IngestReport:
        return self.index.ingest(
            doc_id=doc_id, text=text, version=version, title=title,
            acl=acl or [], chunk_size=chunk_size or self.chunk_size,
            overlap=overlap if overlap is not None else self.overlap,
        )

    def ingest_path(self, path: str | Path, *, doc_id: str | None = None, version: str = "1",
                    acl: list[str] | None = None) -> IngestReport:
        """真实文档链路：loader → splitter → dedupe → index。

        多页 PDF 会按页拆成多个文档（``<stem>#p<page>``），让引用能落到页级。
        """
        path = Path(path)
        documents = load_document(path)
        base_id = doc_id or path.stem
        reports: list[IngestReport] = []

        for position, document in enumerate(documents, start=1):
            page = document.meta.get("page")
            current_id = base_id if len(documents) == 1 else f"{base_id}#p{page or position}"
            chunks = split_text(document.text, self.chunk_size, self.overlap)
            if self.dedupe_threshold > 0:
                chunks = dedupe(chunks, self.index.embedder, self.dedupe_threshold)
            reports.append(self.index.ingest_chunks(
                doc_id=current_id, chunks=chunks, version=version,
                title=document.title, acl=acl or [], raw_chars=len(document.text),
            ))

        return IngestReport(
            doc_id=base_id,
            version=version,
            chunks=sum(report.chunks for report in reports),
            replaced_versions=sum(report.replaced_versions for report in reports),
            chars=sum(report.chars for report in reports),
            duration_ms=round(sum(report.duration_ms for report in reports), 2),
            doc_ids=[report.doc_id for report in reports],
        )

    def ingest_dir(self, directory: str | Path, *, version: str = "1") -> list[IngestReport]:
        directory = Path(directory)
        return [
            self.ingest_path(path, version=version)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        ]

    # 兼容旧入口
    ingest_file = ingest_path
    ingest_pdf = ingest_path

    # ---------- 检索 ----------
    def search(self, query: str, *, top_k: int | None = None,
               acl_groups: set[str] | None = None,
               min_coverage: float | None = None,
               min_vector: float | None = None) -> list[Hit]:
        top_k = top_k or self.default_top_k
        gate: dict[str, float] = {}
        if min_coverage is not None:
            gate["min_coverage"] = min_coverage
        if min_vector is not None:
            gate["min_vector"] = min_vector
        hits = self.index.search(query, top_k=max(top_k, self.default_top_k),
                                 acl_groups=acl_groups, **gate)
        return self.reranker.rerank(query, hits, top_k)

    def documents(self) -> list[dict]:
        return self.index.documents()

    def versions(self, doc_id: str) -> list[dict]:
        return self.index.versions(doc_id)

    def activate_version(self, doc_id: str, version: str) -> bool:
        return self.index.activate(doc_id, version)

    # ---------- 渲染 ----------
    @staticmethod
    def render_context(hits: list[Hit], *, max_chars_per_hit: int = 500, wrap: bool = True) -> str:
        """渲染检索结果。

        ``wrap=False`` 用于工具返回值：工具输出在 Runtime 边界已统一包裹为不可信内容，
        这里再包一层会造成嵌套转义、白白吃掉上下文预算。
        """
        if not hits:
            return ""
        blocks = []
        for i, hit in enumerate(hits, start=1):
            text = hit.text[:max_chars_per_hit]
            title = f"《{hit.title}》" if hit.title else ""
            body = f"[{i}] {title}{hit.citation}\n{text}"
            if wrap:
                body = wrap_untrusted(
                    "retrieved_document", body,
                    {"index": str(i), "doc_id": hit.doc_id, "source": hit.citation})
            blocks.append(body)
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

    def describe(self) -> dict:
        return {
            "embedding": {"provider": getattr(self.index.embedder, "name", "unknown"),
                          "dim": getattr(self.index.embedder, "dim", None)},
            "rerank": {"provider": self.reranker.name},
        }
