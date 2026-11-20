"""证据检索：在本次 run 已抓取的页面里做 BM25 检索，返回带引用编号的片段。"""
from __future__ import annotations

import re

from .registry import ToolContext, ToolSpec

# 专有名词（产品名/公司名）通常是 3 个字母以上的拉丁词；通用行业词不参与实体门控
_ENTITY = re.compile(r"[a-z][a-z0-9\-]{2,}")
_GENERIC = {"crm", "api", "saas", "app", "llm", "rag", "sdk", "erp"}


def _missing_entities(query: str, hits: list[dict]) -> list[str]:
    """实体门控：问的是 Gamma，库里只有 Acme/Beta 时，宁可拒答也不要拿别家资料编。

    只对拉丁专有名词做硬性要求；中文实体靠 BM25 覆盖率自然过滤。
    """
    wanted = {t for t in _ENTITY.findall(query.lower())} - _GENERIC
    if not wanted:
        return []
    corpus = " ".join(f"{h['title']} {h['text']}".lower() for h in hits)
    return sorted(e for e in wanted if e not in corpus)


async def search_evidence(ctx: ToolContext, args: dict) -> dict:
    query = str(args["query"])
    k = int(args.get("k", 4))
    hits = ctx.store.search(query, run_id=ctx.run_id, k=k)
    if not hits:
        return {"ok": True, "hits": [], "note": "证据库中没有匹配内容，请先抓取相关页面。"}
    missing = _missing_entities(query, hits)
    if missing:
        return {"ok": True, "hits": [],
                "note": f"证据库未覆盖 {', '.join(missing)}，请明确说明未找到相关资料，不要用其它公司的数据推断。"}
    lines = [f"[S{h['source_id']}] {h['title']} — {h['text']}" for h in hits]
    return {"ok": True, "hits": hits, "formatted": "\n".join(lines)}


def build_search_tool(settings) -> ToolSpec:
    return ToolSpec(
        name="search_evidence",
        description="在本次调研已抓取的页面中检索相关片段，返回带 [S<编号>] 的证据。",
        schema={"type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"]},
        handler=search_evidence,
        timeout_s=8.0,
        max_retries=0,
        max_result_chars=4000,
    )
