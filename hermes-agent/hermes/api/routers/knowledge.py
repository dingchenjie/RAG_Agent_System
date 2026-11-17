"""知识库域路由：入库、版本管理、检索、文件上传与文档维护。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from pydantic import BaseModel, Field

from ...errors import ApiError, HermesError
from ...security.auth import Principal
from ..deps import get_state, require_scope, who

router = APIRouter()

ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".md", ".txt", ".docx", ".html", ".htm"}
UPLOAD_CHUNK_BYTES = 1024 * 1024


class IngestBody(BaseModel):
    doc_id: str
    text: str
    version: str = "1"
    title: str | None = None
    acl: list[str] = Field(default_factory=list)


@router.post("/v1/knowledge/documents")
async def ingest(body: IngestBody, request: Request,
                 principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "knowledge:write")
    report = await asyncio.to_thread(
        state.knowledge.ingest_text, body.doc_id, body.text, version=body.version,
        title=body.title, acl=body.acl,
        chunk_size=state.settings.chunk_size, overlap=state.settings.chunk_overlap,
    )
    return {"doc_id": report.doc_id, "version": report.version, "chunks": report.chunks,
            "replaced_versions": report.replaced_versions, "chars": report.chars,
            "embedded_chunks": report.embedded_chunks,
            "reused_chunks": report.reused_chunks}


@router.get("/v1/knowledge/documents/{doc_id}/versions")
async def list_kb_versions(doc_id: str, request: Request,
                           principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "knowledge:read")
    items = await asyncio.to_thread(state.knowledge.versions, doc_id)
    if not items:
        raise ApiError("文档不存在", code="document_not_found", status=404)
    return {"doc_id": doc_id, "items": items}


@router.post("/v1/knowledge/documents/{doc_id}/versions/{version}/activate")
async def activate_kb_version(doc_id: str, version: str, request: Request,
                              principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "knowledge:write")
    ok = await asyncio.to_thread(state.knowledge.activate_version, doc_id, version)
    if not ok:
        raise ApiError("版本不存在", code="version_not_found", status=404)
    return {"doc_id": doc_id, "version": version, "activated": True}


@router.get("/v1/knowledge/search")
async def search(q: str, request: Request, top_k: int = Query(4, ge=1, le=10),
                 acl_group: list[str] | None = Query(None),
                 principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "knowledge:read")
    hits = await asyncio.to_thread(state.knowledge.search, q, top_k=top_k,
                                   acl_groups=set(acl_group) if acl_group else None)
    return {"query": q, "hits": [h.to_dict() for h in hits]}


@router.post("/v1/knowledge/files")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    doc_id: str | None = Query(None),
    acl_group: list[str] | None = Query(None),
    principal: Principal = Depends(who),
) -> dict:
    """上传 PDF/MD/TXT/HTML/DOCX 并入库（走完整的 loader → 分块 → 去冗余链路）。

    防护顺序：后缀白名单 → Content-Length 预检 → 分块落盘并累计计数。
    任一步失败都会删掉半成品文件，不留孤儿。
    """
    state = get_state(request)
    require_scope(principal, "knowledge:write")
    raw_name = Path(file.filename or "upload.txt").name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ApiError(f"不支持的文件类型 {suffix or '<无后缀>'}",
                       code="unsupported_file_type", status=415,
                       details={"allowed": sorted(ALLOWED_UPLOAD_SUFFIXES)})
    max_bytes = state.settings.max_upload_mb * 1024 * 1024
    declared = (request.headers.get("content-length") or "").strip()
    if declared.isdigit() and int(declared) > max_bytes + 64 * 1024:
        raise ApiError(f"文件超过 {state.settings.max_upload_mb} MiB 上限",
                       code="file_too_large", status=413)
    upload_dir = Path(state.settings.upload_dir)
    await asyncio.to_thread(upload_dir.mkdir, parents=True, exist_ok=True)
    target = upload_dir / f"{uuid4().hex[:8]}-{raw_name}"
    written = 0
    ok = False
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ApiError(f"文件超过 {state.settings.max_upload_mb} MiB 上限",
                                   code="file_too_large", status=413)
                handle.write(chunk)
        try:
            report = await asyncio.to_thread(
                state.knowledge.ingest_path, target,
                doc_id=doc_id or Path(raw_name).stem, acl=acl_group or [])
        except HermesError as exc:
            raise ApiError(f"解析失败：{exc.message}", code="parse_failed", status=422) from exc
        ok = True
        return {"doc_id": report.doc_id, "doc_ids": report.doc_ids, "version": report.version,
                "chunks": report.chunks, "chars": report.chars,
                "replaced_versions": report.replaced_versions,
                "embedded_chunks": report.embedded_chunks,
                "reused_chunks": report.reused_chunks}
    finally:
        await file.close()
        if not ok:
            await asyncio.to_thread(target.unlink, missing_ok=True)


@router.get("/v1/knowledge/documents")
async def list_documents(request: Request, principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "knowledge:read")
    return {"items": state.knowledge.documents(), "stats": state.knowledge.stats()}


@router.delete("/v1/knowledge/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request,
                          principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "knowledge:write")
    removed = await asyncio.to_thread(state.knowledge.index.delete, doc_id)
    if not removed:
        raise ApiError("文档不存在", code="document_not_found", status=404)
    return {"deleted": doc_id, "chunks": removed}
