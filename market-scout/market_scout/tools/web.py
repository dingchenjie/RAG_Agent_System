"""网页抓取：SSRF 防护 + 域名白名单 + 正文抽取 + 证据入库。"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .registry import ToolContext, ToolSpec

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def assert_safe_url(url: str, allow_domains: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("只允许 http/https")
    host = parsed.hostname or ""
    if not host:
        raise ValueError("URL 缺少主机名")
    allow = [d.strip().lower() for d in allow_domains.split(",") if d.strip()]
    if allow and not any(host.lower() == d or host.lower().endswith("." + d) for d in allow):
        raise ValueError(f"域名不在白名单内: {host}")
    for info in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"拒绝访问内网地址: {host}")


def extract_text(html: str, max_chars: int) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "") or "未命名页面"
    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg", "form"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    return title, text[:max_chars]


async def fetch_page(ctx: ToolContext, args: dict) -> dict:
    url = str(args["url"])
    assert_safe_url(url, ctx.settings.browser_allow_domains)
    async with httpx.AsyncClient(timeout=ctx.settings.tool_timeout_s, follow_redirects=True,
                                 headers={"User-Agent": USER_AGENT}) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        title, text = extract_text(resp.text, ctx.settings.browser_max_chars)
    saved = ctx.store.add_source(ctx.run_id, url, title, text)
    ctx.pages_used += 1
    ctx.pages.append({"url": url, "title": title, "via": "http"})
    return {"ok": True, "citation": f"S{saved['source_id']}", "source_id": saved["source_id"],
            "title": title, "url": url, "text": text}


def build_fetch_tool(settings) -> ToolSpec:
    return ToolSpec(
        name="fetch_page",
        description="抓取一个公开网页并抽取正文，正文会进入本次调研的证据库，可用 [S<编号>] 引用。",
        schema={"type": "object", "properties": {"url": {"type": "string",
                                                         "description": "以 http/https 开头的完整地址"}},
                "required": ["url"]},
        handler=fetch_page,
        timeout_s=settings.tool_timeout_s,
        max_retries=1,
        max_result_chars=6000,
    )
