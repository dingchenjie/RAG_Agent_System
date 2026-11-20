"""Browser Use：真实 Playwright 打开页面，等待渲染后抽取正文并截图。

与 fetch_page 的分工：静态 HTML 用 httpx 更快；需要 JS 渲染、懒加载的页面走这里。
"""
from __future__ import annotations

import os

from .registry import ToolContext, ToolSpec
from .web import USER_AGENT, assert_safe_url


async def browse_page(ctx: ToolContext, args: dict) -> dict:
    url = str(args["url"])
    assert_safe_url(url, ctx.settings.browser_allow_domains)
    from playwright.async_api import async_playwright

    os.makedirs("data/shots", exist_ok=True)
    shot = os.path.join("data/shots", f"{ctx.run_id}-{ctx.pages_used + 1}.png")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=ctx.settings.browser_headless)
        try:
            page = await browser.new_page(user_agent=USER_AGENT)
            resp = await page.goto(url, wait_until="domcontentloaded",
                                   timeout=ctx.settings.browser_timeout_ms)
            await page.wait_for_timeout(700)          # 给前端渲染留一点时间
            title = (await page.title()) or "未命名页面"
            text = await page.inner_text("body")
            try:
                await page.screenshot(path=shot, full_page=False)
            except Exception:
                shot = ""
        finally:
            await browser.close()

    text = " ".join(text.split())[: ctx.settings.browser_max_chars]
    saved = ctx.store.add_source(ctx.run_id, url, title, text)
    ctx.pages_used += 1
    ctx.pages.append({"url": url, "title": title, "via": "playwright"})
    return {"ok": True, "citation": f"S{saved['source_id']}", "source_id": saved["source_id"],
            "title": title, "url": url, "status": resp.status if resp else None,
            "screenshot": shot, "text": text}


def build_browser_tool(settings) -> ToolSpec:
    return ToolSpec(
        name="browse_page",
        description="用无头浏览器打开需要 JS 渲染的页面，返回渲染后的正文与截图路径；正文进入证据库。",
        schema={"type": "object", "properties": {"url": {"type": "string"}},
                "required": ["url"]},
        handler=browse_page,
        timeout_s=max(settings.tool_timeout_s, 30.0),
        max_retries=0,
        max_result_chars=6000,
    )
