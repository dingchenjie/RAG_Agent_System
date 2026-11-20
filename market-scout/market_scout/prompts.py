"""提示词版本管理：v1 → v2 的迭代差异保留在代码里，便于 A/B 与面试复盘。"""
from __future__ import annotations

PROMPTS: dict[str, dict[str, str]] = {
    "v1": {
        "planner": "把用户的调研问题拆成不超过 3 个调研子问题，只输出 JSON：{{\"sections\": [\"...\"]}}。问题：{question}",
        "researcher": (
            "你是竞品调研员。可用工具：fetch_page / browse_page / search_evidence。"
            "先抓取页面，再检索证据，最后用 3-5 句话总结这一节的关键事实。"
            "调研子问题：{section}"
        ),
        "writer": (
            "根据下面的调研笔记写一份简短的竞品调研报告（Markdown）。\n调研问题：{question}\n"
            "笔记：\n{notes}\n"
        ),
        "critic": (
            "检查报告是否存在没有依据的说法，逐条列出问题（没有就输出 NONE）。\n报告：\n{draft}"
        ),
    },
    "v2": {
        "planner": (
            "把用户的竞品调研问题拆成不超过 3 个可独立调研的子问题，覆盖：定价、核心功能、目标用户/定位。"
            "只输出 JSON，不要解释：{{\"sections\": [\"...\"]}}。问题：{question}"
        ),
        "researcher": (
            "你是竞品调研员，必须遵守三条纪律：\n"
            "1) 先证据后结论：没有抓到的内容不许猜，查不到就写「未找到公开信息」；\n"
            "2) 每条事实后面标注来源编号，例如 [S3]；\n"
            "3) 同一动作不要重复超过两次，重复即视为无进展，改为基于已有证据作答。\n"
            "可用工具：fetch_page（静态页）/ browse_page（需 JS 渲染）/ search_evidence（本地证据检索）。\n"
            "调研子问题：{section}"
        ),
        "writer": (
            "你是调研报告撰写人。只用给定证据写作，禁止引入证据之外的事实。\n"
            "要求：Markdown；分「结论 / 分项发现 / 证据来源」三部分；每个事实后带 [S<编号>]；"
            "证据不足的结论必须标注「证据不足」。\n"
            "调研问题：{question}\n证据：\n{notes}\n"
        ),
        "critic": (
            "逐句核对报告中的事实是否被 [S<编号>] 证据支撑。只输出 JSON："
            "{{\"issues\": [\"问题1\", \"问题2\"]}}，没有问题时 issues 为空数组。\n报告：\n{draft}"
        ),
    },
}


def get_prompt(version: str, key: str) -> str:
    return PROMPTS.get(version, PROMPTS["v2"])[key]
