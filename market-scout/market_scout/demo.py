"""离线演示数据：两条虚构竞品的公开信息摘要，用于 mock 模型跑通全链路。"""
from __future__ import annotations

from .store import Store

SOURCES = [
    ("https://example.com/acme-crm/pricing", "Acme CRM 定价页",
     "Acme CRM 提供三档套餐：Starter 每席位每月 12 美元，Growth 每席位 29 美元，"
     "Enterprise 需要联系销售。Growth 及以上包含自动化工作流和 API 访问。"
     "年付可享受 20% 折扣。"),
    ("https://example.com/acme-crm/features", "Acme CRM 功能页",
     "Acme CRM 的核心功能包括销售管道管理、自动化工作流、邮件序列、报表看板与开放 API。"
     "它强调零代码配置，目标用户是 20 到 200 人的中小销售团队。"),
    ("https://example.com/beta-crm/pricing", "Beta CRM 定价页",
     "Beta CRM 分为 Free、Pro 每月 25 美元、Business 每月 49 美元三档，按用户数计费。"
     "Free 档限制 1000 条联系人，Pro 档包含 AI 线索评分。"),
    ("https://example.com/beta-crm/positioning", "Beta CRM 产品定位",
     "Beta CRM 定位为 AI 优先的销售助手，主打线索评分与通话摘要，目标客户是快速扩张的销售组织。"
     "其弱点是对非英语市场的本地化支持有限。"),
]


def seed_demo_sources(store: Store, run_id: str) -> list[dict]:
    return [store.add_source(run_id, url, title, text) for url, title, text in SOURCES]
