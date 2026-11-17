"""检索质量专项评测：只看检索器本身，不经过 Agent 的工具选择。

现有指标（成功率、工具 F1、忠实度）回答的是"Agent 用对了工具没有""答案有没有引用"；
这里补的是"该找到的片段，top-k 里到底有没有"。标注方式不依赖 chunk 序号：
``gold`` 里写 ``doc_id`` + 关键片段，命中该片段即视为找到。

除了 recall@k / MRR，还支持：
- 多 k 扫描（``ks``）：看召回随 top-k 的变化；
- 分类统计（``category``）：定位是近义改写、难负例还是 ACL 边界掉分；
- ``expect_empty``：越界问题必须返回空结果，单独统计"不乱召回"的比例；
- 门控阈值扫描（``gate_thresholds``）：展示"召回 vs 误答"的取舍；
- 多轮追问（``previous``）：先按运行时同一套规则补全 query，再统计召回，
  确保"那上海呢？"这类追问不会因为缺少上文而在评测里被误判为检索能力差。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..knowledge.query import build_retrieval_query


def load_retrieval_tasks(path: str | Path) -> list[dict]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def evaluate_retrieval(
    knowledge: Any,
    tasks: list[dict],
    *,
    k: int = 5,
    ks: tuple[int, ...] | None = None,
    gate_thresholds: tuple[float, ...] | None = None,
) -> dict:
    """对带 ``gold`` 标注的任务计算 recall@k 与 MRR。

    一条任务有多个 gold 时，任意一个命中即算该任务命中；MRR 取最早命中的排名。
    没有 gold 且未声明 ``expect_empty`` 的任务（闲聊、普通权限用例）不参与检索指标。
    """
    ks = tuple(sorted({k, *(ks or ())}))
    max_k = max(ks)
    primary = _run_pass(knowledge, tasks, ks=ks, max_k=max_k)
    scored = primary["scored"]

    by_k = {
        str(kk): {
            "recall_at_k": primary["hits"][kk] / scored if scored else 0.0,
            "mrr": primary["reciprocal"][kk] / scored if scored else 0.0,
            "ndcg_at_k": primary["ndcg"][kk] / scored if scored else 0.0,
        }
        for kk in ks
    }

    details = primary["details"]
    by_category = _by_category(details, k)

    sweep = []
    for threshold in gate_thresholds or ():
        result = _run_pass(knowledge, tasks, ks=(k,), max_k=max_k, min_coverage=threshold)
        sweep.append({
            "min_coverage": threshold,
            "recall_at_k": result["hits"][k] / result["scored"] if result["scored"] else 0.0,
            "mrr": result["reciprocal"][k] / result["scored"] if result["scored"] else 0.0,
            "empty_ok": result["empty_ok"],
            "empty_tasks": result["empty_tasks"],
        })

    return {
        "k": k,
        "ks": list(ks),
        "scored_tasks": scored,
        "hits": primary["hits"][k],
        "recall_at_k": primary["hits"][k] / scored if scored else 0.0,
        "mrr": primary["reciprocal"][k] / scored if scored else 0.0,
        "ndcg_at_k": primary["ndcg"][k] / scored if scored else 0.0,
        "empty_tasks": primary["empty_tasks"],
        "empty_ok": primary["empty_ok"],
        "empty_rate": (primary["empty_ok"] / primary["empty_tasks"]
                       if primary["empty_tasks"] else 1.0),
        "by_k": by_k,
        "by_category": by_category,
        "gate_sweep": sweep,
        "details": details,
    }


def _run_pass(
    knowledge: Any,
    tasks: list[dict],
    *,
    ks: tuple[int, ...],
    max_k: int,
    min_coverage: float | None = None,
) -> dict:
    hits = {kk: 0 for kk in ks}
    reciprocal = {kk: 0.0 for kk in ks}
    ndcg = {kk: 0.0 for kk in ks}
    scored = 0
    empty_tasks = 0
    empty_ok = 0
    details: list[dict] = []

    for task in tasks:
        scopes = set(task.get("scopes") or ["tool:read"])
        groups = {s.split("kb:group:", 1)[1] for s in scopes if s.startswith("kb:group:")}
        kwargs: dict[str, Any] = {"top_k": max_k, "acl_groups": groups}
        if min_coverage is not None:
            kwargs["min_coverage"] = min_coverage
        query = build_retrieval_query(task["input"], previous=task.get("previous"))
        retrieved = knowledge.search(query, **kwargs)
        category = task.get("category", "general")

        if task.get("expect_empty"):
            empty_tasks += 1
            ok = not retrieved
            empty_ok += int(ok)
            details.append({
                "task_id": task.get("id", ""), "category": category, "query": query,
                "expect_empty": True, "ok": ok, "returned": len(retrieved),
            })
            continue

        gold = task.get("gold") or []
        if not gold:
            continue

        scored += 1
        rank: int | None = None
        for index, item in enumerate(retrieved, start=1):
            if any(item.doc_id == ref["doc_id"] and ref["contains"] in item.text for ref in gold):
                rank = index
                break

        ranks = {kk: (rank if rank is not None and rank <= kk else None) for kk in ks}
        for kk in ks:
            if ranks[kk] is not None:
                hits[kk] += 1
                reciprocal[kk] += 1.0 / ranks[kk]
        task_ndcg = {str(kk): _ndcg_at_k(retrieved, gold, kk) for kk in ks}
        for kk in ks:
            ndcg[kk] += task_ndcg[str(kk)]
        details.append({
            "task_id": task.get("id", ""), "category": category, "query": query,
            "rank": rank, "ok": rank is not None, "ranks": ranks, "returned": len(retrieved),
            "ndcg": {key: round(value, 4) for key, value in task_ndcg.items()},
        })

    return {
        "hits": hits, "reciprocal": reciprocal, "ndcg": ndcg, "scored": scored,
        "empty_tasks": empty_tasks, "empty_ok": empty_ok, "details": details,
    }


def _ndcg_at_k(retrieved: list, gold: list[dict], k: int) -> float:
    """二元相关性 nDCG@k：每个 gold 片段最多计一次，IDCG 按 gold 数取满。

    检索返回的是 chunk 而不是文档，所以按片段（而非 doc）去重；这样
    "同一文档里命中了几个关键片段"仍会体现在排序收益上。
    """
    if not gold or k <= 0:
        return 0.0
    matched: set[int] = set()
    dcg = 0.0
    for index, item in enumerate(retrieved[:k], start=1):
        for gold_index, ref in enumerate(gold):
            if gold_index in matched:
                continue
            if item.doc_id == ref["doc_id"] and ref["contains"] in item.text:
                matched.add(gold_index)
                dcg += 1.0 / math.log2(index + 1)
                break
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def _by_category(details: list[dict], k: int) -> dict:
    groups: dict[str, list[dict]] = {}
    for item in details:
        if item.get("expect_empty"):
            continue
        groups.setdefault(item["category"], []).append(item)

    result = {}
    for category, items in sorted(groups.items()):
        hits = sum(1 for item in items if item.get("ok") and item["rank"] <= k)
        reciprocal = sum(1.0 / item["rank"] for item in items
                         if item.get("ok") and item["rank"] <= k)
        ndcg = sum(item.get("ndcg", {}).get(str(k), 0.0) for item in items)
        result[category] = {
            "scored_tasks": len(items),
            "hits": hits,
            "recall_at_k": hits / len(items),
            "mrr": reciprocal / len(items),
            "ndcg_at_k": ndcg / len(items),
        }
    return result
