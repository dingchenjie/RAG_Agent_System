"""长期记忆：确认、召回、纠错、TTL、租户隔离。"""
from __future__ import annotations

from hermes.api.app import build_state
from hermes.llm.mock import MockLLM
from hermes.memory.extractor import MemoryCandidate, RuleMemoryExtractor


def _service(settings):
    return build_state(settings, MockLLM()).memory


def test_extractor_finds_profile_and_preference():
    candidates = RuleMemoryExtractor().extract("我叫丁辰杰，我喜欢用 Python 做数据分析，我的城市是南京。")
    keys = {c.mkey for c in candidates}
    assert "name" in keys
    assert "profile.城市" in keys


def test_pending_memory_not_recalled_until_confirmed(settings):
    memory = _service(settings)
    created = memory.propose(tenant_id="t1", user_id="u1",
                             candidates=[MemoryCandidate(mkey="profile.city", value="南京", mtype="profile", confidence=0.9)])
    assert created and created[0].status == "pending"
    assert memory.recall(tenant_id="t1", user_id="u1", query="我住在哪") == []

    memory.confirm(created[0].id, tenant_id="t1", user_id="u1")
    recalled = memory.recall(tenant_id="t1", user_id="u1", query="我住在哪个城市")
    assert recalled and recalled[0].record.value == "南京"


def test_tenant_isolation(settings):
    memory = _service(settings)
    created = memory.propose(tenant_id="tenantA", user_id="u1",
                             candidates=[MemoryCandidate(mkey="secret", value="A 公司机密", confidence=0.9)])
    memory.confirm(created[0].id, tenant_id="tenantA", user_id="u1")
    assert memory.recall(tenant_id="tenantB", user_id="u1", query="机密") == []
    assert memory.recall(tenant_id="tenantA", user_id="u2", query="机密") == []
    assert memory.recall(tenant_id="tenantA", user_id="u1", query="机密")


def test_correction_supersedes_old_value(settings):
    memory = _service(settings)
    created = memory.propose(tenant_id="t", user_id="u",
                             candidates=[MemoryCandidate(mkey="profile.city", value="北京", confidence=0.9)])
    old = memory.confirm(created[0].id, tenant_id="t", user_id="u")
    new = memory.correct(old.id, tenant_id="t", user_id="u", value="南京")

    assert new.supersedes_id == old.id
    active = memory.list_memories(tenant_id="t", user_id="u", status="active")
    assert [r.value for r in active] == ["南京"]
    archived = memory.list_memories(tenant_id="t", user_id="u", status="archived")
    assert [r.value for r in archived] == ["北京"]


def test_ttl_expiry(settings):
    memory = _service(settings)
    created = memory.propose(tenant_id="t", user_id="u",
                             candidates=[MemoryCandidate(mkey="temp", value="临时信息", confidence=0.9)],
                             ttl_days=1)
    memory.confirm(created[0].id, tenant_id="t", user_id="u")

    future = created[0].expires_at + 86400
    assert memory.recall(tenant_id="t", user_id="u", query="临时", now=future) == []
    assert memory.sweep(now=future) >= 1
