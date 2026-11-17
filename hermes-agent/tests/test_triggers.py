"""周期触发器：参数校验、到期原子推进、错过周期不补跑、租户隔离。"""
from __future__ import annotations

import pytest

from hermes.triggers import TriggerService


@pytest.fixture
def service(state) -> TriggerService:
    return state.triggers


def test_create_rejects_too_short_interval(service):
    with pytest.raises(ValueError):
        service.create(name="too-fast", input="ping", interval_s=5)


def test_due_advances_next_fire_at_atomically(service):
    trigger = service.create(name="t", input="ping", interval_s=10, now=1000.0)
    assert trigger.next_fire_at == 1010.0
    assert service.due(now=1005.0) == []

    fired = service.due(now=1011.0)
    assert [item.id for item in fired] == [trigger.id]
    assert service.get(trigger.id).next_fire_at == 1020.0
    # 同一个周期不会被第二次取走
    assert service.due(now=1011.0) == []


def test_due_skips_missed_cycles_after_downtime(service):
    trigger = service.create(name="t", input="ping", interval_s=10, now=1000.0)

    fired = service.due(now=1100.0)
    assert [item.id for item in fired] == [trigger.id]
    # 停机期间错过的周期直接跳过，恢复瞬间不会补跑一串
    assert service.get(trigger.id).next_fire_at > 1100.0


def test_enable_disable_tenant_filter_and_remove(service):
    tenant_a = service.create(name="a", input="ping", interval_s=10, tenant_id="t1")
    tenant_b = service.create(name="b", input="ping", interval_s=10, tenant_id="t2")

    assert {item.id for item in service.list(tenant_id="t1")} == {tenant_a.id}

    service.set_enabled(tenant_a.id, False)
    assert service.get(tenant_a.id).enabled is False
    fired = service.due(now=tenant_a.next_fire_at + 1)
    assert [item.id for item in fired] == [tenant_b.id]

    assert service.remove(tenant_a.id) is True
    assert service.remove(tenant_a.id) is False
    with pytest.raises(KeyError):
        service.get(tenant_a.id)


def test_mark_fired_records_last_run(service):
    trigger = service.create(name="t", input="ping", interval_s=10)
    service.mark_fired(trigger.id, "run_x", trigger.next_fire_at + 10)

    updated = service.get(trigger.id)
    assert updated.last_run_id == "run_x"
    assert updated.next_fire_at == trigger.next_fire_at + 10
