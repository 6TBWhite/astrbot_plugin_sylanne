"""Warmth v2 invariants: affect, source routing, settlement, decay, and migration."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from sylanne_alpha._engine.sylanne_core.compute.body import AlphaBodyState
from sylanne_alpha._engine.sylanne_core.compute.host import (
    SylanneAlphaHost,
    SylanneAlphaHostEvent,
)
from sylanne_alpha._engine.sylanne_core.compute.kernel import AlphaKernel
from sylanne_alpha.public_api import PublicAPI
from sylanne_alpha.v2core.body_port_v2 import CanonicalKernelBodyPort
from sylanne_alpha.v2core.domains.emotion import EmotionLedger
from sylanne_alpha.v2core.domains.user_model import UserModelDomain


def _event(text: str, now: float, *, flags: list[str] | None = None) -> SylanneAlphaHostEvent:
    return SylanneAlphaHostEvent(
        text=text,
        confidence=1.0,
        flags=list(flags or []),
        now=now,
        event_time={"epoch": now},
    )


def test_three_neutral_first_contact_rounds_do_not_warm_or_claim_relationship(
    tmp_path: Path,
) -> None:
    host = SylanneAlphaHost(root=tmp_path, session_key="neutral")
    neutral = {"valence": 0.0, "arousal": 0.0, "wound_risk": 0.0}

    for turn in range(3):
        now = 1_000.0 + turn * 10.0
        host.on_request(_event(f"中性消息 {turn}", now), assessment=neutral)
        host.on_response(_event(f"普通回复 {turn}", now + 1.0))

    body = host.kernel.body
    assert math.isclose(body.temperature.warmth, 0.45, abs_tol=1e-9)
    assert math.isclose(body.bloodflow.warmth, 0.40, abs_tol=1e-9)

    snapshot = CanonicalKernelBodyPort.from_host(host, "neutral").observe()
    prompt = "·".join(
        (EmotionLedger().prompt_line(snapshot), UserModelDomain().prompt_line())
    )
    for forbidden in ("很暖", "滚烫", "默契", "认识很久"):
        assert forbidden not in prompt


def test_request_and_response_have_explicit_source_phase_and_bounded_audit(
    tmp_path: Path,
) -> None:
    host = SylanneAlphaHost(root=tmp_path, session_key="audit")
    host.on_request(_event("用户消息", 100.0))
    assert host.kernel.last_event["origin"] == "user"
    assert host.kernel.last_event["phase"] == "request"

    host.on_response(_event("Agent 回复", 101.0))
    assert host.kernel.last_event["origin"] == "agent"
    assert host.kernel.last_event["phase"] == "response"
    assert [(item["origin"], item["phase"]) for item in host.kernel.audit["events"]] == [
        ("user", "request"),
        ("agent", "response"),
    ]

    for turn in range(70):
        host.on_request(_event("x", 200.0 + turn))
    assert len(host.kernel.audit["events"]) == 64
    assert all("text" not in item for item in host.kernel.audit["events"])


def test_agent_response_uses_dedicated_settlement_only(tmp_path: Path) -> None:
    host = SylanneAlphaHost(root=tmp_path, session_key="response")
    body = host.kernel.body
    body.pulse.last_tick = 500.0
    body.temperature.warmth = 0.72
    body.bloodflow.warmth = 0.63
    body.needs["need_expression"] = 0.50
    body.muscle.fatigue = 0.10
    body.nerve.repetition = 4
    body._recent_texts.append("用户原话")
    body.memory["relationship"] = {"signals": {"preference_count": 2}}
    before_relationship = body.relationship_memory()

    host.on_response(_event("这是 Agent 自己说的话", 500.0))

    assert math.isclose(body.needs["need_expression"], 0.38, abs_tol=1e-9)
    assert math.isclose(body.muscle.fatigue, 0.12, abs_tol=1e-9)
    assert body.temperature.warmth == 0.72
    assert body.bloodflow.warmth == 0.63
    assert body.nerve.repetition == 4
    assert list(body._recent_texts) == ["用户原话"]
    assert body.relationship_memory() == before_relationship


def test_explicit_safe_only_recovers_threat_state() -> None:
    body = AlphaBodyState()
    body.pulse.last_tick = 100.0
    body.pulse.strain = 0.5
    body.nerve.sensitivity = 0.5
    body.temperature.warmth = 0.45
    body.temperature.repair_heat = 0.4
    body.bloodflow.warmth = 0.40
    body.muscle.fatigue = 0.3
    body.needs["need_contact"] = 0.4
    body.immunity.boundary_pressure = 0.5
    body.immunity.interruption_budget = 0.6
    body.mortality.load = 0.5

    body.apply(flags=["safe"], now=100.0)

    assert body.pulse.strain < 0.5
    assert body.nerve.sensitivity < 0.5
    assert body.immunity.boundary_pressure < 0.5
    assert body.mortality.load < 0.5
    assert body.temperature.warmth == 0.45
    assert body.bloodflow.warmth == 0.40
    assert body.temperature.repair_heat == 0.4
    assert body.muscle.fatigue == 0.3
    assert body.needs["need_contact"] == 0.4
    assert body.immunity.interruption_budget == 0.6


def test_appraisal_caps_and_invalid_assessment_is_zero_update() -> None:
    body = AlphaBodyState()
    body.apply_assessment(
        {"valence": 1.0, "arousal": 0.0, "wound_risk": 0.0},
        event_confidence=1.0,
    )
    assert math.isclose(body.temperature.warmth, 0.47, abs_tol=1e-9)
    assert math.isclose(body.bloodflow.warmth, 0.43, abs_tol=1e-9)

    body.temperature.warmth = 0.5
    body.bloodflow.warmth = 0.5
    body.apply_assessment(
        {"valence": -1.0, "arousal": 0.0, "wound_risk": 1.0},
        event_confidence=1.0,
    )
    assert math.isclose(body.temperature.warmth, 0.44, abs_tol=1e-9)
    assert math.isclose(body.bloodflow.warmth, 0.44, abs_tol=1e-9)

    before = body.state_vector()
    body.apply_assessment({"valence": "bad", "arousal": 1.0, "wound_risk": 1.0})
    body.apply_assessment({"valence": 1.0, "arousal": 0.0})
    body.apply_assessment(None)
    assert body.state_vector() == before


def test_warmth_decays_by_wall_clock_and_clock_rollback_is_zero_interval() -> None:
    body = AlphaBodyState()
    body.pulse.last_tick = 100.0
    body.temperature.warmth = 1.0
    body.bloodflow.warmth = 1.0

    body.apply(now=1_900.0)
    assert math.isclose(body.temperature.warmth, 0.725, abs_tol=1e-9)
    assert math.isclose(body.bloodflow.warmth, 0.70, abs_tol=1e-9)

    body.temperature.warmth = 0.8
    body.bloodflow.warmth = 0.7
    body.apply(now=1_800.0)
    assert body.temperature.warmth == 0.8
    assert body.bloodflow.warmth == 0.7
    assert body.pulse.last_tick == 1_900.0

    body.apply(now=1_900.0 + 30.0 * 24.0 * 3600.0)
    assert math.isclose(body.temperature.warmth, 0.45, abs_tol=1e-9)
    assert math.isclose(body.bloodflow.warmth, 0.40, abs_tol=1e-9)


def test_legacy_kernel_migration_resets_only_warmth_and_is_idempotent() -> None:
    kernel = AlphaKernel.boot("legacy")
    kernel.body.temperature.warmth = 0.98
    kernel.body.bloodflow.warmth = 0.91
    kernel.body.wound.open = 0.33
    kernel.body.wound.scar = 0.44
    kernel.body.needs["need_repair"] = 0.27
    kernel.body.memory["relationship"] = {"signals": {"preference_count": 4}}
    kernel.personality = {"traits": {"edge": 0.77}}
    snapshot = kernel.snapshot()
    snapshot.pop("affect_semantics_version")

    migrated = AlphaKernel.restore(snapshot)
    assert migrated.body.temperature.warmth == 0.45
    assert migrated.body.bloodflow.warmth == 0.40
    assert migrated.body.wound.open == 0.33
    assert migrated.body.wound.scar == 0.44
    assert migrated.body.needs["need_repair"] == 0.27
    assert migrated.body.relationship_memory()["signals"]["preference_count"] == 4
    assert migrated.personality["traits"]["edge"] == 0.77

    migrated.body.temperature.warmth = 0.52
    restored_again = AlphaKernel.restore(migrated.snapshot())
    assert restored_again.body.temperature.warmth == 0.52
    assert restored_again.body.wound.scar == 0.44


class _ExpressionClock:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)

    def set(self, key: str, value: float) -> None:
        self.values[key] = value


@pytest.mark.asyncio
async def test_short_followup_is_not_accepted_but_long_silence_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = SylanneAlphaHost(root=tmp_path, session_key="feedback")
    clock = _ExpressionClock()
    plugin = SimpleNamespace(
        _host=lambda _key: host,
        _store=SimpleNamespace(last_bot_expression_time=clock),
        _event_time=lambda now: {"epoch": now},
        _has_persona_manager=lambda: False,
    )
    api = PublicAPI(plugin)
    outcomes: list[str] = []

    def spy_feedback(_self: object, outcome: str, dt: float = 1.0) -> None:
        _ = dt
        outcomes.append(outcome)

    monkeypatch.setattr(type(host.kernel.computation), "feedback", spy_feedback)
    clock.set("feedback", 100.0)

    await api.observe_request("feedback", text="继续", now=120.0)
    assert outcomes == []

    await api.observe_request("feedback", text="很久以后", now=401.0)
    assert outcomes == ["ignored"]
