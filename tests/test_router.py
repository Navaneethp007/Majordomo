"""Tests for the escalation gate.

The whole point of keeping the gate deterministic is that these tests need no
model, no network and no fixtures — the routing decision is arithmetic.
"""
from __future__ import annotations

import pytest

from majordomo import router
from majordomo.config import RouterConfig

SMALL = RouterConfig(size_threshold_tokens=100)


def test_small_payload_takes_the_cheap_path():
    decision = router.decide("a short line", SMALL)
    assert decision.path == "cheap"


def test_oversized_payload_escalates():
    decision = router.decide("x" * 100_000, SMALL)
    assert decision.path == "escalate"


def test_exactly_at_threshold_stays_cheap():
    """The threshold is a ceiling, not a trigger — 'exceeds' means strictly over."""
    payload = "x" * (SMALL.size_threshold_tokens * 4)
    decision = router.decide(payload, SMALL)
    assert decision.estimated_tokens == SMALL.size_threshold_tokens
    assert decision.path == "cheap"


def test_one_token_over_escalates():
    payload = "x" * ((SMALL.size_threshold_tokens + 1) * 4)
    assert router.decide(payload, SMALL).path == "escalate"


def test_empty_payload_is_cheap():
    assert router.decide("", SMALL).path == "cheap"


def test_decision_reports_its_reasoning():
    """--explain is what makes the fork observable rather than a claim."""
    decision = router.decide("x" * 100_000, SMALL)
    assert "exceeds" in decision.reason
    assert str(SMALL.size_threshold_tokens) in decision.reason


def test_threshold_is_configurable():
    payload = "x" * 4_000  # ~1000 tokens
    assert router.decide(payload, RouterConfig(size_threshold_tokens=100)).path == "escalate"
    assert router.decide(payload, RouterConfig(size_threshold_tokens=99_999)).path == "cheap"


def test_action_path_is_explicitly_deferred():
    """v1 reports; it does not act. The parameter exists so the interface won't
    have to change when it does."""
    with pytest.raises(NotImplementedError) as exc:
        router.decide("anything", SMALL, needs_action=True)
    assert "v2" in str(exc.value)


def test_estimate_tokens_is_four_chars_per_token():
    assert router.estimate_tokens("x" * 400) == 100


def test_path_for_maps_to_the_report_field():
    assert router.path_for(router.decide("small", SMALL)) == "cheap"
    assert router.path_for(router.decide("x" * 100_000, SMALL)) == "escalated"
