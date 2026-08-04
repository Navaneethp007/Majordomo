"""The escalation gate — the multi-agent fork.

Per-source, not global. Deliberately deterministic: spending a model call to
decide whether to use a model would be absurd, so the gate is arithmetic on the
payload size plus a flag.

    fetch → [gate] → within threshold  → cheap summarize   (default)
                     over threshold    → reducer agent for THIS source only

The **Size** trigger is wired for real. The **Action** trigger is designed into
the signature and raises ``NotImplementedError`` — v1 reports, it does not act
(spec §5).

On observability: on an ordinary morning nothing exceeds the threshold, so the
fork would never visibly fire and "we have a multi-agent system" would be a
claim rather than a fact. Hence ``decide`` returns its *reason*, and
``mj brief --explain`` prints which path every source took and why.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from majordomo.config import RouterConfig
from majordomo.models import RoutePath

Path_ = Literal["cheap", "escalate"]


@dataclass(frozen=True)
class Decision:
    path: Path_
    reason: str
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """Rough token count.

    Four characters per token is the usual English approximation. This does not
    need to be exact — it decides between two strategies, and being off by 15%
    only shifts where the boundary sits. A real tokenizer would mean a
    heavyweight dependency for a threshold the user can tune anyway.
    """
    return len(text) // 4


def decide(
    payload: str,
    config: RouterConfig,
    needs_action: bool = False,
) -> Decision:
    """Route one source's raw payload.

    Raises:
        NotImplementedError: ``needs_action`` — the Action path is v2. The
            parameter exists so the interface doesn't have to change later.
    """
    if needs_action:
        raise NotImplementedError(
            "the Action escalation path is deferred to v2 — v1 reports, it does not act"
        )

    tokens = estimate_tokens(payload)
    threshold = config.size_threshold_tokens

    if tokens > threshold:
        return Decision(
            path="escalate",
            reason=f"payload ~{tokens} tokens exceeds threshold {threshold}",
            estimated_tokens=tokens,
        )
    return Decision(
        path="cheap",
        reason=f"payload ~{tokens} tokens within threshold {threshold}",
        estimated_tokens=tokens,
    )


def path_for(decision: Decision) -> RoutePath:
    """Map a gate decision onto the value recorded on a SourceReport."""
    return "escalated" if decision.path == "escalate" else "cheap"
