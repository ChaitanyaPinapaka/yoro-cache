"""Shared lookup / action planning for library and proxy paths.

Both `YORO.solve` and the OpenAI-compatible proxy must implement the same ladder:

  HIT     — high sim, fresh deps -> serve cached outcome
  REPLAY  — high sim, stale deps, derivation present -> re-apply method
  UPDATE  — high sim, escalate without safe replay -> re-reason in place
  COLD    — miss / borderline / novel -> reason and store a new case

Keeping the decision in one place prevents the two surfaces from drifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cache import ReasoningCache, ReasoningCase
from .invalidation import Invalidator
from .matcher import Decision, Matcher


@dataclass
class Lookup:
    """Result of nearest-neighbor + freshness + matcher, with replay eligibility."""

    decision: Decision
    case: Optional[ReasoningCase]
    sim: float
    fresh: bool
    same_case: bool  # sim >= tau_hit (case demonstrably belongs)
    should_replay: bool  # stale same-case with a stored derivation

    @property
    def empty_deps_hit_candidate(self) -> bool:
        """True when a HIT would serve a case that itself has no dependency scope."""
        return (
            self.decision == Decision.HIT
            and self.case is not None
            and not (self.case.deps or {})
        )


def lookup(
    cache: ReasoningCache,
    matcher: Matcher,
    invalidator: Invalidator,
    emb,
    current_deps: Optional[dict] = None,
    *,
    replay: bool = False,
) -> Lookup:
    """Single source of truth for cache routing decisions."""
    case, sim = cache.nearest(emb)
    if case is None:
        return Lookup(
            decision=Decision.MISS,
            case=None,
            sim=-1.0,
            fresh=True,
            same_case=False,
            should_replay=False,
        )
    fresh = invalidator.is_fresh(case, current_deps)
    decision = matcher.decide(sim, fresh)
    same_case = sim >= matcher.tau_hit
    has_derivation = bool((case.reasoning or "").strip() or case.steps)
    # Replay only on stale same-case (deps moved), never on failed-verify of a fresh case.
    should_replay = bool(
        replay and same_case and not fresh and has_derivation
    )
    return Lookup(
        decision=decision,
        case=case,
        sim=sim,
        fresh=fresh,
        same_case=same_case,
        should_replay=should_replay,
    )
