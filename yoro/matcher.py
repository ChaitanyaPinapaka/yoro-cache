"""Matcher — fuzzy retrieval, plus the novelty gate that prevents brittle force-fits.

Given a similarity score and whether the matched case is still fresh, decide:

  HIT      confident + fresh match  -> reuse cached reasoning, NO model call
  ESCALATE related but stale/borderline -> re-reason and UPDATE the case
  MISS     novel -> reason fresh and store a NEW case

The whole speed/quality dial lives here. `novelty_gate` is the brittleness guard:
with it ON, borderline matches ESCALATE (safe, costs a call); with it OFF the cache
force-fits borderline matches into a HIT (fast, but wrong on near-misses).
"""

from __future__ import annotations

from enum import Enum


class Decision(Enum):
    HIT = "hit"
    ESCALATE = "escalate"
    MISS = "miss"


class Matcher:
    def __init__(
        self, tau_hit: float = 0.90, tau_miss: float = 0.60, novelty_gate: bool = True
    ):
        self.tau_hit = tau_hit
        self.tau_miss = tau_miss
        self.novelty_gate = novelty_gate

    def decide(self, sim: float, fresh: bool) -> Decision:
        if sim < self.tau_miss:
            return Decision.MISS  # clearly novel -> reason fresh
        # sim in [tau_miss, 1]
        if sim >= self.tau_hit:
            return (
                Decision.HIT if fresh else Decision.ESCALATE
            )  # right case; reuse if fresh, else refresh
        # borderline: tau_miss <= sim < tau_hit  -- the dangerous band
        if self.novelty_gate:
            return Decision.ESCALATE  # guard ON: don't trust it, re-reason
        return (
            Decision.HIT if fresh else Decision.ESCALATE
        )  # guard OFF: force-fit (brittle)
