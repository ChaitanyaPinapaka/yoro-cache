"""YORO — You Only Reason Once.

Ties the cache, the matcher, and the invalidator together with a *pluggable*
model in one `solve()` loop:

    reason ONCE on a miss  ->  reuse on a hit  ->  re-reason + UPDATE when stale

This is the YOCO ("you only cache once") analogue for cognition: the expensive
deep reasoning of a strong model is computed once and amortized over every later
task that matches — instead of skipped (lossy) or recomputed (wasteful).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .cache import ReasoningCache
from .embeddings import Embedder
from .invalidation import Invalidator
from .keyer import IdentityKeyer, Keyer
from .matcher import Decision, Matcher


@dataclass
class Result:
    outcome: str
    decision: str  # "hit" | "cold" | "update" | "replay"
    reasoned: bool  # did we call the model? (the cost we're trying to avoid)
    case_id: Optional[str]
    sim: float
    version: int
    replayed: bool = False  # re-derived by REPLAYING the cached method (cheap output, not a 0-token hit)


class YORO:
    def __init__(
        self,
        model,
        embedder: Embedder,
        cache: ReasoningCache,
        matcher: Matcher,
        invalidator: Invalidator,
        use_cache: bool = True,
        keyer: Keyer | None = None,
        behaviors=None,
        replay: bool = False,
    ):
        self.model = model
        self.embedder = embedder
        self.cache = cache
        self.matcher = matcher
        self.invalidator = invalidator
        self.use_cache = use_cache  # False = the reason-every-time baseline
        self.keyer = keyer or IdentityKeyer()  # canonicalize before embedding
        self.behaviors = behaviors  # optional BehaviorStore (None = off)
        # Replay: on a stale same-case escalation, re-apply the cached method to the
        # current inputs instead of reasoning from scratch. Deliberately conservative —
        # replay fires ONLY when the case demonstrably belongs (sim >= tau_hit and
        # freshness failed): a confident replayed-but-wrong answer on a genuinely
        # different task is a worse failure than an escalation.
        self.replay = replay

    def solve(
        self,
        task: str,
        current_deps: Optional[dict] = None,
        verify: Optional[Callable[[str, str], bool]] = None,
    ) -> Result:
        # Baseline: no cache, always reason. The reference for "calls saved".
        if not self.use_cache:
            reasoning, outcome = self.model.reason(task)
            return Result(outcome, "cold", True, None, -1.0, 0)

        emb = self.embedder.embed(
            self.keyer.key(task)
        )  # key on the canonical form, store the original task
        case, sim = self.cache.nearest(emb)
        fresh = (
            self.invalidator.is_fresh(case, current_deps) if case is not None else True
        )
        decision = Decision.MISS if case is None else self.matcher.decide(sim, fresh)

        # --- hot path: reuse, no model call ---
        if decision == Decision.HIT:
            outcome = case.outcome
            ok = verify(task, outcome) if verify else True
            self.cache.record_use(case, ok)
            if ok:
                return Result(outcome, "hit", False, case.id, sim, case.version)
            decision = Decision.ESCALATE  # failed reuse -> fall through and re-reason

        # SAME-CASE escalation: sim >= tau_hit (the case demonstrably belongs). Drives in-place UPDATE.
        same_case = (
            decision == Decision.ESCALATE
            and case is not None
            and sim >= self.matcher.tau_hit
        )
        # REPLAY is only safe when the escalation is because the DEPS actually changed (not fresh) —
        # a failed-VERIFY escalation on a FRESH case (fresh==True) means the method just produced a
        # wrong answer, so re-running it would just repeat the failure.
        stale_same_case = same_case and not fresh

        # --- replay path: apply the cached method to current inputs (short, no fresh exploration) ---
        if self.replay and stale_same_case and (case.reasoning or case.steps):
            # inject the RAW trace rather than the extracted steps — step extraction is
            # lossy, and the raw trace replays at least as accurately in our measurements.
            _, outcome = self.model.replay(task, case.reasoning or case.steps)
            # preserve the ORIGINAL trace/steps — the terse replay output would erode the
            # very method injected on the next change. Only outcome + deps/version move.
            c = self.cache.update(
                case.id, task, emb, case.reasoning, outcome, current_deps
            )
            c.steps = case.steps
            return Result(outcome, "replay", True, c.id, sim, c.version, replayed=True)

        # --- cold path: reason ONCE (miss / escalate / failed-hit) ---
        prompt = task
        if self.behaviors is not None:  # retrieve + inject behaviors
            from .behaviors import format_behaviors

            rel = self.behaviors.retrieve(emb, k=3)
            if rel:
                prompt = format_behaviors(rel) + task
        reasoning, outcome = self.model.reason(prompt)
        # Only UPDATE in place when this is clearly the *same* case (high similarity). A merely-similar
        # task (borderline ESCALATE) gets its own new case, so near-misses can't overwrite a good case.
        if same_case:
            c = self.cache.update(case.id, task, emb, reasoning, outcome, current_deps)
            tag = "update"
        else:
            c = self.cache.add(task, emb, reasoning, outcome, current_deps)
            tag = "cold"
        from .structured import to_steps  # store structured steps

        c.steps = to_steps(reasoning)
        if self.behaviors is not None:  # mine new behaviors
            from .behaviors import extract_behaviors

            try:
                extract_behaviors(
                    reasoning,
                    self.model.complete,
                    self.embedder,
                    self.behaviors,
                    from_case=c.id,
                )
            except Exception:
                pass
        return Result(outcome, tag, True, c.id, sim, c.version)
