"""YORO — You Only Reason Once.

Ties the cache, the matcher, and the invalidator together with a *pluggable*
model in one `solve()` loop:

    reason ONCE on a miss  ->  reuse on a hit  ->  re-reason + UPDATE when stale

Routing decisions live in `engine.lookup` so the library path and the proxy share
one ladder (HIT / REPLAY / UPDATE / COLD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .cache import ReasoningCache
from .embeddings import Embedder
from .engine import lookup as engine_lookup
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
        replay_verifier: Optional[Callable[[str, str], bool]] = None,
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
        self.replay_verifier = replay_verifier

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
        found = engine_lookup(
            self.cache,
            self.matcher,
            self.invalidator,
            emb,
            current_deps,
            replay=self.replay,
        )
        case, sim, decision, fresh = found.case, found.sim, found.decision, found.fresh

        # --- hot path: reuse, no model call ---
        if decision == Decision.HIT and case is not None:
            outcome = case.outcome
            ok = verify(task, outcome) if verify else True
            self.cache.record_use(case, ok)
            if ok:
                return Result(outcome, "hit", False, case.id, sim, case.version)
            # failed reuse -> fall through and re-reason (do not replay: method was wrong)
            decision = Decision.ESCALATE
            found = type(found)(
                decision=decision,
                case=case,
                sim=sim,
                fresh=fresh,
                same_case=found.same_case,
                should_replay=False,
            )

        # SAME-CASE escalation: sim >= tau_hit (the case demonstrably belongs).
        same_case = (
            decision == Decision.ESCALATE
            and case is not None
            and found.same_case
        )
        stale_same_case = same_case and not fresh

        # --- replay path: apply the cached method to current inputs ---
        if self.replay and stale_same_case and case is not None and (
            case.reasoning or case.steps
        ):
            # inject the RAW trace rather than the extracted steps — step extraction is
            # lossy, and the raw trace replays at least as accurately in our measurements.
            from .replay import procedure_applicable, validate_output

            if not procedure_applicable(case, task):
                found.should_replay = False
            else:
                _, outcome = self.model.replay(task, case.reasoning or case.steps)
                validator = self.replay_verifier or verify
                valid = validate_output(
                    outcome, verifier=(lambda x: validator(task, x)) if validator else None
                )
                if valid:
                    c = self.cache.update(
                        case.id, task, emb, case.reasoning, outcome, current_deps
                    )
                    c.steps = case.steps
                    c.procedure = dict(case.procedure or {})
                    return Result(outcome, "replay", True, c.id, sim, c.version, replayed=True)

        # --- cold path: reason ONCE (miss / escalate / failed-hit) ---
        prompt = task
        if self.behaviors is not None:  # retrieve + inject behaviors
            from .behaviors import format_behaviors

            rel = self.behaviors.retrieve(emb, k=3)
            if rel:
                prompt = format_behaviors(rel) + task
        reasoning, outcome = self.model.reason(prompt)
        # Only UPDATE in place when this is clearly the *same* case (high similarity).
        if same_case and case is not None:
            c = self.cache.update(case.id, task, emb, reasoning, outcome, current_deps)
            tag = "update"
        else:
            c = self.cache.add(task, emb, reasoning, outcome, current_deps)
            tag = "cold"
        from .structured import to_steps  # store structured steps

        from .structured import ProcedureArtifact
        self.cache.set_artifact(
            c, steps=to_steps(reasoning),
            procedure=ProcedureArtifact.from_reasoning(reasoning, current_deps).to_dict(),
        )
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
            except Exception as e:
                # mining is best-effort; surface once so silent failures are visible
                import warnings

                warnings.warn(f"YORO: behavior extraction failed: {e}", stacklevel=2)
        return Result(outcome, tag, True, c.id, sim, c.version)
