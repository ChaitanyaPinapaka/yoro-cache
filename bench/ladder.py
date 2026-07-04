"""The baseline ladder — the comparison rungs, behind one Strategy interface
so the harness drives them identically over the same prompt stream:

  (a) no-cache            - reason every time (the accuracy + token reference)
  (b) exact-match         - string-keyed cache (the trivial baseline)
  (c) gptcache-semantic   - embed + nearest-neighbour + threshold; NO invalidation, NO
                            novelty gate (the canonical GPTCache-style semantic cache)
  (d) yoro+behaviors      - YORO with mined behaviors injected (the Metacognitive-Reuse
                            comparable)
  (e) yoro                - full YORO: gate + dependency-invalidation + canonical keying


Each rung's solve() returns (outcome, reused) plus the LLM token count, so metrics can
score accuracy/staleness/brittleness against the world's ground truth uniformly.
"""
from __future__ import annotations

from typing import NamedTuple

from yoro import (YORO, ReasoningCache, Matcher, Invalidator, BehaviorStore,
                  format_behaviors, extract_behaviors)


class SolveResult(NamedTuple):
    outcome: str
    reused: bool             # a 0-token serve-hit
    out_tokens: int          # generation tokens (0 on a serve-hit)
    in_tokens: int           # prompt tokens (inflated by an injected plan on a replay)
    replayed: bool = False   # re-derived by replaying the cached method


class Strategy:
    name = "base"

    def solve(self, task, current_deps=None, verify=None) -> "SolveResult":
        raise NotImplementedError


class NoCache(Strategy):
    name = "no-cache"

    def __init__(self, model):
        self.model = model

    def solve(self, task, current_deps=None, verify=None):
        reasoning, outcome = self.model.reason(task)
        o, i = toks(self.model, reasoning, outcome)
        return SolveResult(outcome, False, o, i, False)


class ExactCache(Strategy):
    name = "exact-match"

    def __init__(self, model):
        self.model = model
        self.store: dict[str, str] = {}

    def solve(self, task, current_deps=None, verify=None):
        if task in self.store:
            return SolveResult(self.store[task], True, 0, 0, False)
        reasoning, outcome = self.model.reason(task)
        self.store[task] = outcome
        o, i = toks(self.model, reasoning, outcome)
        return SolveResult(outcome, False, o, i, False)


class SemanticCache(Strategy):
    """GPTCache-style: embed -> nearest -> threshold. Deliberately NO invalidation and NO
    novelty gate, so it shows the failure modes (staleness, force-fit) YORO is built to fix."""
    name = "gptcache-semantic"

    def __init__(self, model, embedder, tau: float = 0.9):
        self.model = model
        self.emb = embedder
        self.cache = ReasoningCache()
        self.tau = tau

    def solve(self, task, current_deps=None, verify=None):
        e = self.emb.embed(task)
        case, sim = self.cache.nearest(e)
        if case is not None and sim >= self.tau:
            return SolveResult(case.outcome, True, 0, 0, False)
        reasoning, outcome = self.model.reason(task)
        self.cache.add(task, e, reasoning, outcome, {})
        o, i = toks(self.model, reasoning, outcome)
        return SolveResult(outcome, False, o, i, False)


class BehaviorsOnly(Strategy):
    """Metacognitive-Reuse comparable, WITHOUT YORO: never reuse an answer (always reason),
    just retrieve + inject mined behaviors to make reasoning cheaper, and mine new ones from
    each trace. Isolates the token savings from behaviors ALONE (no case-cache, no gate)."""
    name = "behaviors-only"

    def __init__(self, model, embedder):
        self.model = model
        self.emb = embedder
        self.behaviors = BehaviorStore()

    def solve(self, task, current_deps=None, verify=None):
        e = self.emb.embed(task)
        rel = self.behaviors.retrieve(e, k=3)
        prompt = (format_behaviors(rel) + task) if rel else task
        reasoning, outcome = self.model.reason(prompt)          # ALWAYS reasons -> reused=False
        try:
            extract_behaviors(reasoning, self.model.complete, self.emb, self.behaviors)
        except Exception:
            pass
        o, i = toks(self.model, reasoning, outcome)
        return SolveResult(outcome, False, o, i, False)


class YOROStrategy(Strategy):
    """Full YORO, or the behaviors-only comparable when behaviors=True + gate/deps relaxed, or the
    graduated serve/replay/reason engine when replay=True."""

    def __init__(self, model, embedder, tau_hit=0.9, tau_miss=0.6, gate=True,
                 use_deps=True, behaviors=False, keyer=None, replay=False, replay_effort=None, name="yoro"):
        self.name = name
        if replay and replay_effort is not None and hasattr(model, "replay_effort"):
            model.replay_effort = replay_effort             # the effort DIAL for this rung (None=default, "low"=cheap)
        self.eng = YORO(
            model, embedder, ReasoningCache(),
            Matcher(tau_hit, tau_miss, gate),
            Invalidator(use_deps=use_deps, use_ttl=False, use_reliability=False),
            keyer=keyer, behaviors=BehaviorStore() if behaviors else None, replay=replay,
        )
        self._model = model

    def solve(self, task, current_deps=None, verify=None):
        r = self.eng.solve(task, current_deps=current_deps, verify=verify)
        reused = not r.reasoned                              # a serve-hit: no model call
        if reused:                                           # 0-token serve
            return SolveResult(r.outcome, True, 0, 0, False)
        out_tok, in_tok = toks(self._model)                  # replay OR full reason: real usage
        return SolveResult(r.outcome, False, out_tok, in_tok, r.replayed)


def _tok(reasoning: str, outcome: str) -> int:
    """Cheap token proxy (~4 chars/token) when the model didn't report usage."""
    return max(1, (len(reasoning) + len(outcome)) // 4)


def toks(model, reasoning: str = "", outcome: str = "") -> tuple[int, int]:
    """(output_tokens, input_tokens) of the model's LAST call — prefer the server's REAL usage
    (last_completion_tokens / last_prompt_tokens, set by VLLMClient & MockPerfect); fall back to the
    char proxy for output when a mock doesn't report it. Input proxy is 0 (unknown) in that case."""
    out = getattr(model, "last_completion_tokens", None)
    inp = getattr(model, "last_prompt_tokens", 0) or 0
    return (out if out else _tok(reasoning, outcome)), inp


def build_ladder(make_model, embedder, tau_hit=0.9, tau_miss=0.6, gptcache_tau=0.9,
                 rungs=()) -> list[Strategy]:
    """The rungs over the same stream. `make_model` is a NO-ARG factory: each rung gets its OWN
    model client, so the rungs can run concurrently (own token-state) and vLLM batches their
    requests. Pass `lambda: shared` to keep one client for a sequential run. `rungs` (a name
    subset, e.g. ("no-cache","gptcache-semantic","yoro")) selects which to build — empty = all 6;
    only the selected rungs' model clients are constructed (cheaper safety sweeps)."""
    builders = {
        "no-cache":          lambda: NoCache(make_model()),
        "exact-match":       lambda: ExactCache(make_model()),
        "gptcache-semantic": lambda: SemanticCache(make_model(), embedder, tau=gptcache_tau),
        "behaviors-only":    lambda: BehaviorsOnly(make_model(), embedder),   # Metacognitive-Reuse (no YORO)
        "yoro+behaviors":    lambda: YOROStrategy(make_model(), embedder, tau_hit, tau_miss,
                                                  gate=False, use_deps=False, behaviors=True,
                                                  name="yoro+behaviors"),
        "yoro":              lambda: YOROStrategy(make_model(), embedder, tau_hit, tau_miss,
                                                  gate=True, use_deps=True, behaviors=False, name="yoro"),
        "yoro-replay":       lambda: YOROStrategy(make_model(), embedder, tau_hit, tau_miss,   # replay:
                                                  gate=True, use_deps=True, behaviors=False,   # default effort
                                                  replay=True, name="yoro-replay"),            # 100% acc, ~35% out-saving
        "yoro-replay-low":   lambda: YOROStrategy(make_model(), embedder, tau_hit, tau_miss,   # the effort DIAL down:
                                                  gate=True, use_deps=True, behaviors=False,   # ~80% acc, ~84% out-saving
                                                  replay=True, replay_effort="low", name="yoro-replay-low"),
    }
    order = ["no-cache", "exact-match", "gptcache-semantic", "behaviors-only",
             "yoro+behaviors", "yoro", "yoro-replay", "yoro-replay-low"]
    unknown = [r for r in rungs if r not in builders]
    if unknown:
        raise ValueError(f"unknown rung(s) {unknown}; valid: {order}")
    selected = [r for r in order if r in rungs] if rungs else order
    return [builders[r]() for r in selected]
