"""Benchmark metrics + statistical aggregation.

Per prompt we record an Outcome; per run we summarize hit-rate / accuracy / staleness /
brittleness / latency / tokens; across >=30 seeds we aggregate to mean +/- std and run a
paired test between rungs (seed alone swings small-benchmark Pass@1 by 5-15pp, so a gain
must clear that band AND be significant). p-values use scipy on the cluster; here we
report the paired t-statistic + mean diff so the harness has zero heavy deps locally.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Outcome:
    task_key: str            # ground-truth identity (for scoring/staleness), NOT given to the cache
    reused: bool             # cache hit -> no LLM generation
    correct: bool            # answer matched ground truth
    stale: bool              # reused an answer whose ground truth had drifted (wrong reuse)
    forced: bool             # reused on a novel/near-miss prompt (brittle force-fit)
    latency_s: float
    llm_tokens: int          # OUTPUT/generation tokens (the primary cost axis; 0 on a hit)
    in_tokens: int = 0       # INPUT/prompt tokens — inflated by an injected plan on a REPLAY; kept
                             # separate so replay's cheap-OUTPUT claim can't hide an inflated input
    replayed: bool = False   # re-derived by REPLAYING the cached method (short output, NOT a 0-token hit)
    replay_wrong: bool = False   # replayed AND wrong — its OWN column (must not hide inside accuracy,
                                 #   where it would be invisible)
    # Split of a wrong same-entity REUSE (these partition `stale`) — the failure taxonomy: a no-invalidation
    # cache serves a genuinely OUTDATED (once-correct) answer; an invalidating engine that then RE-DERIVES
    # WRONGLY caches never-correct garbage and serves THAT — invalidation WORKED, re-derivation didn't.
    outdated: bool = False       # served answer WAS a valid gold for this entity earlier (true staleness)
    repoisoned: bool = False     # served answer was NEVER correct (re-derived wrong, cached, re-served)


def _pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def summarize(outcomes: list[Outcome], nocache_tokens: int | None = None) -> dict:
    n = len(outcomes)
    if not n:
        return {}
    hits = sum(o.reused for o in outcomes)
    replays = sum(o.replayed for o in outcomes)
    tokens = sum(o.llm_tokens for o in outcomes)             # OUTPUT/generation — primary cost axis
    in_tokens = sum(o.in_tokens for o in outcomes)           # INPUT/prompt (replay inflates this)
    out = {
        "n": n,
        "hit_rate": hits / n,
        "accuracy": sum(o.correct for o in outcomes) / n,
        "staleness": (sum(o.stale for o in outcomes) / hits) if hits else 0.0,   # of hits (= outdated + repoisoned)
        "outdated_rate": (sum(o.outdated for o in outcomes) / hits) if hits else 0.0,     # genuinely-stale share
        "repoisoned_rate": (sum(o.repoisoned for o in outcomes) / hits) if hits else 0.0,  # re-derived-wrong share
        "brittleness": sum(o.forced for o in outcomes) / n,
        "replay_rate": replays / n,                          # share of tasks re-derived by plan-replay
        "replay_wrong": sum(o.replay_wrong for o in outcomes) / n,   # replayed AND wrong (own column)
        "tokens_total": tokens,                              # OUTPUT tokens (kept name for back-compat)
        "input_tokens_total": in_tokens,                     # INPUT tokens — report separately (honest)
        "latency_mean": sum(o.latency_s for o in outcomes) / n,
        "latency_p50": _pct([o.latency_s for o in outcomes], 50),
        "latency_p95": _pct([o.latency_s for o in outcomes], 95),
    }
    if nocache_tokens:                       # OUTPUT-token savings vs the no-cache rung
        out["tokens_saved_frac"] = max(0.0, 1.0 - tokens / nocache_tokens)
    return out


def aggregate_seeds(per_seed: list[dict], key: str) -> dict:
    vals = [d[key] for d in per_seed if key in d]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return {"mean": m, "std": math.sqrt(var), "n": len(vals)}


def paired_t(a: list[float], b: list[float]) -> dict:
    """Paired t-statistic for a-vs-b across matched seeds. (Final p-values via
    scipy.stats.ttest_rel / wilcoxon on the cluster; here we return t + mean diff.)"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return {"t": 0.0, "mean_diff": (d[0] if d else 0.0), "n": n}
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n) if sd else float("inf")
    t = m / se if se not in (0.0, float("inf")) else 0.0
    return {"t": t, "mean_diff": m, "n": n}
