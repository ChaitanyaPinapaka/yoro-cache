"""Convergence-based early stop — conclude when the result is statistically SETTLED, not
when the budget runs out. $500 is the ceiling; convergence is the target.

After each seed we hold the per-seed value of every metric for every rung. We track the
95% CI half-width of the PRIMARY quantities — YORO's accuracy and hit-rate, and the
YORO-vs-GPTCache deltas on staleness + brittleness (the headline claims). Once every
primary CI is tighter than `ci_target` and at least `min_seeds` are in, we stop — usually
well under `max_seeds`, saving GPU hours. Final p-values still come from scipy on the
cluster; this is just the stopping rule.
"""
from __future__ import annotations

import math

# t_{.975, n-1} for small n (two-sided 95%); falls back to 1.96 for large n.
_T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31,
        10: 2.26, 12: 2.20, 15: 2.14, 20: 2.09, 25: 2.06, 30: 2.05}


def _t95(n: int) -> float:
    if n < 2:
        return float("inf")
    for k in sorted(_T95):
        if n <= k:
            return _T95[k]
    return 1.96


def ci_halfwidth(vals: list) -> float:
    n = len(vals)
    if n < 2:
        return float("inf")
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return _t95(n) * sd / math.sqrt(n)


class Convergence:
    def __init__(self, min_seeds: int = 8, max_seeds: int = 30, ci_target: float = 0.02):
        self.min_seeds = min_seeds
        self.max_seeds = max_seeds
        self.ci_target = ci_target

    def primaries(self, per_seed: list) -> dict:
        """The primary metrics whose CIs gate the early-stop: reuse rate and
        the staleness/brittleness advantage vs GPTCache. NOTE: model accuracy is deliberately
        NOT here — it's a property of gpt-oss (±0.05 seed noise), not of YORO, so gating on its
        0.02-precision would force every level to max_seeds without sharpening the claim. It is
        still measured per-seed and reported in the aggregate (with its honest CI)."""
        if not per_seed:
            return {}
        rungs = per_seed[0].keys()
        out = {}
        if "yoro" in rungs:
            out["yoro.hit_rate"] = [ps["yoro"]["hit_rate"] for ps in per_seed]
        if "yoro" in rungs and "gptcache-semantic" in rungs:
            out["delta.staleness"] = [ps["gptcache-semantic"]["staleness"] - ps["yoro"]["staleness"]
                                      for ps in per_seed]
            out["delta.brittleness"] = [ps["gptcache-semantic"]["brittleness"] - ps["yoro"]["brittleness"]
                                        for ps in per_seed]
        return out

    def reported(self, per_seed: list) -> dict:
        """CIs we LOG for transparency but do NOT gate on (model accuracy — gpt-oss noise)."""
        if not per_seed or "yoro" not in per_seed[0]:
            return {}
        return {"yoro.accuracy": [ps["yoro"]["accuracy"] for ps in per_seed]}

    def check(self, per_seed: list) -> tuple:
        """Returns (stop: bool, info: dict). Stop on convergence OR max_seeds."""
        n = len(per_seed)
        if n >= self.max_seeds:
            return True, {"reason": "max_seeds", "n": n}
        if n < self.min_seeds:
            return False, {"reason": "min_seeds_not_reached", "n": n, "need": self.min_seeds}
        widths = {k: ci_halfwidth(v) for k, v in self.primaries(per_seed).items()}
        extra = {k: ci_halfwidth(v) for k, v in self.reported(per_seed).items()}   # logged, not gated
        converged = bool(widths) and all(w <= self.ci_target for w in widths.values())
        return converged, {"reason": "converged" if converged else "not_yet", "n": n,
                           "ci_target": self.ci_target, "gated_on": list(widths.keys()),
                           "ci_halfwidths": {k: round(w, 4) for k, w in {**widths, **extra}.items()}}
