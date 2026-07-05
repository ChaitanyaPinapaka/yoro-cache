"""Shared micro-benchmark helpers for the examples.

Workload shape (deliberately small so each example runs in minutes on a local
model): a handful of unique tasks asked repeatedly, then one DRIFT event (the
underlying data changes), then the tasks re-asked with the new values. Every
answer is checkable, so the table reports correctness as well as cost — a cache
that is cheap but wrong under drift loses here, by design.
"""

from __future__ import annotations

import time

PROCEDURE = (
    "Apply the standard intake procedure: add 40 for backlog, then triple for the "
    "three shifts, then subtract 27 for inspection, then double for the safety "
    "reserve. Reply with just the number."
)

SITES = ["Athens depot", "Bergen depot", "Cairo depot", "Denver depot", "Osaka depot"]


def gold(v: int) -> str:
    return str(((v + 40) * 3 - 27) * 2)


def task_text(site: str, value: int) -> str:
    return f"The {site} ledger shows {value} units. {PROCEDURE}"


def build_stream(values: dict, repeats: int = 3) -> list:
    """[(site, task_text, gold)] — each unique task asked `repeats` times."""
    stream = []
    for _ in range(repeats):
        for site, v in values.items():
            stream.append((site, task_text(site, v), gold(v)))
    return stream


def base_values(iteration: int, sites=None) -> dict:
    """Distinct per-iteration values so no prompt text repeats across iterations."""
    return {s: 400 + 30 * i + 7 * iteration for i, s in enumerate(sites or SITES)}


def drift(values: dict, bump: int = 50) -> dict:
    return {site: v + bump for site, v in values.items()}


class Meter:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0          # requests that reached the model
        self.out_tokens = 0
        self.wrong = 0
        self.total = 0
        self.t0 = time.time()

    def record(self, answer: str, expected: str, model_called: bool, out_tokens: int = 0):
        self.total += 1
        if model_called:
            self.calls += 1
            self.out_tokens += out_tokens
        if expected not in (answer or ""):
            self.wrong += 1

    def row(self) -> str:
        secs = time.time() - self.t0
        return (f"{self.name:<22} {self.total:>4}  {self.calls:>6}  {self.out_tokens:>8}  "
                f"{self.wrong:>5}  {secs:>7.1f}s")


HEADER = f"{'config':<22} {'reqs':>4}  {'model':>6}  {'out-tok':>8}  {'wrong':>5}  {'time':>8}"


def print_table(meters: list, note: str = ""):
    print("\n" + HEADER)
    print("-" * len(HEADER))
    for m in meters:
        print(m.row())
    if note:
        print("\n" + note)


def aggregate(pairs: list, label_a: str = "no cache", label_b: str = "yoro"):
    """pairs: [(baseline_meter, yoro_meter)] across iterations -> summary lines."""
    print("\n=== aggregate over", len(pairs), "iterations ===")
    print(HEADER)
    print("-" * len(HEADER))
    for name, idx in ((label_a, 0), (label_b, 1)):
        tot = sum(p[idx].total for p in pairs)
        calls = sum(p[idx].calls for p in pairs)
        toks = sum(p[idx].out_tokens for p in pairs)
        wrong = sum(p[idx].wrong for p in pairs)
        print(f"{name:<22} {tot:>4}  {calls:>6}  {toks:>8}  {wrong:>5}   (sum)")
    b_t = sum(p[0].out_tokens for p in pairs)
    y_t = sum(p[1].out_tokens for p in pairs)
    b_c = sum(p[0].calls for p in pairs)
    y_c = sum(p[1].calls for p in pairs)
    per = [100 * (1 - p[1].out_tokens / max(p[0].out_tokens, 1)) for p in pairs]
    print(f"\noutput tokens saved: {100*(1-y_t/max(b_t,1)):.0f}% overall "
          f"(per-iteration: {', '.join(f'{v:.0f}%' for v in per)})")
    print(f"model calls avoided: {b_c - y_c} of {b_c} "
          f"({100*(1-y_c/max(b_c,1)):.0f}%)")
    print(f"wrong answers: {label_a} {sum(p[0].wrong for p in pairs)}, "
          f"{label_b} {sum(p[1].wrong for p in pairs)}")
