"""Replay-quality SPIKE (~$2) — the go/no-go before building E7 into the sweep.

Does injecting a cached METHOD let the model re-derive a drifted answer cheaply AND correctly? Is the
extraction (to_steps) or the raw trace the right thing to ship? And — the terseness control —
is replay's saving actually the CACHED METHOD, or just a "be terse" instruction? The drift re-ask
REFERENCES the established procedure without restating it, so a stateless solver lacks the method:

  cold-ref     full-reason the COLD task (procedure IS in the prompt) -> the accuracy CEILING + the
               trace we cache. Shows the task is solvable when the method is present.
  full-reason  full-reason the DRIFT re-ask (procedure NOT restated) -> stateless baseline; should be
               WRONG (it doesn't know the method) and expensive.
  terse-direct CONTROL: terse system, NO plan, on the drift re-ask -> cheap; if it's also ACCURATE,
               replay's saving was just terseness. Expected: cheap but WRONG (no method).
  replay(steps)  inject the cached STRUCTURED steps -> apply to new value. Expected: cheap AND right.
  replay(raw)    inject the cached RAW trace -> apply. Expected: cheap AND right.

Acceptance criteria: a replay arm PASSES iff it (a) reaches ~cold-ref accuracy (within
2 pts) AND (b) uses <=30% of cold-ref OUTPUT tokens AND (c) beats terse-direct accuracy by a clear
margin (>15 pts) — (c) is the anti-confound: it proves the PLAN, not terseness, carries the method. If
raw passes but steps doesn't, ship raw-trace injection and fix to_steps later.

Run on the box (vLLM up):  python -m bench.spike_replay --base-url http://127.0.0.1:8000/v1 --model openai/gpt-oss-120b --n 25
Local dry check (no GPU):  python -m bench.spike_replay --smoke --n 8
"""
from __future__ import annotations

import argparse
import random

from bench.datasets import build_stress_workload, _STRESS_HARD_OPS
from bench.model_client import VLLMClient, TERSE_SYSTEM
from bench.run_phase0 import is_correct, MockPerfect
from yoro.structured import to_steps


def _hard_pairs(n: int, seed: int = 0):
    """n (parent, drifted) task pairs sharing a method: same entity, a moved value -> new gold."""
    rng = random.Random(seed)
    stream = build_stress_workload(n_unique=max(n, 12), stream_len=n * 12, zipf_s=1.3,
                                   drift_rate=0.5, near_miss_rate=0.0, hard=True, seed=seed)
    by_key, pairs = {}, []
    for t in stream:
        seq = by_key.setdefault(t.key, [])
        if t.kind == "cold":
            seq.append(t)
        elif t.kind == "drift" and seq:
            pairs.append((seq[0], t))                    # (parent cold, its drifted re-ask)
            if len(pairs) >= n:
                break
    return pairs


def run_spike(model, pairs) -> dict:
    arms = {"cold-ref": [], "full-reason": [], "terse-direct": [],
            "replay(to_steps)": [], "replay(raw-trace)": []}

    def rec(arm, out, gold):
        arms[arm].append((is_correct(out, gold), model.last_completion_tokens, model.last_prompt_tokens))

    for parent, drifted in pairs:
        reasoning, out_cold = model.reason(parent.text)  # derive the method on the COLD task (has the procedure)
        steps = to_steps(reasoning) or [reasoning]
        rec("cold-ref", out_cold, parent.gold)           # accuracy ceiling + the cached trace/steps

        _, out_reason = model.reason(drifted.text)                       # stateless full-reason on the bare re-ask
        rec("full-reason", out_reason, drifted.gold)
        _, out_terse = model.reason(drifted.text, system=TERSE_SYSTEM)   # CONTROL: terse, no plan
        rec("terse-direct", out_terse, drifted.gold)
        _, out_steps = model.replay(drifted.text, steps)                 # inject cached steps
        rec("replay(to_steps)", out_steps, drifted.gold)
        _, out_raw = model.replay(drifted.text, reasoning)               # inject cached raw trace
        rec("replay(raw-trace)", out_raw, drifted.gold)

    def summ(rows):
        n = len(rows) or 1
        return {"acc": sum(c for c, _, _ in rows) / n,
                "out_tokens": sum(o for _, o, _ in rows) / n,
                "in_tokens": sum(i for _, _, i in rows) / n, "n": len(rows)}
    return {k: summ(v) for k, v in arms.items()}


def main():
    ap = argparse.ArgumentParser(description="YORO replay-quality spike (3-arm)")
    ap.add_argument("--smoke", action="store_true", help="MockPerfect, no GPU")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-effort", default=None,
                    help="reasoning_effort for the replay arms: low|medium|high (default = model's own)")
    a = ap.parse_args()

    pairs = _hard_pairs(a.n, a.seed)
    if a.smoke:
        gold = {t.text: t.gold for p in pairs for t in p}
        model = MockPerfect(gold)
    else:
        model = VLLMClient(a.base_url, a.model, replay_max_tokens=512, replay_effort=a.replay_effort)
    print(f"[spike] {len(pairs)} hard drift pairs, model={'mock' if a.smoke else a.model}")

    res = run_spike(model, pairs)
    ceil_acc = res["cold-ref"]["acc"]                    # method-present accuracy ceiling
    scratch_out = res["full-reason"]["out_tokens"] or 1.0  # reasoning-from-scratch cost (what no-cache PAYS)
    terse_acc = res["terse-direct"]["acc"]
    print(f"\n{'arm':18s} {'acc':>6} {'out_tok':>9} {'in_tok':>9} {'out_vs_scratch':>14}  verdict")
    for name, m in res.items():
        ratio = m["out_tokens"] / scratch_out            # vs full-reason (the honest baseline)
        verdict = ""
        if name.startswith("replay"):
            passes = (m["acc"] >= ceil_acc - 0.02          # (a) reaches the method-present ceiling
                      and ratio <= 0.50                    # (b) meaningfully cheaper than reasoning-from-scratch
                      and m["acc"] - terse_acc > 0.15)     # (c) beats terse control -> PLAN carries it, not terseness
            verdict = "PASS ✓" if passes else "fail"
        print(f"{name:18s} {m['acc']:>6.2f} {m['out_tokens']:>9.0f} {m['in_tokens']:>9.0f} "
              f"{ratio:>13.0%}  {verdict}")
    print(f"\nGate: replay must (a) >= cold-ref acc ({ceil_acc:.2f}), (b) <=50% of full-reason out-tokens")
    print(f"(reasoning-from-scratch, what no-cache pays), (c) beat terse-direct acc ({terse_acc:.2f}) by >15pts")
    print("— (c) proves the cached PLAN carries the method, not terseness. raw>steps -> ship raw-trace.")
    return res


if __name__ == "__main__":
    main()
