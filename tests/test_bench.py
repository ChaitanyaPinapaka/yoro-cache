"""Phase-0 harness foundation tests — budget guard, metrics, ladder. No GPU, no model,
no new deps (mock model + HashEmbedder)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile

from yoro import HashEmbedder
from bench import (BudgetGuard, Outcome, summarize, aggregate_seeds, paired_t,
                   build_ladder, NoCache, ExactCache, SemanticCache, YOROStrategy,
                   EventLog, Checkpoint)


class MockModel:
    name = "mock"

    def __init__(self):
        self.calls = 0

    def reason(self, task):
        self.calls += 1
        return (f"reasoning for {task}", f"ans::{task}")

    def complete(self, prompt, max_tokens=None):
        return ""                                   # behaviors no-op


def test_budget_guard():
    t = [1000.0]
    fired = []
    g = BudgetGuard(ceiling_usd=500, hourly_usd=100, shutdown_frac=0.9,
                    clock=lambda: t[0], on_shutdown=lambda: fired.append(1))
    assert g.spent() == 0.0
    t[0] = 1000.0 + 3600 * 4                         # 4h -> $400 (< $450 soft cap)
    assert abs(g.spent() - 400) < 1e-6 and not g.check() and not fired
    t[0] = 1000.0 + 3600 * 4.6                       # 4.6h -> $460 (>= soft cap)
    assert g.check() and fired == [1]
    assert g.check() and fired == [1]                # idempotent: fires exactly once
    assert g.status()["stopped"] is True
    print("ok budget_guard")


def test_metrics():
    outs = [
        Outcome("k1", reused=False, correct=True,  stale=False, forced=False, latency_s=1.0,  llm_tokens=100),
        Outcome("k1", reused=True,  correct=True,  stale=False, forced=False, latency_s=0.01, llm_tokens=0),
        Outcome("k2", reused=True,  correct=False, stale=True,  forced=False, latency_s=0.01, llm_tokens=0),
    ]
    s = summarize(outs, nocache_tokens=300)
    assert s["n"] == 3
    assert abs(s["hit_rate"] - 2 / 3) < 1e-9
    assert abs(s["accuracy"] - 2 / 3) < 1e-9
    assert abs(s["staleness"] - 0.5) < 1e-9          # 1 of 2 hits was stale
    assert s["tokens_total"] == 100
    assert abs(s["tokens_saved_frac"] - (1 - 100 / 300)) < 1e-9
    agg = aggregate_seeds([{"hit_rate": 0.6}, {"hit_rate": 0.8}], "hit_rate")
    assert abs(agg["mean"] - 0.7) < 1e-9 and agg["n"] == 2
    pt = paired_t([0.9, 0.8, 0.85], [0.6, 0.55, 0.65])
    assert pt["mean_diff"] > 0 and pt["n"] == 3 and pt["t"] > 0
    # taxonomy split: staleness partitions into outdated (served once-correct) + re-poisoned (never-correct)
    tax = [
        Outcome("k", True, False, stale=True, forced=False, latency_s=0, llm_tokens=0, outdated=True),
        Outcome("k", True, False, stale=True, forced=False, latency_s=0, llm_tokens=0, repoisoned=True),
        Outcome("k", True, True,  stale=False, forced=False, latency_s=0, llm_tokens=0),
    ]
    st = summarize(tax)
    assert abs(st["staleness"] - 2/3) < 1e-9                          # 2 of 3 hits stale
    assert abs(st["outdated_rate"] - 1/3) < 1e-9 and abs(st["repoisoned_rate"] - 1/3) < 1e-9  # split 50/50
    assert abs((st["outdated_rate"] + st["repoisoned_rate"]) - st["staleness"]) < 1e-9        # they partition staleness
    # the CLASSIFIER itself (correctness-lineage), pinning the definition the paper uses:
    from bench.run_phase0 import classify_stale
    assert classify_stale("5400", ["5400", "5760"]) == (True, False)   # served a PAST gold -> outdated (was true once)
    assert classify_stale("9999", ["5400", "5760"]) == (False, True)   # served a NEVER-correct answer -> re-poisoned
    assert classify_stale("5400", []) == (False, True)                 # no prior correct gold -> re-poisoned
    print("ok metrics")


def test_ladder_reuse():
    emb = HashEmbedder()
    stream = ["what is 2+2", "what is 2+2", "capital of france", "what is 2+2"]
    cases = [
        (lambda m: NoCache(m), 0),
        (lambda m: ExactCache(m), 2),                                  # 2nd + 4th reuse
        (lambda m: SemanticCache(m, emb, tau=0.95), 2),
        (lambda m: YOROStrategy(m, emb, 0.95, 0.6, gate=True, use_deps=True, name="yoro"), 2),
    ]
    for factory, expect in cases:
        m = MockModel()
        strat = factory(m)
        reused = 0
        for task in stream:
            r = strat.solve(task)
            reused += int(r.reused)
            if r.reused:
                assert r.out_tokens == 0 and r.in_tokens == 0          # a serve-hit spends no tokens
        assert reused == expect, (strat.name, reused, expect)
    assert len(build_ladder(lambda: MockModel(), emb)) == 8            # 8 rungs (incl yoro-replay + -low)
    print("ok ladder_reuse")


def test_replay_tier():
    """Replay: on a STALE SAME-CASE (drift), yoro-replay REPLAYS the cached method (short output)
    instead of full re-reasoning; plain yoro re-reasons. Near-miss must NOT replay in v1."""
    from bench.ladder import YOROStrategy
    emb = HashEmbedder()

    class RModel:                                                       # records reason vs replay + tokens
        def __init__(self):
            self.reasoned = self.replayed = 0
            self.last_completion_tokens = self.last_prompt_tokens = 0
        def reason(self, task):
            self.reasoned += 1
            self.last_completion_tokens, self.last_prompt_tokens = 100, 30   # full CoT: big output
            return ("step a\nstep b\nstep c", f"ans::{task[:8]}")
        def replay(self, task, plan):
            self.replayed += 1
            self.last_completion_tokens, self.last_prompt_tokens = 12, 90     # short output, plan-inflated input
            return ("applied", f"ans::{task[:8]}")
        def complete(self, p, max_tokens=None): return ""

    # cold populate then a DRIFT of the same entity (same text/key, dep version bumped)
    m = RModel()
    strat = YOROStrategy(m, emb, tau_hit=0.9, tau_miss=0.6, gate=True, use_deps=True,
                         replay=True, name="yoro-replay")
    strat.solve("compute load for reactor R7", current_deps={"R7": 1})      # cold -> reason
    assert m.reasoned == 1 and m.replayed == 0
    r = strat.solve("compute load for reactor R7", current_deps={"R7": 2})  # drift -> REPLAY
    assert r.replayed is True and m.replayed == 1                           # replayed, not re-reasoned
    assert r.out_tokens < 100 and r.in_tokens >= 30                         # short output, plan-inflated input

    # plain yoro (replay off) re-reasons the same drift instead of replaying
    m2 = RModel()
    plain = YOROStrategy(m2, emb, tau_hit=0.9, tau_miss=0.6, gate=True, use_deps=True, name="yoro")
    plain.solve("compute load for reactor R7", current_deps={"R7": 1})
    r2 = plain.solve("compute load for reactor R7", current_deps={"R7": 2})
    assert r2.replayed is False and m2.replayed == 0 and m2.reasoned == 2
    print("ok replay_tier")


class _CaptureSink:
    def __init__(self): self.events = []
    def write(self, ev, line): self.events.append((ev, line))
    def flush(self): pass
    def close(self): pass


def test_eventlog_jsonl():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "events.jsonl")
    cap = _CaptureSink()
    log = EventLog(p, run_id="run-1", sinks=[cap], clock=lambda: 123.0)
    log.progress(domain="math", done=10, total=100)
    log.result(task_key="k1", rung="yoro", reused=True, correct=True)
    log.error("boom", where="rung:yoro")
    log.close()
    lines = [json.loads(x) for x in open(p)]
    assert len(lines) == 3
    assert [e["kind"] for e in lines] == ["progress", "result", "error"]
    assert lines[0]["run"] == "run-1" and lines[0]["seq"] == 0 and lines[0]["t"] == 123.0
    assert lines[2]["msg"] == "boom"
    assert len(cap.events) == 3                          # sink received every event
    print("ok eventlog_jsonl")


def test_checkpoint_resume():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "ckpt.json")
    ck = Checkpoint(p)                                    # no S3 -> local only
    assert ck.load() is None                             # nothing yet
    ck.save({"cursor": 42, "outcomes": {"yoro": [1, 0, 1]}})
    assert os.path.exists(p)
    state = Checkpoint(p).load()                          # fresh handle (simulates resume)
    assert state["cursor"] == 42 and state["outcomes"]["yoro"] == [1, 0, 1]
    ck.save({"cursor": 99})                               # overwrite atomically
    assert Checkpoint(p).load()["cursor"] == 99
    print("ok checkpoint_resume")


def test_convergence():
    from bench import Convergence

    def seed(acc_y, stale_g, stale_y, brit_g, brit_y, hit_y=0.5):
        return {"yoro": {"accuracy": acc_y, "hit_rate": hit_y, "staleness": stale_y, "brittleness": brit_y},
                "gptcache-semantic": {"accuracy": 0.9, "hit_rate": 0.5, "staleness": stale_g, "brittleness": brit_g}}

    c = Convergence(min_seeds=3, max_seeds=10, ci_target=0.02)
    s, i = c.check([seed(1, 0.1, 0, 0.1, 0)])
    assert not s and i["reason"] == "min_seeds_not_reached"                 # below min
    s, i = c.check([seed(1.0, 0.1, 0.0, 0.1, 0.0) for _ in range(3)])
    assert s and i["reason"] == "converged"                                # zero-variance -> tight CI
    noisy = [seed(a, 0.1, sv, 0.1, 0.0) for a, sv in
             [(0.6, 0.0), (1.0, 0.3), (0.7, 0.0), (1.0, 0.2), (0.6, 0.1)]]
    s, i = c.check(noisy)
    assert not s and i["reason"] == "not_yet"                              # noisy -> keep going
    s, i = c.check([seed(1, 0.1, 0, 0.1, 0)] * 10)
    assert s and i["reason"] == "max_seeds"                                # ceiling
    print("ok convergence")


def test_workload_recurrence():
    from collections import Counter
    from bench.datasets import build_workload
    pool = [(f"task number {i}?", f"ans{i}", f"k{i}", "math") for i in range(10)]
    stream = build_workload(pool, stream_len=200, zipf_s=1.3, n_paraphrases=3, paraphraser=None, seed=1)
    assert len(stream) == 200
    keys = [t.key for t in stream]
    assert len(set(keys)) <= 10                                     # recurrence: reuse across requests
    seen = set()
    for t in stream:                                               # first sighting cold, rest repeat
        assert t.kind == ("cold" if t.key not in seen else "repeat")
        seen.add(t.key)
    c = Counter(keys)
    assert max(c.values()) >= 3 * min(c.values()) or len(c) < 10    # Zipf head concentration
    golds = {t.key: t.gold for t in stream}
    assert all(golds[f"k{i}"] == f"ans{i}" for i in range(10) if f"k{i}" in golds)   # gold preserved
    print("ok workload_recurrence")


def test_load_hf_construction():
    import sys
    import types

    class FakeDS:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    def fake_load(repo, config=None, split=None):
        if repo == "openai/gsm8k":
            return FakeDS([{"question": f"g{i}?", "answer": f"work #### {i}"} for i in range(8)])
        if repo == "HuggingFaceH4/MATH-500":
            return FakeDS([{"problem": f"m{i}", "answer": str(i)} for i in range(8)])
        if repo == "cais/mmlu":
            return FakeDS([{"question": f"u{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4} for i in range(8)])
        if repo == "lukaemon/bbh":
            return FakeDS([{"input": f"b{i}", "target": f"t{i}"} for i in range(8)])
        if repo == "nyu-mll/glue":
            return FakeDS([{"question1": f"x{i}?", "question2": f"y{i}?", "label": i % 2} for i in range(6)])
        raise ValueError(repo)

    fake = types.ModuleType("datasets")
    fake.load_dataset = fake_load
    sys.modules["datasets"] = fake
    try:
        from bench.datasets import load_hf
        tasks = load_hf(["math", "qa", "reasoning"], n_unique=9, stream_len=120, zipf_s=1.1,
                        n_paraphrases=2, pairs=("qqp",), n_pairs=4, drift_probes=True, seed=0)
        kinds = {}
        for t in tasks:
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
        assert kinds.get("cold", 0) >= 1 and kinds.get("repeat", 0) >= 1     # recurrence present
        assert kinds.get("populate", 0) == 4                                 # qqp q1 seeds
        dec = [t for t in tasks if t.expect_reuse is not None]
        assert len(dec) == 4 and all(t.gold == "" for t in dec)              # decision-mode pairs
        assert any(t.kind == "drift" for t in tasks)                         # controlled probe overlay
    finally:
        del sys.modules["datasets"]
    print("ok load_hf_construction")


def test_resume_per_seed():
    """Simulate a preemption between seeds: craft a checkpoint with seed 0 DONE (sentinel
    tokens=99999 on every rung) + seed_idx=1, then confirm a 2-seed run RESUMES at seed 1 —
    it must reuse seed 0 from the checkpoint, not recompute it (which would be ~440)."""
    from bench.run_phase0 import Config, run
    RUNGS = ["no-cache", "exact-match", "gptcache-semantic", "behaviors-only", "yoro+behaviors", "yoro"]
    summ = {"n": 21, "hit_rate": 0.5, "accuracy": 1.0, "staleness": 0.0, "brittleness": 0.0,
            "tokens_total": 99999}
    out = tempfile.mkdtemp()
    ck = {"seed_idx": 1, "per_seed": [{r: dict(summ) for r in RUNGS}]}     # seed 0 already done
    with open(os.path.join(out, "ckpt.json"), "w") as f:
        json.dump(ck, f)
    rep = run(Config(smoke=True, seeds=2, out=out, run_id="rt"))
    assert rep["seeds"] == 2                                              # resumed seed 0 + computed seed 1
    # seed 0's sentinel 99999 must be in the no-cache mean (else it was recomputed -> mean ~440)
    assert rep["rungs"]["no-cache"]["tokens_total"]["mean"] > 1000, rep["rungs"]["no-cache"]
    finalck = json.load(open(os.path.join(out, "ckpt.json")))
    assert finalck["seed_idx"] == 2 and "seed_stream" not in finalck
    print("ok resume_per_seed")


def test_vast_credit():
    from bench import VastCredit
    v = VastCredit(api_key="x", min_usd=2.0)
    v._fetch_credit = lambda: 10.0
    assert v.remaining() == 10.0 and v.low() is False                 # plenty
    v._fetch_credit = lambda: 1.5
    assert v.low() is True                                            # below buffer -> stop
    def boom(): raise RuntimeError("api down")
    v._fetch_credit = boom
    assert v.remaining() == 1.5 and v.low() is True                   # API error -> keep last value, fail safe
    nokey = VastCredit(api_key=None, min_usd=2.0)
    assert nokey.remaining() is None and nokey.low() is False         # no key -> disabled, fall back to BudgetGuard
    print("ok vast_credit")


def test_budget_external_stop():
    from bench import BudgetGuard
    g = BudgetGuard(ceiling_usd=500, hourly_usd=0, clock=lambda: 0.0)
    assert not g.stopped
    g.stop()                                                          # external signal (low Vast credit)
    assert g.stopped and g.check()                                   # stays stopped
    print("ok budget_external_stop")


def test_stress_workload():
    from collections import Counter
    from bench.datasets import build_stress_workload
    ts = build_stress_workload(n_unique=30, stream_len=500, zipf_s=1.1,
                               drift_rate=0.3, near_miss_rate=0.2, seed=3)
    assert len(ts) == 500
    kinds = Counter(t.kind for t in ts)
    rec = sum(kinds[k] for k in ("repeat", "drift", "near_miss"))     # recurrences (not cold)
    assert 0.20 <= kinds["drift"] / rec <= 0.40                       # ~drift_rate
    assert 0.10 <= kinds["near_miss"] / rec <= 0.30                   # ~near_miss_rate
    nm = [t for t in ts if t.kind == "near_miss"]
    assert all("|nm" in t.key for t in nm)                           # near-miss = sibling of a parent key
    assert len(set(t.key for t in nm)) == len(nm)                    # each near-miss is a DISTINCT entity (unique key)
    byk = {}
    for t in ts:
        byk.setdefault(t.key, []).append(t)
    drifted = [v for v in byk.values() if any(t.kind == "drift" for t in v) and any(t.kind == "cold" for t in v)]
    assert drifted, "expected an entity with a cold->drift"
    v = drifted[0]
    cold = next(t for t in v if t.kind == "cold")
    dr = next(t for t in v if t.kind == "drift")
    assert dr.gold != cold.gold                                       # DRIFT: answer changed (else no staleness)
    assert list(dr.deps.values())[0] > list(cold.deps.values())[0]    # dependency version bumped (YORO can invalidate)
    print("ok stress_workload")


def test_stress_separability():
    """Distinct base subjects must be embedding-SEPARABLE (a
    semantic cache won't confuse them -> accurate at rate 0), while an injected near-miss must be
    CLOSE to its parent (a naive cache force-fits it). Uses the HashEmbedder proxy for a fast,
    dependency-free check of the geometry the real all-MiniLM run confirmed (base max<0.85, nm>=0.85)."""
    from bench.datasets import _STRESS_SUBJECTS, _STRESS_OPS, _stress_text, _near_miss_of
    import random
    emb = HashEmbedder()

    def cos(a, b):
        import math
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return d / (na * nb + 1e-9)
    rng = random.Random(0)
    # subjects themselves must be distinct strings (no accidental dupes in the pool)
    subs = [s for s, u, t in _STRESS_SUBJECTS]
    assert len(set(subs)) == len(subs)
    tags = [t for s, u, t in _STRESS_SUBJECTS]
    assert len(set(tags)) == len(tags)                               # distinct dep-key tags
    # a near-miss shares most of its parent's surface -> more tokens in common than two base subjects
    i = 0
    parent = {"subject": _STRESS_SUBJECTS[i][0], "unit": _STRESS_SUBJECTS[i][1],
              "op": _STRESS_OPS[0], "v": 300, "key": "ops:x", "ver": 1}
    nm = _near_miss_of(parent, rng)
    assert nm["key"] != parent["key"] and nm["v"] != parent["v"]     # distinct entity + different answer
    assert parent["subject"] in nm["subject"]                        # sibling = parent surface + a qualifier
    # real-embedder regression guard: base subjects must stay embedding-far (< tau) so a
    # semantic cache doesn't confuse DISTINCT entities. Skip only if sentence-transformers is absent.
    try:
        from yoro import SentenceTransformerEmbedder
        import itertools
        se = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

        def _cos(a, b):
            import math
            d = sum(x * y for x, y in zip(a, b))
            return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)
        E = [se.embed(_stress_text(s, u, _STRESS_OPS[i % 5][0], 300, rng)) for i, (s, u, t) in enumerate(_STRESS_SUBJECTS)]
        mx = max(_cos(E[i], E[j]) for i, j in itertools.combinations(range(len(E)), 2))
        assert mx < 0.80, f"base subjects too close (max cos {mx:.3f}); diversify until < 0.80 (well below tau 0.9)"
    except ImportError:
        pass
    print("ok stress_separability")


def test_hard_separability():
    """Invariant: the HARD workload's short REF re-asks must stay ABOVE the hard τ (0.80)
    in similarity to their own cold text — else they'd escalate to full-reason on a method-less re-ask
    and get it wrong for surface reasons, cratering the sweep. And cross-entity must stay BELOW τ (no false
    hits). Enforces min(same ref↔cold) > 0.80 and max(cross-entity) < 0.80, so it's checked not lucky.
    Skips only if sentence-transformers is absent (the real-MiniLM geometry is what matters)."""
    import importlib.util
    if importlib.util.find_spec("sentence_transformers") is None:
        print("ok hard_separability (skipped: no sentence-transformers)")
        return
    from yoro import SentenceTransformerEmbedder
    from bench.datasets import build_stress_workload
    import itertools
    HARD_TAU = 0.80
    se = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

    def _cos(a, b):
        import math
        d = sum(x * y for x, y in zip(a, b))
        return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)
    ts = build_stress_workload(n_unique=12, stream_len=400, zipf_s=1.3, drift_rate=0.4,
                               near_miss_rate=0, hard=True, seed=1)
    byk = {}
    for t in ts:
        byk.setdefault(t.key, []).append(t)
    same, cold_emb = [], {}
    for k, seq in byk.items():
        colds = [t for t in seq if t.kind == "cold"]
        if not colds:
            continue
        ce = se.embed(colds[0].text); cold_emb[k] = ce
        for t in seq:
            if t.kind in ("drift", "repeat"):
                same.append(_cos(ce, se.embed(t.text)))
    cross = [_cos(cold_emb[a], cold_emb[b]) for a, b in itertools.combinations(cold_emb, 2)]
    assert same and min(same) > HARD_TAU, \
        f"a hard ref re-ask sits at {min(same):.3f} <= tau {HARD_TAU}: it would escalate-and-fail. Raise ref similarity or lower TAU_HIT."
    assert max(cross) < HARD_TAU, f"cross-entity max {max(cross):.3f} >= tau {HARD_TAU}: false-hit risk"
    print(f"ok hard_separability (same min {min(same):.3f} > {HARD_TAU} > cross max {max(cross):.3f})")


def test_post_drift_repeat():
    """After a DRIFT permanently mutates an entity's value, a LATER
    REPEAT of that entity must carry the NEW gold — so a cache serving the pre-drift answer on that
    repeat is genuinely STALE. (Scoring now counts it: stale = reused AND wrong on same-entity kinds,
    not only on the kind=='drift' occurrence.)"""
    from bench.datasets import build_stress_workload
    ts = build_stress_workload(n_unique=15, stream_len=800, zipf_s=1.4,
                               drift_rate=0.35, near_miss_rate=0.0, seed=11)
    byk = {}
    for t in ts:
        byk.setdefault(t.key, []).append(t)
    found = False
    for seq in byk.values():                                        # find an entity: cold ... drift ... repeat
        kinds = [t.kind for t in seq]
        if "drift" in kinds and kinds.index("drift") < len(kinds) - 1:
            di = kinds.index("drift")
            after = [t for t in seq[di + 1:] if t.kind == "repeat"]
            if after:
                found = True
                assert after[0].gold == seq[di].gold                # post-drift repeat carries the DRIFTED gold
                assert after[0].gold != seq[0].gold                 # ...which differs from the original pre-drift gold
                break
    assert found, "expected an entity with cold -> drift -> repeat (the post-drift stale window)"
    print("ok post_drift_repeat")


def test_stress_fidelity():
    from bench.datasets import build_stress_workload                  # E4: silent drift (fidelity 0) => no dep bump
    ts = build_stress_workload(n_unique=20, stream_len=400, zipf_s=1.2,
                               drift_rate=0.5, near_miss_rate=0.0, inval_fidelity=0.0, seed=5)
    byk = {}
    for t in ts:
        byk.setdefault(t.key, []).append(t)
    for v in byk.values():
        cold = [t for t in v if t.kind == "cold"]
        for dr in [t for t in v if t.kind == "drift"]:
            if cold:
                assert list(dr.deps.values())[0] == list(cold[0].deps.values())[0]   # version NOT bumped
    print("ok stress_fidelity")


def test_ladder_rung_subset():
    emb = HashEmbedder()
    full = build_ladder(lambda: MockModel(), emb)
    assert [s.name for s in full] == ["no-cache", "exact-match", "gptcache-semantic", "behaviors-only",
                                      "yoro+behaviors", "yoro", "yoro-replay", "yoro-replay-low"]
    sub = build_ladder(lambda: MockModel(), emb, rungs=("yoro-replay-low", "no-cache", "yoro-replay"))
    assert [s.name for s in sub] == ["no-cache", "yoro-replay", "yoro-replay-low"]   # subset, canonical order kept
    try:
        build_ladder(lambda: MockModel(), emb, rungs=("bogus",))
        assert False, "should reject unknown rung"
    except ValueError:
        pass
    print("ok ladder_rung_subset")


def test_sweep_suffix():
    from bench.run_phase0 import _level_suffix
    assert _level_suffix("zipf_s", 0.6) == "z0.6"                     # back-compat with existing S3 paths
    assert _level_suffix("drift_rate", 0.3) == "drift0.3"
    assert _level_suffix("near_miss_rate", 0.2) == "nm0.2"
    assert _level_suffix("inval_fidelity", 0.5) == "fid0.5"
    print("ok sweep_suffix")


if __name__ == "__main__":
    tests = [test_budget_guard, test_metrics, test_ladder_reuse, test_replay_tier,
             test_eventlog_jsonl, test_checkpoint_resume, test_convergence,
             test_workload_recurrence, test_load_hf_construction, test_resume_per_seed,
             test_vast_credit, test_budget_external_stop,
             test_stress_workload, test_stress_separability, test_hard_separability,
             test_post_drift_repeat, test_stress_fidelity, test_ladder_rung_subset, test_sweep_suffix]
    for fn in tests:
        fn()
    print(f"\nALL {len(tests)} BENCH TESTS PASSED")
