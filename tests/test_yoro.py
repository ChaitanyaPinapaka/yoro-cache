"""Plain-assert tests. Run directly (`.venv/bin/python tests/test_yoro.py`) or with
pytest. Covers the data structure, the three mechanisms, the end-to-end loop, and
the decision-tree router."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yoro import (
    YORO,
    Decision,
    HashEmbedder,
    Invalidator,
    Matcher,
    ReasoningCache,
    ReasoningCase,
    ReasoningTreeRouter,
    cosine,
)


class World:
    """Minimal drifting world: each task family has a current answer + a version
    fingerprint; drift() changes both. Enough to exercise reuse/invalidate/update."""

    def __init__(self):
        self.ver: dict[str, int] = {}

    def register(self, fam):
        self.ver.setdefault(fam, 1)

    def family_of(self, task):
        return task.split()[0]

    def answer(self, fam):
        return f"{fam}-answer-v{self.ver.get(fam, 1)}"

    def deps(self, fam):
        self.register(fam)
        return {fam: f"{fam}#{self.ver[fam]}"}

    def drift(self, fam):
        self.ver[fam] = self.ver.get(fam, 1) + 1


class MockModel:
    """Perfect-but-counted reasoner over the World — every measured error is a
    CACHE error (stale/brittle reuse), never a model error."""

    name = "mock"

    def __init__(self, world):
        self.world = world
        self.calls = 0

    def reason(self, task):
        self.calls += 1
        fam = self.world.family_of(task)
        ans = self.world.answer(fam)
        return (f"[reason] family={fam} -> {ans}", ans)

    def complete(self, prompt, max_tokens=None):
        return ""


def test_embedder_similarity():
    e = HashEmbedder(64)
    a = e.embed("famA aw0 aw1 aw2 aw3")
    b = e.embed("famA aw0 aw1 aw2 aw3")
    c = e.embed("famZ zz0 zz1 zz2 zz3")
    assert cosine(a, b) > 0.999  # identical -> ~1
    assert cosine(a, c) < 0.5  # disjoint -> low


def test_cache_roundtrip():
    e = HashEmbedder(64)
    c = ReasoningCache()
    c.add("t1", e.embed("famA aw0 aw1 aw2 aw3"), "r1", "o1", {"src": "A#1"})
    c.add("t2", e.embed("famB bw0 bw1 bw2 bw3"), "r2", "o2", {"src": "B#1"})
    case, sim = c.nearest(e.embed("famA aw0 aw1 aw2 aw3"))
    assert case.outcome == "o1" and sim > 0.99
    f = tempfile.mktemp(suffix=".json")
    c.save(f)
    c2 = ReasoningCache().load(f)
    assert len(c2) == 2 and c2.cases[0].outcome == "o1"
    assert isinstance(c2.cases[0].embedding, np.ndarray)
    os.remove(f)


def test_matcher():
    m = Matcher(tau_hit=0.9, tau_miss=0.6, novelty_gate=True)
    assert m.decide(0.95, True) == Decision.HIT
    assert m.decide(0.95, False) == Decision.ESCALATE  # right case but stale
    assert m.decide(0.75, True) == Decision.ESCALATE  # borderline + gate -> safe
    assert m.decide(0.40, True) == Decision.MISS
    m2 = Matcher(tau_hit=0.9, tau_miss=0.6, novelty_gate=False)
    assert m2.decide(0.75, True) == Decision.HIT  # gate off -> force-fit


def test_invalidator():
    c = ReasoningCase(
        id="x",
        task="t",
        embedding=np.zeros(4, dtype=np.float32),
        reasoning="r",
        outcome="o",
        deps={"src": "A#1"},
    )
    inv = Invalidator(use_deps=True, use_ttl=False, use_reliability=False)
    assert inv.is_fresh(c, {"src": "A#1"}) is True
    assert inv.is_fresh(c, {"src": "A#2"}) is False  # premise changed -> stale
    # case has deps but request carries no signal -> refuse (do not pretend invalidation works)
    assert inv.is_fresh(c, {}) is False
    assert inv.is_fresh(c, None) is False
    inv_loose = Invalidator(
        use_deps=True, use_ttl=False, use_reliability=False, require_signal=False
    )
    assert inv_loose.is_fresh(c, {}) is True  # opt-out of empty-signal guard
    inv_strict = Invalidator(
        use_deps=True, use_ttl=False, use_reliability=False, strict_deps=True
    )
    assert inv_strict.is_fresh(c, {"src": "A#1", "extra": "x"}) is True  # case keys covered
    assert inv_strict.is_fresh(
        ReasoningCase(
            id="y",
            task="t",
            embedding=np.zeros(4, dtype=np.float32),
            reasoning="r",
            outcome="o",
            deps={"src": "A#1", "other": "B#1"},
        ),
        {"src": "A#1"},
    ) is False  # missing other under strict
    inv_off = Invalidator(use_deps=False, use_ttl=False, use_reliability=False)
    assert inv_off.is_fresh(c, {"src": "A#2"}) is True  # deps toggle off: drift ignored


def test_engine_lookup_shared():
    """Library and proxy share engine.lookup for HIT / replay routing."""
    from yoro import lookup as engine_lookup

    e = HashEmbedder(64)
    cache = ReasoningCache()
    emb = e.embed("famA aw0 aw1 aw2 aw3")
    cache.add("t", emb, "method: add then triple", "42", {"src": "v1"})
    matcher = Matcher(0.9, 0.6, True)
    inv = Invalidator(use_deps=True, use_ttl=False, use_reliability=False)

    hit = engine_lookup(cache, matcher, inv, emb, {"src": "v1"}, replay=True)
    assert hit.decision == Decision.HIT and hit.should_replay is False

    stale = engine_lookup(cache, matcher, inv, emb, {"src": "v2"}, replay=True)
    assert stale.decision == Decision.ESCALATE and stale.same_case and stale.should_replay

    empty = engine_lookup(cache, matcher, inv, emb, {}, replay=True)
    assert empty.fresh is False and empty.decision == Decision.ESCALATE


def test_cache_eviction_and_sqlite(tmp_path):
    e = HashEmbedder(32)
    c = ReasoningCache(max_cases=3)
    for i in range(5):
        case = c.add(f"t{i}", e.embed(f"fam tokens {i} uniq{i}"), "r", f"o{i}", {})
        if i < 2:
            c.record_use(case, True)
            c.record_use(case, True)  # pin early cases via higher use count
    assert len(c) == 3
    assert c._evicted == 2
    # heavily used early cases survive
    ids = {x.outcome for x in c.cases}
    assert "o0" in ids and "o1" in ids

    p = str(tmp_path / "cache.sqlite")
    sc = ReasoningCache(p, flush_every=10)
    sc.add("a", e.embed("alpha beta gamma"), "r", "oa", {"k": "1"})
    assert not os.path.exists(p)  # write-behind: not flushed yet
    sc.add("b", e.embed("delta epsilon zeta"), "r", "ob", {"k": "1"})
    sc.flush()
    assert os.path.exists(p)
    loaded = ReasoningCache(p).load()
    assert len(loaded) == 2
    case, sim = loaded.nearest(e.embed("alpha beta gamma"))
    assert case.outcome == "oa" and sim > 0.9


def test_end_to_end_reason_once_reuse_update():
    world = World()
    world.register("famA")
    e = HashEmbedder(64)
    model = MockModel(world)
    engine = YORO(
        model,
        e,
        ReasoningCache(),
        Matcher(0.9, 0.6, True),
        Invalidator(use_deps=True, use_ttl=False, use_reliability=False),
    )
    txt = "famA famaw0 famaw1 famaw2 famaw3"

    r1 = engine.solve(txt, current_deps=world.deps("famA"))
    assert r1.reasoned and r1.decision == "cold" and model.calls == 1

    r2 = engine.solve(txt, current_deps=world.deps("famA"))
    assert (
        (not r2.reasoned) and r2.decision == "hit" and model.calls == 1
    )  # the YORO win: reused

    world.drift("famA")  # the world changed
    r3 = engine.solve(txt, current_deps=world.deps("famA"))
    assert (
        r3.reasoned and r3.decision == "update" and model.calls == 2
    )  # caught stale -> re-reason + UPDATE
    assert r3.outcome == world.answer("famA")  # returns the NEW answer

    rn = engine.solve(
        "famZ famzz0 famzz1 famzz2 famzz3", current_deps=world.deps("famZ")
    )
    assert rn.reasoned and rn.decision == "cold"  # novel -> reason fresh


def test_tree_router():
    e = HashEmbedder(64)
    c = ReasoningCache()
    c.add("a", e.embed("famA aw0 aw1 aw2 aw3"), "r", "o1")
    c.add("b", e.embed("famB bw0 bw1 bw2 bw3"), "r", "o2")
    rt = ReasoningTreeRouter().fit(c)
    assert rt.route(e.embed("famA aw0 aw1 aw2 aw3")) == c.cases[0].id


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")
