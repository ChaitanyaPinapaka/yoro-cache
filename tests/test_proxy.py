"""Proxy logic tests — pure functions + ProxyCache, no socket, no model, no torch."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yoro import Decision, HashEmbedder, Invalidator, Matcher, ReasoningCache
from yoro.proxy import (
    ProxyCache,
    extract_task,
    is_cacheable,
    parse_deps,
    synth_completion,
)


def test_extract_task():
    assert (
        extract_task(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "hello"}]
        )
        == "hello"
    )
    assert (
        extract_task(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                }
            ]
        )
        == "a b"
    )
    assert extract_task([{"role": "assistant", "content": "hi"}]) == ""  # no user turn


def test_parse_deps():
    assert parse_deps("a.py:1,b.toml:2") == {"a.py": "1", "b.toml": "2"}
    assert parse_deps(None) == {}


def test_cacheable_policy():
    base = {"messages": [{"role": "user", "content": "q"}]}
    assert is_cacheable(base, None, "safe") is True
    assert (
        is_cacheable({**base, "tools": [{"x": 1}]}, None, "safe") is False
    )  # agentic -> skip
    assert (
        is_cacheable({**base, "tools": [{"x": 1}]}, "1", "safe") is True
    )  # header forces on
    assert (
        is_cacheable({**base, "temperature": 0.7}, None, "safe") is False
    )  # sampled -> skip
    assert is_cacheable({**base, "tools": [{"x": 1}]}, None, "aggressive") is True
    assert is_cacheable(base, "0", "safe") is False  # header forces off
    hist = {
        "messages": [{"role": "user", "content": "q"}, {"role": "tool", "content": "r"}]
    }
    assert is_cacheable(hist, None, "safe") is False  # tool result in history


def test_proxycache_hit_miss():
    pc = ProxyCache(
        HashEmbedder(),
        ReasoningCache(),
        Matcher(0.95, 0.6, True),
        Invalidator(use_deps=True, use_ttl=False, use_reliability=False),
    )
    d, _, _ = pc.lookup("sum of integers 1 to 100", {})
    assert d == Decision.MISS  # empty cache
    pc.store("sum of integers 1 to 100", "5050", None, {})
    d, case, sim = pc.lookup("sum of integers 1 to 100", {})
    assert (
        d == Decision.HIT and case.outcome == "5050" and sim > 0.99
    )  # exact recurrence
    d2, _, _ = pc.lookup("capital city of france please", {})
    assert d2 != Decision.HIT  # unrelated -> not a hit
    assert pc.stats.stored == 1


def test_proxycache_dep_invalidation():
    pc = ProxyCache(
        HashEmbedder(),
        ReasoningCache(),
        Matcher(0.95, 0.6, True),
        Invalidator(use_deps=True, use_ttl=False, use_reliability=False),
    )
    pc.store("read config value", "old", None, {"config.toml": "v1"})
    assert pc.lookup("read config value", {"config.toml": "v1"})[0] == Decision.HIT
    # same question, but the dependency changed -> must NOT serve the stale answer
    assert pc.lookup("read config value", {"config.toml": "v2"})[0] != Decision.HIT


def test_synth_completion():
    c = synth_completion("m", "hi", "because", 123.0)
    assert c["object"] == "chat.completion"
    assert c["choices"][0]["message"]["content"] == "hi"
    assert c["choices"][0]["message"]["reasoning_content"] == "because"


if __name__ == "__main__":
    tests = [
        test_extract_task,
        test_parse_deps,
        test_cacheable_policy,
        test_proxycache_hit_miss,
        test_proxycache_dep_invalidation,
        test_synth_completion,
    ]
    for fn in tests:
        fn()
        print("  ok ", fn.__name__)
    print(f"\nALL {len(tests)} PROXY TESTS PASSED")
