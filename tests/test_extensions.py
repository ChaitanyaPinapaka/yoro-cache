"""Coverage for the extension-point API: ModelKeyer parsing, OpenAIEmbedder
normalization, StructuredReasoning round-trip. All stubbed — no network, no torch."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yoro import ModelKeyer, OpenAIEmbedder, StructuredReasoning


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeRequests:
    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def post(self, *a, **k):
        self.calls += 1
        return _Resp(self._payload)


def _chat(content):
    return {"choices": [{"message": {"content": content}}]}


def test_modelkeyer_parses_canon_and_memoizes():
    k = ModelKeyer()
    k._requests = _FakeRequests(_chat("thinking...</think>\nCANON: factorial of 6"))
    assert k.key("What is 6 factorial?") == "factorial of 6"  # CANON line wins, think-block stripped
    assert k.key("What is 6 factorial?") == "factorial of 6"  # memoized ->
    assert k._requests.calls == 1  # no second model call


def test_modelkeyer_fallbacks():
    k = ModelKeyer()
    k._requests = _FakeRequests(_chat("no marker here\njust a last line"))
    assert k.key("task A") == "just a last line"  # no CANON -> last non-empty line
    k2 = ModelKeyer()
    k2._requests = _FakeRequests(_chat(""))
    assert k2.key("task B") == "task B"  # empty response -> raw task


def test_openai_embedder_normalizes():
    e = OpenAIEmbedder(dim=2)
    e._requests = _FakeRequests({"data": [{"embedding": [3.0, 4.0]}]})
    v = e.embed("x")
    assert np.allclose(v, [0.6, 0.8])  # unit-normed
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6


def test_structured_reasoning_roundtrip():
    sr = StructuredReasoning.parse("1. read the config\n2. add the offsets\n3. report the total")
    assert sr.steps == ["read the config", "add the offsets", "report the total"]
    assert sr.edges == [(0, 1), (1, 2)]  # linear chain
    d = sr.to_dict()
    assert d["steps"][0] == "read the config" and d["edges"] == [[0, 1], [1, 2]]


if __name__ == "__main__":
    for fn in (test_modelkeyer_parses_canon_and_memoizes, test_modelkeyer_fallbacks,
               test_openai_embedder_normalizes, test_structured_reasoning_roundtrip):
        fn()
        print("  ok ", fn.__name__)
