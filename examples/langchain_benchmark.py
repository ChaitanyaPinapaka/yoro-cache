"""Example 3 — YORO as the LangChain LLM cache.

`set_llm_cache(YoroLangChainCache(...))` in front of a minimal LLM that calls a
local OpenAI-compatible endpoint. Same workload, with and without the cache.

    llama-server -hf deepreinforce-ai/Ornith-1.0-35B-GGUF --port 8000
    pip install "yoro-cache[embed,langchain]"
    python examples/langchain_benchmark.py --api-base http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import requests

sys.path.insert(0, os.path.dirname(__file__))
from _bench import Meter, aggregate, base_values, build_stream, drift, print_table  # noqa: E402

SITES4 = ["Athens depot", "Bergen depot", "Cairo depot", "Denver depot"]
COUNTS = {"calls": 0, "out_tokens": 0}


def make_llm(api_base: str):
    from langchain_core.language_models.llms import LLM

    class LocalLLM(LLM):
        cache: bool | None = None

        @property
        def _llm_type(self) -> str:
            return "local-openai"

        @property
        def _identifying_params(self):
            return {"api_base": api_base, "model": "local"}

        def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
            COUNTS["calls"] += 1
            r = requests.post(api_base + "/chat/completions", timeout=600, json={
                "model": "local", "temperature": 0, "max_tokens": 700,
                "messages": [{"role": "user", "content": prompt}],
            })
            r.raise_for_status()
            j = r.json()
            COUNTS["out_tokens"] += int((j.get("usage") or {}).get("completion_tokens") or 0)
            return j["choices"][0]["message"]["content"] or ""

    return LocalLLM()


def run_stream(llm, values: dict, meter: Meter, repeats: int):
    for _, text, expected in build_stream(values, repeats=repeats):
        before = COUNTS["calls"]
        toks0 = COUNTS["out_tokens"]
        answer = llm.invoke(text)
        called = COUNTS["calls"] > before
        meter.record(answer, expected, model_called=called,
                     out_tokens=COUNTS["out_tokens"] - toks0)
        print(f"  [{'MODEL' if called else 'CACHE':>5}] {text[:52]}… -> {answer.strip()[:12]}")


def main():
    from langchain_core.globals import set_llm_cache
    from yoro import deps as depsmod
    from yoro.integrations.langchain_cache import YoroLangChainCache

    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()

    pairs = []
    for it in range(a.iters):
        print(f"\n################ iteration {it + 1}/{a.iters} ################")
        values = base_values(it, SITES4)
        v2 = drift(values)
        sidecar = os.path.join(tempfile.mkdtemp(prefix=f"yoro-lc-{it}-"), "deps.json")
        open(sidecar, "w").write(json.dumps({"ledger": f"v1-{it}"}))
        llm = make_llm(a.api_base)

        print("\n=== baseline: no cache ===")
        set_llm_cache(None)
        base = Meter("langchain no cache")
        run_stream(llm, values, base, 3)
        print("  -- DRIFT --")
        run_stream(llm, v2, base, 1)

        print("\n=== langchain + YoroLangChainCache (deps-file signal) ===")
        set_llm_cache(YoroLangChainCache(deps_file=sidecar))
        m = Meter("langchain + yoro")
        run_stream(llm, values, m, 3)
        print("  -- DRIFT: sidecar publishes the new fingerprint --")
        open(sidecar, "w").write(json.dumps({"ledger": f"v2-{it}"}))
        depsmod._CACHE.clear()
        run_stream(llm, v2, m, 1)
        set_llm_cache(None)

        print_table([base, m])
        pairs.append((base, m))

    aggregate(pairs, "langchain no cache", "langchain + yoro")


if __name__ == "__main__":
    main()
