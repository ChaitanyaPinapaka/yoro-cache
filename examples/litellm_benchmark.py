"""Example 2 — YORO inside LiteLLM's cache slot.

The same recurring-tasks-then-drift workload, driven through litellm.completion
against a local OpenAI-compatible endpoint, with and without YoroSemanticCache.
The adapter serves and invalidates (no replay in a cache slot — that is the
proxy's tier), so the win here is: repeats are free, and drift never serves stale.

    llama-server -hf deepreinforce-ai/Ornith-1.0-35B-GGUF --port 8000
    pip install "yoro-cache[embed,litellm]"
    python examples/litellm_benchmark.py --api-base http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from _bench import Meter, aggregate, base_values, build_stream, drift, print_table  # noqa: E402

SITES4 = ["Athens depot", "Bergen depot", "Cairo depot", "Denver depot"]


def run_stream(values: dict, meter: Meter, yc, repeats: int, api_base: str):
    import litellm

    for _, text, expected in build_stream(values, repeats=repeats):
        hits_before = yc.hits if yc else 0
        r = litellm.completion(
            model="openai/local", api_base=api_base, api_key="unused",
            temperature=0, max_tokens=700, caching=yc is not None,
            messages=[{"role": "user", "content": text}],
        )
        answer = r.choices[0].message.content or ""
        served = yc is not None and yc.hits > hits_before
        toks = 0 if served else int(getattr(r.usage, "completion_tokens", 0) or 0)
        meter.record(answer, expected, model_called=not served, out_tokens=toks)
        print(f"  [{'CACHE' if served else 'MODEL':>5}] {text[:52]}… -> {answer.strip()[:12]}")


def main():
    import litellm
    from litellm.caching.caching import Cache
    from yoro import deps as depsmod
    from yoro.integrations.litellm_cache import YoroSemanticCache

    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()

    pairs = []
    for it in range(a.iters):
        print(f"\n################ iteration {it + 1}/{a.iters} ################")
        values = base_values(it, SITES4)
        sidecar = os.path.join(tempfile.mkdtemp(prefix=f"yoro-litellm-{it}-"), "deps.json")
        open(sidecar, "w").write(json.dumps({"ledger": f"v1-{it}"}))

        print("\n=== baseline: litellm, no cache ===")
        litellm.cache = None
        base = Meter("litellm no cache")
        run_stream(values, base, None, 3, a.api_base)
        v2 = drift(values)
        print("  -- DRIFT --")
        run_stream(v2, base, None, 1, a.api_base)

        print("\n=== litellm + YoroSemanticCache (deps-file signal) ===")
        yc = YoroSemanticCache(deps_file=sidecar)
        litellm.cache = Cache()
        litellm.cache.cache = yc
        m = Meter("litellm + yoro")
        run_stream(values, m, yc, 3, a.api_base)
        print("  -- DRIFT: sidecar publishes the new fingerprint --")
        open(sidecar, "w").write(json.dumps({"ledger": f"v2-{it}"}))
        depsmod._CACHE.clear()
        run_stream(v2, m, yc, 1, a.api_base)
        litellm.cache = None

        print_table([base, m])
        pairs.append((base, m))

    aggregate(pairs, "litellm no cache", "litellm + yoro")
    print("\nnote: cache-slot adapters serve + invalidate; replay is the proxy's tier, so "
          "post-drift asks here are full fresh calls (correct, never stale).")


if __name__ == "__main__":
    main()
