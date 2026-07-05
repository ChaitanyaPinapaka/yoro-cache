"""Example 1 — the proxy with git as the invalidation signal.

A coding-agent-shaped scenario: recurring questions about workspace data, then the
data changes (edit + commit), then the questions recur with the new values. Run the
same stream twice — straight to the upstream model, and through `yoro serve --git` —
and compare model calls, output tokens, wrongness, and wall time.

    # upstream: any OpenAI-compatible endpoint (llama.cpp shown)
    llama-server -hf deepreinforce-ai/Ornith-1.0-35B-GGUF --port 8000
    python examples/proxy_git_benchmark.py --upstream http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from _bench import SITES, Meter, aggregate, base_values, build_stream, drift, print_table  # noqa: E402


def ask(url: str, text: str, deps: dict | None = None, max_tokens: int = 700):
    headers = {"Content-Type": "application/json"}
    if deps:
        headers["X-YORO-Deps"] = ",".join(f"{k}:{v}" for k, v in deps.items())
    r = requests.post(url + "/chat/completions", headers=headers, timeout=600, json={
        "model": "local", "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": text}],
    })
    r.raise_for_status()
    j = r.json()
    tag = r.headers.get("X-YORO-Cache", "UPSTREAM")
    usage = j.get("usage") or {}
    return j["choices"][0]["message"]["content"], tag, int(usage.get("completion_tokens") or 0)


def run_stream(url: str, values: dict, meter: Meter, repeats: int = 3):
    for _, text, expected in build_stream(values, repeats=repeats):
        answer, tag, toks = ask(url, text)
        meter.record(answer, expected, model_called=tag in ("UPSTREAM", "MISS", "REPLAY"), out_tokens=toks)
        print(f"  [{tag:>7}] {text[:52]}… -> {answer.strip()[:12]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--port", type=int, default=8411)
    ap.add_argument("--iters", type=int, default=3, help="independent iterations (fresh cache + workspace each)")
    a = ap.parse_args()

    pairs = []
    for it in range(a.iters):
        print(f"\n################ iteration {it + 1}/{a.iters} ################")
        # a real git workspace whose state IS the dependency (fresh every iteration)
        ws = tempfile.mkdtemp(prefix=f"yoro-example-{it}-")
        run = lambda *c: subprocess.run(["git", "-C", ws, *c], capture_output=True)
        run("init", "-b", "main"); run("config", "user.email", "e@x"); run("config", "user.name", "ex")
        values = base_values(it)
        open(os.path.join(ws, "ledger.json"), "w").write(json.dumps(values))
        run("add", "-A"); run("commit", "-m", "ledger v1")

        env = dict(os.environ, YORO_UPSTREAM=a.upstream, YORO_PORT=str(a.port + it),
                   YORO_GIT=ws, YORO_CACHE_PATH=os.path.join(ws, ".yoro-cache.json"))
        proxy = subprocess.Popen([sys.executable, "-m", "yoro.cli", "serve"], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(120):
                try:
                    requests.get(f"http://127.0.0.1:{a.port + it}/yoro/health", timeout=2); break
                except Exception:
                    time.sleep(1)

            print("\n=== baseline: straight to the model ===")
            base = Meter("no cache")
            run_stream(a.upstream, values, base)
            v2 = drift(values)
            print("  -- DRIFT: ledger values change --")
            run_stream(a.upstream, v2, base, repeats=1)

            print("\n=== through YORO (serve / replay / reason, git signal) ===")
            yoro_url = f"http://127.0.0.1:{a.port + it}/v1"
            m = Meter("yoro --git")
            run_stream(yoro_url, values, m)
            print("  -- DRIFT: edit + commit the ledger --")
            open(os.path.join(ws, "ledger.json"), "w").write(json.dumps(v2))
            run("add", "-A"); run("commit", "-m", "ledger v2")
            time.sleep(2.5)  # the git fingerprint source caches for ~2s
            run_stream(yoro_url, v2, m, repeats=1)

            stats = requests.get(f"http://127.0.0.1:{a.port + it}/yoro/stats", timeout=5).json()
            print_table([base, m], note=f"proxy stats: {stats}")
            pairs.append((base, m))
        finally:
            proxy.terminate()

    aggregate(pairs, "no cache", "yoro --git")


if __name__ == "__main__":
    main()
