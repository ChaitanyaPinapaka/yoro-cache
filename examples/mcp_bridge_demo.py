"""Example 4 — MCP resources as the invalidation signal, end to end.

An in-process MCP server exposes a ledger resource. The bridge snapshots it into
a deps-file; `yoro serve --deps-file` consumes it. When the resource changes, the
next ask REPLAYs instead of serving stale — no application code computed a single
fingerprint.

    llama-server -hf deepreinforce-ai/Ornith-1.0-35B-GGUF --port 8000
    pip install "yoro-cache[embed,mcp]"
    python examples/mcp_bridge_demo.py --upstream http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from _bench import PROCEDURE, gold  # noqa: E402


def ask(port: int, text: str):
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", timeout=600, json={
        "model": "local", "temperature": 0, "max_tokens": 1500,
        "messages": [{"role": "user", "content": text}],
    })
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], r.headers.get("X-YORO-Cache", "?")


def main():
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session
    from yoro.integrations.mcp_bridge import run_bridge

    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--port", type=int, default=8431)
    ap.add_argument("--iters", type=int, default=2)
    a = ap.parse_args()

    for it in range(a.iters):
        print(f"\n################ iteration {it + 1}/{a.iters} ################")
        state = {"units": 500 + 11 * it}
        server = FastMCP("ledger")

        @server.resource("ledger://osaka")
        def osaka() -> str:
            return f"units={state['units']}"

        tmp = tempfile.mkdtemp(prefix=f"yoro-mcp-{it}-")
        deps_file = os.path.join(tmp, "mcp_deps.json")

        async def snap():
            async with create_connected_server_and_client_session(server._mcp_server) as s:
                await run_bridge(s, deps_file, cycles=1)

        asyncio.run(snap())
        print(f"bridge wrote {deps_file}: {open(deps_file).read()}")

        env = dict(os.environ, YORO_UPSTREAM=a.upstream, YORO_PORT=str(a.port + it),
                   YORO_DEPS_FILE=deps_file,
                   YORO_CACHE_PATH=os.path.join(tmp, "cache.json"))
        proxy = subprocess.Popen([sys.executable, "-m", "yoro.cli", "serve"], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(120):
                try:
                    requests.get(f"http://127.0.0.1:{a.port + it}/yoro/health", timeout=2)
                    break
                except Exception:
                    time.sleep(1)

            q = lambda: (f"The Osaka depot ledger shows {state['units']} units. {PROCEDURE}")
            expect = lambda: gold(state["units"])

            ans, tag = ask(a.port + it, q())
            print(f"[{tag:>7}] cold ask        -> {ans.strip()[:12]}  (expected {expect()})")
            ans, tag = ask(a.port + it, q())
            print(f"[{tag:>7}] repeat          -> {ans.strip()[:12]}  (0 tokens, served)")

            state["units"] += 50  # the world moves: the MCP resource changes
            asyncio.run(snap())   # the bridge notices; no app code involved
            from yoro import deps as depsmod
            depsmod._CACHE.clear()
            time.sleep(2.5)
            ans, tag = ask(a.port + it, q())
            ok = expect() in ans
            print(f"[{tag:>7}] after change    -> {ans.strip()[:12]}  "
                  f"(expected {expect()}, {'correct' if ok else 'WRONG'})")
            if tag != "REPLAY" or not ok:
                print("unexpected result — check the run"); sys.exit(1)
        finally:
            proxy.terminate()
    print("\nMCP resource change -> bridge fingerprint -> REPLAY with the correct new answer.")


if __name__ == "__main__":
    main()
