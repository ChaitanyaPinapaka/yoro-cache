"""MCP resources as YORO's invalidation signal (experimental).

MCP is the first standardized change-feed for agent context: servers expose
resources by URI and can notify on updates. This bridge turns that feed into
dependency fingerprints the proxy understands, by maintaining a deps-file
({uri: fingerprint}) that `yoro serve --deps-file` (or the LiteLLM/LangChain
adapters) read on every request.

    # terminal 1: the bridge, watching an MCP server's resources
    yoro mcp-bridge --server "python my_mcp_server.py" --deps-file ~/.yoro/mcp_deps.json

    # terminal 2: the proxy, consuming the signal
    yoro serve --deps-file ~/.yoro/mcp_deps.json

v1 polls `resources/list` + `resources/read` and fingerprints content (works with
every MCP server; subscriptions are an optional capability). When the server
supports `resources/subscribe`, notifications tighten the loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Iterable


def fingerprint(content: bytes | str) -> str:
    data = content.encode() if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()[:12]


def write_deps_file(path: str, deps: dict) -> None:
    """Atomic write so the proxy never reads a torn file."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(deps, f, indent=0, sort_keys=True)
    os.replace(tmp, path)


async def snapshot(session) -> dict:
    """{'mcp:<uri>': fingerprint} for every resource the server currently exposes."""
    deps: dict = {}
    listed = await session.list_resources()
    for res in listed.resources:
        uri = str(res.uri)
        try:
            read = await session.read_resource(res.uri)
            parts = []
            for c in read.contents:
                parts.append(getattr(c, "text", None) or getattr(c, "blob", "") or "")
            deps["mcp:" + uri] = fingerprint("\x00".join(str(p) for p in parts))
        except Exception:
            # unreadable resource: fingerprint its identity so at least appearance/
            # disappearance is a signal
            deps["mcp:" + uri] = "unreadable"
    return deps


async def run_bridge(session, deps_file: str, interval: float = 5.0, cycles: int | None = None) -> dict:
    """Poll the server and keep deps_file current. `cycles` bounds the loop for tests;
    None means run forever."""
    import asyncio

    last: dict = {}
    n = 0
    while cycles is None or n < cycles:
        deps = await snapshot(session)
        if deps != last:
            write_deps_file(deps_file, deps)
            last = deps
        n += 1
        if cycles is None or n < cycles:
            await asyncio.sleep(interval)
    return last


def main(argv: Iterable[str] | None = None) -> int:
    import argparse
    import asyncio
    import shlex

    ap = argparse.ArgumentParser(
        prog="yoro mcp-bridge",
        description="Mirror an MCP server's resources into a YORO deps-file.",
    )
    ap.add_argument("--server", required=True, help="command that runs the MCP server (stdio)")
    ap.add_argument("--deps-file", required=True)
    ap.add_argument("--interval", type=float, default=5.0, help="poll seconds (default 5)")
    a = ap.parse_args(list(argv) if argv is not None else None)

    async def run():
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        cmd = shlex.split(a.server)
        params = StdioServerParameters(command=cmd[0], args=cmd[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"[yoro mcp-bridge] watching resources -> {a.deps_file}", flush=True)
                await run_bridge(session, a.deps_file, interval=a.interval)

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
