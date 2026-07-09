"""The `yoro` command — thin dispatch over the proxy and its stats endpoint.

yoro serve   run the caching proxy (env-configured; see yoro/proxy.py docstring)
yoro stats   pretty-print /yoro/stats from a running proxy
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="yoro",
        description="YORO — You Only Reason Once. Cache the plan, not the answer.",
    )
    sub = ap.add_subparsers(dest="cmd")

    sv = sub.add_parser("serve", help="run the OpenAI-compatible caching proxy")
    sv.add_argument(
        "--upstream", default=None, help="upstream base URL (default $YORO_UPSTREAM)"
    )
    sv.add_argument(
        "--port",
        type=int,
        default=None,
        help="listen port (default $YORO_PORT or 8400)",
    )
    sv.add_argument(
        "--policy",
        choices=("safe", "aggressive"),
        default=None,
        help="caching policy (default $YORO_POLICY or safe)",
    )
    sv.add_argument(
        "--git",
        default=None,
        metavar="REPO",
        help="fingerprint this git working tree as an automatic dependency "
        "(any commit or edit invalidates entries scoped to it)",
    )
    sv.add_argument(
        "--git-mode",
        choices=("repo", "mentioned", "watch", "off"),
        default=None,
        help="git signal granularity: whole-repo (default), paths mentioned in the "
        "task, explicit --watch paths, or off",
    )
    sv.add_argument(
        "--watch",
        default=None,
        metavar="PATHS",
        help="comma-separated paths under --git to fingerprint (implies git-mode=watch)",
    )
    sv.add_argument(
        "--workspace",
        default=None,
        help="opaque workspace id stored as a dependency (multi-tenant isolation)",
    )
    sv.add_argument(
        "--deps-file",
        default=None,
        metavar="JSON",
        help="JSON file of {name: fingerprint} maintained by a sidecar (file watcher, MCP bridge)",
    )
    sv.add_argument(
        "--cache-max",
        type=int,
        default=None,
        help="evict least-used cases when the store exceeds this size",
    )
    sv.add_argument(
        "--cache-flush-every",
        type=int,
        default=None,
        help="write-behind: flush to disk every N mutations (default 1 = sync)",
    )
    sv.add_argument(
        "--strict-deps",
        action="store_true",
        default=None,
        help="require every stored dep key to be present and matching on lookup",
    )

    mb = sub.add_parser("mcp-bridge", help="mirror an MCP server's resources into a deps-file")
    mb.add_argument("--server", required=True, help="command that runs the MCP server (stdio)")
    mb.add_argument("--deps-file", required=True)
    mb.add_argument("--interval", type=float, default=5.0)

    st = sub.add_parser("stats", help="show a running proxy's cache stats")
    st.add_argument(
        "--url", default="http://127.0.0.1:8400", help="proxy base (default :8400)"
    )

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        # flags override env so `yoro serve --upstream …` works without exports
        if a.upstream:
            os.environ["YORO_UPSTREAM"] = a.upstream
        if a.port:
            os.environ["YORO_PORT"] = str(a.port)
        if a.policy:
            os.environ["YORO_POLICY"] = a.policy
        if a.git:
            os.environ["YORO_GIT"] = a.git
        if a.git_mode:
            os.environ["YORO_GIT_MODE"] = a.git_mode
        if a.watch:
            os.environ["YORO_WATCH"] = a.watch
            os.environ.setdefault("YORO_GIT_MODE", "watch")
        if a.workspace:
            os.environ["YORO_WORKSPACE"] = a.workspace
        if a.deps_file:
            os.environ["YORO_DEPS_FILE"] = a.deps_file
        if a.cache_max is not None:
            os.environ["YORO_CACHE_MAX"] = str(a.cache_max)
        if a.cache_flush_every is not None:
            os.environ["YORO_CACHE_FLUSH_EVERY"] = str(a.cache_flush_every)
        if a.strict_deps:
            os.environ["YORO_STRICT_DEPS"] = "1"
        from .proxy import main as serve_main

        serve_main()
        return 0
    if a.cmd == "mcp-bridge":
        from .integrations.mcp_bridge import main as bridge_main

        return bridge_main([
            "--server", a.server, "--deps-file", a.deps_file, "--interval", str(a.interval),
        ])
    if a.cmd == "stats":
        import requests

        r = requests.get(a.url.rstrip("/") + "/yoro/stats", timeout=10)
        print(json.dumps(r.json(), indent=2))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
