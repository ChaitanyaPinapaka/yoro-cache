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
        from .proxy import main as serve_main

        serve_main()
        return 0
    if a.cmd == "stats":
        import requests

        r = requests.get(a.url.rstrip("/") + "/yoro/stats", timeout=10)
        print(json.dumps(r.json(), indent=2))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
