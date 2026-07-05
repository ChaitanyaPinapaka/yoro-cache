"""Dependency-signal sources for the proxy.

YORO's correctness guarantee runs on the invalidation signal. Clients can always
send `X-YORO-Deps` explicitly; these sources let the proxy compute a signal when
they don't:

  * git_fingerprint(repo)  - the workspace as one coarse dependency: HEAD commit
                             plus a hash of the dirty state. Any commit or edit
                             changes the fingerprint. Coarse but CORRECT: a moved
                             workspace can only cost hit rate, never staleness.
                             The natural signal for coding agents.
  * file_deps(path)        - a JSON file of {name: fingerprint} maintained by any
                             sidecar (a file watcher, a git hook, the MCP bridge).

Merge order (resolve_deps): deps-file, then git, then request header - the most
explicit source wins on key collisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

_CACHE: dict = {}  # source key -> (expires_at, value)
_TTL = 2.0  # seconds; keeps subprocess/IO cost off the request hot path


def _cached(key: str, compute):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    val = compute()
    _CACHE[key] = (now + _TTL, val)
    return val


def git_fingerprint(repo: str) -> dict:
    """{'git:<name>': '<HEAD12>+<dirty8>'} for the working tree at `repo`.
    Empty dict if `repo` isn't a git checkout (degrades to no signal, loudly once)."""

    def compute():
        try:
            head = subprocess.run(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if not head:
                return {}
            dirty = subprocess.run(
                ["git", "-C", repo, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            fp = head[:12] + "+" + hashlib.sha256(dirty.encode()).hexdigest()[:8]
            name = "git:" + (os.path.basename(os.path.abspath(repo)) or "repo")
            return {name: fp}
        except Exception:
            return {}

    return _cached("git:" + repo, compute)


def file_deps(path: str) -> dict:
    """{name: fingerprint} from a sidecar-maintained JSON file; {} if absent/invalid."""

    def compute():
        try:
            with open(path) as f:
                data = json.load(f)
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except Exception:
            return {}

    return _cached("file:" + path, compute)


def resolve_deps(header_deps: dict, git_repo: str = "", deps_file: str = "") -> dict:
    """Merge the configured sources; the request header is the most explicit and wins."""
    out: dict = {}
    if deps_file:
        out.update(file_deps(deps_file))
    if git_repo:
        out.update(git_fingerprint(git_repo))
    out.update(header_deps or {})
    return out
