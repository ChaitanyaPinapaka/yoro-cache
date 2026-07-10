"""Dependency-signal sources for the proxy.

YORO's correctness guarantee runs on the invalidation signal. Clients can always
send `X-YORO-Deps` explicitly; these sources let the proxy compute a signal when
they don't:

  * git_fingerprint(repo)  - the workspace as one coarse dependency: HEAD commit
                             plus a hash of the dirty state. Any commit or edit
                             changes the fingerprint. Coarse but CORRECT: a moved
                             workspace can only cost hit rate, never staleness.
  * file_fingerprints(...) - per-path content hashes (`file:<relpath>`). Finer than
                             whole-repo git: only listed files invalidate.
  * mentioned_paths(task)  - heuristic extraction of path-like tokens from the ask,
                             so file deps can track what the task actually names.
  * file_deps(path)        - a JSON file of {name: fingerprint} maintained by any
                             sidecar (a file watcher, a git hook, the MCP bridge).

Merge order (resolve_deps): deps-file, then git (coarse and/or file-level), then
request header - the most explicit source wins on key collisions.

Scope helpers:
  * scope_deps(...) adds `model` / `workspace` keys so different models never share
    entries and multi-tenant caches stay separated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import warnings
from typing import Iterable, Optional

_CACHE: dict = {}  # source key -> (expires_at, value)
_TTL = 2.0  # seconds; keeps subprocess/IO cost off the request hot path
_WARNED: set[str] = set()

# path-like tokens: foo/bar.py, src/a-b/c.toml, ./pkg/mod.rs (not bare "v1.2.3")
_PATH_RE = re.compile(
    r"(?<![\w./-])((?:\./|\.\./)?(?:[\w.+-]+/)+\.?[\w.+-]+\.[\w]{1,12})(?![\w./-])"
)


def _cached(key: str, compute):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    val = compute()
    _CACHE[key] = (now + _TTL, val)
    return val


def _warn_once(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(msg, stacklevel=3)


def git_fingerprint(repo: str) -> dict:
    """{'git:<name>': '<HEAD12>+<dirty8>'} for the working tree at `repo`.
    Empty dict if `repo` isn't a git checkout (degrades to no signal, loudly once)."""

    def compute():
        try:
            head = subprocess.run(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if not head:
                _warn_once(
                    "git:" + repo,
                    f"YORO: git fingerprint empty for {repo!r}; workspace invalidation disabled",
                )
                return {}
            dirty = subprocess.run(
                ["git", "-C", repo, "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            fp = head[:12] + "+" + hashlib.sha256(dirty.encode()).hexdigest()[:8]
            name = "git:" + (os.path.basename(os.path.abspath(repo)) or "repo")
            return {name: fp}
        except Exception as e:
            _warn_once(
                "git:" + repo,
                f"YORO: git fingerprint failed for {repo!r} ({e}); workspace invalidation disabled",
            )
            return {}

    return _cached("git:" + repo, compute)


def content_fingerprint(data: bytes | str) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()[:12]


def file_fingerprint(path: str) -> Optional[str]:
    """SHA-256 prefix of a single file's contents, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            return content_fingerprint(f.read())
    except OSError:
        return None


def file_fingerprints(
    paths: Iterable[str], root: str = "", *, prefix: str = "file:"
) -> dict:
    """Per-file content fingerprints. Paths may be absolute or relative to `root`.
    Missing files get fingerprint `missing` so appearance/disappearance is a signal."""
    out: dict = {}
    root = os.path.abspath(root) if root else ""
    for p in paths:
        p = p.strip()
        if not p:
            continue
        full = p if os.path.isabs(p) else (os.path.join(root, p) if root else p)
        rel = os.path.relpath(full, root) if root else p
        key = prefix + rel.replace("\\", "/")
        fp = file_fingerprint(full)
        out[key] = fp if fp is not None else "missing"
    return out


def mentioned_paths(text: str) -> list[str]:
    """Heuristic: path-like tokens in the task text (foo/bar.py, ./src/a.toml)."""
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for m in _PATH_RE.finditer(text):
        p = m.group(1)
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def file_deps(path: str) -> dict:
    """{name: fingerprint} from a sidecar-maintained JSON file; {} if absent/invalid."""

    def compute():
        try:
            with open(path) as f:
                data = json.load(f)
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except Exception as e:
            _warn_once(
                "file:" + path,
                f"YORO: deps-file {path!r} unreadable ({e}); sidecar invalidation disabled",
            )
            return {}

    return _cached("file:" + path, compute)


def scope_deps(
    deps: Optional[dict] = None,
    *,
    model: str = "",
    workspace: str = "",
) -> dict:
    """Attach model / workspace scope so entries never cross those boundaries."""
    out = dict(deps or {})
    if model:
        out["model"] = str(model)
    if workspace:
        out["workspace"] = str(workspace)
    return out


def resolve_deps(
    header_deps: dict,
    git_repo: str = "",
    deps_file: str = "",
    *,
    git_mode: str = "repo",
    task: str = "",
    watch_paths: Optional[list[str]] = None,
    model: str = "",
    workspace: str = "",
) -> dict:
    """Merge configured sources; request header is most explicit and wins.

    git_mode:
      * "repo"      — coarse HEAD+dirty fingerprint (default; any edit invalidates)
      * "mentioned" — only paths mentioned in `task` under git_repo (finer)
      * "watch"     — only paths in `watch_paths` under git_repo
      * "off"       — no git-derived signal
    """
    out: dict = {}
    if deps_file:
        sidecar = file_deps(deps_file)
        out.update(sidecar)
        # Persist source health as a dependency.  If a healthy sidecar later becomes
        # unreadable, "ok" -> "unavailable" invalidates every case that relied on it.
        out["source:deps-file:" + os.path.abspath(deps_file)] = (
            "ok" if sidecar or _valid_empty_deps_file(deps_file) else "unavailable"
        )

    mode = (git_mode or "repo").lower()
    if git_repo and mode not in ("off", "none", "0", "false"):
        git_values: dict = {}
        if mode == "mentioned":
            paths = mentioned_paths(task)
            if paths:
                git_values = file_fingerprints(paths, root=git_repo)
            # if the task names no files, fall back to coarse git so we still have a signal
            else:
                git_values = git_fingerprint(git_repo)
        elif mode == "watch":
            paths = list(watch_paths or [])
            if paths:
                git_values = file_fingerprints(paths, root=git_repo)
            else:
                git_values = git_fingerprint(git_repo)
        else:
            git_values = git_fingerprint(git_repo)
        out.update(git_values)
        out["source:git:" + os.path.abspath(git_repo)] = "ok" if git_values else "unavailable"

    out.update(header_deps or {})
    # Key-set coverage is itself a signal: dropping one explicitly reported dependency
    # must invalidate even when partial-dependency compatibility is enabled.
    out["source:request-deps"] = content_fingerprint(
        json.dumps(sorted(str(k) for k in (header_deps or {})))
    )
    return scope_deps(out, model=model, workspace=workspace)


def _valid_empty_deps_file(path: str) -> bool:
    try:
        with open(path) as f:
            return json.load(f) == {}
    except Exception:
        return False
