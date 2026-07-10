"""Application-side dependency capture for library and upstream integrations."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Iterator

from .deps import content_fingerprint, file_fingerprint


class DependencyTracker(Mapping):
    """Collect facts actually read while producing an answer.

    Pass ``dict(tracker)`` as ``current_deps`` or emit ``tracker.header()`` as the
    upstream ``X-YORO-Deps`` response header. Values are content fingerprints, never
    raw potentially-sensitive content.
    """
    def __init__(self):
        self._deps: dict[str, str] = {}

    def add(self, name: str, value) -> str:
        raw = value if isinstance(value, (bytes, str)) else json.dumps(
            value, sort_keys=True, default=str
        )
        fp = content_fingerprint(raw)
        self._deps[str(name)] = fp
        return fp

    def file(self, path: str) -> str:
        fp = file_fingerprint(path) or "missing"
        self._deps["file:" + path] = fp
        return fp

    def resource(self, uri: str, content) -> str:
        return self.add("mcp:" + uri, content)

    def query(self, system: str, query: str, result) -> str:
        key = hashlib.sha256(query.encode()).hexdigest()[:12]
        return self.add(f"query:{system}:{key}", result)

    def url(self, url: str, content) -> str:
        return self.add("url:" + url, content)

    def tool_output(self, tool: str, arguments, result) -> str:
        key = hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return self.add(f"tool:{tool}:{key}", result)

    def header(self) -> str:
        return json.dumps(self._deps, sort_keys=True, separators=(",", ":"))

    def __getitem__(self, key): return self._deps[key]
    def __iter__(self) -> Iterator[str]: return iter(self._deps)
    def __len__(self) -> int: return len(self._deps)
