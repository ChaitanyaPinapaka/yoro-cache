"""YORO as a LiteLLM cache plugin.

Drop the YORO engine into LiteLLM's pluggable cache slot: semantic matching with a
novelty gate (no force-fits) and dependency invalidation (no stale serves).

LiteLLM's read path forwards only `messages` to cache backends, so the CURRENT
dependency fingerprints come from an environment source configured on the plugin
(a git working tree, a sidecar deps-file, or any callable). Entries can be scoped
further at write time via metadata={"yoro_deps": {...}}.

    import litellm
    from litellm.caching.caching import Cache
    from yoro.integrations.litellm_cache import YoroSemanticCache

    litellm.cache = Cache()                      # any type; the backend is replaced:
    litellm.cache.cache = YoroSemanticCache(     # YORO takes over storage + matching
        git_repo=".",                            # and/or deps_file=..., deps_source=...
    )

Scope note: a cache layer can serve and invalidate, but it cannot call the model,
so the replay tier is not available here; entries whose dependencies changed simply
miss (correct, never stale). For replay, put `yoro serve` in front of the endpoint.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

from litellm.caching.base_cache import BaseCache

from ..cache import ReasoningCache
from ..deps import resolve_deps
from ..engine import lookup as engine_lookup
from ..invalidation import Invalidator
from ..matcher import Decision, Matcher


def _task_of(kwargs: dict) -> str:
    for m in reversed(kwargs.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict)).strip()
    return ""


def _deps_of(kwargs: dict) -> dict:
    for holder in (kwargs.get("metadata"), (kwargs.get("litellm_params") or {}).get("metadata")):
        if isinstance(holder, dict) and isinstance(holder.get("yoro_deps"), dict):
            return {str(k): str(v) for k, v in holder["yoro_deps"].items()}
    return {}


class YoroSemanticCache(BaseCache):
    """LiteLLM BaseCache backed by the YORO engine (gate + dependency invalidation)."""

    def __init__(self, embedder=None, tau_hit: float = 0.95, tau_miss: float = 0.6,
                 cache_path: Optional[str] = None, default_ttl: int = 60,
                 git_repo: str = "", deps_file: str = "", deps_source=None,
                 git_mode: str = "repo", workspace: str = "",
                 max_cases: Optional[int] = None, flush_every: int = 1):
        super().__init__(default_ttl=default_ttl)
        self.git_repo = git_repo
        self.deps_file = deps_file
        self.deps_source = deps_source
        self.git_mode = git_mode
        self.workspace = workspace
        if embedder is None:
            from ..embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        self.embedder = embedder
        self.store = ReasoningCache(
            cache_path, max_cases=max_cases, flush_every=flush_every
        )
        if cache_path:
            self.store.load()
        self.matcher = Matcher(tau_hit=tau_hit, tau_miss=tau_miss, novelty_gate=True)
        self.invalidator = Invalidator(
            use_deps=True, use_ttl=False, use_reliability=False, require_signal=True
        )
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _current_deps(self, task: str = "", model: str = "") -> dict:
        extra = self.deps_source() if callable(self.deps_source) else {}
        return resolve_deps(
            extra or {},
            git_repo=self.git_repo,
            deps_file=self.deps_file,
            git_mode=self.git_mode,
            task=task,
            model=model,
            workspace=self.workspace,
        )

    # ---- LiteLLM sync interface ----
    def get_cache(self, key, **kwargs):
        task = _task_of(kwargs)
        if not task:
            return None
        model = str(kwargs.get("model") or "")
        emb = self.embedder.embed(task)
        with self._lock:
            found = engine_lookup(
                self.store,
                self.matcher,
                self.invalidator,
                emb,
                self._current_deps(task=task, model=model),
            )
            if found.decision == Decision.HIT and found.case is not None:
                self.hits += 1
                self.store.record_use(found.case, True)
                return json.loads(found.case.outcome)
            self.misses += 1
            return None

    def set_cache(self, key, value, **kwargs):
        task = _task_of(kwargs)
        if not task:
            return
        model = str(kwargs.get("model") or "")
        emb = self.embedder.embed(task)
        payload = json.dumps(value, default=str)
        deps = {**self._current_deps(task=task, model=model), **_deps_of(kwargs)}
        with self._lock:
            self.store.add(task, emb, payload, payload, deps)
            self.store.flush()

    # ---- async delegates (the engine is in-process and lock-guarded) ----
    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key, **kwargs)

    async def async_set_cache(self, key, value, **kwargs):
        return self.set_cache(key, value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        for key, value in cache_list:
            self.set_cache(key, value, **kwargs)

    def batch_cache_write(self, key, value, **kwargs):
        self.set_cache(key, value, **kwargs)

    def disconnect(self):
        pass
