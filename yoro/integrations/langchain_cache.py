"""YORO as a LangChain / LangGraph LLM cache.

The same slot semantic caches plug into (`langchain_core.caches.BaseCache` →
`set_llm_cache`), backed by the YORO engine: novelty gate against force-fits,
dependency invalidation against stale serves.

    from langchain_core.globals import set_llm_cache
    from yoro.integrations.langchain_cache import YoroLangChainCache

    set_llm_cache(YoroLangChainCache(git_repo="."))   # and/or deps_file=..., deps_source=...

The `llm_string` (model + params) is stored as a dependency fingerprint, so two
different models never serve each other's entries even when prompts match.

Scope note: like any LLM-cache slot this can serve and invalidate but not call the
model, so the replay tier is not available here; a changed dependency simply misses
(correct, never stale). For replay, put `yoro serve` in front of the endpoint.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Optional

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads

from ..cache import ReasoningCache
from ..deps import resolve_deps
from ..invalidation import Invalidator
from ..matcher import Decision, Matcher


class YoroLangChainCache(BaseCache):
    """LangChain BaseCache backed by the YORO engine (gate + dependency invalidation)."""

    def __init__(self, embedder=None, tau_hit: float = 0.95, tau_miss: float = 0.6,
                 cache_path: Optional[str] = None,
                 git_repo: str = "", deps_file: str = "", deps_source=None):
        if embedder is None:
            from ..embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        self.embedder = embedder
        self.store = ReasoningCache(cache_path)
        if cache_path:
            self.store.load()
        self.matcher = Matcher(tau_hit=tau_hit, tau_miss=tau_miss, novelty_gate=True)
        self.invalidator = Invalidator(use_deps=True, use_ttl=False, use_reliability=False)
        self.git_repo = git_repo
        self.deps_file = deps_file
        self.deps_source = deps_source
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _deps(self, llm_string: str) -> dict:
        extra = self.deps_source() if callable(self.deps_source) else {}
        deps = resolve_deps(extra or {}, git_repo=self.git_repo, deps_file=self.deps_file)
        deps["llm"] = hashlib.sha256(llm_string.encode()).hexdigest()[:12]
        return deps

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        emb = self.embedder.embed(prompt)
        with self._lock:
            case, sim = self.store.nearest(emb)
            if case is None:
                self.misses += 1
                return None
            fresh = self.invalidator.is_fresh(case, self._deps(llm_string))
            if self.matcher.decide(sim, fresh) == Decision.HIT:
                self.hits += 1
                self.store.record_use(case, True)
                return loads(case.outcome)
            self.misses += 1
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        emb = self.embedder.embed(prompt)
        payload = dumps(list(return_val))
        with self._lock:
            self.store.add(prompt, emb, payload, payload, self._deps(llm_string))
            self.store.save()

    def clear(self, **kwargs: Any) -> None:
        with self._lock:
            self.store.cases = []
            self.store._E = None
            self.store.save()

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        return self.lookup(prompt, llm_string)

    async def aupdate(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        return self.update(prompt, llm_string, return_val)

    async def aclear(self, **kwargs: Any) -> None:
        return self.clear(**kwargs)
