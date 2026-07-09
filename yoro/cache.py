"""The data structure at the heart of YORO (You Only Reason Once).

A `ReasoningCase` is one memoized reasoning episode: the task, its embedding, the
reasoning trace the model produced, the outcome, and the *dependency fingerprints*
the reasoning relied on (so we can tell when it goes stale). `ReasoningCache` is a
versioned, embedding-indexed store of those cases with brute-force cosine nearest-
neighbor (fine for thousands of cases; swap in FAISS/hnswlib for millions).

Persistence backends:
  * JSON  (default for `.json` paths) — human-readable, portable
  * SQLite (`.sqlite` / `.db` paths, or `backend="sqlite"`) — better for larger stores

Write-behind: `flush_every=N` batches N mutations before disk I/O (1 = sync every
write, the historical default). Call `flush()` / `close()` to force a write.

Eviction: `max_cases` caps memory; lowest (uses, updated_at) entries are dropped first.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ReasoningCase:
    id: str
    task: str
    embedding: np.ndarray  # unit vector
    reasoning: str  # the cached reasoning trace (text or serialized plan/graph)
    outcome: str  # the result/decision the reasoning produced
    deps: dict = field(default_factory=dict)  # {name: fingerprint} for invalidation
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # seconds; None = no time-based expiry
    uses: int = 0
    successes: int = 0
    failures: int = 0
    steps: list = field(default_factory=list)  # structured form of `reasoning` (see structured.py)

    def reliability(self) -> float:
        return self.successes / self.uses if self.uses else 1.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningCase":
        d = dict(d)
        d["embedding"] = np.asarray(d["embedding"], dtype=np.float32)
        # tolerate older payloads missing optional fields
        d.setdefault("steps", [])
        d.setdefault("deps", {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _detect_backend(path: Optional[str], backend: Optional[str]) -> str:
    if backend:
        return backend.lower()
    if not path:
        return "json"
    lower = path.lower()
    if lower.endswith((".sqlite", ".sqlite3", ".db")):
        return "sqlite"
    return "json"


class ReasoningCache:
    def __init__(
        self,
        path: Optional[str] = None,
        max_cases: Optional[int] = None,
        flush_every: int = 1,
        backend: Optional[str] = None,
    ):
        self.cases: list[ReasoningCase] = []
        self.path = path
        self.max_cases = max_cases
        self.flush_every = max(1, int(flush_every))
        self.backend = _detect_backend(path, backend)
        self._E = None  # cached embedding matrix; rebuilt lazily after any write
        self._dirty = 0  # mutations since last successful save
        self._evicted = 0

    # ---- writes ----
    def add(
        self, task, embedding, reasoning, outcome, deps=None, ttl=None
    ) -> ReasoningCase:
        c = ReasoningCase(
            id=uuid.uuid4().hex[:12],
            task=task,
            embedding=np.asarray(embedding, dtype=np.float32),
            reasoning=reasoning,
            outcome=outcome,
            deps=dict(deps or {}),
            ttl=ttl,
        )
        self.cases.append(c)
        self._E = None
        self._mark_dirty()
        self._evict_if_needed()
        self._maybe_flush()
        return c

    def update(
        self, case_id, task, embedding, reasoning, outcome, deps=None
    ) -> ReasoningCase:
        """Re-reason an existing case: bump version, refresh content + freshness,
        and reset reliability counters (it's effectively a new belief)."""
        c = self.get(case_id)
        c.task = task
        c.embedding = np.asarray(embedding, dtype=np.float32)
        c.reasoning = reasoning
        c.outcome = outcome
        if deps is not None:
            c.deps = dict(deps)
        c.version += 1
        c.updated_at = time.time()
        c.uses = c.successes = c.failures = 0
        self._E = None
        self._mark_dirty()
        self._maybe_flush()
        return c

    def record_use(self, case: ReasoningCase, success: bool) -> None:
        case.uses += 1
        if success:
            case.successes += 1
        else:
            case.failures += 1
        # mark dirty so flush/close persists counters, but do not flush on the HIT
        # hot path (would rewrite the whole store on every reuse)
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty += 1

    def _evict_if_needed(self) -> None:
        if not self.max_cases or self.max_cases <= 0:
            return
        while len(self.cases) > self.max_cases:
            # drop least-used, then oldest updated
            i = min(
                range(len(self.cases)),
                key=lambda j: (self.cases[j].uses, self.cases[j].updated_at),
            )
            self.cases.pop(i)
            self._evicted += 1
            self._E = None

    def _maybe_flush(self) -> None:
        if self.path and self._dirty >= self.flush_every:
            self.save()

    # ---- reads ----
    def nearest(self, embedding) -> tuple[Optional[ReasoningCase], float]:
        if not self.cases:
            return None, -1.0
        q = np.asarray(embedding, dtype=np.float32)
        if self._E is None or self._E.shape[0] != len(self.cases):
            self._E = np.stack([c.embedding for c in self.cases])
        sims = self._E @ q
        i = int(np.argmax(sims))
        return self.cases[i], float(sims[i])

    def get(self, case_id) -> ReasoningCase:
        for c in self.cases:
            if c.id == case_id:
                return c
        raise KeyError(case_id)

    # ---- persistence ----
    def flush(self) -> None:
        """Force a disk write if there are pending mutations."""
        if self.path and self._dirty:
            self.save()

    def close(self) -> None:
        self.flush()

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.path
        if not p:
            return
        backend = _detect_backend(p, self.backend if path is None else None)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        if backend == "sqlite":
            self._save_sqlite(p)
        else:
            self._save_json(p)
        self._dirty = 0

    def _save_json(self, p: str) -> None:
        tmp = p + ".tmp"  # atomic: a concurrent reader/crash never sees a torn file
        with open(tmp, "w") as f:
            json.dump([c.to_dict() for c in self.cases], f, indent=2)
        os.replace(tmp, p)

    def _save_sqlite(self, p: str) -> None:
        tmp = p + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        conn = sqlite3.connect(tmp)
        try:
            conn.execute(
                "CREATE TABLE cases (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO cases (id, payload) VALUES (?, ?)",
                [(c.id, json.dumps(c.to_dict())) for c in self.cases],
            )
            conn.commit()
        finally:
            conn.close()
        os.replace(tmp, p)

    def load(self, path: Optional[str] = None) -> "ReasoningCache":
        p = path or self.path
        if p and os.path.exists(p):
            backend = _detect_backend(p, self.backend if path is None else None)
            if backend == "sqlite":
                self._load_sqlite(p)
            else:
                self._load_json(p)
            self._E = None
            self._dirty = 0
        return self

    def _load_json(self, p: str) -> None:
        with open(p) as f:
            self.cases = [ReasoningCase.from_dict(d) for d in json.load(f)]

    def _load_sqlite(self, p: str) -> None:
        conn = sqlite3.connect(p)
        try:
            rows = conn.execute("SELECT payload FROM cases").fetchall()
        finally:
            conn.close()
        self.cases = [ReasoningCase.from_dict(json.loads(r[0])) for r in rows]

    def __len__(self) -> int:
        return len(self.cases)

    def __enter__(self) -> "ReasoningCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
