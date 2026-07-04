"""The data structure at the heart of YORO (You Only Reason Once).

A `ReasoningCase` is one memoized reasoning episode: the task, its embedding, the
reasoning trace the model produced, the outcome, and the *dependency fingerprints*
the reasoning relied on (so we can tell when it goes stale). `ReasoningCache` is a
versioned, embedding-indexed store of those cases with brute-force cosine nearest-
neighbor (fine for thousands of cases; swap in FAISS/hnswlib for millions) and a
single-JSON-file persistence so a cache is human-readable and portable.
"""

from __future__ import annotations

import json
import os
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
        return cls(**d)


class ReasoningCache:
    def __init__(self, path: Optional[str] = None):
        self.cases: list[ReasoningCase] = []
        self.path = path
        self._E = None  # cached embedding matrix; rebuilt lazily after any write

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
        return c

    def record_use(self, case: ReasoningCase, success: bool) -> None:
        case.uses += 1
        if success:
            case.successes += 1
        else:
            case.failures += 1

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

    # ---- persistence (single readable JSON) ----
    def save(self, path: Optional[str] = None) -> None:
        p = path or self.path
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"  # atomic: a concurrent reader/crash never sees a torn file
        with open(tmp, "w") as f:
            json.dump([c.to_dict() for c in self.cases], f, indent=2)
        os.replace(tmp, p)

    def load(self, path: Optional[str] = None) -> "ReasoningCache":
        p = path or self.path
        if p and os.path.exists(p):
            with open(p) as f:
                self.cases = [ReasoningCase.from_dict(d) for d in json.load(f)]
            self._E = None
        return self

    def __len__(self) -> int:
        return len(self.cases)
