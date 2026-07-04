"""Behaviors — Metacognitive-Reuse-style reusable reasoning fragments.

A Behavior is a NAMED, general sub-procedure mined from a reasoning trace
("arithmetic_series_sum: to add 1..n, use n(n+1)/2"). Unlike a ReasoningCase
(a whole task -> answer), a behavior helps DIFFERENT problems that share method —
so it extends reuse beyond same-task recurrence (the real 'reuse anywhere' win).

On YORO's COLD path: RETRIEVE relevant behaviors -> INJECT them into the prompt
(cheaper, more consistent reasoning) -> EXTRACT new behaviors from the fresh trace.
The HIT path is untouched (still a zero-call answer serve).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from .embeddings import cosine


@dataclass
class Behavior:
    id: str
    name: str
    instruction: str
    embedding: np.ndarray
    from_cases: list = field(default_factory=list)
    uses: int = 0
    successes: int = 0
    failures: int = 0
    created_at: float = field(default_factory=time.time)


class BehaviorStore:
    def __init__(self):
        self.items: list[Behavior] = []

    def add(
        self, name: str, instruction: str, embedding, from_case: str | None = None
    ) -> Behavior:
        name = name.strip()
        for x in self.items:  # dedup by name; merge provenance
            if x.name.lower() == name.lower():
                if from_case:
                    x.from_cases.append(from_case)
                return x
        b = Behavior(
            id=uuid.uuid4().hex[:12],
            name=name,
            instruction=instruction.strip(),
            embedding=np.asarray(embedding, dtype=np.float32),
            from_cases=[from_case] if from_case else [],
        )
        self.items.append(b)
        return b

    def retrieve(self, embedding, k: int = 3, tau: float = 0.25) -> list[Behavior]:
        if not self.items:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        scored = sorted(
            ((cosine(q, b.embedding), b) for b in self.items), key=lambda t: -t[0]
        )
        return [b for s, b in scored[:k] if s >= tau]

    def __len__(self):
        return len(self.items)


EXTRACT_PROMPT = (
    "From the REASONING below, extract 0-3 REUSABLE named procedures — general methods "
    "that would help solve OTHER, different problems (NOT facts specific to this one, NOT "
    "the final answer). For each, output exactly one line:\n"
    "BEHAVIOR: <short_snake_case_name> | <one concise imperative instruction>\n"
    "If there are no reusable procedures, output exactly: NONE\n\n"
    "REASONING:\n{trace}"
)


def extract_behaviors(
    trace: str, complete, embedder, store: BehaviorStore, from_case: str | None = None
) -> list[Behavior]:
    """`complete`: a callable(prompt:str)->str (a raw model completion). Parses BEHAVIOR: lines."""
    text = complete(EXTRACT_PROMPT.format(trace=trace[:4000])) or ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*BEHAVIOR:\s*(.+?)\s*\|\s*(.+)", line)
        if not m:
            continue
        name, instr = m.group(1).strip(), m.group(2).strip()
        if not name or not instr:
            continue
        emb = embedder.embed(f"{name}: {instr}")
        out.append(store.add(name, instr, emb, from_case))
    return out


def format_behaviors(behaviors: list[Behavior]) -> str:
    if not behaviors:
        return ""
    lines = "\n".join(f"- {b.name}: {b.instruction}" for b in behaviors)
    return f"You may use these known methods if relevant:\n{lines}\n\n"
