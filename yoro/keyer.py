"""Keyers — normalize a task BEFORE it's embedded, so 'reuse anywhere' isn't fooled
by surface wording. Because the embedding is the cache key, keying the *text* keys
the cache (surface-invariance: the same ask in different words reuses the same entry).

  IdentityKeyer - embed the raw task (the default).
  ModelKeyer    - ask an LLM to CANONICALIZE the task: strip phrasing/filler but
                  PRESERVE every number/entity/unit/constraint that determines the
                  answer. So "What is 6 factorial?" and "compute 6!" -> same key
                  (reuse), while "5 factorial" stays distinct (no near-miss collision).
                  Memoized: one model call per distinct task, then free on repeats.

Prompt design notes (a reasoning model fights canonicalization):
  * a distinct CANON: marker (an ANSWER: marker makes the model *solve* the task);
  * few-shot examples that keep the number (so 5! and 6! don't collide);
  * "do NOT solve"; and robust parsing (strip any <think>, take the last CANON:).
"""

from __future__ import annotations


class Keyer:
    def key(self, task: str) -> str:
        raise NotImplementedError


class IdentityKeyer(Keyer):
    def key(self, task: str) -> str:
        return task


class ModelKeyer(Keyer):
    """Extension point — not used by the proxy. Plug in via `YORO(keyer=ModelKeyer(...))`
    when surface wording varies enough that raw-text embeddings miss reuse."""

    PROMPT = (
        "Rewrite the TASK as a minimal canonical cache key. Do NOT solve it. Keep every "
        "number, entity, unit, and constraint that affects the answer; drop wording and "
        "filler. Normalize spelled-out numbers to digits (e.g. 'one hundred' -> 100) and "
        "synonym phrasings to one canonical form (e.g. 'add up'/'total of' -> 'sum'). "
        "Same-answer tasks must give the SAME key; different-answer tasks must "
        "give DIFFERENT keys. Reply with exactly one line: CANON: <key>\n\n"
        "Examples:\n"
        "TASK: What is 6 factorial?\nCANON: factorial of 6\n"
        "TASK: Compute 5! (factorial).\nCANON: factorial of 5\n"
        "TASK: Add together all the numbers from one to ten.\nCANON: sum of integers from 1 to 10\n"
        "TASK: A train travels at 60 mph for 2.5 hours. How far does it go?\n"
        "CANON: distance traveled at 60 mph for 2.5 hours\n"
        "TASK: How many minutes are in 4.5 hours?\nCANON: minutes in 4.5 hours\n\n"
        "TASK: {task}"
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "local",
        max_tokens: int = 1024,
        api_key: str = "sk-local",
        cache: dict | None = None,
    ):
        import re

        import requests  # lazy

        self._requests = requests
        self._re = re
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._cache = {} if cache is None else cache
        self.calls = 0

    def key(self, task: str) -> str:
        if task in self._cache:
            return self._cache[task]
        self.calls += 1
        r = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "user", "content": self.PROMPT.format(task=task)}
                ],
            },
            timeout=120,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"].get("content") or ""
        if "</think>" in content:
            content = content.split("</think>")[-1]
        hits = self._re.findall(r"CANON:\s*(.+)", content)
        if hits:
            canon = hits[-1].strip()
        elif content.strip():
            canon = content.strip().splitlines()[-1].strip()  # last-line fallback
        else:
            canon = task
        self._cache[task] = canon or task
        return self._cache[task]
