"""YORO — You Only Reason Once.

A small, model-pluggable tool that memoizes an LLM's reasoning so it's computed
once and reused, addressing the three hard problems of reasoning reuse:

  1. matching      (fuzzy retrieval)        -> Matcher + Embedder
  2. invalidation  (staleness/update)       -> Invalidator + cache.update()
  3. brittleness   (don't force-fit novel)  -> Matcher.novelty_gate

See README.md for the design and the measured results.
"""

from .behaviors import Behavior, BehaviorStore, extract_behaviors, format_behaviors
from .cache import ReasoningCache, ReasoningCase
from .core import YORO, Result
from .embeddings import (
    Embedder,
    HashEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    cosine,
)
from .engine import Lookup, lookup
from .invalidation import Invalidator
from .keyer import IdentityKeyer, Keyer, ModelKeyer
from .matcher import Decision, Matcher
from .structured import StructuredReasoning, to_steps
from .tree import ReasoningTreeRouter

__all__ = [
    "ReasoningCase",
    "ReasoningCache",
    "Matcher",
    "Decision",
    "Invalidator",
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "cosine",
    "YORO",
    "Result",
    "Lookup",
    "lookup",
    "ReasoningTreeRouter",
    "Keyer",
    "IdentityKeyer",
    "ModelKeyer",
    "Behavior",
    "BehaviorStore",
    "extract_behaviors",
    "format_behaviors",
    "to_steps",
    "StructuredReasoning",
]
