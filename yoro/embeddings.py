"""Pluggable embedders: task text -> unit vector.

The matcher's whole job is "is this new task close to one I've already reasoned
about?" — which is a similarity question in embedding space. YORO never assumes a
particular embedder, so you can trade fidelity for speed:

  * HashEmbedder              - deterministic, fast, zero model. Right for tests
                               and synthetic workloads.
  * SentenceTransformerEmbedder - real semantics. Use for real reasoning tasks.
  * OpenAIEmbedder           - any /v1/embeddings server (incl. a local one).
"""

from __future__ import annotations

import hashlib

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Vectors are unit-normed at creation, so dot == cosine similarity."""
    return float(np.dot(a, b))


class Embedder:
    dim: int

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Signed feature-hashing of tokens. Shared tokens -> high similarity, disjoint
    tokens -> ~0. Reproducible and model-free — runs in seconds with no GPU,
    which makes it right for tests and offline experiments."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h // self.dim) % 2 == 0 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class SentenceTransformerEmbedder(Embedder):
    """Real semantic embeddings (`pip install "yoro-cache[embed]"`). The default
    model is small + fast and runs on CPU, CUDA, or Apple MPS."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy

        self._m = SentenceTransformer(model, device=device)
        self.dim = self._m.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(
            self._m.encode(text, normalize_embeddings=True), dtype=np.float32
        )


class OpenAIEmbedder(Embedder):
    """Any OpenAI-compatible /v1/embeddings endpoint (llama-server, vLLM, ...).

    Extension point — the proxy defaults to SentenceTransformerEmbedder; use this to
    keep embeddings on a server you already run."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "text-embedding",
        api_key: str = "sk-local",
        dim: int = 1024,
    ):
        import requests  # lazy

        self._requests = requests
        self.base_url, self.model, self.api_key, self.dim = (
            base_url.rstrip("/"),
            model,
            api_key,
            dim,
        )

    def embed(self, text: str) -> np.ndarray:
        r = self._requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text},
            timeout=120,
        )
        r.raise_for_status()
        v = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
