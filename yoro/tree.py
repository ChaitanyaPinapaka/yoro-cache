"""Decision-tree view of the reasoning cache (the 'decision tree' representation).

The matcher routes by nearest-neighbor in embedding space. An alternative view
is a *searchable decision tree*: fit a shallow tree that maps an
embedding to the cached case it should reuse. Two uses:

  * inspection - `export_text()` shows, in plain words, how tasks get routed to
                 reasoning (auditable: "reasoning is sticky and you can see why").
  * routing    - `route(emb)` returns a case_id, a fast O(depth) alternative to
                 brute-force NN once the cache is large.

Built with scikit-learn. Refit periodically (it's cheap), not every insert.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ReasoningTreeRouter:
    """Inspection/extension utility — not used by the proxy's request path."""

    def __init__(self, max_depth: int = 8):
        # sklearn only loads if you actually use the tree view — the core install stays light
        from sklearn.tree import DecisionTreeClassifier  # noqa: F401 (import check)

        self.max_depth = max_depth
        self.tree = None

    def fit(self, cache) -> "ReasoningTreeRouter":
        """Train on the cache's own cases (embedding -> case id). With one example
        per case it memorizes the routing; feed historical (task, case) pairs for a
        generalizing tree."""
        if len(cache) < 2:
            self.tree = None
            return self
        from sklearn.tree import DecisionTreeClassifier

        X = np.stack([c.embedding for c in cache.cases])
        y = [c.id for c in cache.cases]
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=0)
        self.tree.fit(X, y)
        return self

    def route(self, embedding) -> Optional[str]:
        if self.tree is None:
            return None
        return str(
            self.tree.predict(np.asarray(embedding, dtype=np.float32)[None, :])[0]
        )

    def export_text(self) -> str:
        if self.tree is None:
            return "(tree empty: need >=2 cached cases)"
        from sklearn.tree import export_text

        feats = [f"dim{i}" for i in range(self.tree.n_features_in_)]
        return export_text(self.tree, feature_names=feats)
