"""Optional approximate nearest-neighbor index, partitioned by exact scope."""
from __future__ import annotations

import json
import numpy as np


class HNSWIndex:
    def __init__(self, ef_search: int = 64, ef_construction: int = 200, m: int = 16):
        import hnswlib
        self._hnswlib = hnswlib
        self.ef_search, self.ef_construction, self.m = ef_search, ef_construction, m
        self._parts = {}

    @staticmethod
    def _key(scope) -> str:
        return json.dumps(scope, sort_keys=True, separators=(",", ":"))

    def search(self, cases, query, scope=None):
        eligible = [(i, c) for i, c in enumerate(cases) if scope is None or c.scope == scope]
        if not eligible:
            return None, -1.0
        key = self._key(scope)
        signature = tuple((c.id, c.version, c.updated_at) for _, c in eligible)
        cached = self._parts.get(key)
        if cached is None or cached[0] != signature:
            matrix = np.stack([c.embedding for _, c in eligible]).astype(np.float32)
            index = self._hnswlib.Index(space="cosine", dim=matrix.shape[1])
            index.init_index(max_elements=len(eligible), ef_construction=self.ef_construction, M=self.m)
            index.add_items(matrix, np.arange(len(eligible)))
            index.set_ef(min(self.ef_search, len(eligible)))
            cached = (signature, index, eligible)
            self._parts[key] = cached
        _, index, ordered = cached
        labels, distances = index.knn_query(np.asarray(query, dtype=np.float32), k=1)
        pos = int(labels[0][0])
        return ordered[pos][1], 1.0 - float(distances[0][0])

    def clear(self):
        self._parts.clear()
