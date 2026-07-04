"""Invalidator — when has cached reasoning gone stale?

Reasoning is only safe to reuse while the world it assumed still holds. Three
independent triggers, each independently toggleable:

  * deps        - dependency fingerprints changed (the facts/files/premises the
                  reasoning relied on were edited). The strongest signal.
  * ttl         - the case is older than its time-to-live.
  * reliability - the case has started producing failures on reuse (needs a
                  verifier in the loop to mark failures).

`is_fresh(case, current_deps) == False` flips a would-be HIT into an ESCALATE, so
the next use re-reasons and UPDATEs the case in place.
"""

from __future__ import annotations

import time
from typing import Optional


class Invalidator:
    def __init__(
        self,
        use_deps: bool = True,
        use_ttl: bool = True,
        use_reliability: bool = True,
        fail_ratio: float = 0.5,
        min_uses: int = 3,
    ):
        self.use_deps = use_deps
        self.use_ttl = use_ttl
        self.use_reliability = use_reliability
        self.fail_ratio = fail_ratio
        self.min_uses = min_uses

    def is_fresh(self, case, current_deps: Optional[dict] = None) -> bool:
        if self.use_deps and current_deps:
            for k, v in current_deps.items():
                if case.deps.get(k) != v:  # a premise changed -> stale
                    return False
        if self.use_ttl and case.ttl is not None:
            if time.time() - case.updated_at > case.ttl:
                return False
        if self.use_reliability and case.uses >= self.min_uses:
            if (
                1.0 - case.reliability()
            ) > self.fail_ratio:  # reuse keeps failing -> stale
                return False
        return True
