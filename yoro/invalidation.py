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

Dep-signal policy (require_signal / strict_deps):
  * A case that was stored *with* deps is never treated as fresh when the request
    carries no signal — empty current deps used to silently disable invalidation.
  * `strict_deps` requires every stored key to be present and matching (full
    coverage both ways on the case's keys). Default is off so partial request
    signals still work for incremental adoption.
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
        require_signal: bool = True,
        strict_deps: bool = False,
    ):
        self.use_deps = use_deps
        self.use_ttl = use_ttl
        self.use_reliability = use_reliability
        self.fail_ratio = fail_ratio
        self.min_uses = min_uses
        # case has deps but request provides none -> refuse (never pretend invalidation works)
        self.require_signal = require_signal
        # every key on the case must appear in current_deps with the same fingerprint
        self.strict_deps = strict_deps

    def is_fresh(self, case, current_deps: Optional[dict] = None) -> bool:
        if self.use_deps:
            current = dict(current_deps or {})
            case_deps = dict(case.deps or {})
            if case_deps:
                if not current:
                    if self.require_signal:
                        return False  # scoped entry, no signal -> not safe to serve
                elif self.strict_deps:
                    for k, v in case_deps.items():
                        if current.get(k) != v:
                            return False
                else:
                    # request keys must match what was stored (extra request keys that
                    # the case never recorded count as a changed world)
                    for k, v in current.items():
                        if case_deps.get(k) != v:
                            return False
            elif current:
                # case was stored with no deps; any request signal is new context
                for k, v in current.items():
                    if case_deps.get(k) != v:
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
