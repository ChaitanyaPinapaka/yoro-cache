"""BudgetGuard — the safety feature that makes a multi-day, unattended, $500 rented-GPU
run safe to leave alone.

It tracks spend = instance $/hr x elapsed + any per-token API cost, and once spend
crosses a soft fraction of the hard ceiling it fires a provider-specific shutdown hook
(e.g. `vastai destroy instance <id>`) exactly once. Pair it with frequent checkpointing
so the auto-shutdown (or a spot preemption) never loses results.

The clock is injectable so the logic is unit-testable without waiting hours.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class BudgetGuard:
    ceiling_usd: float                                  # HARD cap (e.g. 500)
    hourly_usd: float                                   # instance price, e.g. Vast.ai spot $/hr
    shutdown_frac: float = 0.9                          # auto-shutdown at 90% of the ceiling
    clock: Callable[[], float] = time.time             # injectable for tests
    on_shutdown: Optional[Callable[[], None]] = None    # provider terminate hook (fired once)
    token_cost_usd: float = 0.0                         # accrued per-token cost (0 when self-hosting)
    started_at: Optional[float] = None
    _stopped: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = self.clock()

    def add_token_cost(self, usd: float) -> None:
        self.token_cost_usd += max(0.0, usd)

    def spent(self) -> float:
        hours = max(0.0, (self.clock() - self.started_at) / 3600.0)
        return hours * self.hourly_usd + self.token_cost_usd

    def remaining(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent())

    def soft_cap(self) -> float:
        return self.ceiling_usd * self.shutdown_frac

    def should_stop(self) -> bool:
        return self.spent() >= self.soft_cap()

    def check(self) -> bool:
        """Call this periodically (e.g. every checkpoint). Returns True once spend crosses
        the soft cap, firing the shutdown hook exactly once. Idempotent thereafter."""
        if self._stopped:
            return True
        if self.should_stop():
            self._stopped = True
            if self.on_shutdown is not None:
                try:
                    self.on_shutdown()
                except Exception:
                    pass
            return True
        return False

    def stop(self) -> None:
        """Force-stop from an EXTERNAL signal (e.g. low real Vast credit), so a sweep that
        shares this guard also halts — not just the current level."""
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def status(self) -> dict:
        return {
            "spent_usd": round(self.spent(), 2),
            "remaining_usd": round(self.remaining(), 2),
            "ceiling_usd": self.ceiling_usd,
            "soft_cap_usd": round(self.soft_cap(), 2),
            "hourly_usd": self.hourly_usd,
            "stopped": self._stopped,
        }
