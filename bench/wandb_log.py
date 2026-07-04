"""W&B logger shim — the live dashboard for the run.

Lazy-imports wandb so the harness has no hard dependency locally: if wandb is absent or
WANDB disabled, it falls back to stdout. On the cluster it streams per-step metrics
(hit-rate, accuracy, staleness, brittleness, $ spent, per-domain) so the run is watchable
in real time, which is exactly how an unattended multi-day run should be monitored.
"""
from __future__ import annotations

from typing import Optional


class WandbLogger:
    def __init__(self, project: str = "yoro-benchmark", name: Optional[str] = None,
                 config: Optional[dict] = None, enabled: bool = True):
        self.run = None
        self._step = 0
        if not enabled:
            print(f"[wandb off] project={project} name={name}")
            return
        try:
            import wandb
            self.run = wandb.init(project=project, name=name, config=config or {})
            print(f"[wandb] live at {getattr(self.run, 'url', '(local)')}")
        except Exception as e:                       # missing pkg / not logged in / offline
            print(f"[wandb unavailable -> stdout] ({type(e).__name__}: {str(e)[:80]})")

    def log(self, data: dict, step: Optional[int] = None) -> None:
        if step is None:
            step = self._step
            self._step += 1
        if self.run is not None:
            self.run.log(data, step=step)
        else:
            print(f"[step {step}] " + "  ".join(f"{k}={_fmt(v)}" for k, v in data.items()))

    def summary(self, data: dict) -> None:
        if self.run is not None:
            self.run.summary.update(data)
        else:
            print("[summary] " + "  ".join(f"{k}={_fmt(v)}" for k, v in data.items()))

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)
