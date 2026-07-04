"""YORO benchmark harness.

A baseline ladder over labelled prompt streams, runnable locally with a mock model
(`--smoke`, no GPU) or against any OpenAI-compatible endpoint, with per-level
checkpoint/resume, optional cloud sinks (S3 / CloudWatch / W&B), and a hard
auto-shutdown budget cap for rented GPUs.

Modules:
  budget   - BudgetGuard: spend tracking + auto-shutdown before the ceiling.
  metrics  - per-prompt Outcome, run summary, cross-seed aggregation + significance.
  ladder   - the five rungs (no-cache / exact / gptcache-semantic / behaviors / YORO).
  wandb_log- thin W&B logger shim (falls back to stdout if wandb is absent).
"""
from .budget import BudgetGuard
from .metrics import Outcome, summarize, aggregate_seeds, paired_t
from .ladder import (Strategy, NoCache, ExactCache, SemanticCache, BehaviorsOnly,
                     YOROStrategy, build_ladder)
from .wandb_log import WandbLogger
from .eventlog import EventLog, S3FileSink, CloudWatchSink
from .checkpoint import Checkpoint
from .convergence import Convergence, ci_halfwidth
from .vast import VastCredit, stop_self

__all__ = [
    "BudgetGuard",
    "Outcome", "summarize", "aggregate_seeds", "paired_t",
    "Strategy", "NoCache", "ExactCache", "SemanticCache", "BehaviorsOnly", "YOROStrategy", "build_ladder",
    "WandbLogger",
    "EventLog", "S3FileSink", "CloudWatchSink",
    "Checkpoint",
    "Convergence", "ci_halfwidth",
    "VastCredit", "stop_self",
]
