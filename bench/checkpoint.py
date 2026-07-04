"""Checkpoint + resume — so a Vast.ai spot preemption or the budget auto-shutdown never
loses work. A checkpoint is a JSON snapshot of run state (cursor into the prompt stream,
per-rung outcomes so far, cache contents, accumulated spend). It's written ATOMICALLY
(temp file + os.replace, so a kill mid-write can't corrupt it) and mirrored to S3, so a
fresh instance can pull the latest and continue from the cursor.

Usage:
    ck = Checkpoint("runs/phase0/ckpt.json", s3=("my-bucket", "yoro/phase0/ckpt.json"))
    state = ck.load() or {"cursor": 0, "outcomes": {}}
    ...                      # run from state["cursor"], periodically:
    ck.save(state)
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional, Tuple


def _s3():
    try:
        import boto3
        return boto3.client("s3")
    except Exception:
        return None


class Checkpoint:
    def __init__(self, path: str, s3: Optional[Tuple[str, str]] = None):
        self.path = path
        self.s3 = s3                                     # (bucket, key) or None
        self._c = _s3() if s3 else None

    def save(self, state: dict) -> None:
        d = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, default=str)
            os.replace(tmp, self.path)                   # atomic on POSIX
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if self._c:
            try:
                self._c.upload_file(self.path, self.s3[0], self.s3[1])
            except Exception as e:
                print(f"[ckpt s3 err {str(e)[:70]}]")

    def load(self) -> Optional[dict]:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass                                     # fall through to S3
        if self._c:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
                self._c.download_file(self.s3[0], self.s3[1], self.path)   # boto3 won't mkdir the target dir
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None
