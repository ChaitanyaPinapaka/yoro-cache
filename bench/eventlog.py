"""Structured event log — the detailed, replayable record of a run.

EVERY error, progress tick, per-prompt result, raw model output, and metric is written as
one self-describing JSON line:
    {"t": epoch, "run": id, "seq": n, "kind": "result|progress|error|model_output|metric", ...}
A JSONL stream like this is trivial to load later for plotting, debugging, or future
development (pandas.read_json(lines=True), jq, etc.) — nothing is lost to a pretty-printer.

Sinks are pluggable and FAIL-SAFE (a sink error never breaks the run):
  * local JSONL file        — always on (the source of truth).
  * S3FileSink              — periodically uploads the JSONL to s3://bucket/key (your acct).
  * CloudWatchSink          — streams each event to a CloudWatch Logs group/stream (live).
Both AWS sinks lazy-import boto3 and degrade to no-ops if it's absent or creds are missing.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Optional


def _boto3_client(service: str):
    try:
        import boto3
        return boto3.client(service)
    except Exception as e:
        print(f"[aws {service} unavailable -> skipped] ({type(e).__name__}: {str(e)[:70]})")
        return None


class EventLog:
    def __init__(self, path: str, run_id: str, sinks: Optional[list] = None,
                 clock: Callable[[], float] = time.time):
        self.path = path
        self.run_id = run_id
        self.clock = clock
        self.sinks = sinks or []
        self.seq = 0
        self._lock = threading.Lock()                  # concurrent rungs write results -> serialize
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._f = open(path, "a")

    def emit(self, kind: str, **fields) -> dict:
        with self._lock:
            ev = {"t": self.clock(), "run": self.run_id, "seq": self.seq, "kind": kind, **fields}
            self.seq += 1
            line = json.dumps(ev, default=str)
            self._f.write(line + "\n")
            self._f.flush()
            for s in self.sinks:
                try:
                    s.write(ev, line)
                except Exception as e:
                    print(f"[sink err {type(s).__name__}: {str(e)[:60]}]")
        return ev

    # convenience kinds
    def progress(self, **f):     return self.emit("progress", **f)
    def result(self, **f):       return self.emit("result", **f)
    def error(self, msg, **f):   return self.emit("error", msg=str(msg), **f)
    def model_output(self, **f): return self.emit("model_output", **f)
    def metric(self, **f):       return self.emit("metric", **f)

    def flush(self):
        for s in self.sinks:
            try:
                s.flush()
            except Exception:
                pass

    def close(self):
        self.flush()
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass
        self._f.close()


class S3FileSink:
    """Mirror the JSONL file to s3://bucket/key every `every` events (and on flush/close)."""
    def __init__(self, local_path: str, bucket: str, key: str, every: int = 200):
        self.local_path, self.bucket, self.key, self.every = local_path, bucket, key, every
        self.n = 0
        self._c = _boto3_client("s3")

    def _upload(self):
        if self._c:
            try:
                self._c.upload_file(self.local_path, self.bucket, self.key)
            except Exception as e:
                print(f"[s3 upload err {str(e)[:70]}]")

    def write(self, ev, line):
        self.n += 1
        if self.n % self.every == 0:
            self._upload()

    def flush(self): self._upload()
    def close(self): self._upload()


class CloudWatchSink:
    """Stream each event line to CloudWatch Logs (batched). Live tail in the AWS console."""
    def __init__(self, group: str, stream: str, batch: int = 50):
        self.group, self.stream, self.batch = group, stream, batch
        self.buf: list = []
        self._c = _boto3_client("logs")
        if self._c:
            for create, kw in (("create_log_group", {"logGroupName": group}),
                               ("create_log_stream", {"logGroupName": group, "logStreamName": stream})):
                try:
                    getattr(self._c, create)(**kw)
                except Exception:
                    pass                                  # already exists / no perms -> tolerate

    def write(self, ev, line):
        ts = int(ev.get("t", time.time()) * 1000)
        self.buf.append({"timestamp": ts, "message": line[:250_000]})
        if len(self.buf) >= self.batch:
            self.flush()

    def flush(self):
        if not (self._c and self.buf):
            return
        try:
            self._c.put_log_events(logGroupName=self.group, logStreamName=self.stream,
                                   logEvents=sorted(self.buf, key=lambda e: e["timestamp"]))
            self.buf.clear()
        except Exception as e:
            print(f"[cloudwatch err {str(e)[:70]}]")
            self.buf.clear()

    def close(self): self.flush()
