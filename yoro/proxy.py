"""YORO as a drop-in OpenAI-compatible caching proxy.

Point any OpenAI-compatible client (OpenCode, Codex, an OpenAI SDK) at this proxy
instead of your real model endpoint. It intercepts `/v1/chat/completions`, and:

  * HIT  — a semantically-matching, fresh case exists -> replay the cached
           completion with ZERO upstream call (the "you only reason once" win).
  * MISS — forward to the real upstream, return it, and store it for next time.
  * SKIP — caching is unsafe for this request (see the policy) -> pure passthrough.

Everything else (`/v1/models`, `/v1/embeddings`, ...) is transparently proxied.

WHY SAFE-BY-DEFAULT. This sits in front of agentic coding tools that EDIT FILES.
A false HIT there returns stale/wrong code. So the default `safe` policy refuses to
cache any request that is agentic (carries `tools`, or whose history contains tool
calls / tool results) or sampled (`temperature > 0.2`). What's left — plain,
deterministic Q&A — is where a hit can't corrupt your tree. Graduating to caching
mutating turns is exactly what the benchmark is meant to validate; until then,
`aggressive` mode and the `X-YORO-Cache` / `X-YORO-Deps` headers are explicit opt-ins.

Run:
    YORO_UPSTREAM=http://127.0.0.1:8000/v1 python -m yoro.proxy      # serves :8400
Then set the client's base URL to http://127.0.0.1:8400/v1.

Observability: every response carries `X-YORO-Cache: HIT|MISS|SKIP` and `X-YORO-Sim`;
GET /yoro/stats returns running totals.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .cache import ReasoningCache
from .invalidation import Invalidator
from .matcher import Decision, Matcher

# ---------------------------------------------------------------- pure helpers


def extract_task(messages: list) -> str:
    """The cache key is the human's actual ask — the LAST user message — not the whole
    transcript (system prompt + tool defs + history differ wildly across sessions, so
    embedding all of it would never match). Freshness/scope is handled by deps instead."""
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):  # OpenAI "parts" form
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict)).strip()
    return ""


def parse_deps(header: Optional[str]) -> dict:
    """`X-YORO-Deps: file_a.py:9f3,config.toml:1ab` -> {name: fingerprint}. A hit only
    serves if these still match what was stored, so the caller can scope a cache entry
    to workspace state it depends on."""
    out: dict = {}
    if not header:
        return out
    for part in header.split(","):
        part = part.strip()
        if ":" in part:
            name, fp = part.split(":", 1)
            out[name.strip()] = fp.strip()
    return out


def cacheable_reason(
    body: dict, cache_header: Optional[str], policy: str
) -> Optional[str]:
    """The safety gate, but it explains itself: returns None if the request is cacheable,
    else a SHORT reason it was skipped (shown in logs + the X-YORO-Cache header).
    `X-YORO-Cache: 0/1` forces the decision; otherwise `policy` decides. `safe` (default)
    refuses agentic (tool-bearing) and sampled turns."""
    if cache_header == "0":
        return "forced-off"
    if not extract_task(body.get("messages", [])):
        return "no-user-msg"
    temp = body.get("temperature")
    forced = cache_header == "1"
    if temp is not None and temp > 0.2 and not forced:
        return "sampled"  # caller wants variety, not a replay
    if forced or policy == "aggressive":
        return None
    # safe policy: never cache an agentic / tool-using turn (a stale hit could break code)
    if body.get("tools") or body.get("functions"):
        return "tools"
    for m in body.get("messages", []):
        if m.get("role") == "tool" or m.get("tool_calls"):
            return "tool-history"
    return None


def is_cacheable(body: dict, cache_header: Optional[str], policy: str) -> bool:
    return cacheable_reason(body, cache_header, policy) is None


def synth_message(content: str, reasoning: Optional[str] = None) -> dict:
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = (
            reasoning  # faithful replay for llama.cpp-style clients
        )
    return msg


def synth_completion(
    model: str, content: str, reasoning: Optional[str], created: float
) -> dict:
    return {
        "id": "chatcmpl-yoro-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(created),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": synth_message(content, reasoning),
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "yoro_cache": "hit",
        },
    }


def sse_chunk(model: str, content: str, created: float) -> bytes:
    chunk = {
        "id": "chatcmpl-yoro-" + uuid.uuid4().hex[:24],
        "object": "chat.completion.chunk",
        "created": int(created),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def sse_stop(model: str, created: float) -> bytes:
    chunk = {
        "id": "chatcmpl-yoro-" + uuid.uuid4().hex[:24],
        "object": "chat.completion.chunk",
        "created": int(created),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def sse_usage(model: str, created: float) -> bytes:
    """OpenAI emits a final choices-empty usage chunk when stream_options.include_usage
    is set; some clients block waiting for it, so the cached replay must send one too."""
    chunk = {
        "id": "chatcmpl-yoro-" + uuid.uuid4().hex[:24],
        "object": "chat.completion.chunk",
        "created": int(created),
        "model": model,
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


SSE_DONE = b"data: [DONE]\n\n"


def accumulate_sse(raw: bytes) -> tuple:
    """Reassemble a captured SSE byte stream into (content, reasoning) for caching."""
    content, reasoning = [], []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            delta = (json.loads(payload).get("choices") or [{}])[0].get("delta", {})
        except Exception:
            continue
        if delta.get("content"):
            content.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
    return "".join(content).strip(), ("".join(reasoning).strip() or None)


# ---------------------------------------------------------------- the cache core


@dataclass
class Stats:
    hit: int = 0
    miss: int = 0
    skip: int = 0
    stored: int = 0

    def as_dict(self) -> dict:
        served = self.hit + self.miss + self.skip
        return {
            **self.__dict__,
            "served": served,
            "hit_rate": round(self.hit / served, 3) if served else 0.0,
        }


class ProxyCache:
    """Wraps the YORO cache with the proxy's lookup/store decisions. Embedder is injected
    so tests can use a cheap one."""

    def __init__(
        self,
        embedder,
        cache: ReasoningCache,
        matcher: Matcher,
        invalidator: Invalidator,
    ):
        self.embedder = embedder
        self.cache = cache
        self.matcher = matcher
        self.invalidator = invalidator
        self.stats = Stats()

    def lookup(self, task: str, deps: dict):
        """Returns (decision, case, sim). HIT means: replay case.outcome."""
        emb = self.embedder.embed(task)
        case, sim = self.cache.nearest(emb)
        if case is None:
            return Decision.MISS, None, -1.0
        fresh = self.invalidator.is_fresh(case, deps)
        return self.matcher.decide(sim, fresh), case, sim

    def store(
        self, task: str, content: str, reasoning: Optional[str], deps: dict
    ) -> None:
        emb = self.embedder.embed(task)
        self.cache.add(task, emb, reasoning or content, content, deps)
        self.stats.stored += 1
        self.cache.save()


def build_proxy_cache(cfg: "Config") -> ProxyCache:
    from .embeddings import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(cfg.embed_model)
    cache = ReasoningCache(cfg.cache_path).load()
    matcher = Matcher(tau_hit=cfg.tau_hit, tau_miss=cfg.tau_miss, novelty_gate=True)
    inval = Invalidator(use_deps=True, use_ttl=False, use_reliability=False)
    return ProxyCache(emb, cache, matcher, inval)


# ---------------------------------------------------------------- config + server


@dataclass
class Config:
    upstream: str = field(
        default_factory=lambda: os.environ.get(
            "YORO_UPSTREAM", "http://127.0.0.1:8000/v1"
        ).rstrip("/")
    )
    port: int = field(default_factory=lambda: int(os.environ.get("YORO_PORT", "8400")))
    policy: str = field(default_factory=lambda: os.environ.get("YORO_POLICY", "safe"))
    tau_hit: float = field(
        default_factory=lambda: float(os.environ.get("YORO_TAU_HIT", "0.95"))
    )
    tau_miss: float = field(
        default_factory=lambda: float(os.environ.get("YORO_TAU_MISS", "0.6"))
    )
    embed_model: str = field(
        default_factory=lambda: os.environ.get("YORO_EMBED", "all-MiniLM-L6-v2")
    )
    cache_path: str = field(
        default_factory=lambda: os.path.expanduser(
            os.environ.get("YORO_CACHE_PATH", "~/.yoro/proxy_cache.json")
        )
    )


def make_handler(cfg: Config, pcache: ProxyCache):
    import requests

    def upstream_url(path: str) -> str:
        # incoming "/v1/chat/completions" -> upstream base (".../v1") + "/chat/completions"
        tail = path[3:] if path.startswith("/v1") else path
        return cfg.upstream + tail

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet; we print our own one-liners
            pass

        def handle(self):  # a client closing a keep-alive socket is normal
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError):
                pass

        def _auth(self) -> dict:
            h = {"Content-Type": "application/json"}
            if self.headers.get("Authorization"):
                h["Authorization"] = self.headers["Authorization"]
            return h

        def _send_json(
            self, obj: dict, status: int = 200, extra: Optional[dict] = None
        ):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        # ---- GET: stats + transparent proxy (e.g. /v1/models) ----
        def do_GET(self):
            if self.path == "/yoro/stats":
                return self._send_json(pcache.stats.as_dict())
            if self.path in ("/health", "/yoro/health"):
                return self._send_json(
                    {"ok": True, "upstream": cfg.upstream, "policy": cfg.policy}
                )
            try:
                r = requests.get(
                    upstream_url(self.path), headers=self._auth(), timeout=60
                )
                return self._send_json(r.json(), r.status_code)
            except Exception as e:
                return self._send_json({"error": str(e)}, 502)

        # ---- POST: intercept chat/completions, else passthrough ----
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            if not self.path.endswith("/chat/completions"):
                return self._passthrough(raw)
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                return self._passthrough(raw)
            return self._handle_chat(body, raw)

        def _passthrough(self, raw: bytes):
            try:
                r = requests.post(
                    upstream_url(self.path), headers=self._auth(), data=raw, timeout=600
                )
                self.send_response(r.status_code)
                self.send_header(
                    "Content-Type", r.headers.get("Content-Type", "application/json")
                )
                self.send_header("Content-Length", str(len(r.content)))
                self.end_headers()
                self.wfile.write(r.content)
            except Exception as e:
                self._send_json({"error": str(e)}, 502)

        def _handle_chat(self, body: dict, raw: bytes):
            model = body.get("model", "yoro")
            stream = bool(body.get("stream"))
            cache_hdr = self.headers.get("X-YORO-Cache")
            deps = parse_deps(self.headers.get("X-YORO-Deps"))
            task = extract_task(body.get("messages", []))

            why = cacheable_reason(body, cache_hdr, cfg.policy)
            if why is not None:
                pcache.stats.skip += 1
                self._note(f"SKIP:{why}", task)
                return self._proxy_chat(
                    body, raw, store_task=None, deps=deps, tag=f"SKIP:{why}"
                )

            decision, case, sim = pcache.lookup(task, deps)
            if decision == Decision.HIT and case is not None:
                pcache.stats.hit += 1
                self._note("HIT", task, sim)
                pcache.cache.record_use(case, True)
                created = time.time()
                hdr = {"X-YORO-Cache": "HIT", "X-YORO-Sim": f"{sim:.3f}"}
                reasoning = case.reasoning if case.reasoning != case.outcome else None
                if stream:
                    iu = bool((body.get("stream_options") or {}).get("include_usage"))
                    return self._stream_cached(model, case.outcome, created, hdr, iu)
                return self._send_json(
                    synth_completion(model, case.outcome, reasoning, created), 200, hdr
                )

            pcache.stats.miss += 1
            self._note("MISS", task, sim)
            return self._proxy_chat(body, raw, store_task=task, deps=deps, tag="MISS")

        # replay a cached answer as a one-shot SSE stream
        def _stream_cached(
            self,
            model: str,
            content: str,
            created: float,
            hdr: dict,
            include_usage: bool = False,
        ):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header(
                "Connection", "close"
            )  # SSE has no length; close marks the end
            for k, v in hdr.items():
                self.send_header(k, v)
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(sse_chunk(model, content, created))
                self.wfile.write(sse_stop(model, created))
                if include_usage:
                    self.wfile.write(sse_usage(model, created))
                self.wfile.write(SSE_DONE)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        # forward to upstream; on MISS, also accumulate + store the result
        def _proxy_chat(
            self,
            body: dict,
            raw: bytes,
            store_task: Optional[str],
            deps: dict,
            tag: str,
        ):
            stream = bool(body.get("stream"))
            try:
                if stream:
                    return self._proxy_stream(raw, store_task, deps, tag)
                r = requests.post(
                    upstream_url(self.path), headers=self._auth(), data=raw, timeout=600
                )
                obj = r.json()
                if store_task and r.status_code == 200:
                    msg = (obj.get("choices") or [{}])[0].get("message", {})
                    content = (msg.get("content") or "").strip()
                    reasoning = (msg.get("reasoning_content") or "").strip() or None
                    if content:
                        pcache.store(store_task, content, reasoning, deps)
                return self._send_json(obj, r.status_code, {"X-YORO-Cache": tag})
            except Exception as e:
                return self._send_json({"error": str(e)}, 502)

        def _proxy_stream(
            self, raw: bytes, store_task: Optional[str], deps: dict, tag: str
        ):
            try:
                r = requests.post(
                    upstream_url(self.path),
                    headers=self._auth(),
                    data=raw,
                    stream=True,
                    timeout=600,
                )
            except Exception as e:
                return self._send_json({"error": str(e)}, 502)
            buf = bytearray()
            with r:
                self.send_response(r.status_code)
                self.send_header(
                    "Content-Type", r.headers.get("Content-Type", "text/event-stream")
                )
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-YORO-Cache", tag)
                self.send_header(
                    "Connection", "close"
                )  # length-less stream: close marks the end
                self.end_headers()
                self.close_connection = True
                try:
                    for chunk in r.iter_content(
                        chunk_size=None
                    ):  # forward raw bytes, unframed
                        if not chunk:
                            continue
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if store_task:
                            buf += chunk
                except (BrokenPipeError, ConnectionResetError):
                    return  # client hung up mid-stream
            if store_task and buf:
                content, reasoning = accumulate_sse(bytes(buf))
                if content:
                    pcache.store(store_task, content, reasoning, deps)

        def _note(self, tag: str, task: str, sim: float = -1.0):
            t = (task[:60] + "…") if len(task) > 60 else task
            s = f" sim={sim:.3f}" if sim >= 0 else ""
            print(f"  [{tag}]{s}  {t!r}", flush=True)

    return Handler


BUILD = "yoro-proxy 0.1.0"


def main():
    cfg = Config()
    print(f"YORO proxy  ::{cfg.port}  ->  {cfg.upstream}")
    print(f"  build: {BUILD}")
    print(f"  policy={cfg.policy}  tau_hit={cfg.tau_hit}  cache={cfg.cache_path}")
    print("  loading embedder…", flush=True)
    pcache = build_proxy_cache(cfg)
    print(
        f"  ready ({len(pcache.cache)} cached cases). Point your client's base URL at "
        f"http://127.0.0.1:{cfg.port}/v1",
        flush=True,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", cfg.port), make_handler(cfg, pcache))
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")


if __name__ == "__main__":
    main()
