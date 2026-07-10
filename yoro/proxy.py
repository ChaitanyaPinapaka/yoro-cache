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
import threading
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .cache import ReasoningCache
from .deps import resolve_deps
from .engine import lookup as engine_lookup
from .invalidation import Invalidator
from .matcher import Decision, Matcher
from .request_identity import has_non_text_input, request_identity
from .telemetry import Telemetry

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


def extract_responses_task(value) -> str:
    """Extract the latest user text from Responses API `input`."""
    if isinstance(value, str):
        return value.strip()
    messages = value if isinstance(value, list) else []
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(
                str(p.get("text", "")) for p in content
                if isinstance(p, dict) and p.get("type") in ("input_text", "text")
            ).strip()
    return ""


def parse_deps(header: Optional[str]) -> dict:
    """`X-YORO-Deps: file_a.py:9f3,config.toml:1ab` -> {name: fingerprint}. A hit only
    serves if these still match what was stored, so the caller can scope a cache entry
    to workspace state it depends on."""
    out: dict = {}
    if not header:
        return out
    if header.lstrip().startswith("{"):
        try:
            value = json.loads(header)
            return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}
        except Exception:
            return {}
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
    if has_non_text_input(body) and not forced:
        return "multimodal"
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


REPLAY_SYSTEM = (
    "You are given a validated procedure that solved a very similar task. Apply it "
    "directly to the new inputs: do not re-derive, re-plan, or explore; execute the "
    "procedure's steps on the new values and state the result. Be terse."
)


def replay_body(body: dict, task: str, derivation: str) -> dict:
    """The upstream request for the replay tier: same client fields, but the messages
    inject the cached derivation and ask for direct application to the new task."""
    out = dict(body)
    out["messages"] = [
        {"role": "system", "content": REPLAY_SYSTEM},
        {"role": "user", "content": f"Validated procedure:\n{derivation}\n\nApply it to this task:\n{task}"},
    ]
    return out


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


def synth_response(model: str, content: str, created: float) -> dict:
    rid = "resp_yoro_" + uuid.uuid4().hex[:24]
    mid = "msg_yoro_" + uuid.uuid4().hex[:24]
    return {
        "id": rid, "object": "response", "created_at": int(created),
        "status": "completed", "model": model, "background": False,
        "completed_at": int(created), "error": None, "incomplete_details": None,
        "instructions": None, "max_output_tokens": None,
        "output": [{"id": mid, "type": "message", "status": "completed",
                    "role": "assistant", "content": [
                        {"type": "output_text", "text": content, "annotations": [], "logprobs": []}
                    ]}],
        "parallel_tool_calls": True, "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None}, "store": False,
        "temperature": 0.0, "text": {"format": {"type": "text"}},
        "tool_choice": "auto", "tools": [], "top_p": 1.0,
        "truncation": "disabled", "user": None, "metadata": {},
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def response_output_text(obj: dict) -> str:
    parts = []
    for item in obj.get("output", []):
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                parts.append(part.get("text", ""))
    return "".join(parts).strip()


def response_sse(model: str, content: str, created: float) -> bytes:
    response = synth_response(model, content, created)
    item = response["output"][0]
    part = item["content"][0]
    events = [
        ("response.created", {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}),
        ("response.in_progress", {"type": "response.in_progress", "response": {**response, "status": "in_progress", "output": []}}),
        ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0,
                                        "item": {**item, "status": "in_progress", "content": []}}),
        ("response.content_part.added", {"type": "response.content_part.added", "item_id": item["id"],
                                         "output_index": 0, "content_index": 0,
                                         "part": {"type": "output_text", "text": "", "annotations": []}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": item["id"],
                                        "output_index": 0, "content_index": 0, "delta": content}),
        ("response.output_text.done", {"type": "response.output_text.done", "item_id": item["id"],
                                       "output_index": 0, "content_index": 0, "text": content}),
        ("response.content_part.done", {"type": "response.content_part.done", "item_id": item["id"],
                                        "output_index": 0, "content_index": 0, "part": part}),
        ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item}),
        ("response.completed", {"type": "response.completed", "response": response}),
    ]
    chunks = []
    for sequence_number, (name, payload) in enumerate(events):
        payload["sequence_number"] = sequence_number
        chunks.append(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
    return b"".join(chunks)


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


def accumulate_response_sse(raw: bytes) -> tuple[str, None]:
    content = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except Exception:
            continue
        if obj.get("type") == "response.output_text.delta":
            content.append(obj.get("delta", ""))
        elif obj.get("type") == "response.completed" and not content:
            content.append(response_output_text(obj.get("response", {})))
    return "".join(content).strip(), None


# ---------------------------------------------------------------- the cache core


@dataclass
class Stats:
    hit: int = 0
    miss: int = 0
    skip: int = 0
    replay: int = 0
    stored: int = 0
    hit_no_deps: int = 0  # HITs whose case had no dependency scope (semantic-only)

    def as_dict(self) -> dict:
        served = self.hit + self.miss + self.skip + self.replay
        return {
            **self.__dict__,
            "served": served,
            "hit_rate": round(self.hit / served, 3) if served else 0.0,
            "hit_no_deps_rate": (
                round(self.hit_no_deps / self.hit, 3) if self.hit else 0.0
            ),
        }


class ProxyCache:
    """Wraps the YORO cache with the proxy's lookup/store decisions. Embedder is injected
    so tests can use a cheap one. Routing uses `engine.lookup` (shared with YORO.solve).

    Thread safety: ThreadingHTTPServer serves each request on its own thread, so the
    case store, stats, and disk writes are guarded by one lock, and the embedder by
    another (torch encoders are not guaranteed re-entrant). Embedding happens outside
    the store lock so a slow encode never blocks cache reads."""

    def __init__(
        self,
        embedder,
        cache: ReasoningCache,
        matcher: Matcher,
        invalidator: Invalidator,
        replay: bool = True,
        telemetry: Optional[Telemetry] = None,
    ):
        self.embedder = embedder
        self.cache = cache
        self.matcher = matcher
        self.invalidator = invalidator
        self.replay = replay
        self.telemetry = telemetry or Telemetry()
        self.stats = Stats()
        self._lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._flights: dict[str, threading.Event] = {}

    def _embed(self, task: str):
        with self._embed_lock:
            return self.embedder.embed(task)

    def lookup(self, task: str, deps: dict, scope: Optional[dict] = None):
        """Returns (decision, case, sim, emb, fresh, should_replay).
        Embedding is returned so a following store() never re-encodes."""
        emb = self._embed(task)
        with self._lock:
            self.cache.refresh()
            found = engine_lookup(
                self.cache,
                self.matcher,
                self.invalidator,
                emb,
                deps,
                replay=self.replay,
                scope=scope,
            )
            self.telemetry.event("yoro.lookup", decision=found.decision.value,
                                 similarity=found.sim, fresh=found.fresh)
            return (
                found.decision,
                found.case,
                found.sim,
                emb,
                found.fresh,
                found.should_replay,
            )

    def store(
        self, task: str, content: str, reasoning: Optional[str], deps: dict, emb=None,
        scope: Optional[dict] = None,
    ) -> None:
        if emb is None:
            emb = self._embed(task)
        with self._lock:
            self.cache.add(task, emb, reasoning or content, content, deps, scope=scope)
            c = self.cache.cases[-1]
            from .structured import ProcedureArtifact
            self.cache.set_artifact(
                c, steps=ProcedureArtifact.from_reasoning(reasoning or content, deps).steps,
                procedure=ProcedureArtifact.from_reasoning(reasoning or content, deps).to_dict(),
            )
            self.stats.stored += 1
            # write-behind: ReasoningCache flushes on its own schedule; force nothing extra

    def store_replay(self, case, task: str, content: str, deps: dict, emb=None,
                     scope: Optional[dict] = None) -> None:
        """A replayed answer refreshes the case in place: new outcome + deps/version,
        original derivation preserved (the terse replay output must never erode the
        method that will be injected on the next change)."""
        if emb is None:
            emb = self._embed(task)
        with self._lock:
            keep = case.reasoning
            c = self.cache.update(case.id, task, emb, keep, content, deps, scope=scope)
            c.steps = case.steps
            self.stats.replay += 1

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self.stats, field, getattr(self.stats, field) + 1)

    def record_hit(self, case) -> None:
        with self._lock:
            self.cache.record_use(case, True)
            self.stats.hit += 1
            if not (case.deps or {}):
                self.stats.hit_no_deps += 1

    def stats_dict(self) -> dict:
        with self._lock:
            d = self.stats.as_dict()
            d["evicted"] = getattr(self.cache, "_evicted", 0)
            d["cases"] = len(self.cache)
            return d

    def begin_fill(self, key: str) -> tuple[bool, threading.Event]:
        """Singleflight: exactly one leader fills a missing request identity."""
        with self._lock:
            event = self._flights.get(key)
            if event is not None:
                return False, event
            event = threading.Event()
            self._flights[key] = event
            return True, event

    def finish_fill(self, key: str) -> None:
        with self._lock:
            event = self._flights.pop(key, None)
            if event:
                event.set()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in ("off", "0", "false", "no")


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def build_proxy_cache(cfg: "Config") -> ProxyCache:
    from .embeddings import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(cfg.embed_model)
    vector_index = None
    if cfg.vector_index == "hnsw":
        from .vector_index import HNSWIndex
        vector_index = HNSWIndex()
    cache = ReasoningCache(
        cfg.cache_path,
        max_cases=cfg.cache_max,
        flush_every=cfg.cache_flush_every,
        vector_index=vector_index,
        refresh_interval=cfg.cache_refresh_seconds,
    ).load()
    matcher = Matcher(tau_hit=cfg.tau_hit, tau_miss=cfg.tau_miss, novelty_gate=True)
    inval = Invalidator(
        use_deps=True,
        use_ttl=False,
        use_reliability=False,
        require_signal=cfg.require_signal,
        strict_deps=cfg.strict_deps,
    )
    return ProxyCache(emb, cache, matcher, inval, replay=cfg.replay)


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
    replay: bool = field(
        default_factory=lambda: _env_bool("YORO_REPLAY", True)
    )
    git_repo: str = field(default_factory=lambda: os.environ.get("YORO_GIT", ""))
    deps_file: str = field(default_factory=lambda: os.environ.get("YORO_DEPS_FILE", ""))
    # repo | mentioned | watch | off — finer than whole-tree git when "mentioned"/"watch"
    git_mode: str = field(
        default_factory=lambda: os.environ.get("YORO_GIT_MODE", "repo")
    )
    watch_paths: list = field(
        default_factory=lambda: [
            p.strip()
            for p in os.environ.get("YORO_WATCH", "").split(",")
            if p.strip()
        ]
    )
    workspace: str = field(
        default_factory=lambda: os.environ.get("YORO_WORKSPACE", "")
    )
    require_signal: bool = field(
        default_factory=lambda: _env_bool("YORO_REQUIRE_SIGNAL", True)
    )
    strict_deps: bool = field(
        default_factory=lambda: _env_bool("YORO_STRICT_DEPS", False)
    )
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
    cache_max: Optional[int] = field(
        default_factory=lambda: _env_int("YORO_CACHE_MAX", None)
    )
    cache_flush_every: int = field(
        default_factory=lambda: int(os.environ.get("YORO_CACHE_FLUSH_EVERY", "1"))
    )
    vector_index: str = field(
        default_factory=lambda: os.environ.get("YORO_VECTOR_INDEX", "numpy").lower()
    )
    cache_refresh_seconds: float = field(
        default_factory=lambda: float(os.environ.get("YORO_CACHE_REFRESH_SECONDS", "1"))
    )


def make_handler(cfg: Config, pcache: ProxyCache):
    import requests

    sess = requests.Session()  # connection pooling: no per-request TCP/TLS handshake

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
                return self._send_json(pcache.stats_dict())
            if self.path in ("/health", "/yoro/health"):
                return self._send_json(
                    {
                        "ok": True,
                        "upstream": cfg.upstream,
                        "policy": cfg.policy,
                        "git_mode": cfg.git_mode,
                        "replay": cfg.replay,
                    }
                )
            try:
                r = sess.get(
                    upstream_url(self.path), headers=self._auth(), timeout=60
                )
                return self._send_json(r.json(), r.status_code)
            except Exception as e:
                return self._send_json({"error": str(e)}, 502)

        # ---- POST: intercept chat/completions, else passthrough ----
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            if not self.path.endswith(("/chat/completions", "/responses")):
                return self._passthrough(raw)
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                return self._passthrough(raw)
            if self.path.endswith("/responses"):
                return self._handle_responses(body, raw)
            return self._handle_chat(body, raw)

        def _handle_responses(self, body: dict, raw: bytes):
            task = extract_responses_task(body.get("input"))
            model = str(body.get("model") or "yoro")
            stream = bool(body.get("stream"))
            normalized = dict(body)
            normalized["messages"] = (
                [{"role": "user", "content": task}] if isinstance(body.get("input"), str)
                else body.get("input", [])
            )
            why = cacheable_reason(normalized, self.headers.get("X-YORO-Cache"), cfg.policy)
            if why is None and (
                body.get("store", True) is not False
                or body.get("previous_response_id")
                or body.get("conversation")
            ) and self.headers.get("X-YORO-Cache") != "1":
                why = "stateful-responses"
            if why is not None:
                pcache.bump("skip")
                return self._proxy_stream(raw, None, {}, f"SKIP:{why}") if stream else self._passthrough(raw, f"SKIP:{why}")
            identity = request_identity(body, workspace=cfg.workspace, operation="responses")
            deps = resolve_deps(
                parse_deps(self.headers.get("X-YORO-Deps")), git_repo=cfg.git_repo,
                deps_file=cfg.deps_file, git_mode=cfg.git_mode, task=task,
                watch_paths=cfg.watch_paths, model=model, workspace=cfg.workspace,
            )
            decision, case, sim, emb, _, should_replay = pcache.lookup(
                task, deps, scope=identity.scope
            )
            if decision == Decision.HIT and case is not None:
                pcache.record_hit(case)
                hdr = {"X-YORO-Cache": "HIT", "X-YORO-Sim": f"{sim:.3f}"}
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    for k, v in hdr.items(): self.send_header(k, v)
                    self.end_headers()
                    self.close_connection = True
                    self.wfile.write(response_sse(model, case.outcome, time.time()))
                    self.wfile.flush()
                    return
                return self._send_json(synth_response(model, case.outcome, time.time()), 200, hdr)
            if should_replay and case is not None and not stream:
                replay_request = dict(body)
                replay_request["instructions"] = REPLAY_SYSTEM
                replay_request["input"] = (
                    f"Validated procedure:\n{case.reasoning}\n\nApply it to this task:\n{task}"
                )
                try:
                    rr = sess.post(upstream_url(self.path), headers=self._auth(),
                                   data=json.dumps(replay_request).encode(), timeout=600)
                    robj = rr.json()
                    replayed = response_output_text(robj)
                    from .replay import validate_output
                    if rr.status_code == 200 and validate_output(replayed, body):
                        replay_deps = dict(deps)
                        replay_deps.update(parse_deps(rr.headers.get("X-YORO-Deps")))
                        pcache.store_replay(case, task, replayed, replay_deps, emb=emb,
                                            scope=identity.scope)
                        return self._send_json(robj, 200, {"X-YORO-Cache": "REPLAY"})
                except Exception:
                    pass
            pcache.bump("miss")
            if stream:
                # Forward typed upstream events faithfully. A later non-streaming request
                # can seed the same identity; buffering typed streams is the next optimization.
                return self._proxy_stream(
                    raw, task, deps, "MISS", emb=emb, scope=identity.scope,
                    response_api=True,
                )
            fill_key = hashlib.sha256(json.dumps(
                [task, identity.scope, deps], sort_keys=True
            ).encode()).hexdigest()
            leader, event = pcache.begin_fill(fill_key)
            if not leader:
                event.wait(timeout=600)
                return self._handle_responses(body, raw)
            try:
                r = sess.post(upstream_url(self.path), headers=self._auth(), data=raw, timeout=600)
                obj = r.json()
                content = response_output_text(obj)
                if r.status_code == 200 and content:
                    store_deps = dict(deps)
                    store_deps.update(parse_deps(r.headers.get("X-YORO-Deps")))
                    pcache.store(task, content, None, store_deps, emb=emb, scope=identity.scope)
                pcache.finish_fill(fill_key)
                return self._send_json(obj, r.status_code, {"X-YORO-Cache": "MISS"})
            except Exception as e:
                pcache.finish_fill(fill_key)
                return self._send_json({"error": str(e)}, 502)

        def _passthrough(self, raw: bytes, tag: Optional[str] = None):
            try:
                r = sess.post(
                    upstream_url(self.path), headers=self._auth(), data=raw, timeout=600
                )
                self.send_response(r.status_code)
                self.send_header(
                    "Content-Type", r.headers.get("Content-Type", "application/json")
                )
                self.send_header("Content-Length", str(len(r.content)))
                if tag:
                    self.send_header("X-YORO-Cache", tag)
                self.end_headers()
                self.wfile.write(r.content)
            except Exception as e:
                self._send_json({"error": str(e)}, 502)

        def _handle_chat(self, body: dict, raw: bytes):
            model = body.get("model", "yoro")
            stream = bool(body.get("stream"))
            cache_hdr = self.headers.get("X-YORO-Cache")
            task = extract_task(body.get("messages", []))
            identity = request_identity(body, workspace=cfg.workspace)
            # model + optional workspace always scope the entry; git_mode picks
            # coarse repo vs per-file (mentioned/watch) fingerprints
            deps = resolve_deps(
                parse_deps(self.headers.get("X-YORO-Deps")),
                git_repo=cfg.git_repo,
                deps_file=cfg.deps_file,
                git_mode=cfg.git_mode,
                task=task,
                watch_paths=cfg.watch_paths,
                model=str(model or ""),
                workspace=cfg.workspace,
            )

            why = cacheable_reason(body, cache_hdr, cfg.policy)
            if why is not None:
                pcache.bump("skip")
                self._note(f"SKIP:{why}", task)
                return self._proxy_chat(
                    body, raw, store_task=None, deps=deps, tag=f"SKIP:{why}"
                )

            decision, case, sim, emb, fresh, should_replay = pcache.lookup(
                task, deps, scope=identity.scope
            )
            if decision == Decision.HIT and case is not None:
                self._note("HIT", task, sim)
                pcache.record_hit(case)
                created = time.time()
                hdr = {"X-YORO-Cache": "HIT", "X-YORO-Sim": f"{sim:.3f}"}
                # after a replay refresh (version > 1) the stored derivation belongs to the
                # ORIGINAL inputs; echoing it beside the refreshed answer would mislead.
                reasoning = (
                    case.reasoning
                    if case.version == 1 and case.reasoning != case.outcome
                    else None
                )
                if stream:
                    iu = bool((body.get("stream_options") or {}).get("include_usage"))
                    return self._stream_cached(model, case.outcome, created, hdr, iu)
                return self._send_json(
                    synth_completion(model, case.outcome, reasoning, created), 200, hdr
                )

            # engine.lookup already encodes "stale same-case + derivation" as should_replay
            if should_replay and case is not None and not stream:
                self._note("REPLAY", task, sim)
                b2 = replay_body(body, task, case.reasoning)
                return self._proxy_chat(
                    b2, json.dumps(b2).encode(), store_task=task, deps=deps,
                    tag="REPLAY", emb=emb, replay_case=case, scope=identity.scope,
                    fallback_body=body, fallback_raw=raw,
                )

            pcache.bump("miss")
            self._note("MISS", task, sim)
            fill_key = None
            if not stream:
                fill_key = hashlib.sha256(json.dumps(
                    [task, identity.scope, deps], sort_keys=True
                ).encode()).hexdigest()
                leader, event = pcache.begin_fill(fill_key)
                if not leader:
                    event.wait(timeout=600)
                    return self._handle_chat(body, raw)
            return self._proxy_chat(
                body, raw, store_task=task, deps=deps, tag="MISS", emb=emb,
                scope=identity.scope,
                fill_key=fill_key,
            )

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
            emb=None,
            replay_case=None,
            scope=None,
            fallback_body=None,
            fallback_raw=None,
            fill_key=None,
        ):
            stream = bool(body.get("stream"))
            try:
                if stream:
                    return self._proxy_stream(raw, store_task, deps, tag, emb, replay_case, scope)
                r = sess.post(
                    upstream_url(self.path), headers=self._auth(), data=raw, timeout=600
                )
                obj = r.json()
                if store_task and r.status_code == 200:
                    store_deps = dict(deps)
                    store_deps.update(parse_deps(r.headers.get("X-YORO-Deps")))
                    msg = (obj.get("choices") or [{}])[0].get("message", {})
                    content = (msg.get("content") or "").strip()
                    reasoning = (msg.get("reasoning_content") or "").strip() or None
                    if replay_case is not None:
                        from .replay import validate_output
                        if not validate_output(content, fallback_body):
                            pcache.bump("miss")
                            return self._proxy_chat(
                                fallback_body, fallback_raw, store_task, deps, "MISS",
                                emb=emb, scope=scope,
                            )
                    if content and replay_case is not None:
                        pcache.store_replay(replay_case, store_task, content, store_deps, emb=emb, scope=scope)
                    elif content:
                        pcache.store(store_task, content, reasoning, store_deps, emb=emb, scope=scope)
                    else:  # e.g. a reasoning model exhausted max_tokens thinking
                        self._note(f"{tag}:not-stored(empty content)", store_task)
                result = self._send_json(obj, r.status_code, {"X-YORO-Cache": tag})
                if fill_key:
                    pcache.finish_fill(fill_key)
                return result
            except Exception as e:
                if fill_key:
                    pcache.finish_fill(fill_key)
                return self._send_json({"error": str(e)}, 502)

        def _proxy_stream(
            self, raw: bytes, store_task: Optional[str], deps: dict, tag: str, emb=None,
            replay_case=None, scope=None, response_api=False,
        ):
            try:
                r = sess.post(
                    upstream_url(self.path),
                    headers=self._auth(),
                    data=raw,
                    stream=True,
                    timeout=600,
                )
            except Exception as e:
                return self._send_json({"error": str(e)}, 502)
            buf = bytearray()
            store_deps = dict(deps)
            store_deps.update(parse_deps(r.headers.get("X-YORO-Deps")))
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
                parser = accumulate_response_sse if response_api else accumulate_sse
                content, reasoning = parser(bytes(buf))
                if content and replay_case is not None:
                    pcache.store_replay(replay_case, store_task, content, store_deps, emb=emb, scope=scope)
                elif content:
                    pcache.store(store_task, content, reasoning, store_deps, emb=emb, scope=scope)

        def _note(self, tag: str, task: str, sim: float = -1.0):
            t = (task[:60] + "…") if len(task) > 60 else task
            s = f" sim={sim:.3f}" if sim >= 0 else ""
            print(f"  [{tag}]{s}  {t!r}", flush=True)

    return Handler


BUILD = "yoro-proxy 0.2.0"


def main():
    cfg = Config()
    print(f"YORO proxy  ::{cfg.port}  ->  {cfg.upstream}")
    print(f"  build: {BUILD}")
    print(
        f"  policy={cfg.policy}  tau_hit={cfg.tau_hit}  git_mode={cfg.git_mode}  "
        f"cache={cfg.cache_path}"
    )
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
        pcache.cache.flush()


if __name__ == "__main__":
    main()
