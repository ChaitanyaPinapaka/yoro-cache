# YORO — You Only Reason Once

![tests](https://github.com/ChaitanyaPinapaka/yoro-cache/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue)

YORO is an OpenAI-compatible caching proxy for LLM applications: procedural
memory for LLM systems. Unlike a plain semantic cache, it tracks what each cached
answer depends on and invalidates entries when those dependencies change, so it
never serves an answer whose premises moved. And when a dependency does change,
it does not re-derive from scratch: it *replays* the cached derivation against
the new inputs. Remember how, invalidate when the world moves, replay.

Website: [yorocache.com](https://yorocache.com)

## Why

Semantic caches (GPTCache and similar) serve a cached answer whenever a new request
is embedding-similar to a previous one. This saves tokens, but it has a failure mode
that standard cache metrics do not surface: when the world changes, the cache keeps
serving the old answer. In my measurements, a drift rate of just 5% (5% of recurring
tasks whose true answer has changed) already makes over half of a naive cache's hits
wrong, because popular items drift too and every later hit serves the dead answer.

Adding invalidation alone is not sufficient. In agent workloads, the *method* behind
an answer often lives in earlier interactions rather than in the current request. An
invalidating cache correctly drops the stale entry, then re-derives without the
method, caches the wrong result, and serves it — a failure mode I call
*re-poisoning*. YORO addresses both failure modes: dependency fingerprints handle
detection, and replay of the stored reasoning handles re-derivation.

## Install

```bash
pip install "yoro-cache[embed]"
# or install the latest from main:
pip install "yoro-cache[embed] @ git+https://github.com/ChaitanyaPinapaka/yoro-cache"
```

Requires Python 3.10+. The `[embed]` extra installs `sentence-transformers` for
semantic matching; without it the library still works with the hash embedder or an
external embedding endpoint.

## Usage

Run the proxy in front of any OpenAI-compatible endpoint (vLLM, llama.cpp server,
OpenRouter, ...), then point your client at it. The worked example below is the
setup this README was tested on — a local 35B reasoning model on an M-series Mac:

```bash
# 1. serve a local model via llama.cpp        (brew install llama.cpp)
llama-server -hf deepreinforce-ai/Ornith-1.0-35B-GGUF --port 8000

# 2. put YORO in front of it
YORO_UPSTREAM=http://127.0.0.1:8000/v1 yoro serve    # listens on :8400

# 3. point any OpenAI-compatible client at the proxy
export OPENAI_BASE_URL=http://127.0.0.1:8400/v1
```

On this setup, a repeated ask serves from cache in ~12 ms against ~3.3 s upstream,
with the cached derivation preserved in the response.

To use YORO under [OpenCode](https://opencode.ai), register the proxy as a custom
provider in `opencode.json`:

```json
{ "provider": { "yoro": {
    "npm": "@ai-sdk/openai-compatible",
    "options": { "baseURL": "http://127.0.0.1:8400/v1" },
    "models": { "ornith-35b": {} } } } }
```

The safe policy caches OpenCode's plain question turns and passes its tool-bearing
(agentic) turns through untouched.

To scope a cache entry to workspace state, pass dependency fingerprints — or let
the proxy compute them:

```bash
# whole working tree (coarse but correct: any edit invalidates)
yoro serve --git .

# only paths named in the task (finer — unmentioned edits do not thrash the cache)
yoro serve --git . --git-mode mentioned

# explicit path list
yoro serve --git . --watch data/rollup.csv,config.toml

# sidecar JSON (file watcher, git hook, or MCP bridge)
yoro serve --deps-file deps.json
yoro mcp-bridge --server "<cmd>" --deps-file deps.json
```

Entries only serve while fingerprints match. When they change, the proxy
**replays** the stored derivation against the new inputs (`X-YORO-Cache: REPLAY`)
instead of serving stale or re-deriving blind. The proxy also scopes every entry
by `model` (from the request body) and optional `--workspace` / `YORO_WORKSPACE`,
so different models never share answers. Cases stored with deps refuse to HIT
when the request carries no signal (`YORO_REQUIRE_SIGNAL=on`, default).

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8400/v1", api_key="unused-locally")
r = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Recompute the rollup for March"}],
    extra_headers={"X-YORO-Deps": "rollup.csv:9f3ab2"},
)
```

Every response reports the cache decision, and `yoro stats` (or
`GET /yoro/stats`) shows running totals — including `hit_no_deps` (semantic-only
hits with no dependency scope) and `evicted` when a size cap is set.

Persistence defaults to a JSON file; set `YORO_CACHE_PATH` to a `.sqlite` path
for SQLite, `YORO_CACHE_MAX` to bound size (least-used eviction), and
`YORO_CACHE_FLUSH_EVERY` for write-behind batching.

YORO separates the semantic task key from an exact request identity. System and
developer instructions, prior context, output schemas, model settings, and non-text
parts are fingerprinted into an exact scope. The same question under different
instructions therefore creates a separate case instead of replaying across contexts.
Multimodal requests bypass the cache by default unless explicitly forced.

| Header | Direction | Meaning |
|---|---|---|
| `X-YORO-Deps` | request or upstream response | JSON `{name:fingerprint}` or legacy `name:fingerprint,...`; entry serves only while these match |
| `X-YORO-Cache: 0` / `1` | request | force caching off / on for this call |
| `X-YORO-Cache` | response | `HIT`, `REPLAY`, `MISS`, or `SKIP:<reason>` |
| `X-YORO-Sim` | response | similarity of the matched entry (on hits) |

### Configuration

| Variable | Default | |
|---|---|---|
| `YORO_UPSTREAM` | `http://127.0.0.1:8000/v1` | upstream OpenAI-compatible endpoint |
| `YORO_PORT` | `8400` | proxy listen port |
| `YORO_POLICY` | `safe` | `safe` refuses to cache tool-bearing or sampled turns; `aggressive` caches them |
| `YORO_TAU_HIT` / `YORO_TAU_MISS` | `0.95` / `0.6` | reuse-acceptance / novelty thresholds |
| `YORO_EMBED` | `all-MiniLM-L6-v2` | sentence-transformers model for matching |
| `YORO_CACHE_PATH` | `~/.yoro/proxy_cache.json` | persistent cache (`.sqlite` / `.db` uses SQLite) |
| `YORO_CACHE_MAX` | unset | max cases before least-used eviction |
| `YORO_CACHE_FLUSH_EVERY` | `1` | write-behind: flush every N mutations |
| `YORO_VECTOR_INDEX` | `numpy` | `numpy` brute-force or scoped `hnsw` (`pip install yoro-cache[ann]`) |
| `YORO_CACHE_REFRESH_SECONDS` | `1` | SQLite cross-process visibility interval; `0` disables refresh |
| `YORO_GIT` | unset | workspace root for automatic git/file deps |
| `YORO_GIT_MODE` | `repo` | `repo` (whole tree), `mentioned` (paths in task), `watch`, or `off` |
| `YORO_WATCH` | unset | comma-separated paths for `git_mode=watch` |
| `YORO_WORKSPACE` | unset | opaque id stored as a dependency (multi-tenant) |
| `YORO_STRICT_DEPS` | `off` | require full coverage of every stored dep key |
| `YORO_REQUIRE_SIGNAL` | `on` | refuse HITs for scoped cases when the request has no deps |

The default policy is deliberately conservative: requests that carry tools, contain
tool history, or use `temperature > 0.2` pass through uncached, because a stale hit
in an agentic flow can corrupt real work. Caching such turns is an explicit opt-in.

## How it works

Each request is embedded and matched against the case store, then routed to the
cheapest tier that is safe:

1. **Serve** — the matched entry is fresh and similarity is high: return the cached
   answer with no model call.
2. **Replay** — same entry, but its dependencies changed: inject the stored
   derivation (the model's visible working, or a structured plan where none is
   exposed) and apply it to the new inputs. Short output; no re-exploration.
   Disable with `YORO_REPLAY=off`.
3. **Reason** — novel or borderline request: full reasoning upstream; the trace,
   answer, and dependency fingerprints are cached.

A novelty gate escalates look-alike-but-different requests to re-reasoning instead
of force-fitting them into a near-match — trading some hit rate for correctness.

Library and proxy share one decision path (`yoro.engine.lookup`): HIT / ESCALATE /
REPLAY cannot drift between surfaces.

The proxy supports both `/v1/chat/completions` and `/v1/responses`. Responses inputs
use typed request identity and cached responses reconstruct typed output and SSE
events. Streaming misses are forwarded byte-for-byte and stored after completion.
Because cached synthetic response IDs are not upstream conversation objects, the safe
policy caches only stateless Responses requests (`store: false`, without
`previous_response_id` or `conversation`); stateful requests pass through unchanged.

Applications can report the data actually read without exposing its contents:

```python
from yoro import DependencyTracker

deps = DependencyTracker()
deps.file("config.toml")
deps.resource("docs://policy", policy_text)
deps.query("postgres", sql, rows)
response_headers = {"X-YORO-Deps": deps.header()}
```

The proxy merges an upstream `X-YORO-Deps` response header into the stored case.
Configured Git and sidecar sources also store health markers, so a source becoming
unreadable invalidates prior cases instead of silently losing coverage.

## Integrations

- **LiteLLM** — drop the engine into LiteLLM's cache slot:
  `pip install "yoro-cache[embed,litellm]"`, then
  `litellm.cache = Cache(); litellm.cache.cache = YoroSemanticCache(git_repo=".")`
  (`yoro.integrations.litellm_cache`).
- **LangChain / LangGraph** — `set_llm_cache(YoroLangChainCache(git_repo="."))`
  (`yoro.integrations.langchain_cache`); the `llm_string` is stored as a
  dependency so models never serve each other's entries.
- **MCP (experimental)** — `yoro mcp-bridge` mirrors an MCP server's resources
  into a deps-file: the ecosystem's standardized change-feed becomes the
  invalidation signal with zero application code.

Cache-slot adapters serve and invalidate but cannot call the model, so the replay
tier applies to the proxy; in the adapters a changed dependency simply misses
(correct, never stale).

## Examples

[`examples/`](examples/) has one runnable, measured benchmark per surface: the
proxy with git as the signal, the LiteLLM cache slot, the LangChain LLM cache,
and the MCP bridge end to end. Each runs the same recurring-tasks-then-drift
workload with and without YORO, over multiple independent iterations, against a
local model, and checks every answer against a deterministic gold value. Measured
on Ornith-1.0-35B: 48-50% of output tokens saved and roughly half of model calls
avoided, with zero stale answers after drift. See
[`examples/README.md`](examples/README.md) for the tables.

## Evaluation

The claims above are measured, on gpt-oss-120B (H100, vLLM) and reproduced on
Qwen2.5-32B-Instruct-AWQ (4-bit, one consumer RTX 5090), across controlled sweeps of
drift rate, near-miss rate, and invalidation-signal fidelity — 25 sweep levels,
1,027 runs, 616,200 scored queries, 72.7M tokens in total. Selected results at
drift 0.4 on the method-in-history workload:

| | GPTCache-style | YORO serve-only | YORO replay | YORO replay (low effort) | no cache |
|---|---|---|---|---|---|
| Accuracy | 0.16 | 0.16 | **0.96** | 0.92 | 0.07 |
| Output tokens vs no-cache | 4% | 42% | 21% | 10% | 100% |

- On self-contained workloads, a no-invalidation cache reaches staleness 0.90
  (share of hits serving a wrong answer) as drift rises; YORO holds ~0.00 at the
  same matched thresholds, with accuracy 1.00.
- Wrong serves split into two mechanistically different failure modes: *outdated*
  (served an answer that was once correct) and *re-poisoned* (served an answer that
  was never correct). The no-invalidation cache fails mostly outdated; an
  invalidating cache without replay fails ~99% re-poisoned; replay reduces both to
  near zero. Accuracy alone cannot distinguish these; the taxonomy metrics
  (`outdated_rate`, `repoisoned_rate`) can.
- Weakening the invalidation signal degrades YORO gracefully — staleness tracks the
  share of missed signals and converges to naive-cache behavior at zero signal.

The full benchmark harness (sweep driver, workload generators, taxonomy metrics,
and the runbook behind these numbers) lives in [`bench/`](bench/).

## Scope and limitations

- The replay result is measured in the *method-in-history* regime, where re-asks
  reference a procedure established earlier — the normal case for long-running
  agents. If every request restates its full context, a plain cache with
  invalidation performs equally well on correctness.
- Replay is validated on multi-step arithmetic procedures; non-numeric procedures
  (extraction rules, rubrics, tool plans) have not yet been evaluated.
- Replay quality depends on the invalidation signal. Without dependency
  fingerprints, YORO falls back to conservative matching and behaves like a
  gated semantic cache; scoped cases with no request signal refuse to HIT rather
  than pretending invalidation is active.
- Related work: Buffer of Thoughts, Metacognitive Reuse, and Analogical Prompting
  reuse reasoning templates. YORO's contribution is making reuse safe and
  accounted for: invalidation, the failure-mode taxonomy, and separate input/output
  token accounting.

## Repository layout

```
yoro/      library and proxy: engine, cache, matcher, invalidation, replay, deps, CLI
bench/     the benchmark harness: rungs, sweeps, taxonomy metrics, result curves, runbook
examples/  runnable with-and-without benchmarks for each integration surface
tests/     library, proxy, and benchmark tests; no GPU required
site/      yorocache.com (static)
```

## License

MIT. Built and measured by [Chaitanya Pinapaka](https://github.com/ChaitanyaPinapaka).
