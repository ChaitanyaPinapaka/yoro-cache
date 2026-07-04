# YORO — You Only Reason Once

![tests](https://github.com/ChaitanyaPinapaka/yoro-cache/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue)

YORO is an OpenAI-compatible caching proxy for LLM applications. Unlike a plain
semantic cache, it tracks what each cached answer depends on and invalidates
entries when those dependencies change, so it never serves an answer whose
premises moved. The engine can also re-apply the cached reasoning to new inputs
(*replay*) instead of re-deriving from scratch — available in the library today,
wired into the proxy in the next release.

Website: [yorocache.com](https://yorocache.com)

## Why

Semantic caches (GPTCache and similar) serve a cached answer whenever a new request
is embedding-similar to a previous one. This saves tokens, but it has a failure mode
that standard cache metrics do not surface: when the world changes, the cache keeps
serving the old answer. In our measurements, a drift rate of just 5% (5% of recurring
tasks whose true answer has changed) already makes over half of a naive cache's hits
wrong, because popular items drift too and every later hit serves the dead answer.

Adding invalidation alone is not sufficient. In agent workloads, the *method* behind
an answer often lives in earlier interactions rather than in the current request. An
invalidating cache correctly drops the stale entry, then re-derives without the
method, caches the wrong result, and serves it — a failure mode we call
*re-poisoning*. YORO addresses both failure modes: dependency fingerprints handle
detection, and replay of the stored reasoning handles re-derivation.

## Install

```bash
pip install "yoro-cache[embed]"
# before the first PyPI release:
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
with the cached reasoning trace preserved in the response.

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

To scope a cache entry to workspace state, pass dependency fingerprints. An entry
only serves while its fingerprints match what was stored; when they change, the
entry stops serving and the request is re-reasoned upstream:

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
`GET /yoro/stats`) shows running totals.

| Header | Direction | Meaning |
|---|---|---|
| `X-YORO-Deps` | request | `name:fingerprint,...` — entry serves only while these match |
| `X-YORO-Cache: 0` / `1` | request | force caching off / on for this call |
| `X-YORO-Cache` | response | `HIT`, `MISS`, or `SKIP:<reason>` |
| `X-YORO-Sim` | response | similarity of the matched entry (on hits) |

### Configuration

| Variable | Default | |
|---|---|---|
| `YORO_UPSTREAM` | `http://127.0.0.1:8000/v1` | upstream OpenAI-compatible endpoint |
| `YORO_PORT` | `8400` | proxy listen port |
| `YORO_POLICY` | `safe` | `safe` refuses to cache tool-bearing or sampled turns; `aggressive` caches them |
| `YORO_TAU_HIT` / `YORO_TAU_MISS` | `0.95` / `0.6` | reuse-acceptance / novelty thresholds |
| `YORO_EMBED` | `all-MiniLM-L6-v2` | sentence-transformers model for matching |
| `YORO_CACHE_PATH` | `~/.yoro/proxy_cache.json` | persistent cache location |

The default policy is deliberately conservative: requests that carry tools, contain
tool history, or use `temperature > 0.2` pass through uncached, because a stale hit
in an agentic flow can corrupt real work. Caching such turns is an explicit opt-in.

## How it works

Each request is embedded and matched against the case store, then routed to the
cheapest tier that is safe:

1. **Serve** — the matched entry is fresh and similarity is high: return the cached
   answer with no model call.
2. **Replay** — same entry, but its dependencies changed: inject the stored
   reasoning trace and apply it to the new inputs. Short output; no re-exploration.
   (Library + benchmark today; proxy integration lands in the next release.)
3. **Reason** — novel or borderline request: full reasoning upstream; the trace,
   answer, and dependency fingerprints are cached.

A novelty gate escalates look-alike-but-different requests to re-reasoning instead
of force-fitting them into a near-match — trading some hit rate for correctness.

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
and the result curves behind these numbers) lands in this repository in an upcoming
release.

## Scope and limitations

- The replay result is measured in the *method-in-history* regime, where re-asks
  reference a procedure established earlier — the normal case for long-running
  agents. If every request restates its full context, a plain cache with
  invalidation performs equally well on correctness.
- Replay is validated on multi-step arithmetic procedures; non-numeric procedures
  (extraction rules, rubrics, tool plans) have not yet been evaluated.
- Replay quality depends on the invalidation signal. Without dependency
  fingerprints, YORO falls back to conservative matching and behaves like a
  gated semantic cache.
- Related work: Buffer of Thoughts, Metacognitive Reuse, and Analogical Prompting
  reuse reasoning templates. YORO's contribution is making reuse safe and
  accounted for: invalidation, the failure-mode taxonomy, and separate input/output
  token accounting.

## Repository layout

```
yoro/    library and proxy: cache, matcher, invalidation, replay, CLI
bench/   the benchmark harness: rungs, sweeps, taxonomy metrics, result curves, runbook
tests/   library, proxy, and benchmark tests; no GPU required
site/    yorocache.com (static)
```

## License

MIT. Built and measured by [Chaitanya Pinapaka](https://github.com/ChaitanyaPinapaka).
