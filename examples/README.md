# Examples: measured with and without YORO

Four runnable examples, one per integration surface. Each drives the same shape of
workload: a handful of recurring tasks, then one drift event (the underlying data
changes), then the tasks re-asked with the new values. Every answer is checkable
against a deterministic gold value, so the tables report correctness as well as
cost. A cache that is cheap but wrong under drift loses here, by design.

All numbers below are real runs against a local
[Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF)
served by llama.cpp on one machine, temperature 0. Each example runs multiple
independent iterations: fresh cache, fresh workspace, and different ledger values
every iteration, so no prompt text repeats across iterations.

Setup for all examples:

```bash
llama-server -hf deepreinforce-ai/Ornith-1.0-35B-GGUF --port 8000
pip install "yoro-cache[embed]"   # plus the extra named in each example
```

## 1. Proxy + git signal (`proxy_git_benchmark.py`)

The full three-tier policy (serve / replay / reason) with a real git workspace as
the dependency. The scenario is coding-agent-shaped: recurring questions about
workspace data, then the data changes (edit + commit), then the questions recur.

```bash
python examples/proxy_git_benchmark.py --upstream http://127.0.0.1:8000/v1
```

Measured, 3 iterations, 60 requests per side:

| config     | reqs | model calls | out tokens | wrong |
|------------|-----:|------------:|-----------:|------:|
| no cache   |   60 |          60 |     19,761 |     1 |
| yoro --git |   60 |          33 |     10,242 |     0 |

**Output tokens saved: 48%** (per-iteration: 50%, 50%, 44%). Model calls avoided:
27 of 60 (45%). The baseline also produced one wrong answer across its 60 fresh
calls; YORO produced zero, because repeats are served from a validated entry and
post-drift asks REPLAY the stored derivation against the new values instead of
re-deriving from scratch.

## 2. LiteLLM cache slot (`litellm_benchmark.py`)

`YoroSemanticCache` plugged into LiteLLM's cache interface, with a deps-file
sidecar as the invalidation signal. A cache slot can serve and invalidate but
cannot call the model, so there is no replay tier here: the win is that repeats
are free and drift never serves stale.

```bash
pip install "yoro-cache[embed,litellm]"
python examples/litellm_benchmark.py --api-base http://127.0.0.1:8000/v1
```

Measured, 3 iterations, 48 requests per side:

| config           | reqs | model calls | out tokens | wrong |
|------------------|-----:|------------:|-----------:|------:|
| litellm no cache |   48 |          48 |     14,898 |     0 |
| litellm + yoro   |   48 |          24 |      7,394 |     0 |

**Output tokens saved: 50%** (per-iteration: 51%, 50%, 50%). Model calls avoided:
24 of 48 (50%). Zero wrong answers on both sides; after the sidecar publishes the
new fingerprint, every post-drift ask goes to the model fresh instead of serving
the stale cached answer.

## 3. LangChain LLM cache (`langchain_benchmark.py`)

The same workload through `set_llm_cache(YoroLangChainCache(...))` in front of a
minimal LangChain LLM. Same contract as the LiteLLM slot: serve + invalidate.

```bash
pip install "yoro-cache[embed,langchain]"
python examples/langchain_benchmark.py --api-base http://127.0.0.1:8000/v1
```

Measured, 3 iterations, 48 requests per side:

| config             | reqs | model calls | out tokens | wrong |
|--------------------|-----:|------------:|-----------:|------:|
| langchain no cache |   48 |          48 |     14,898 |     0 |
| langchain + yoro   |   48 |          24 |      7,394 |     0 |

**Output tokens saved: 50%**, model calls avoided: 24 of 48. The totals match the
LiteLLM run exactly because it is the same deterministic workload at temperature 0
through a different integration surface, which is a consistency check in itself.

## 4. MCP resources as the change feed (`mcp_bridge_demo.py`)

End to end: an in-process MCP server exposes a ledger resource, the bridge
snapshots it into a deps-file, and the proxy consumes that file. When the resource
changes, the next ask REPLAYs with the correct new answer. No application code
computes a single fingerprint.

```bash
pip install "yoro-cache[embed,mcp]"
python examples/mcp_bridge_demo.py --upstream http://127.0.0.1:8000/v1
```

The demo asserts the full chain each iteration and exits non-zero otherwise.
Measured, 2 iterations, both passing:

```
[   MISS] cold ask     -> 3186  (expected 3186)
[    HIT] repeat       -> 3186  (0 tokens, served)
[ REPLAY] after change -> 3486  (expected 3486, correct)
```

## Reading the numbers

- The savings percentage is workload-dependent. These streams ask each unique task
  three times before drift, so roughly half the requests are repeats; heavier
  repetition saves more, lighter repetition saves less. The point of the examples
  is not the exact percentage but the shape: repeats cost zero, drift never serves
  stale, and (in the proxy) post-drift asks are cheaper than cold asks because the
  derivation is replayed rather than re-derived.
- "Wrong" counts answers that do not contain the gold value. With a deterministic
  gold function, every one of the ~250 answers across these runs was checked.
- Replay exists only where YORO can reach the model: the proxy (examples 1 and 4).
  Cache-slot adapters (examples 2 and 3) serve and invalidate only, and the
  examples say so honestly in their output.
