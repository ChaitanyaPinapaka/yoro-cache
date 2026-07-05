# Changelog

## 0.1.2 — 2026-07-05

The replay release: the proxy now ships the full graduated serve → replay → reason
policy, and the invalidation signal gets first-class sources (git, sidecar file,
MCP resources) plus adapters for LiteLLM and LangChain. YORO's category in one
line: procedural memory for LLM systems — remember *how*, invalidate when the
world moves, replay against new inputs.

### Replay in the proxy
- On a stale same-case escalation (similarity ≥ `tau_hit` and dependency
  fingerprints changed), the proxy no longer re-reasons blind: it injects the
  stored derivation and asks the upstream model to apply it to the new inputs.
  Response carries `X-YORO-Cache: REPLAY`; `/yoro/stats` gains a `replay` counter.
- The replayed answer refreshes the cache entry in place (new outcome, new deps,
  version bump) while **preserving the original derivation**, so the method never
  erodes across successive replays.
- Entries refreshed by replay (version > 1) serve answer-only on later hits: the
  stored derivation belongs to the original inputs, and echoing it beside a
  refreshed answer would mislead.
- `YORO_REPLAY=off` restores 0.1.1 behavior (serve + invalidate only).
- Verified live against Ornith-1.0-35B via llama.cpp: drift produced
  `X-YORO-Cache: REPLAY` with the correct new answer at 39% of the cold-reasoning
  tokens, and the follow-up hit served the refreshed answer.

### Dependency-signal sources (`yoro/deps.py`)
- `yoro serve --git <repo>` fingerprints a git working tree (HEAD + dirty state)
  as an automatic dependency: any commit or edit invalidates entries scoped to the
  workspace. Coarse but *correct* — a moved workspace can only cost hit rate,
  never staleness. The natural zero-setup signal for coding agents.
- `yoro serve --deps-file <json>` reads `{name: fingerprint}` maintained by any
  sidecar (file watcher, git hook, the MCP bridge below).
- Merge order: deps-file, then git, then the request's `X-YORO-Deps` header (the
  most explicit source wins). Sources are cached ~2 s off the request hot path.

### MCP resource bridge (experimental)
- `yoro mcp-bridge --server "<cmd>" --deps-file <json>` mirrors an MCP server's
  resources into a deps-file: every resource URI becomes a dependency whose
  fingerprint moves when its content changes. MCP is the first standardized
  change-feed for agent context; this makes it YORO's invalidation signal with
  zero application code. Polls `resources/list` + `read` (works with every
  server); subscription support is the upgrade path.

### LiteLLM adapter (`yoro.integrations.litellm_cache.YoroSemanticCache`)
- Drop-in backend for LiteLLM's pluggable cache slot: semantic matching with the
  novelty gate and dependency invalidation inside existing LiteLLM deployments.
- Current fingerprints come from plugin-level sources (`git_repo`, `deps_file`,
  `deps_source`) because LiteLLM's read path forwards only `messages`; write-time
  `metadata={"yoro_deps": ...}` additionally scopes individual entries.

### LangChain / LangGraph adapter (`yoro.integrations.langchain_cache.YoroLangChainCache`)
- `BaseCache` implementation for `set_llm_cache`: the same slot semantic caches
  plug into, with the gate and invalidation. The `llm_string` is stored as a
  dependency, so different models never serve each other's entries.

### Scope notes
- Cache-slot adapters (LiteLLM, LangChain) can serve and invalidate but cannot
  call the model, so the replay tier applies only to the proxy; a changed
  dependency in the adapters simply misses (correct, never stale).
- Packaging: new optional extras `[litellm]`, `[langchain]`, `[mcp]`.

### Tests
- Suite grows 38 → 44: replay-preserves-derivation, replay request shape, git
  fingerprint end-to-end (edit and commit both invalidate), LiteLLM adapter
  against the real library (mock transport), LangChain adapter against
  langchain-core, MCP bridge against a real in-process SDK server.

## 0.1.1 — 2026-07-04

- The benchmark: the drift / near-miss / invalidation-fidelity stress harness
  behind every published number, the baseline ladder, the outdated vs re-poisoned
  failure taxonomy as first-class metrics (`outdated_rate`, `repoisoned_rate`),
  per-level checkpoint/resume with an in-run budget guard, the replay-quality
  spike, the runbook, and the five result curves from the published experiments.
- Proxy hardening: thread-safe cache/stats under ThreadingHTTPServer, atomic
  cache persistence (tmp + rename), cached similarity matrix, one embed per miss,
  pooled upstream connections, and a log note when an empty completion is not
  cached (reasoning models exhausting `max_tokens`).

## 0.1.0 — 2026-07-04

- Initial public release: the YORO library (cache, matcher with novelty gate,
  dependency invalidator, replay engine, keyers, embedders, behaviors) and the
  OpenAI-compatible caching proxy (`yoro serve`) with safe-by-default policy,
  `X-YORO-Deps` scoping, and `/yoro/stats`.
