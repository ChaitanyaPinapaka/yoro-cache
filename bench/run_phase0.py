"""Phase-0 driver — runs the whole benchmark loop end to end.

Builds the labelled stream, runs the 5-rung ladder over N seeds, scores each prompt
against ground truth (correct / stale / forced), logs every event (JSONL + W&B + optional
S3/CloudWatch), checkpoints for resume, and enforces the budget cap. Aggregates to
mean +/- std with a paired test of YORO vs the GPTCache-style baseline.

  --smoke   run locally with a mock 'perfect reasoner' + the built-in sample. No GPU, no
            AWS, no spend — proves the harness before a dollar is rented.
  (real)    point --base-url at the vLLM endpoint on the rented H100; S3/CloudWatch
            default to the resources created for this project (override via env).

Cloud sinks are OPT-IN via env: set S3_BUCKET (and optionally CW_LOG_GROUP) to
mirror events/checkpoints/reports; unset, everything stays local under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace, asdict

from bench.ladder import build_ladder
from bench.metrics import Outcome, summarize, aggregate_seeds, paired_t
from bench.budget import BudgetGuard
from bench.convergence import Convergence
from bench.vast import VastCredit
from bench.eventlog import EventLog, S3FileSink, CloudWatchSink
from bench.checkpoint import Checkpoint
from bench.wandb_log import WandbLogger


def _envf(name: str, default: str) -> float:
    """float() from env, treating an EMPTY string (e.g. `export X=` in the launcher) as absent —
    so an unset-but-exported knob falls back to its default instead of crashing float('')."""
    return float(os.environ.get(name) or default)


def _envi(name: str, default: str) -> int:
    return int(os.environ.get(name) or default)


@dataclass
class Config:
    smoke: bool = True
    seeds: int = 2
    out: str = "runs/phase0-smoke"
    run_id: str = "phase0"
    # model
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "openai/gpt-oss-120b"
    embed_model: str = "all-MiniLM-L6-v2"
    fast_embed: bool = False
    # cache knobs. gptcache_tau defaults to tau_hit so the baseline runs at the SAME threshold as
    # YORO — any YORO win then isolates the gate+invalidation, not a stricter cutoff.
    # Set GPTCACHE_TAU=0.85 to also run the looser classic GPTCache as an extra baseline. TAU_HIT is
    # env-driven because the HARD workload's short ref re-asks sit at ~0.82-0.94 sim to their cold
    # text — so E7 runs matched at TAU_HIT=GPTCACHE_TAU=0.80 (cross-entity max ~0.51: a clean gap).
    tau_hit: float = field(default_factory=lambda: _envf("TAU_HIT", "0.9"))
    gptcache_tau: float = field(default_factory=lambda: _envf("GPTCACHE_TAU", "0.9"))
    # real-suite data (customer-workload model: variety x recurrence)
    domains: tuple = ("math", "qa", "reasoning")
    n_unique: int = field(default_factory=lambda: _envi("N_UNIQUE", "40"))     # distinct tasks
    stream_len: int = field(default_factory=lambda: _envi("STREAM_LEN", "600"))  # total requests
    n_pairs: int = field(default_factory=lambda: _envi("N_PAIRS", "120"))       # QQP/PAWS probe pairs
    zipf_s: float = 1.1          # popularity skew (higher = more head concentration / more reuse)
    # workload: "hf" (verified HF suite + Zipf recurrence) | "stress" (synthetic tunable drift/near-miss)
    workload: str = field(default_factory=lambda: os.environ.get("WORKLOAD", "hf"))
    drift_rate: float = field(default_factory=lambda: _envf("DRIFT_RATE", "0"))       # E1
    near_miss_rate: float = field(default_factory=lambda: _envf("NEAR_MISS_RATE", "0"))  # E2
    inval_fidelity: float = field(default_factory=lambda: _envf("INVAL_FIDELITY", "1.0"))  # E4
    hard_workload: bool = field(default_factory=lambda: os.environ.get("HARD", "") not in ("", "0"))  # E7: multi-step chains
    # sweep SWEEP_PARAM over SWEEP_VALUES (generic; ZIPF_SWEEP kept for back-compat -> maps to zipf_s)
    zipf_sweep: tuple = field(default_factory=lambda: tuple(
        float(x) for x in os.environ.get("ZIPF_SWEEP", "").split(",") if x.strip()))
    sweep_param: str = field(default_factory=lambda: os.environ.get("SWEEP_PARAM", "zipf_s"))
    sweep_values: tuple = field(default_factory=lambda: tuple(
        float(x) for x in os.environ.get("SWEEP_VALUES", "").split(",") if x.strip()))
    rungs: tuple = field(default_factory=lambda: tuple(                # rung subset (empty = all 6)
        x.strip() for x in os.environ.get("RUNGS", "").split(",") if x.strip()))
    # convergence early-stop (seeds = the MAX; we stop earlier when CIs are tight)
    min_seeds: int = field(default_factory=lambda: _envi("MIN_SEEDS", "12"))   # randomized probes give the
    ci_target: float = field(default_factory=lambda: _envf("CI_TARGET", "0.02"))
    # concurrency: run (seed,rung) units in one shared pool — vLLM batches the requests.
    # ~`workers` requests in flight at once; seed_batch seeds built per batch (batch x 6 rungs ~= workers).
    workers: int = field(default_factory=lambda: _envi("WORKERS", "24"))
    seed_batch: int = field(default_factory=lambda: _envi("SEED_BATCH", "4"))
    # budget (env-driven on the cluster; 0 hourly in smoke -> never triggers)
    ceiling_usd: float = field(default_factory=lambda: _envf("CEILING_USD", "480"))
    hourly_usd: float = field(default_factory=lambda: _envf("HOURLY_USD", "0"))
    # GROUND-TRUTH stop: poll real Vast.ai credit and stop with this USD buffer (0 = disable)
    vast_min_credit_usd: float = field(default_factory=lambda: _envf("VAST_MIN_CREDIT_USD", "0"))
    # aws / wandb (default to the resources created for this project)
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-west-2"))
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", ""))  # empty = no S3 mirroring
    s3_prefix: str = field(default_factory=lambda: os.environ.get("S3_PREFIX", "phase0"))
    cw_log_group: str = field(default_factory=lambda: os.environ.get("CW_LOG_GROUP", "/yoro/benchmark"))
    wandb: bool = False
    wandb_project: str = "yoro-benchmark"


# ---- scoring ----
def _toks(s: str) -> set:
    return set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def is_correct(pred: str, gold: str) -> bool:
    g, p = _toks(gold), _toks(pred)
    return bool(g) and (g <= p or p <= g)              # token-set containment either way


def classify_stale(served: str, past_golds) -> tuple:
    """Split a wrong same-entity SERVE into (outdated, repoisoned) — the paper's taxonomy, by the
    CORRECTNESS-LINEAGE definition: OUTDATED iff the served answer was a correct gold for this entity
    at an EARLIER point (it was true, the entity then drifted); RE-POISONED iff it was NEVER correct
    (a wrong answer entered the cache and was served). These partition `staleness`. NB this differs
    from a version-lineage classifier only on cached model-errors (never-correct-from-cache), which
    this counts as re-poisoned; we report correctness-lineage because 'stale' should mean 'once true'."""
    outdated = any(is_correct(served, pg) for pg in past_golds)
    return outdated, (not outdated)


class MockPerfect:
    """A perfect-but-counted reasoner over the stream's gold answers — makes every measured
    error a CACHE error (stale/brittle reuse), never a model error. Swap in VLLMClient for
    the real run."""
    name = "mock-perfect"

    def __init__(self, gold: dict):
        self.gold = gold
        self.calls = 0
        self.last_completion_tokens = 0
        self.last_prompt_tokens = 0

    def reason(self, text, system=None):
        self.calls += 1
        g = self.gold.get(text, "?")
        self.last_completion_tokens = max(1, len(g) // 4 + 20)   # full CoT: sizeable output
        self.last_prompt_tokens = max(1, len(text) // 4)
        return (f"reasoning -> {g}", g)

    def replay(self, text, plan):
        """Perfect replay: applies the (perfect) method to the current task -> correct answer, but with
        SHORT output (no exploration) and a plan-inflated INPUT — so the token axes move like a real replay."""
        self.calls += 1
        g = self.gold.get(text, "?")
        self.last_completion_tokens = max(1, len(g) // 4 + 4)    # short output (≈ answer only)
        plen = len(plan if isinstance(plan, str) else "\n".join(map(str, plan)))
        self.last_prompt_tokens = max(1, (len(text) + plen) // 4)   # input inflated by the injected plan
        return (f"replay -> {g}", g)

    def complete(self, prompt, max_tokens=None):
        return ""


def _embedder(cfg: Config):
    if cfg.fast_embed:
        from yoro import HashEmbedder
        return HashEmbedder()
    from yoro import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(cfg.embed_model)


def _build_sinks(cfg: Config, jsonl_path: str) -> list:
    if cfg.smoke:
        return []                                       # local only
    sinks = []
    if cfg.s3_bucket:
        sinks.append(S3FileSink(jsonl_path, cfg.s3_bucket,
                                f"{cfg.s3_prefix}/{cfg.run_id}/events.jsonl"))
    if cfg.cw_log_group:
        sinks.append(CloudWatchSink(cfg.cw_log_group, f"{cfg.run_id}-events"))
    return sinks


def _upload_artifact(cfg: Config, local_path: str, suffix: str) -> None:
    """Mirror a result artifact (report/curve) to S3 so it survives instance death — the
    raw per-rung results are already checkpointed to S3, but this keeps the aggregates too."""
    if cfg.smoke or not cfg.s3_bucket or not os.path.exists(local_path):
        return
    try:
        import boto3
        boto3.client("s3").upload_file(local_path, cfg.s3_bucket, f"{cfg.s3_prefix}/{cfg.run_id}/{suffix}")
    except Exception as e:
        print(f"[s3 artifact upload err {str(e)[:70]}]")


def run(cfg: Config, guard=None) -> dict:
    os.makedirs(cfg.out, exist_ok=True)
    jsonl = os.path.join(cfg.out, "events.jsonl")
    log = EventLog(jsonl, cfg.run_id, sinks=_build_sinks(cfg, jsonl))
    ck = Checkpoint(os.path.join(cfg.out, "ckpt.json"),
                    s3=None if cfg.smoke else (cfg.s3_bucket, f"{cfg.s3_prefix}/{cfg.run_id}/ckpt.json"))
    wb = WandbLogger(cfg.wandb_project, name=cfg.run_id, config=vars(cfg), enabled=cfg.wandb)
    guard = guard or BudgetGuard(cfg.ceiling_usd, cfg.hourly_usd)   # shared across a sweep
    conv = Convergence(cfg.min_seeds, cfg.seeds, cfg.ci_target)   # cfg.seeds = the MAX
    vast = VastCredit(min_usd=cfg.vast_min_credit_usd) if (cfg.vast_min_credit_usd > 0 and not cfg.smoke) else None

    emb = _embedder(cfg)
    from bench.datasets import build_smoke_stream, gold_map, load_hf, build_stress_workload, Task

    state = ck.load() or {"seed_idx": 0, "per_seed": []}
    per_seed = state["per_seed"]
    log.progress(phase="start", suite=("smoke" if cfg.smoke else "hf"),
                 seeds=cfg.seeds, from_seed=state["seed_idx"], resumed=bool(per_seed))
    if cfg.wandb and wb.run is None:                    # don't let the dashboard vanish silently
        log.error("W&B disabled — live dashboard will be MISSING (check WANDB_API_KEY in Secrets Manager)")

    from collections import defaultdict

    def build_seed(s):                                  # a seed's stream + per-rung ladder (each rung its own model)
        if cfg.workload == "stress":                    # synthetic tunable drift/near-miss (E1/E2/E4)
            st = build_stress_workload(cfg.n_unique, cfg.stream_len, cfg.zipf_s, cfg.drift_rate,
                                       cfg.near_miss_rate, cfg.inval_fidelity, hard=cfg.hard_workload, seed=s)
            mk = ((lambda gm: (lambda: MockPerfect(gm)))(gold_map(st)) if cfg.smoke
                  else (lambda: _real_model(cfg)))
        elif cfg.smoke:
            st = build_smoke_stream(s)
            mk = (lambda gm: (lambda: MockPerfect(gm)))(gold_map(st))
        else:
            bm = _real_model(cfg)
            st = load_hf(list(cfg.domains), n_unique=cfg.n_unique, stream_len=cfg.stream_len,
                         zipf_s=cfg.zipf_s, n_pairs=cfg.n_pairs, seed=s, paraphraser=bm.complete)
            mk = lambda: _real_model(cfg)
        return st, build_ladder(mk, emb, tau_hit=cfg.tau_hit, gptcache_tau=cfg.gptcache_tau, rungs=cfg.rungs)

    def run_unit(unit):                                 # one (seed,rung): ordered internally, own model
        s, rung, st = unit
        outs = []
        gold_hist: dict = {}                             # key -> [golds seen earlier] (for the outdated/re-poison split)
        for t in st:
            try:
                r = rung.solve(t.text, current_deps=t.deps)
            except Exception as e:
                log.error(e, rung=rung.name, seed=s, task=t.text[:60])
                continue
            if t.kind == "populate":
                continue
            outcome, reused, out_tok, in_tok, replayed = (
                r.outcome, r.reused, r.out_tokens, r.in_tokens, r.replayed)
            if t.expect_reuse is not None:              # decision-mode (QQP/PAWS): gold is the reuse choice
                ok = (reused == t.expect_reuse)
                stale = False
                forced = reused and (t.expect_reuse is False)
            else:                                       # answer-mode (verifiable / drift / near-miss)
                ok = is_correct(outcome, t.gold)
                if t.kind == "near_miss":               # a DISTINCT look-alike entity: ANY reuse is a force-fit
                    forced = reused
                    stale = False
                elif t.kind == "cold":                  # FIRST sighting: there is no "own answer" yet, so a
                    forced = reused and (not ok)        # wrong reuse is a cross-entity force-fit (brittle), not stale
                    stale = False
                else:                                   # repeat/drift = same-entity lineage: a wrong reuse =
                    forced = False                      # an OUTDATED own-answer served => STALE (counts the drift
                    stale = reused and (not ok)         # occurrence AND every post-drift repeat served stale)
            replay_wrong = replayed and (not ok)        # replayed the method but got it wrong — its own column
            # split a stale serve: did we serve a once-CORRECT answer (outdated) or never-correct garbage
            # that got re-derived + cached (re-poisoned)? distinguishes true staleness from a re-derive failure.
            outdated = repoisoned = False
            if stale:                                   # served a once-correct answer (outdated) vs never-correct (re-poisoned)
                outdated, repoisoned = classify_stale(outcome, gold_hist.get(t.key, []))
            if t.expect_reuse is None and t.gold:
                gold_hist.setdefault(t.key, []).append(t.gold)   # record AFTER classifying (past = strictly earlier)
            outs.append(Outcome(t.key, reused, ok, stale, forced, 0.0, out_tok,
                                in_tokens=in_tok, replayed=replayed, replay_wrong=replay_wrong,
                                outdated=outdated, repoisoned=repoisoned))
            ver = next(iter(t.deps.values()), None) if t.deps else None
            log.result(seed=s, rung=rung.name, domain=t.domain, task_kind=t.kind, key=t.key,
                       version=ver, gold=t.gold, served=str(outcome)[:60], reused=reused, correct=ok,
                       stale=stale, forced=forced, replayed=replayed, replay_wrong=replay_wrong,
                       outdated=outdated, repoisoned=repoisoned, out_tokens=out_tok, in_tokens=in_tok)
        return s, rung.name, outs

    si = state["seed_idx"]
    while si < cfg.seeds:
        seeds_now = list(range(si, min(si + max(1, cfg.seed_batch), cfg.seeds)))
        units = []                                      # (seed, rung, stream) — the unit of concurrent work
        for s in seeds_now:
            st, ladder = build_seed(s)
            for rung in ladder:
                units.append((s, rung, st))
        log.progress(phase="batch_start", seeds=seeds_now, units=len(units), workers=cfg.workers)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
            res = list(ex.map(run_unit, units))         # ~workers requests in flight -> vLLM batches them
        by_seed = defaultdict(dict)
        for s, name, outs in res:
            by_seed[s][name] = outs
        for s in seeds_now:
            d = by_seed[s]
            for name, outs in d.items():                # total failure (model down) -> fail LOUD, ckpt kept
                if not outs:
                    log.error("rung produced NO scored outcomes — model unavailable?", rung=name, seed=s)
                    log.close()
                    raise RuntimeError(f"rung '{name}' empty at seed {s} — aborting (checkpoint preserved)")
            nocache_tokens = sum(o.llm_tokens for o in d.get("no-cache", []))
            seed_res = {}
            for name, outs in d.items():
                sm = summarize(outs, nocache_tokens)
                seed_res[name] = sm
                log.metric(seed=s, rung=name, **sm)
                wb.log({f"{name}/{k}": v for k, v in sm.items() if isinstance(v, (int, float))}, step=s)
            per_seed.append(seed_res)
        si = seeds_now[-1] + 1
        state["seed_idx"] = si
        state["per_seed"] = per_seed
        ck.save(state)                                  # checkpoint per BATCH (resume rebuilds an unfinished batch)
        log.progress(phase="batch_done", through_seed=seeds_now[-1], secs=round(time.time() - t0, 1),
                     vast_credit=(vast.last if vast else None), **guard.status())
        cred = vast.remaining() if vast else None        # budget / Vast-credit stop (per batch)
        credit_low = cred is not None and cred <= cfg.vast_min_credit_usd
        if guard.check() or credit_low:
            if credit_low:
                guard.stop()
            log.error("stopping cleanly — budget cap or low Vast credit (resume on top-up)",
                      vast_credit=cred, **guard.status())
            break
        stop, info = conv.check(per_seed)                # conclude early once the headline CIs are tight enough
        log.progress(phase="convergence", **info)
        wb.log({"convergence/n": info["n"],
                **{f"ci/{k}": v for k, v in info.get("ci_halfwidths", {}).items()}}, step=seeds_now[-1])
        if stop and info["reason"] in ("converged", "max_seeds"):
            log.progress(phase="early_stop", **info)
            break

    state["done"] = not guard.stopped                  # finished on its own (converged/max_seeds), not budget-cut
    ck.save(state)                                      # -> a sweep resume SKIPS this level instead of re-touching it

    report = _aggregate(cfg, per_seed)
    with open(os.path.join(cfg.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    _upload_artifact(cfg, os.path.join(cfg.out, "report.json"), "report.json")   # survive instance death
    wb.summary({"seeds": report["seeds"], **{f"{r}/hit_rate": report["rungs"][r]["hit_rate"]["mean"]
                                              for r in report["rungs"]}})
    wb.finish()
    log.progress(phase="done", **report.get("headline", {}))
    log.close()
    return report


def _aggregate(cfg: Config, per_seed: list) -> dict:
    report = {"run": cfg.run_id, "seeds": len(per_seed), "rungs": {}}
    if not per_seed:
        return report
    common = set(per_seed[0])                          # only rungs present (non-empty) in EVERY seed
    for ps in per_seed[1:]:
        common &= set(ps)
    rungs = [r for r in per_seed[0] if r in common and per_seed[0][r]]
    keys = ("hit_rate", "accuracy", "staleness", "outdated_rate", "repoisoned_rate", "brittleness",
            "tokens_total", "input_tokens_total", "replay_rate", "replay_wrong")
    for rn in rungs:
        report["rungs"][rn] = {k: aggregate_seeds([ps[rn] for ps in per_seed], k) for k in keys}
    if "yoro" in rungs and "gptcache-semantic" in rungs:
        report["yoro_vs_gptcache"] = {
            k: paired_t([ps["yoro"][k] for ps in per_seed],
                        [ps["gptcache-semantic"][k] for ps in per_seed])
            for k in ("accuracy", "staleness", "brittleness")}
    return report


def _real_model(cfg: Config):
    from bench.model_client import VLLMClient
    return VLLMClient(cfg.base_url, cfg.model)


def _headline(rep: dict) -> dict:
    """The few numbers that go on the curve: YORO's metrics + its deltas vs GPTCache."""
    rungs = rep.get("rungs", {})
    y = rungs.get("yoro", {})
    out = {"seeds": rep.get("seeds")}
    for m in ("hit_rate", "accuracy", "staleness", "brittleness", "tokens_total"):
        if m in y:
            out[f"yoro_{m}"] = y[m]["mean"]
    for m in ("accuracy", "staleness", "brittleness"):
        if m in rep.get("yoro_vs_gptcache", {}):
            out[f"delta_{m}"] = rep["yoro_vs_gptcache"][m]["mean_diff"]
    return out


def _load_curve(cfg: Config, curve_path: str) -> list:
    """Resume: the curve points already computed in a prior (pre-top-up) run. Prefer the local
    file (intact on a STOPPED instance); fall back to S3 (survives a destroyed one)."""
    if os.path.exists(curve_path):
        try:
            with open(curve_path) as f:
                return json.load(f).get("curve", [])
        except Exception:
            pass
    if not cfg.smoke and cfg.s3_bucket:
        try:
            import boto3
            os.makedirs(os.path.dirname(curve_path) or ".", exist_ok=True)
            boto3.client("s3").download_file(cfg.s3_bucket, f"{cfg.s3_prefix}/{cfg.run_id}/curve.json", curve_path)
            with open(curve_path) as f:
                return json.load(f).get("curve", [])
        except Exception:
            pass
    return []


_SWEEP_ABBR = {"zipf_s": "z", "drift_rate": "drift", "near_miss_rate": "nm",
               "tau_hit": "tau", "inval_fidelity": "fid", "gptcache_tau": "gtau"}


def _level_suffix(param: str, val: float) -> str:
    return f"{_SWEEP_ABBR.get(param, param)}{val}"      # zipf_s=0.6 -> 'z0.6' (keeps existing S3 paths)


def _level_done(cfg: Config, param: str, val: float) -> bool:
    """True iff this sweep level's checkpoint is marked done (finished on its own, not budget-cut).
    A budget-interrupted level is NOT done -> resume re-enters it and continues from its seed_idx."""
    rid = f"{cfg.run_id}-{_level_suffix(param, val)}"
    ck = Checkpoint(os.path.join(cfg.out, _level_suffix(param, val), "ckpt.json"),
                    s3=None if cfg.smoke else (cfg.s3_bucket, f"{cfg.s3_prefix}/{rid}/ckpt.json"))
    st = ck.load()
    return bool(st and st.get("done"))


def run_sweep(cfg: Config) -> dict:
    """Sweep cfg.sweep_param over cfg.sweep_values -> a curve of YORO benefit/safety vs the swept
    axis (zipf_s | drift_rate | near_miss_rate | tau_hit | inval_fidelity). One shared BudgetGuard
    bounds the TOTAL $; convergence early-stop keeps each level cheap. RESUMABLE across top-ups:
    completed levels skip (checkpoint 'done' marker), a budget-cut level continues."""
    param = cfg.sweep_param
    guard = BudgetGuard(cfg.ceiling_usd, cfg.hourly_usd)
    os.makedirs(cfg.out, exist_ok=True)
    curve_path = os.path.join(cfg.out, "curve.json")
    curve = _load_curve(cfg, curve_path)               # keep prior levels' points across a resume

    def _persist():                                    # save + upload the curve-so-far after each level
        with open(curve_path, "w") as f:
            json.dump({"run": cfg.run_id, "sweep_param": param,
                       "metric": f"yoro benefit/safety vs {param}", "curve": curve}, f, indent=2, default=str)
        _upload_artifact(cfg, curve_path, "curve.json")

    for val in cfg.sweep_values:
        if _level_done(cfg, param, val):               # completed in a prior run -> skip, keep its curve point
            print(f"[sweep] {param}={val} already complete — skipping (resume)")
            continue
        sub = replace(cfg, run_id=f"{cfg.run_id}-{_level_suffix(param, val)}",
                      out=os.path.join(cfg.out, _level_suffix(param, val)),
                      zipf_sweep=(), sweep_values=(), **{param: val})
        rep = run(sub, guard=guard)
        if guard.stopped:                              # budget cap / low credit hit DURING this level:
            break                                      # leave it un-marked so the next top-up resumes it
        curve = [pt for pt in curve                    # replace any stale point for this value, then record fresh
                 if round(float(pt.get("sweep_val", pt.get("zipf_s", -9))), 4) != round(float(val), 4)]
        curve.append({"sweep_param": param, "sweep_val": val, param: val, **_headline(rep)})
        _persist()
    wb = WandbLogger(cfg.wandb_project, name=f"{cfg.run_id}-curve", config=vars(cfg), enabled=cfg.wandb)
    for pt in curve:
        wb.log({k: v for k, v in pt.items() if isinstance(v, (int, float))})
    wb.finish()
    return {"curve": curve}


def _selfstop(cfg: Config) -> None:
    """Stop this Vast instance from inside on ANY run exit (complete / budget cap / crash), so a
    finished-or-dead run can never idle-bill while an external monitor is blind. No-op off-cluster (needs VAST_INSTANCE_ID)."""
    iid = os.environ.get("VAST_INSTANCE_ID")
    if iid and not cfg.smoke:
        try:
            from bench.vast import stop_self
            stop_self(iid, cfg.aws_region)
        except Exception as e:
            print(f"[selfstop] failed: {type(e).__name__}: {str(e)[:80]}")


def _arm_watchdog(cfg: Config) -> None:
    """Hard wall-clock backstop for a HUNG run (one that never reaches the per-batch budget check):
    force a self-stop + exit after MAX_RUNTIME_H (default = 1.15x the $/hr budget horizon). No network
    needed to fire. Daemon thread -> dies silently if the run finishes first."""
    if cfg.smoke or not os.environ.get("VAST_INSTANCE_ID"):
        return
    import threading
    horizon = float(os.environ.get("MAX_RUNTIME_H") or "0") or \
        (cfg.ceiling_usd / max(cfg.hourly_usd, 0.01)) * 1.15   # empty env ('') -> fall back, don't crash

    def _wd():
        time.sleep(max(600.0, horizon * 3600.0))
        print(f"[watchdog] hard max runtime ~{horizon:.1f}h exceeded — self-stopping instance + exiting")
        _selfstop(cfg)
        os._exit(2)

    threading.Thread(target=_wd, daemon=True).start()
    print(f"[watchdog] armed: hard self-stop at ~{horizon:.1f}h")


def main():
    ap = argparse.ArgumentParser(description="YORO Phase-0 benchmark driver")
    ap.add_argument("--smoke", action="store_true", help="local mock run — no GPU/AWS/spend")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default="phase0")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--fast-embed", action="store_true", help="HashEmbedder (fast, no semantic hits)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--zipf-sweep", default=os.environ.get("ZIPF_SWEEP", ""),
                    help="comma list e.g. 0.6,0.9,1.1,1.4,1.8 -> benefit-vs-repetitiveness curve")
    a = ap.parse_args()
    zsweep = tuple(float(x) for x in a.zipf_sweep.split(",") if x.strip())
    # default the out-dir to runs/<run_id> so DIFFERENT runs with the SAME sweep suffixes (e.g. e1-drift
    # and e7-drift both make drift0.1/) can't share LOCAL level dirs — a prior run's done=True checkpoint
    # would else make the new run SKIP every level. (S3 keys were already run_id-scoped; this fixes local.)
    cfg = Config(smoke=a.smoke, seeds=a.seeds, run_id=a.run_id, base_url=a.base_url,
                 model=a.model, fast_embed=a.fast_embed, wandb=a.wandb, zipf_sweep=zsweep,
                 out=a.out or f"runs/{a.run_id}{'-smoke' if a.smoke else ''}")
    if not cfg.sweep_values and cfg.zipf_sweep:         # back-compat: ZIPF_SWEEP -> generic sweep over zipf_s
        cfg = replace(cfg, sweep_param="zipf_s", sweep_values=cfg.zipf_sweep)
    try:
        _arm_watchdog(cfg)                              # inside try: even if IT raises, finally self-stops
        if cfg.sweep_values:
            rep = run_sweep(cfg)
            print(f"\n=== YORO sweep over {cfg.sweep_param} ===")
            for pt in rep["curve"]:
                v = pt.get("sweep_val", pt.get("zipf_s"))
                print(f"  {cfg.sweep_param}={v}  yoro_hit={pt.get('yoro_hit_rate', 0):.2f}  "
                      f"yoro_acc={pt.get('yoro_accuracy', 0):.2f}  "
                      f"Δstale={pt.get('delta_staleness', 0):+.3f}  Δbrittle={pt.get('delta_brittleness', 0):+.3f}")
            return
        report = run(cfg)
        print("\n=== Phase-0 report (mean over seeds) ===")
        for rn, m in report["rungs"].items():
            rp = m.get("replay_rate", {}).get("mean", 0.0)
            rw = m.get("replay_wrong", {}).get("mean", 0.0)
            it = m.get("input_tokens_total", {}).get("mean", 0.0)
            print(f"  {rn:14s} hit={m['hit_rate']['mean']:.2f} acc={m['accuracy']['mean']:.2f} "
                  f"stale={m['staleness']['mean']:.2f} brittle={m['brittleness']['mean']:.2f} "
                  f"out_tok={m['tokens_total']['mean']:.0f} in_tok={it:.0f} "
                  f"replay={rp:.2f} replay_wrong={rw:.2f}")
        if "yoro_vs_gptcache" in report:
            v = report["yoro_vs_gptcache"]
            print(f"  YORO vs GPTCache: acc Δ={v['accuracy']['mean_diff']:+.2f}  "
                  f"stale Δ={v['staleness']['mean_diff']:+.2f}  brittle Δ={v['brittleness']['mean_diff']:+.2f}")
    finally:
        _selfstop(cfg)                                  # stop the instance from inside on ANY exit


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        # last-resort: a crash BEFORE main()'s try/finally (argparse/Config) must still free the GPU,
        # else the box idles until the external monitor catches it.
        print(f"[fatal] {type(e).__name__}: {str(e)[:120]} — self-stopping instance")
        iid = os.environ.get("VAST_INSTANCE_ID")
        if iid and os.environ.get("WORKLOAD") != "__nostop__":
            try:
                from bench.vast import stop_self
                stop_self(iid, os.environ.get("AWS_REGION", "us-west-2"))
            except Exception:
                pass
        raise
