# Benchmark runbook

Everything runs through one driver, `bench/run_phase0.py`, configured by env vars.
`--smoke` needs no GPU or cloud: a mock "perfect reasoner" makes every measured error a
*cache* error, which is what the safety metrics isolate.

```bash
pip install "yoro-cache[embed]"
python -m bench.run_phase0 --smoke              # full pipeline, no spend
```

## Real runs

Point the driver at any OpenAI-compatible endpoint serving your model (we used vLLM):

```bash
python -m bench.run_phase0 --seeds 15 --run-id myrun \
  --base-url http://127.0.0.1:8000/v1 --model <served-model-name>
```

Optional cloud mirroring: set `S3_BUCKET` (and `CW_LOG_GROUP`) to stream events,
checkpoints, and reports off-box; unset, results stay local under `--out`.

## The sweeps behind the released curves

Each sweep writes one `curve.json` (aggregates) plus per-level `report.json` and
`events.jsonl` (every scored query — re-derive any metric offline). `RUNGS` selects
the ladder subset; sweeps checkpoint per level and resume losslessly.

```bash
# E1' — staleness vs drift (easy workload, matched tau=0.90)
RUNGS=no-cache,gptcache-semantic,yoro WORKLOAD=stress \
SWEEP_PARAM=drift_rate SWEEP_VALUES=0,0.05,0.1,0.2,0.3,0.4 \
N_UNIQUE=40 STREAM_LEN=600 MIN_SEEDS=12 python -m bench.run_phase0 --seeds 15 --run-id e1

# E2' — brittleness vs near-miss rate
... SWEEP_PARAM=near_miss_rate SWEEP_VALUES=0,0.05,0.1,0.2,0.3,0.4 DRIFT_RATE=0.05 --run-id e2

# E4 — staleness vs invalidation-signal fidelity (the oracle ablation)
... SWEEP_PARAM=inval_fidelity SWEEP_VALUES=1.0,0.9,0.7,0.5,0.0 DRIFT_RATE=0.3 --run-id e4

# E7 — the four-tier Pareto (hard method-in-history workload, matched tau=0.80)
RUNGS=no-cache,gptcache-semantic,yoro,yoro-replay,yoro-replay-low WORKLOAD=stress HARD=1 \
TAU_HIT=0.80 GPTCACHE_TAU=0.80 SWEEP_PARAM=drift_rate SWEEP_VALUES=0,0.1,0.2,0.3,0.4 \
python -m bench.run_phase0 --seeds 15 --run-id e7

# E5 — second-model replication (any OpenAI-compatible server; we used Qwen2.5-32B-AWQ)
#   as E7 with SWEEP_VALUES=0,0.2,0.4 and RUNGS=...,yoro-replay (no effort dial on non-reasoning models)
```

Replay-quality spike (the cheap go/no-go before E7-style runs):

```bash
python -m bench.spike_replay --base-url http://127.0.0.1:8000/v1 --model <name> --n 25
```

## Key knobs

| env | default | meaning |
|---|---|---|
| `DRIFT_RATE` / `NEAR_MISS_RATE` | 0 | per-recurrence injection rates |
| `INVAL_FIDELITY` | 1.0 | probability a dep-change signal is delivered |
| `HARD` | off | 4-step method-in-history workload (replay regime) |
| `TAU_HIT` / `GPTCACHE_TAU` | 0.90 / 0.90 | matched reuse thresholds |
| `RUNGS` | all | ladder subset |
| `MIN_SEEDS` / `CI_TARGET` | 12 / 0.02 | convergence early-stop |
| `CEILING_USD` / `HOURLY_USD` | 480 / 0 | in-run budget guard (self-stop hooks) |

Metrics per rung: hit-rate, accuracy, staleness (= `outdated_rate` + `repoisoned_rate`
— the failure taxonomy), brittleness, `replay_rate`, `replay_wrong`, and output/input
tokens separately.
