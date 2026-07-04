"""Prompt stream for the benchmark — labelled with ground truth so the harness can score
accuracy / staleness / brittleness. Prompt situations:

  cold      - first sighting of a task          -> should MISS (reason fresh)
  repeat    - a paraphrase of a known task      -> should HIT (reuse)
  drift     - a known task whose answer CHANGED -> should re-reason (serving old = stale)
  near_miss - a NOVEL task that LOOKS similar    -> should MISS (force-fitting = brittle)

`build_smoke_stream` is a small, high-quality, BALANCED built-in sample (math / code / QA /
reasoning / long-context) so the whole harness runs locally with no downloads. The real
Phase-0/1 suites (GSM8K / MATH / HumanEval / MBPP / MMLU / BBH / LongMemEval ...) load via
HuggingFace `datasets` on the cluster and produce Tasks in this exact shape — see load_hf().
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    text: str                       # what the cache sees
    key: str                        # ground-truth identity (NOT given to the cache)
    gold: str                       # correct answer AT THIS STEP ("" for populate/decision tasks)
    kind: str                       # cold | repeat | drift | near_miss | populate
    domain: str                     # math | code | qa | reasoning | longctx | pair
    deps: dict = field(default_factory=dict)   # dependency fingerprint at this step
    expect_reuse: Optional[bool] = None        # decision-mode (PAWS/QQP): the gold REUSE decision,
                                               # not an answer. True=should reuse, False=must refuse.


# (domain, key, gold, base_text, [paraphrases], near_miss=(text, gold) | None)
_BASE = [
    ("math", "sum_1_100", "5050", "What is the sum of all integers from 1 to 100?",
        ["Add up every whole number from 1 through 100.", "Compute the total of 1+2+...+100."],
        ("What is the sum of all integers from 1 to 50?", "1275")),
    ("math", "fact_6", "720", "What is 6 factorial?",
        ["Compute 6! (the factorial).", "What is the value of 6 factorial?"],
        ("What is 5 factorial?", "120")),
    ("qa", "cap_japan", "Tokyo", "What is the capital of Japan?",
        ["Name Japan's capital city.", "Japan's capital city is what?"], None),
    ("code", "reverse_str", "s[::-1]", "In Python, how do you reverse a string s?",
        ["Pythonic one-liner to reverse the string s?", "How do I reverse string s in Python?"], None),
    ("reasoning", "primes_10", "2 3 5 7", "List the prime numbers below 10.",
        ["Which prime numbers are less than 10?", "Enumerate the primes under 10."],
        ("List the prime numbers below 20.", "2 3 5 7 11 13 17 19")),
    ("longctx", "policy_branch", "main",
        "Per our repository policy document, what is the name of the default branch?",
        ["According to the policy doc, what's the default git branch?"], None),
]

DRIFT_KEY = "policy_branch"
DRIFT_QUERY = ("After the policy update, what is the repository's default branch now?", "trunk")


def build_smoke_stream(seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []

    def deps_for(key, ver=1):
        return {"doc": f"{key}#{ver}"} if key == DRIFT_KEY else {}

    # 1) cold pass — first sighting of each base task
    for dom, key, gold, text, paras, nm in _BASE:
        tasks.append(Task(text, key, gold, "cold", dom, deps_for(key)))

    # 2) repeat pass — paraphrases that SHOULD reuse
    rep = []
    for dom, key, gold, text, paras, nm in _BASE:
        for p in paras:
            rep.append(Task(p, key, gold, "repeat", dom, deps_for(key)))
    rng.shuffle(rep)
    tasks += rep

    # 3) drift event — the policy answer changes; its dependency fingerprint bumps
    tasks.append(Task(DRIFT_QUERY[0], DRIFT_KEY, DRIFT_QUERY[1], "drift", "longctx",
                      deps_for(DRIFT_KEY, ver=2)))

    # 4) near-miss pass — novel tasks that LOOK like a cached one (force-fit = brittle)
    nm_tasks = []
    for dom, key, gold, text, paras, nm in _BASE:
        if nm:
            nm_tasks.append(Task(nm[0], key + "_nm", nm[1], "near_miss", dom, {}))
    rng.shuffle(nm_tasks)
    tasks += nm_tasks
    return tasks


def gold_map(tasks: list[Task]) -> dict:
    """text -> correct answer, for the mock 'perfect reasoner' used in --smoke."""
    return {t.text: t.gold for t in tasks}


_PARA_TEMPLATES = ["Solve the following: {q}", "Please answer this — {q}",
                   "Consider the question: {q}", "I need the answer to: {q}"]


def _paraphrases(q: str, paraphraser=None, k: int = 2) -> list[str]:
    """Surface variants of a prompt to drive 'repeat' (should-reuse) signals. If a model
    `paraphraser(prompt)->str` is given, use it (real paraphrases); else cheap templates.
    A model paraphraser yields higher-quality paraphrases when available."""
    if paraphraser:
        try:
            out = paraphraser(f"Paraphrase this question {k} different ways, one per line, "
                              f"preserving the exact meaning and every number/entity:\n{q}")
            ps = [ln.strip("-•* ").strip() for ln in out.splitlines() if ln.strip()][:k]
            if len(ps) >= 1:
                return ps
        except Exception:
            pass
    return [t.format(q=q) for t in _PARA_TEMPLATES[:k]]


# ---- base verifiable loaders (answer-mode). HF ids VERIFIED 2026-06. ----
# NB original Hendrycks MATH (hendrycks/competition_math) is DMCA-disabled -> use the mirrors.
def _gsm8k(ld, n, rng):
    ds = ld("openai/gsm8k", "main", split="test")
    return [(ds[i]["question"], ds[i]["answer"].split("####")[-1].strip(), f"gsm8k:{i}", "math")
            for i in rng.sample(range(len(ds)), min(n, len(ds)))]


def _math500(ld, n, rng):
    ds = ld("HuggingFaceH4/MATH-500", split="test")                 # clean explicit `answer` column
    return [(ds[i]["problem"], str(ds[i]["answer"]).strip(), f"math500:{i}", "math")
            for i in rng.sample(range(len(ds)), min(n, len(ds)))]


def _mmlu(ld, n, rng):
    ds = ld("cais/mmlu", "all", split="test")
    L = "ABCD"
    out = []
    for i in rng.sample(range(len(ds)), min(n, len(ds))):
        r = ds[i]                                                   # `answer` is an int 0-3
        q = r["question"] + "\nOptions: " + "; ".join(f"{L[j]}) {c}" for j, c in enumerate(r["choices"]))
        out.append((q, f"{L[r['answer']]}) {r['choices'][r['answer']]}", f"mmlu:{i}", "qa"))
    return out


def _bbh(ld, n, rng):
    ds = ld("lukaemon/bbh", "causal_judgement", split="test")
    return [(ds[i]["input"], str(ds[i]["target"]).strip(), f"bbh:{i}", "reasoning")
            for i in rng.sample(range(len(ds)), min(n, len(ds)))]


BASE_LOADERS = {"math": [_gsm8k, _math500], "qa": [_mmlu], "reasoning": [_bbh]}
# code (HumanEval/MBPP — need an exec sandbox) + longctx (LongMemEval) are Phase-1.


# ---- human-labeled paraphrase / near-miss PAIRS (decision-mode). label 1=equivalent, 0=near-miss. ----
def _qqp_pairs(ld, n, rng):
    ds = ld("nyu-mll/glue", "qqp", split="validation")              # train/val labeled; TEST is unlabeled (-1)
    return [(ds[i]["question1"], ds[i]["question2"], int(ds[i]["label"]), f"qqp:{i}")
            for i in rng.sample(range(len(ds)), min(n, len(ds)))]


def _paws_pairs(ld, n, rng):
    ds = ld("google-research-datasets/paws", "labeled_final", split="validation")
    return [(ds[i]["sentence1"], ds[i]["sentence2"], int(ds[i]["label"]), f"paws:{i}")
            for i in rng.sample(range(len(ds)), min(n, len(ds)))]


PAIR_LOADERS = {"qqp": _qqp_pairs, "paws": _paws_pairs}


def build_workload(pool, stream_len: int, zipf_s: float = 1.1, n_paraphrases: int = 4,
                   paraphraser=None, seed: int = 0) -> list[Task]:
    """Turn a pool of distinct tasks into a realistic CUSTOMER request stream.

    Datasets give variety; this gives the recurrence that the whole thesis rests on. Task
    popularity follows a Zipf law (skew `zipf_s`) — a small HEAD recurs often, a long TAIL
    once — and each occurrence is a paraphrase, so recurrence is SEMANTIC (varied surface),
    not literal. First sighting of a task = cold (must reason); later paraphrased sightings =
    repeat (should reuse). The reuse rate EMERGES from zipf_s and stream_len/len(pool);
    tune those to a customer's real recurrence rather than hardcoding a hit-rate.
    """
    rng = random.Random(seed)
    variants = {}                                       # key -> (paraphrase pool, gold, domain)
    for prompt, gold, key, dom in pool:
        variants[key] = ([prompt] + _paraphrases(prompt, paraphraser, n_paraphrases - 1), gold, dom)
    keys = [item[2] for item in pool]
    weights = [1.0 / ((r + 1) ** zipf_s) for r in range(len(keys))]   # Zipf over (shuffled) ranks
    rng.shuffle(keys)
    seen = set()
    tasks = []
    for _ in range(stream_len):
        key = rng.choices(keys, weights=weights, k=1)[0]
        texts, gold, dom = variants[key]
        kind = "cold" if key not in seen else "repeat"
        seen.add(key)
        tasks.append(Task(rng.choice(texts), key, gold, kind, dom, {}))
    return tasks


# ---- synthetic STRESS workload: tunable DRIFT + NEAR-MISS for the safety sweeps ----
# HF tasks can't drift (their gold never changes) or supply answer-mode near-misses, so we
# synthesize parameterized operational queries whose answer is a small computation (real
# reasoning tokens, a real model answers them fresh). Each entity can then be:
#   DRIFT     - SAME entity, value changes -> new gold + dependency-version bump. A cache with
#               NO invalidation serves the old cached answer (stale); YORO's dep-invalidation
#               re-reasons (not stale).
#   NEAR-MISS - a DIFFERENT entity built as a look-alike sibling of a cached one -> embedding-close
#               but different gold. A cache with NO gate force-fits the neighbour's answer (brittle);
#               YORO's gate + distinct dep-key decline it.
#
# CRITICAL DESIGN: uniform templates would make ALL entities mutually confusable and floor a
# semantic cache's accuracy even at rate 0 — so the base pool draws DISTINCT SUBJECTS
# (different place+system+unit), so their embeddings are far apart and a semantic cache handles
# them correctly. Only the DELIBERATELY-injected near-miss is a close sibling of its parent — so
# GPTCache's error comes from the injected risk, not from base-pool confusion.
_STRESS_SUBJECTS = [
    ("the Frankfurt distribution warehouse", "pallets in stock", "logi"),
    ("the Singapore payments gateway", "transactions per second", "pay"),
    ("the Oregon ML training cluster", "GPUs online", "infra"),
    ("the London payroll ledger", "employees on record", "hr"),
    ("the Mumbai cold-storage facility", "vaccine crates held", "cold"),
    ("the Dublin support queue", "open tickets", "supp"),
    ("the Toronto solar array", "kilowatts generated", "energy"),
    ("the Nairobi microfinance branch", "active loans", "fin"),
    ("the Seattle coffee roastery", "kilograms roasted daily", "food"),
    ("the Osaka robotics line", "units assembled per hour", "mfg"),
    ("the Berlin bikeshare network", "bicycles docked", "transit"),
    ("the Cairo desalination plant", "cubic metres purified", "water"),
    ("the Denver hospital pharmacy", "prescriptions filled", "health"),
    ("the Lisbon fishing fleet", "tonnes landed", "marine"),
    ("the Helsinki data centre", "server racks powered", "infra2"),
    ("the Bogota bus depot", "coaches in service", "transit2"),
    ("the Perth wheat silo", "tonnes stored", "agri"),
    ("the Reykjavik geothermal well", "megawatts tapped", "energy2"),
    ("the Austin semiconductor fab", "wafers etched per shift", "chip"),
    ("the Manila call centre", "agents on shift", "bpo"),
    ("the Zurich vault", "gold bars held", "bank"),
    ("the Jakarta rice mill", "sacks milled", "agri2"),
    ("the Glasgow shipyard", "hull sections welded", "ship"),
    ("the Santiago vineyard", "barrels fermenting", "wine"),
    ("the Prague brewery", "hectolitres brewed", "beer"),
    ("the Accra cocoa depot", "cocoa sacks graded", "cocoa"),
    ("the Hanoi textile mill", "bolts of fabric woven", "textile"),
    ("the Calgary oil terminal", "barrels in the tank farm", "oil"),
    ("the Amsterdam tulip auction", "flower lots sold", "flora"),
    ("the Wellington wind farm", "turbines spinning", "wind"),
    ("the Chennai steel plant", "tonnes rolled", "steel"),
    ("the Montreal print shop", "reams printed", "print"),
    ("the Casablanca port crane", "containers lifted", "port"),
    ("the Warsaw glassworks", "sheets of glass cast", "glass"),
    ("the Buenos Aires cattle ranch", "head of cattle", "ranch"),
    ("the Stockholm recycling plant", "tonnes sorted", "recycle"),
    ("the Kyoto tea estate", "chests of tea cured", "tea"),
    ("the Lagos telecom tower", "cell sites linked", "telco"),
    ("the Vancouver sawmill", "cubic metres cut", "timber"),
    ("the Doha aquarium", "tanks under filtration", "aqua"),
    ("the Quito flower farm", "rose stems bunched", "flora2"),
    ("the Tbilisi wine cellar", "amphorae aging", "wine2"),
    ("the Bergen salmon farm", "pens stocked", "fish"),
    ("the Marrakesh tannery", "hides cured", "leather"),
    ("the Adelaide almond grove", "tonnes harvested", "nut"),
    ("the Riga amber workshop", "pendants polished", "craft"),
    ("the Sapporo ski resort", "lifts running", "ski"),
    ("the Medellin textile co-op", "garments stitched", "textile2"),
    ("the Tallinn e-gov datacenter", "services hosted", "egov"),
    ("the Muscat dhow harbour", "vessels moored", "harbour"),
]
# arithmetic ops (exact-integer so the model's answer is unambiguous). Distinct question phrasings.
_STRESS_OPS = [
    ("if that figure triples next quarter, what will it be", lambda v: v * 3),
    ("after 18 are removed, how many remain", lambda v: v - 18),
    ("summed across 6 identical sites, what is the total", lambda v: v * 6),
    ("if 45 more are added, what is the new total", lambda v: v + 45),
    ("at double the current level, what is the count", lambda v: v * 2),
]
_STRESS_LEADS = ["", "Current status: ", "For the record, ", "Latest reading — "]

# HARD ops: 4-step DEPENDENT chains (each step consumes the previous result). Deriving the chain from
# prose (with distractors) is expensive; APPLYING a known chain to a new value is cheap — exactly the
# gap plan-replay exploits. Integer-closed so the gold is unambiguous.
_STRESS_HARD_OPS = [
    ("add 40 for backlog, then triple for the three shifts, then subtract 27 for inspection, then double for the safety reserve",
     lambda v: ((v + 40) * 3 - 27) * 2),
    ("subtract 15 for spoilage, then multiply by 4 for the four depots, then add 60 for transfers, then halve for the audited count",
     lambda v: ((v - 15) * 4 + 60) // 2),
    ("double for the mirror site, then add 90 for reserves, then subtract 33 for losses, then triple for the annual projection",
     lambda v: ((v * 2 + 90) - 33) * 3),
    ("add 25 for intake, then multiply by 5 for the regions, then subtract 100 for overhead, then subtract 40 for tax",
     lambda v: ((v + 25) * 5 - 100) - 40),
]
_HARD_DISTRACTORS = ["the site logs {a} incidents and runs {b} audits yearly",
                     "there are {a} supervisors across {b} buildings",
                     "the ledger shows {a} vendors and {b} contracts"]


def _stress_text(subject, unit, opq, v, rng) -> str:
    lead = rng.choice(_STRESS_LEADS)                     # light surface variation -> embedding-close repeats
    return f"{lead}{subject} currently reports {v} {unit}. Compute: {opq}?"


def _hard_text_full(subject, unit, steps, v, rng) -> str:
    """COLD form: states the whole procedure — the model DERIVES it here (expensive), and it's cached."""
    d = rng.choice(_HARD_DISTRACTORS).format(a=rng.randint(3, 19), b=rng.randint(2, 9))
    lead = rng.choice(_STRESS_LEADS)
    return (f"{lead}{subject} holds {v} {unit}. Apply the standard procedure: {steps}. "
            f"(Unrelated: {d}.) What is the final {unit} count?")


def _hard_text_ref(subject, unit, steps, v, rng) -> str:
    """DRIFT/REPEAT re-ask: REFERENCES the established procedure by entity WITHOUT restating the steps.
    So a stateless solver (no-cache) lacks the method and must fail/guess — only a cache that carried
    the METHOD forward can answer cheaply. This makes the injected plan LOAD-BEARING, not dead weight:
    replay's savings can't be re-explained as 'be terse'. no-cache accuracy honestly drops."""
    lead = rng.choice(_STRESS_LEADS)
    return (f"{lead}{subject} now holds {v} {unit}. Recompute the final {unit} count using the "
            f"previously established procedure for {subject}. Give only the number.")


def _stress_task(ent, kind, ver, rng) -> Task:
    opq, opf = ent["op"]
    if ent.get("hard"):                                  # multi-step chain (expensive to derive)
        if kind in ("cold", "near_miss"):               # self-contained: STATE the procedure
            text = _hard_text_full(ent["subject"], ent["unit"], opq, ent["v"], rng)
        else:                                           # drift/repeat: REFERENCE it (no restate) -> plan is load-bearing
            text = _hard_text_ref(ent["subject"], ent["unit"], opq, ent["v"], rng)
    else:
        text = _stress_text(ent["subject"], ent["unit"], opq, ent["v"], rng)
    return Task(text, key=ent["key"], gold=str(opf(ent["v"])), kind=kind, domain="ops",
                deps={ent["key"]: ver})


_NM_QUALIFIERS = ["(annex site)", "(secondary line)", "(overflow unit)", "(east wing)",
                  "(night shift)", "(backup facility)", "(satellite branch)"]


def _near_miss_of(parent, rng) -> dict:
    """A look-alike SIBLING of `parent`: SAME subject text + a small qualifier (embedding-close so a
    naive cache force-fits the parent), but a DISTINCT key + a different value (so the correct answer
    differs, and YORO's dep-key/gate can decline it)."""
    qual = rng.choice(_NM_QUALIFIERS)
    v2 = parent["v"] + rng.randint(11, 60)               # different value -> different gold
    uid = rng.randint(10_000, 999_999)                   # UNIQUE key: each near-miss is a DISTINCT entity
    return {"subject": f"{parent['subject']} {qual}", "unit": parent["unit"], "op": parent["op"],
            "v": v2, "key": f"{parent['key']}|nm{uid}", "ver": 1, "hard": parent.get("hard", False)}


def build_stress_workload(n_unique: int = 40, stream_len: int = 600, zipf_s: float = 1.1,
                          drift_rate: float = 0.0, near_miss_rate: float = 0.0,
                          inval_fidelity: float = 1.0, hard: bool = False, seed: int = 0) -> list[Task]:
    """Zipf recurrence stream over DISTINCT operational subjects, with a fraction of recurrences
    turned into DRIFT (same entity, answer changed) or NEAR-MISS (a close sibling, different answer).
    Rates are per-recurrence and randomized per seed, so staleness/brittleness get real seed-to-seed
    variance. Base subjects are mutually distinguishable -> a semantic cache is only wrong on the
    INJECTED risk, not on base-pool confusion. `hard=True` uses 4-step
    dependent chains (expensive to derive, cheap to apply) — the regime where plan-REPLAY pays off."""
    rng = random.Random(seed * 7919 + 1)
    ops = _STRESS_HARD_OPS if hard else _STRESS_OPS
    ents = []
    for i in range(max(1, n_unique)):
        subject, unit, tag = _STRESS_SUBJECTS[i % len(_STRESS_SUBJECTS)]
        ents.append({"subject": subject, "unit": unit, "op": ops[i % len(ops)],
                     "v": rng.randint(120, 900), "key": f"ops:{tag}", "ver": 1, "hard": hard})
    idxs = list(range(len(ents)))
    weights = [1.0 / ((r + 1) ** zipf_s) for r in range(len(ents))]
    rng.shuffle(idxs)                                    # decouple popularity rank from subject order
    seen: set = set()
    tasks: list[Task] = []
    for _ in range(max(1, stream_len)):
        idx = rng.choices(idxs, weights=weights, k=1)[0]
        ent = ents[idx]
        if idx not in seen:                             # first sighting -> cold (must reason)
            seen.add(idx)
            tasks.append(_stress_task(ent, "cold", ent["ver"], rng))
            continue
        roll = rng.random()
        if roll < drift_rate:                            # DRIFT: same entity, value changed -> new gold
            ent["v"] += rng.randint(20, 80)
            if rng.random() < inval_fidelity:            # signal reaches YORO w.p. fidelity (E4);
                ent["ver"] += 1                          # a "silent" drift (no bump) -> YORO can't invalidate
            tasks.append(_stress_task(ent, "drift", ent["ver"], rng))
        elif roll < drift_rate + near_miss_rate:         # NEAR-MISS: a close SIBLING (distinct key)
            nm = _near_miss_of(ent, rng)
            tasks.append(_stress_task(nm, "near_miss", nm["ver"], rng))
        else:                                            # REPEAT: same entity+value -> should reuse
            tasks.append(_stress_task(ent, "repeat", ent["ver"], rng))
    return tasks


def load_hf(domains=None, n_unique: int = 40, stream_len: int = 600, zipf_s: float = 1.1,
            n_paraphrases: int = 4, pairs=("qqp",), n_pairs: int = 120,
            drift_probes: bool = True, seed: int = 0, paraphraser=None) -> list[Task]:
    """A realistic customer-workload stream — NOT a bag of unique questions.

      1. BASE POOL: `n_unique` distinct verifiable tasks across `domains` (the variety).
      2. RECURRENCE: replay a length-`stream_len` Zipf-popularity request stream of
         PARAPHRASES of those tasks (build_workload) — the head recurs, the tail is novel.
         This is the thesis: repeated reasoning a customer shouldn't redo.
      3. MATCHER PROBE: human-labeled QQP/PAWS pairs — populate q1, then a DECISION q2
         (label 1 => should reuse, 0 => near-miss must refuse). Precision/recall of reuse.
      4. CONTROLLED PROBES: drift (staleness) + synthetic near-miss (brittleness) with gold.

    Possible extensions: lm-sys/llm-decontaminator equivalence validation; real temporal drift
    (SituatedQA-Temp / TempLAMA / LongMemEval knowledge-update); big-context LongMemEval /
    LoCoMo; code with an execution sandbox. (HF ids verified 2026-06.)"""
    try:
        from datasets import load_dataset as ld
    except Exception as e:
        raise RuntimeError("load_hf needs HuggingFace `datasets` (installed on the instance "
                           "by the bootstrap); use build_smoke_stream locally.") from e
    rng = random.Random(seed)
    domains = list(domains or ["math", "qa", "reasoning"])

    # 1) base pool of distinct tasks across domains
    pool = []
    per = max(1, n_unique // max(1, len(domains)))
    for dom in domains:
        for loader in BASE_LOADERS.get(dom, []):
            pool += loader(ld, per, rng)
    pool = pool[:n_unique]

    # 2) the recurrence workload (the thesis core)
    tasks = build_workload(pool, stream_len, zipf_s, n_paraphrases, paraphraser, seed)

    # 3) human-labeled matcher precision/recall pairs
    for pname in pairs:
        loader = PAIR_LOADERS.get(pname)
        if not loader:
            continue
        for q1, q2, label, key in loader(ld, n_pairs, rng):
            tasks.append(Task(q1, key, "", "populate", "pair", {}))                 # seed cache, unscored
            tasks.append(Task(q2, key, "", "repeat" if label == 1 else "near_miss",
                              "pair", {}, expect_reuse=(label == 1)))                # decision-mode

    # 4) controlled staleness + brittleness probes (gold answers)
    if drift_probes:
        tasks += build_smoke_stream(seed)
    return tasks
