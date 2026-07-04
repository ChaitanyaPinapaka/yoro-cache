"""Mine reusable behaviors from your OpenCode sessions -> AGENTS.md.

For an all-agentic tool like OpenCode: instead of
caching whole turns (which the safe proxy correctly refuses), it harvests the
reusable METHODS from your past sessions and writes them where every future
session (OpenCode, Codex, Claude Code) will read them — a self-populating AGENTS.md.

OpenCode stores sessions in SQLite (~/.local/share/opencode/opencode.db):
  message(role=assistant) -> part(type=reasoning|text) holds the thinking traces.
We read those (READ-ONLY), mine named procedures with YORO's behavior extractor
(one model.complete call per trace), rank by how often they recur, and write the
top-N into a MARKED block in AGENTS.md so re-runs update just that block and never
touch your own content.

Run (needs your model endpoint up — it mines with one call per trace):
    YORO_UPSTREAM=http://127.0.0.1:8000/v1 python -m yoro.opencode_behaviors --out AGENTS.md
    ... --dir ~/Work/myproject  # only sessions in that project
    ... --dry-run               # mine + print the block, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3

from .behaviors import BehaviorStore, extract_behaviors
from .embeddings import SentenceTransformerEmbedder

DB_DEFAULT = os.path.expanduser("~/.local/share/opencode/opencode.db")
BEGIN = "<!-- YORO:BEGIN (auto-generated reusable methods — do not edit inside) -->"
END = "<!-- YORO:END -->"
MIN_TRACE = 200  # skip trivial traces


def read_traces(
    db: str, dir_filter: str | None = None, limit: int | None = None
) -> list[str]:
    """One trace per assistant message: its reasoning + text parts joined. Read-only."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()
    sess_dir = {
        sid: (d or "") for sid, d in cur.execute("SELECT id, directory FROM session")
    }
    assistant_msgs = []
    for mid, sid, data in cur.execute(
        "SELECT id, session_id, data FROM message ORDER BY time_created"
    ):
        try:
            j = json.loads(data)
        except Exception:
            continue
        if j.get("role") != "assistant":
            continue
        if dir_filter and dir_filter not in sess_dir.get(sid, ""):
            continue
        assistant_msgs.append(mid)
    traces = []
    for mid in assistant_msgs:
        chunks = []
        for (pd,) in cur.execute(
            "SELECT data FROM part WHERE message_id=? ORDER BY time_created", (mid,)
        ):
            try:
                pj = json.loads(pd)
            except Exception:
                continue
            if pj.get("type") in ("reasoning", "text") and pj.get("text"):
                chunks.append(pj["text"])
        trace = "\n".join(chunks).strip()
        if len(trace) >= MIN_TRACE:
            traces.append(trace)
    con.close()
    return traces[:limit] if limit else traces


def make_completer(
    upstream: str, model: str, api_key: str = "sk-local", max_tokens: int = 2048
):
    """A bare OpenAI-compatible completion callable — avoids importing the models pkg."""
    import requests

    base = upstream.rstrip("/")

    def complete(prompt: str) -> str:
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        src = (msg.get("content") or "").strip() or (
            msg.get("reasoning_content") or ""
        ).strip()
        if "</think>" in src:
            src = src.split("</think>")[-1].strip()
        return src

    return complete


def render_block(store: BehaviorStore, top: int) -> str:
    behs = sorted(store.items, key=lambda b: -len(b.from_cases))[:top]
    lines = [
        BEGIN,
        "## Reusable methods (mined by YORO from past sessions)",
        "_Auto-generated. Edit anything OUTSIDE this block; re-running the exporter "
        "rewrites only what's between the YORO markers._",
        "",
    ]
    for b in behs:
        n = len(b.from_cases)
        seen = f"  _(seen in {n} traces)_" if n > 1 else ""
        lines.append(f"- **{b.name}**: {b.instruction}{seen}")
    lines.append(END)
    return "\n".join(lines)


def write_agents(out: str, block: str) -> str:
    """Idempotent: replace the marked block if present, else append. Returns the mode."""
    existing = ""
    if os.path.exists(out):
        with open(out) as f:
            existing = f.read()
    if BEGIN in existing and END in existing:
        pre, post = existing.split(BEGIN)[0], existing.split(END, 1)[1]
        new, mode = pre + block + post, "updated"
    elif existing.strip():
        new, mode = existing.rstrip() + "\n\n" + block + "\n", "appended"
    else:
        new, mode = block + "\n", "created"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(new)
    return mode


def main():
    ap = argparse.ArgumentParser(
        description="Mine reusable behaviors from OpenCode sessions -> AGENTS.md"
    )
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--out", default="AGENTS.md")
    ap.add_argument(
        "--dir", default=None, help="only sessions whose directory contains this string"
    )
    ap.add_argument(
        "--limit", type=int, default=40, help="max traces to mine (bounds model calls)"
    )
    ap.add_argument(
        "--top", type=int, default=15, help="behaviors to write into AGENTS.md"
    )
    ap.add_argument(
        "--upstream",
        default=os.environ.get("YORO_UPSTREAM", "http://127.0.0.1:8000/v1"),
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("YORO_MODEL", "local"),
    )
    ap.add_argument(
        "--max-tokens", type=int, default=2048,
        help="completion budget per mining call (reasoning models think first; keep generous)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="mine + print the block, write nothing"
    )
    a = ap.parse_args()

    traces = read_traces(a.db, a.dir, a.limit)
    print(
        f"found {len(traces)} assistant reasoning traces"
        f"{' in ' + a.dir if a.dir else ''}",
        flush=True,
    )
    if not traces:
        print("nothing to mine — try without --dir, or use OpenCode a bit first")
        return

    print("loading embedder…", flush=True)
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    store = BehaviorStore()
    complete = make_completer(a.upstream, a.model, max_tokens=a.max_tokens)
    print(f"mining behaviors via {a.upstream} ({a.model})…", flush=True)
    for i, tr in enumerate(traces, 1):
        before = len(store)
        try:
            extract_behaviors(tr, complete, emb, store, from_case=f"trace-{i}")
        except Exception as e:
            print(
                f"  [{i}/{len(traces)}] skip ({type(e).__name__}: {str(e)[:60]})",
                flush=True,
            )
            continue
        print(
            f"  [{i}/{len(traces)}] behaviors={len(store)} (+{len(store) - before})",
            flush=True,
        )

    print(f"\nmined {len(store)} unique behaviors", flush=True)
    if not store.items:
        print("no behaviors extracted (is the model endpoint up?)")
        return
    block = render_block(store, a.top)
    if a.dry_run:
        print("\n--- AGENTS.md block (dry-run, not written) ---\n" + block)
        return
    mode = write_agents(a.out, block)
    print(
        f"{mode} {min(len(store.items), a.top)} behaviors -> {os.path.abspath(a.out)}"
    )


if __name__ == "__main__":
    main()
