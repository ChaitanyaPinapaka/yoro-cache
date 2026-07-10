"""Structured form — store reasoning as ordered STEPS instead of a
flat text blob. This is the foundation for (a) inspection ("see why a task routed to
this reasoning"), (b) partial / step-level reuse, and (c) the decision-tree view.

`to_steps` is a dependency-free heuristic parser (numbered markers -> sentences). A
`StructuredReasoning` bundles the steps with a simple linear edge list (step i -> i+1);
richer DAGs (a step depending on several priors) are a later upgrade — the schema holds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ENUM = r"(?:\d+[.\)]|step\s+\d+\s*:|[-*])\s+"  # "1." "2)" "Step 3:" "- "

# Framing lines that are not reasoning steps. Two kinds:
#  * the model's own meta-preamble ("Thinking Process:", "Analyze the Request:"), and
#  * the behavior/format scaffolding WE inject, which a chat model echoes into its trace
#    ("You may use these known methods…", "Allowed methods: …", "Output format: …").
# Matched markdown-tolerantly so bold-wrapped framing ("**Analyze the Request:**") is caught.
_FRAMING_WHOLE = re.compile(
    r"^(?:here'?s\s+(?:a|my)\s+thinking\s+process"
    r"|thinking\s+process"
    r"|let'?s\s+think(?:\s+step[-\s]by[-\s]step)?"
    r"|reasoning"
    r"|analyze\s+(?:the\s+)?(?:request|user\s+input))\s*:?\s*$",
    re.IGNORECASE,
)
_FRAMING_PREFIX = re.compile(
    r"^(?:you\s+may\s+use\s+these\s+known\s+methods"
    r"|allowed\s+methods"
    r"|output\s+format)\b",
    re.IGNORECASE,
)


def _is_noise(line: str) -> bool:
    s = line.strip().lstrip("*#>-• ").rstrip("*").strip()  # tolerate markdown emphasis
    return bool(_FRAMING_WHOLE.match(s) or _FRAMING_PREFIX.match(s))


def to_steps(reasoning: str, max_steps: int = 24) -> list[str]:
    text = reasoning or ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # Drop trailing final-answer lines — the outcome is stored separately on the case,
    # so it shouldn't pollute the structured steps.
    lines = text.splitlines()
    while lines and re.match(
        r"\s*(?:final\s+answer|answer)\s*:", lines[-1], re.IGNORECASE
    ):
        lines.pop()
    # Drop framing/scaffolding lines ANYWHERE: the model's meta-preamble and the
    # behavior/format block we inject (which a chat model echoes mid-trace, not just
    # at the top), so they don't get glued into an adjacent real step.
    lines = [ln for ln in lines if ln.strip() and not _is_noise(ln)]
    text = "\n".join(lines)
    # Split on explicit step markers (matched at start-of-text too, not just after \n).
    parts = re.split(rf"(?:^|\n)\s*{_ENUM}", text, flags=re.IGNORECASE)
    steps = [p.strip() for p in parts if p.strip()]
    if len(steps) <= 1:  # fall back to sentence split
        steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    # Strip any residual leading enumerator (e.g. a "Step 1:" that opened the text).
    steps = [re.sub(rf"^\s*{_ENUM}", "", s, flags=re.IGNORECASE).strip() for s in steps]
    steps = [s for s in steps if s and not _is_noise(s)]  # drop meta-framing lines
    return steps[:max_steps]


@dataclass
class StructuredReasoning:
    """Extension point — a step-graph schema over `to_steps`; not used by the proxy."""

    steps: list[str] = field(default_factory=list)
    edges: list = field(default_factory=list)  # [(i, j)] linear chain by default

    @classmethod
    def parse(cls, reasoning: str) -> "StructuredReasoning":
        steps = to_steps(reasoning)
        edges = [(i, i + 1) for i in range(len(steps) - 1)]
        return cls(steps=steps, edges=edges)

    def to_dict(self) -> dict:
        return {"steps": self.steps, "edges": [list(e) for e in self.edges]}


@dataclass
class ProcedureArtifact:
    """Portable replay artifact; intentionally provider-neutral and serializable."""

    steps: list[str] = field(default_factory=list)
    edges: list = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    @classmethod
    def from_reasoning(cls, reasoning: str, deps: dict | None = None) -> "ProcedureArtifact":
        sr = StructuredReasoning.parse(reasoning)
        return cls(steps=sr.steps, edges=sr.edges,
                   dependencies=sorted((deps or {}).keys()))

    def to_dict(self) -> dict:
        return {
            "steps": self.steps, "edges": [list(e) for e in self.edges],
            "inputs": self.inputs, "outputs": self.outputs,
            "invariants": self.invariants, "dependencies": self.dependencies,
        }
