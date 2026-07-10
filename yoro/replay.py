"""Replay applicability and deterministic post-replay validation."""
from __future__ import annotations

import json
from typing import Callable


def procedure_applicable(case, task: str) -> bool:
    """Conservative structural gate in addition to the embedding threshold."""
    artifact = getattr(case, "procedure", None) or {}
    steps = artifact.get("steps") if isinstance(artifact, dict) else None
    return bool((steps or case.steps or (case.reasoning or "").strip()) and task.strip())


def validate_output(content: str, request_body: dict | None = None,
                    verifier: Callable[[str], bool] | None = None) -> bool:
    if not (content or "").strip():
        return False
    if verifier is not None and not verifier(content):
        return False
    body = request_body or {}
    fmt = body.get("response_format") or (body.get("text") or {}).get("format") or {}
    typ = fmt.get("type") if isinstance(fmt, dict) else None
    if typ not in ("json_object", "json_schema"):
        return True
    try:
        value = json.loads(content)
    except Exception:
        return False
    if typ == "json_object" and not isinstance(value, dict):
        return False
    schema = fmt.get("json_schema", {}).get("schema", {}) if typ == "json_schema" else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []
    return isinstance(value, dict) and all(k in value for k in required)
