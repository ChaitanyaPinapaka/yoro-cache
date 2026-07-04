"""VLLMClient — the benchmark's model backend: any OpenAI-compatible endpoint (vLLM
serving gpt-oss-120B on the rented H100). Same (reasoning, outcome) interface as YORO's
other models, plus:
  * REAL token usage from the server (usage.completion_tokens) -> exposed as
    `last_completion_tokens`, so the token-savings axis is measured, not proxied.
  * retries + timeout, so a busy or just-booted server doesn't abort a multi-day run.
"""
from __future__ import annotations

import time

DEFAULT_SYSTEM = (
    "You are a careful reasoner. Think step by step. "
    "End your reply with a final line exactly of the form 'ANSWER: <final answer>'."
)

# REPLAY: apply a cached, already-validated method to new inputs — no fresh exploration. The
# 'reason once, then replay the plan' tier: cheap OUTPUT (short), at a plan-inflated INPUT.
REPLAY_SYSTEM = (
    "You are given a VALIDATED procedure that solved a very similar task. Apply it DIRECTLY to the new "
    "inputs: do not re-derive, re-plan, or explore — execute the procedure's steps on the new numbers "
    "and state the result. Be terse. End with a final line exactly of the form 'ANSWER: <final answer>'."
)

# TERSE control (spike arm, no plan): same 'be terse, don't explore' instruction as REPLAY but WITHOUT
# an injected procedure — isolates how much of replay's saving is just terseness vs the cached method.
TERSE_SYSTEM = (
    "Answer directly and tersely. Do not show working or explore. "
    "End with a final line exactly of the form 'ANSWER: <final answer>'."
)


class VLLMClient:
    name = "vllm"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 system: str = DEFAULT_SYSTEM, temperature: float = 0.0,
                 max_tokens: int = 2048, replay_max_tokens: int = 512, replay_effort: str = None,
                 timeout: int = 600, retries: int = 4):
        import requests
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.replay_max_tokens = replay_max_tokens          # replay is short by design (no exploration)
        self.replay_effort = replay_effort                  # None=model default (100% acc); "low"=cheap (80%) — the dial
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.last_completion_tokens = 0
        self.last_prompt_tokens = 0                          # INPUT tokens of the last call (replay inflates it)

    def _post(self, payload: dict) -> dict:
        last = None
        for i in range(self.retries):
            try:
                r = self._requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:                       # connection / 5xx / timeout
                last = e
                time.sleep(min(2 ** i, 20))              # backoff: 1,2,4,8,16,20…
        raise RuntimeError(f"vLLM endpoint failed after {self.retries} tries: {last}")

    @staticmethod
    def _parse(msg: dict):
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        src = content if content else reasoning
        if "</think>" in src:
            src = src.split("</think>")[-1].strip()
        outcome = src
        for line in reversed(src.splitlines()):
            s = line.strip()
            if s.upper().startswith("ANSWER:") or s.upper().startswith("FINAL ANSWER:"):
                outcome = s.split(":", 1)[1].strip()
                break
        return (reasoning or content), outcome

    def _usage(self, data: dict):
        u = data.get("usage") or {}
        self.last_completion_tokens = int(u.get("completion_tokens") or 0)
        self.last_prompt_tokens = int(u.get("prompt_tokens") or 0)

    def reason(self, task: str, system: str = None):
        self.calls += 1
        data = self._post({
            "model": self.model, "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system or self.system},
                         {"role": "user", "content": task}],
        })
        self._usage(data)
        return self._parse(data["choices"][0]["message"])

    def replay(self, task: str, plan):
        """Apply a cached METHOD to the current task — short, no exploration. `plan` is the cached
        steps (list) or raw reasoning (str). Returns (reasoning_text, outcome); token usage lands in
        last_completion_tokens (small OUTPUT) + last_prompt_tokens (plan-inflated INPUT).

        Sets reasoning_effort=low: the model is HANDED the method, so it shouldn't burn a full thinking
        trace re-deriving it — that's what makes replay's OUTPUT cheap on a reasoning model like gpt-oss
        (measured: low effort ~35% fewer completion tokens at equal accuracy). Honest: still counted."""
        self.calls += 1
        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)) if isinstance(plan, list) else str(plan)
        user = f"Validated procedure:\n{plan_text}\n\nApply it to this task:\n{task}"
        body = {
            "model": self.model, "temperature": self.temperature,
            "max_tokens": self.replay_max_tokens,
            "messages": [{"role": "system", "content": REPLAY_SYSTEM},
                         {"role": "user", "content": user}],
        }
        if self.replay_effort:                              # only send when set -> None uses the model default
            body["reasoning_effort"] = self.replay_effort
        data = self._post(body)
        self._usage(data)
        return self._parse(data["choices"][0]["message"])

    def complete(self, prompt: str, max_tokens=None) -> str:
        data = self._post({
            "model": self.model, "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}]})
        msg = data["choices"][0]["message"]
        src = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
        if "</think>" in src:
            src = src.split("</think>")[-1].strip()
        return src
