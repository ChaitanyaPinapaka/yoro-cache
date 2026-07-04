"""Vast.ai remaining-credit checker — ground-truth budget for the incremental-top-up
workflow. Instead of estimating spend from $/hr x elapsed, ask Vast for the ACTUAL account
balance and stop cleanly (flush checkpoint + results to S3) before it hits zero, so a
credit-out is a graceful pause, not a hard kill.

The Vast API key is pulled from Secrets Manager (yoro/vast-api-key) via the instance's
assumed role — no static Vast key ever lands on the box. Degrades safely: if the key or
API is unavailable, remaining() returns None and the run falls back to the $/hr BudgetGuard.
"""
from __future__ import annotations

from typing import Optional


class VastCredit:
    API = "https://console.vast.ai/api/v0/users/current/"

    def __init__(self, api_key: Optional[str] = None, min_usd: float = 2.0,
                 secret_id: str = "yoro/vast-api-key", region: str = "us-west-2"):
        self.api_key = api_key or self._key_from_secrets(secret_id, region)
        self.min_usd = min_usd
        self.last: Optional[float] = None

    @staticmethod
    def _key_from_secrets(secret_id: str, region: str) -> Optional[str]:
        try:
            import boto3
            v = boto3.client("secretsmanager", region_name=region).get_secret_value(SecretId=secret_id)
            return (v.get("SecretString") or "").strip() or None
        except Exception as e:
            print(f"[vast key from secrets unavailable: {str(e)[:60]}]")
            return None

    def _fetch_credit(self) -> Optional[float]:
        import requests
        r = requests.get(self.API, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=20)
        r.raise_for_status()
        d = r.json()
        c = d.get("credit", d.get("balance"))          # `credit` = prepaid balance remaining
        return float(c) if c is not None else None

    def remaining(self) -> Optional[float]:
        """Current Vast account credit in USD, or None if unavailable (fall back to BudgetGuard)."""
        if not self.api_key:
            return None
        try:
            self.last = self._fetch_credit()
        except Exception as e:
            print(f"[vast credit check failed: {str(e)[:60]}]")
        return self.last

    def low(self) -> bool:
        c = self.remaining()
        return (c is not None) and (c <= self.min_usd)


def stop_self(instance_id, region: str = "us-west-2", api_key: Optional[str] = None,
              secret_id: str = "yoro/vast-api-key") -> bool:
    """Stop THIS Vast instance via the Vast API, from inside the box. Invoked on the budget cap
    and on any run exit (see run_phase0._selfstop), so the shutdown needs NO external SSH monitor
    (network-independent: a hung monitor can never strand the instance). Key from Secrets Manager via the assumed
    role. Best-effort with retries; logs loudly on total failure (local monitor stays as backstop)."""
    import requests
    key = api_key or VastCredit._key_from_secrets(secret_id, region)
    if not key or not instance_id:
        print(f"[stop_self] CANNOT self-stop (key={bool(key)} id={instance_id}) — free the GPU manually!")
        return False
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"
    for attempt in (1, 2, 3):
        try:
            r = requests.put(url, headers={"Authorization": f"Bearer {key}"},
                             json={"state": "stopped"}, timeout=30)
            r.raise_for_status()
            print(f"[stop_self] requested STOP of Vast instance {instance_id} -> HTTP {r.status_code}")
            return True
        except Exception as e:
            print(f"[stop_self] attempt {attempt}/3 failed: {type(e).__name__}: {str(e)[:80]}")
    print(f"[stop_self] ALL attempts failed — instance {instance_id} may keep billing; monitor is backstop")
    return False
