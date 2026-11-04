"""Minimal client for the Evencut HTTP service (Python standard library only).

    python evencut_client.py                 # uses http://127.0.0.1:8000
    EVENCUT_URL=http://host:8000 python evencut_client.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

USER_AGENT = "evencut-client-python/1.0.0"


class EvencutAPIError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _detail(err: urllib.error.HTTPError) -> str:
    """Returns the error message of a response, whether or not its body is JSON."""
    body = err.read() or b""
    try:
        return str(json.loads(body).get("detail", ""))
    except (ValueError, AttributeError):  # plain-text or HTML bodies (e.g. a proxy's 502)
        return body.decode("utf-8", errors="replace").strip() or str(err)


@dataclass(frozen=True)
class Partition:
    objective: str
    parts: int
    ends: list[int]
    loads: list[int]
    max_load: int
    total: int
    sum_of_squares: str
    imbalance: float


class EvencutClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 60.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base_url + path, data=data, method="POST", headers={
                "Content-Type": "application/json", "User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.retries:
                    raise EvencutAPIError(e.code, _detail(e)) from None
            except urllib.error.URLError:
                if attempt == self.retries:
                    raise
            time.sleep(min(2 ** attempt * 0.25, 4.0))  # exponential backoff for 429 / 5xx / network
        raise RuntimeError("unreachable")

    def partition(self, weights: list[int], k: int, objective: str = "min_max", **options) -> Partition:
        return Partition(**self._post("/v1/partition", {"weights": weights, "k": k, "objective": objective, **options}))

    def pipeline_stages(self, layer_costs: list[float], num_stages: int, **options) -> dict:
        return self._post("/v1/ai/pipeline-stages", {"layer_costs": layer_costs, "num_stages": num_stages, **options})


if __name__ == "__main__":
    client = EvencutClient(os.environ.get("EVENCUT_URL", "http://127.0.0.1:8000"))
    p = client.partition([5, 3, 8, 2, 7, 4, 6, 1, 9, 5], 3)
    print("partition:", p.ends, p.loads, "max", p.max_load)
    plan = client.pipeline_stages([1.0] * 12, 3, embedding_cost=2.0, head_cost=3.0)
    print("pipeline:", plan["layers_per_stage"], plan["megatron_layout"])
    try:
        client.partition([1, 2, 3], 0)
    except EvencutAPIError as e:
        print("error handled:", e.status)
