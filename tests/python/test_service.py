"""Tests for the HTTP service (FastAPI TestClient; no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from evencut import service  # noqa: E402

client = TestClient(service.app)
EXAMPLE = [5, 3, 8, 2, 7, 4, 6, 1, 9, 5]


def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["backend"] in ("native", "pure")


def test_partition_min_max_and_sum_squares():
    r = client.post("/v1/partition", json={"weights": EXAMPLE, "k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ends"] == [4, 7, 10] and body["max_load"] == 18 and body["sum_of_squares"] == "838"
    r = client.post("/v1/partition", json={"weights": EXAMPLE, "k": 3, "objective": "sum_squares"})
    assert r.status_code == 200 and r.json()["sum_of_squares"] == "838"


def test_big_integers_survive_json():
    w = [(1 << 63), (1 << 63) - 1]
    r = client.post("/v1/partition", json={"weights": w, "k": 2})
    assert r.status_code == 200
    assert r.json()["total"] == (1 << 64) - 1
    assert r.json()["sum_of_squares"] == str(w[0] ** 2 + w[1] ** 2)


@pytest.mark.parametrize("payload, status", [
    ({"weights": EXAMPLE, "k": 0}, 422),                        # schema: k >= 1
    ({"weights": [], "k": 1}, 422),                             # schema: non-empty
    ({"weights": [1.5, 2], "k": 1}, 422),                       # not an integer
    ({"weights": [1 << 64], "k": 1}, 422),                      # above 2^64 - 1
    ({"weights": EXAMPLE, "k": 2, "colour": "red"}, 422),       # unknown field
    ({"weights": EXAMPLE, "k": 2, "forbidden_cuts": [10]}, 400),  # library: position out of range
    ({"weights": EXAMPLE, "k": 4, "secondary": [1] * 10, "secondary_limit": 2}, 400),  # infeasible
    ({"weights": [(1 << 64) - 1, 1], "k": 1}, 400),             # total overflows 64 bits
    ({"weights": EXAMPLE, "k": 2, "forbidden_cuts": [1 << 64]}, 422),  # beyond 64 bits
])
def test_errors(payload, status):
    r = client.post("/v1/partition", json=payload)
    assert r.status_code == status, r.text
    if status == 400:
        assert r.json()["detail"].startswith("evencut:")


def test_evaluate_tree_pipeline_batches():
    r = client.post("/v1/evaluate", json={"weights": EXAMPLE, "ends": [4, 7, 10]})
    assert r.status_code == 200 and r.json()["loads"] == [18, 17, 15]
    r = client.post("/v1/tree-partition", json={"parents": [None, 0, 0, 1], "weights": [4, 3, 5, 2], "k": 2})
    assert r.status_code == 200 and r.json()["max_load"] == 9
    r = client.post("/v1/tree-partition", json={"parents": [-1, 0, 0, 1], "weights": [4, 3, 5, 2], "k": 2})
    assert r.status_code == 200 and r.json()["max_load"] == 9
    r = client.post("/v1/ai/pipeline-stages",
                    json={"layer_costs": [1.0] * 12, "num_stages": 3, "embedding_cost": 2.0, "head_cost": 3.0})
    assert r.status_code == 200, r.text
    plan = r.json()
    assert plan["bottleneck"] == pytest.approx(6.0)
    assert plan["megatron_layout"].startswith("E") and plan["megatron_layout"].endswith("L")
    assert len(plan["torchtitan_split_points"]) == 2 and plan["deepspeed_parts"][-1] == 12
    r = client.post("/v1/ai/token-budget-batches", json={"sample_tokens": [512, 900, 300, 1200, 700, 450], "max_tokens": 2048})
    assert r.status_code == 200 and r.json()["max_load"] <= 2048


def test_body_limits():
    small = TestClient(service._BodyLimit(service.api, 64))
    r = small.post("/v1/partition", json={"weights": list(range(100)), "k": 2})
    assert r.status_code == 413
    r = client.post("/v1/partition", content=(chunk for chunk in [b'{"weights":[1,2],', b'"k":1}']),
                    headers={"content-type": "application/json"})
    assert r.status_code == 411


def test_openapi_quality_checklist():
    spec = service.api.openapi()
    ids = [op["operationId"] for path in spec["paths"].values() for op in path.values()]
    assert len(ids) == len(set(ids)) and all(ids)
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert "200" in op["responses"]
            if method == "post":
                assert {"400", "411", "413", "422"} <= set(op["responses"]), path
                assert op.get("tags"), path
    schemas = spec["components"]["schemas"]
    assert "examples" in schemas["PartitionRequest"]
    assert {t["name"] for t in spec["tags"]} == {"core", "ai", "meta"}


def test_huge_pipeline_costs_give_400_not_500():
    r = client.post("/v1/ai/pipeline-stages", json={"layer_costs": [1e308, 1e308], "num_stages": 2})
    assert r.status_code == 400 and "too large" in r.json()["detail"]
