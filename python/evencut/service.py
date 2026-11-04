"""HTTP/JSON service for Evencut (FastAPI). Any language can use the engine over HTTP.

Run locally:   uvicorn evencut.service:app --host 127.0.0.1 --port 8000
Docs:          http://127.0.0.1:8000/docs   (OpenAPI: /openapi.json)

Requires the ``service`` extra: ``pip install evencut[service]``.

Limits (environment variables): EVENCUT_MAX_ITEMS (default 2,000,000 items per list) and
EVENCUT_MAX_BODY_BYTES (default 64 MiB). Requests must send Content-Length. There is no
authentication: bind to localhost or put the service behind your own gateway.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import evencut
from evencut import EvencutError, ai

U64 = (1 << 64) - 1
MAX_ITEMS = int(os.environ.get("EVENCUT_MAX_ITEMS", "2000000"))
MAX_BODY_BYTES = int(os.environ.get("EVENCUT_MAX_BODY_BYTES", str(64 * 1024 * 1024)))

U64Int = Annotated[int, Field(ge=0, le=U64)]
Index = Annotated[int, Field(ge=0, le=U64)]
Count = Annotated[int, Field(ge=1, le=MAX_ITEMS)]
Real = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]

_WEIGHTS_DOC = "Non-negative integers (up to 2^64-1; the total must also fit). JavaScript clients: keep values below 2^53."


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str = Field(examples=["evencut: k must be at least 1"])


class PartitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"weights": [5, 3, 8, 2, 7, 4, 6, 1, 9, 5], "k": 3, "objective": "min_max"}]})
    weights: list[U64Int] = Field(min_length=1, max_length=MAX_ITEMS, description=_WEIGHTS_DOC)
    k: Count = Field(description="Number of parts.")
    objective: Literal["min_max", "sum_squares"] = Field(
        default="min_max", description="min_max: smallest largest part. sum_squares: smallest sum of squared loads.")
    forbidden_cuts: list[Index] = Field(default_factory=list, max_length=MAX_ITEMS,
                                        description="Cut positions numbered like `ends`: b keeps items b-1 and b "
                                                    "together, so b never appears in `ends`. Valid: 1 .. n-1.")
    secondary: Optional[list[U64Int]] = Field(default=None, max_length=MAX_ITEMS,
                                              description="Second value per item, e.g. memory (min_max only).")
    secondary_limit: Optional[U64Int] = Field(default=None, description="Per-part limit for `secondary`.")
    polish: bool = Field(default=True, description="min_max: even out the non-maximal parts.")


class PartitionResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "objective": "min_max", "parts": 3, "ends": [4, 7, 10], "loads": [18, 17, 15], "max_load": 18,
        "total": 50, "sum_of_squares": "838", "imbalance": 1.08}]})
    objective: str
    parts: int
    ends: list[int] = Field(description="Exclusive end index of each part.")
    loads: list[int]
    max_load: int
    total: int
    sum_of_squares: str = Field(description="Exact decimal value (can exceed 2^53).")
    imbalance: float = Field(description="max_load / (total / parts); 1.0 is perfectly even.")


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"weights": [5, 3, 8, 2, 7, 4, 6, 1, 9, 5], "ends": [4, 7, 10]}]})
    weights: list[U64Int] = Field(min_length=1, max_length=MAX_ITEMS, description=_WEIGHTS_DOC)
    ends: list[Index] = Field(min_length=1, max_length=MAX_ITEMS, description="Strictly increasing; last == len(weights).")


class TreeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"parents": [None, 0, 0, 1], "weights": [4, 3, 5, 2], "k": 2}]})
    parents: list[Optional[Annotated[int, Field(ge=-1)]]] = Field(
        min_length=1, max_length=MAX_ITEMS, description="Parent of each node; null or -1 for the root.")
    weights: list[U64Int] = Field(min_length=1, max_length=MAX_ITEMS, description=_WEIGHTS_DOC)
    k: Count


class TreeResponse(BaseModel):
    parts: int
    part_of: list[int]
    loads: list[int]
    cut_nodes: list[int]
    max_load: int
    total: int
    sum_of_squares: str
    imbalance: float


class PipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "layer_costs": [1.0] * 12, "num_stages": 3, "embedding_cost": 2.0, "head_cost": 3.0}]})
    layer_costs: list[Real] = Field(min_length=1, max_length=MAX_ITEMS,
                                    description="Measured time (or FLOPs) of each transformer layer.")
    num_stages: Count
    embedding_cost: Real = 0.0
    head_cost: Real = 0.0
    layer_memory: Optional[list[Real]] = Field(default=None, max_length=MAX_ITEMS)
    embedding_memory: Real = 0.0
    head_memory: Real = 0.0
    memory_limit: Optional[Real] = None


class PipelineResponse(BaseModel):
    stages: list[list[int]] = Field(description="[first_layer, end_layer) per stage.")
    stage_costs: list[float]
    stage_memory: Optional[list[float]]
    bottleneck: float
    balance: float = Field(description="Mean stage cost / slowest stage cost.")
    layers_per_stage: list[int]
    torchtitan_split_points: list[str]
    megatron_layout: str
    deepspeed_parts: list[int]


class BatchesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"sample_tokens": [512, 900, 300, 1200, 700, 450], "max_tokens": 2048}]})
    sample_tokens: list[U64Int] = Field(min_length=1, max_length=MAX_ITEMS)
    max_tokens: Annotated[int, Field(ge=1, le=U64)]


class HealthResponse(BaseModel):
    status: str
    backend: str
    version: str


# ---------------------------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------------------------
_ERRORS = {
    400: {"model": ErrorResponse, "description": "Invalid input or infeasible limits"},
    411: {"model": ErrorResponse, "description": "Content-Length header missing"},
    413: {"model": ErrorResponse, "description": "Request body larger than EVENCUT_MAX_BODY_BYTES"},
    422: {"description": "Request does not match the schema"},
    500: {"model": ErrorResponse, "description": "Internal error"},
}

api = FastAPI(
    title="Evencut",
    version=evencut.__version__,
    description=("Exact, balanced partitioning of ordered workloads: split costs into k contiguous parts "
                 "(or a tree into k connected parts) so the parts are as even as possible."),
    openapi_tags=[
        {"name": "core", "description": "Partition, evaluate, trees"},
        {"name": "ai", "description": "Helpers for AI training workloads"},
        {"name": "meta", "description": "Health"},
    ],
    servers=[{"url": "http://127.0.0.1:8000", "description": "Local"}],
)


@api.exception_handler(EvencutError)
def _evencut_error(_request, exc: EvencutError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@api.get("/healthz", operation_id="health", tags=["meta"], response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe; also reports which backend (native or pure) is active."""
    return HealthResponse(status="ok", backend=evencut.backend, version=evencut.__version__)


# CPU-bound handlers are plain `def`: FastAPI runs them in a worker thread, so they never block
# the event loop.
@api.post("/v1/partition", operation_id="partition", tags=["core"], response_model=PartitionResponse,
          responses=_ERRORS)
def partition(req: PartitionRequest) -> PartitionResponse:
    """Split weights into k contiguous parts (exact optimum, with the cut positions)."""
    r = evencut.partition(req.weights, req.k, objective=req.objective, forbidden_cuts=req.forbidden_cuts,
                          secondary=req.secondary, secondary_limit=req.secondary_limit, polish=req.polish)
    return PartitionResponse(**r.to_dict())


@api.post("/v1/evaluate", operation_id="evaluate", tags=["core"], response_model=PartitionResponse,
          responses=_ERRORS)
def evaluate(req: EvaluateRequest) -> PartitionResponse:
    """Measure an existing split, e.g. a framework default, to compare with the optimum."""
    return PartitionResponse(**evencut.evaluate(req.weights, req.ends).to_dict())


@api.post("/v1/tree-partition", operation_id="partitionTree", tags=["core"], response_model=TreeResponse,
          responses=_ERRORS)
def tree_partition(req: TreeRequest) -> TreeResponse:
    """Split a tree into k connected parts, minimising the largest part."""
    return TreeResponse(**evencut.partition_tree(req.parents, req.weights, req.k).to_dict())


@api.post("/v1/ai/pipeline-stages", operation_id="pipelineStages", tags=["ai"], response_model=PipelineResponse,
          responses=_ERRORS)
def pipeline_stages(req: PipelineRequest) -> PipelineResponse:
    """Assign transformer layers to pipeline stages so the slowest stage is as fast as possible."""
    plan = ai.pipeline_stages(req.layer_costs, req.num_stages, embedding_cost=req.embedding_cost,
                              head_cost=req.head_cost, layer_memory=req.layer_memory,
                              embedding_memory=req.embedding_memory, head_memory=req.head_memory,
                              memory_limit=req.memory_limit)
    return PipelineResponse(
        stages=[list(s) for s in plan.stages], stage_costs=list(plan.stage_costs),
        stage_memory=list(plan.stage_memory) if plan.stage_memory is not None else None,
        bottleneck=plan.bottleneck, balance=plan.balance, layers_per_stage=plan.layers_per_stage(),
        torchtitan_split_points=plan.torchtitan_split_points(), megatron_layout=plan.megatron_layout(),
        deepspeed_parts=plan.deepspeed_parts())


@api.post("/v1/ai/token-budget-batches", operation_id="tokenBudgetBatches", tags=["ai"],
          response_model=PartitionResponse, responses=_ERRORS)
def token_budget_batches(req: BatchesRequest) -> PartitionResponse:
    """Fewest consecutive batches within max_tokens, evened out."""
    return PartitionResponse(**ai.token_budget_batches(req.sample_tokens, req.max_tokens).to_dict())


# ---------------------------------------------------------------------------------------------
# request-size guard (outermost ASGI layer)
# ---------------------------------------------------------------------------------------------
class _BodyLimit:
    """Rejects bodies larger than `limit` bytes (413) and bodies without Content-Length (411)
    before any JSON parsing happens."""

    def __init__(self, inner, limit: int):
        self.inner = inner
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            headers = dict(scope.get("headers") or [])
            raw = headers.get(b"content-length")
            if raw is None:
                return await self._reply(send, 411, "evencut: send a Content-Length header")
            try:
                size = int(raw)
            except ValueError:
                return await self._reply(send, 400, "evencut: invalid Content-Length header")
            if size > self.limit:
                return await self._reply(send, 413, f"evencut: request body exceeds {self.limit} bytes")
        await self.inner(scope, receive, send)

    @staticmethod
    async def _reply(send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


app = _BodyLimit(api, MAX_BODY_BYTES)
"""ASGI application to serve (``uvicorn evencut.service:app``)."""
