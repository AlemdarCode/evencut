"""Measured time on a real machine: what ``evencut-train probe`` writes and what the diagnosis and
``evencut-train pick`` read. Pure Python, so reading a probe file needs no torch.

Why measure: a GPU does not get proportionally faster on smaller batches. On an RTX 4060 a 30M
model takes ~24 ms for any batch up to ~3k tokens and ~40-47 ms up to ~25k; only above that does
time grow with the work. A cost model alone (padded tokens) therefore overrates splits that make
many small batches; the measured curve prices them right (bench/results/backtest_probe.md).

* :class:`TimeCurve`  milliseconds of one micro-batch (forward + backward) on one GPU, as a
  function of its cost ``members x longest x (a + longest)`` (:class:`~evencut.train.AttentionCost`)
* :class:`AllReduce`  milliseconds to all-reduce a buffer over the machine's GPUs, by size
* :class:`Probe`  one machine: the curve of its slowest GPU, gradient sync, optimizer, power, links
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

from .. import EvencutError

FORMAT = "evencut-probe/1"


def isotonic(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> list[float]:
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    w = list(weights) if weights is not None else [1.0] * len(values)
    blocks: list[list[float]] = []                 # [weighted sum, weight, count]
    for y, wt in zip(values, w):
        blocks.append([y * wt, wt, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, wt2, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += wt2
            blocks[-1][2] += n
    out: list[float] = []
    for s, wt, n in blocks:
        out += [s / wt] * int(n)
    return out


@dataclass(frozen=True)
class TimeCurve:
    """Time of one micro-batch on one GPU (ms) against its cost, from measured points.

    The fit is non-decreasing (more work never takes less time). Below the smallest measured cost
    the time stays at the smallest measurement (the floor of a nearly idle GPU); above the largest
    it grows in proportion. ``best`` is the lowest time per cost unit measured: what the work
    would take on a fully busy GPU. The gap between ``ms`` and ``best x cost`` is what small
    batches waste.
    """

    a: int
    points: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise EvencutError("evencut: a time curve needs at least one measured point")
        prev_c, prev_t = 0.0, 0.0
        for c, t in self.points:
            if not (c > prev_c and t >= prev_t and t > 0):
                raise EvencutError("evencut: time-curve points must have increasing cost and non-decreasing time")
            prev_c, prev_t = c, t

    @classmethod
    def fit(cls, a: int, samples: Iterable[Tuple[float, float]]) -> "TimeCurve":
        """``samples``: (cost, ms) pairs, any order; equal costs are averaged."""
        merged: dict[float, list[float]] = {}
        for c, t in samples:
            if not (c > 0 and t > 0):
                raise EvencutError("evencut: time-curve samples need positive cost and time")
            merged.setdefault(float(c), []).append(float(t))
        costs = sorted(merged)
        means = [sum(merged[c]) / len(merged[c]) for c in costs]
        fitted = isotonic(means, [len(merged[c]) for c in costs])
        return cls(int(a), tuple(zip(costs, fitted)))

    def ms(self, cost: float) -> float:
        pts = self.points
        if cost <= pts[0][0]:
            return pts[0][1]
        if cost >= pts[-1][0]:
            return pts[-1][1] * cost / pts[-1][0]
        lo, hi = 0, len(pts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if pts[mid][0] <= cost:
                lo = mid
            else:
                hi = mid
        (c0, t0), (c1, t1) = pts[lo], pts[hi]
        return t0 + (t1 - t0) * (cost - c0) / (c1 - c0)

    def cost(self, count: int, longest: int) -> int:
        return count * longest * (self.a + longest)

    def batch_ms(self, count: int, longest: int) -> float:
        return self.ms(self.cost(count, longest)) if count else 0.0

    @property
    def best(self) -> float:
        """Lowest measured milliseconds per cost unit."""
        return min(t / c for c, t in self.points)

    def linear_ms(self, count: int, longest: int) -> float:
        """What the batch would take at the best measured efficiency (no small-batch overhead)."""
        return self.best * self.cost(count, longest)

    def scaled(self, factor: float) -> "TimeCurve":
        if not factor > 0:
            raise EvencutError("evencut: a time-curve scale factor must be positive")
        return TimeCurve(self.a, tuple((c, t * factor) for c, t in self.points))

    def to_dict(self) -> dict:
        return {"a": self.a, "points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, d: dict) -> "TimeCurve":
        return cls(int(d["a"]), tuple((float(c), float(t)) for c, t in d["points"]))


def ring_factor(gpus: int) -> float:
    """Bytes each GPU sends in a ring all-reduce, per byte of the buffer: 2 (g - 1) / g."""
    return 2.0 * (gpus - 1) / gpus if gpus > 1 else 0.0


@dataclass(frozen=True)
class AllReduce:
    """Measured all-reduce times on ``gpus`` GPUs: ((megabytes, ms), ...). Between points the time
    is interpolated; beyond the largest it grows at the bandwidth of the two largest points. For
    another GPU count of the same kind of machine it scales with :func:`ring_factor`."""

    gpus: int
    points: Tuple[Tuple[float, float], ...]

    def ms(self, nbytes: float, gpus: Optional[int] = None) -> float:
        g = self.gpus if gpus is None else gpus
        if g <= 1 or nbytes <= 0:
            return 0.0
        if self.gpus <= 1 or not self.points:
            raise EvencutError("evencut: this probe ran on one GPU, so it has no all-reduce time to scale")
        mb = nbytes / 1e6
        pts = self.points
        if mb <= pts[0][0]:
            t = pts[0][1]
        elif len(pts) == 1:
            t = pts[0][1] * mb / pts[0][0]
        else:
            i = next((k for k in range(1, len(pts)) if mb <= pts[k][0]), len(pts) - 1)
            (m0, t0), (m1, t1) = pts[i - 1], pts[i]
            t = t0 + (t1 - t0) * (mb - m0) / (m1 - m0)
            t = max(t, t0)
        return t * ring_factor(g) / ring_factor(self.gpus)

    def bus_gbps(self) -> float:
        """Bus bandwidth (GB/s) at the largest measured size, the figure nccl-tests reports."""
        mb, t = self.points[-1]
        return mb * ring_factor(self.gpus) / t      # MB/ms == GB/s

    def to_dict(self) -> dict:
        return {"gpus": self.gpus, "points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, d: dict) -> "AllReduce":
        return cls(int(d["gpus"]), tuple((float(m), float(t)) for m, t in d["points"]))


@dataclass(frozen=True)
class Sync:
    """Gradient sync time a data-parallel step cannot hide. DDP all-reduces gradient buckets while
    the backward pass of the last micro-batch is still running, so part of the all-reduce overlaps
    with it. How much depends on the batch: a small, launch-bound batch leaves the GPU idle between
    kernels and the all-reduce fills those gaps; a large one keeps it busy and the all-reduce mostly
    waits. ``points`` are probe measurements ((micro-batch ms, exposed share of the all-reduce), ...):
    the exposed share is interpolated between them (held flat beyond). Without points, ``overlap``
    is the older proportional model, exposed = max(0, allreduce - overlap x last micro-batch ms);
    with neither, the whole all-reduce is exposed."""

    allreduce_ms: float
    overlap: float = 0.0
    points: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(sorted((float(t), float(f)) for t, f in self.points)))

    def share(self, last_ms: float) -> float:
        """Exposed share of the all-reduce after a last micro-batch of ``last_ms``."""
        pts = self.points
        if not pts:
            if not self.allreduce_ms:
                return 0.0
            return max(0.0, self.allreduce_ms - self.overlap * last_ms) / self.allreduce_ms
        if last_ms <= pts[0][0]:
            f = pts[0][1]
        elif last_ms >= pts[-1][0]:
            f = pts[-1][1]
        else:
            i = next(k for k in range(1, len(pts)) if last_ms <= pts[k][0])
            (t0, f0), (t1, f1) = pts[i - 1], pts[i]
            f = f0 + (f1 - f0) * (last_ms - t0) / (t1 - t0) if t1 > t0 else f1
        return min(1.0, max(0.0, f))

    def __call__(self, last_ms: float) -> float:
        return self.allreduce_ms * self.share(last_ms)


@dataclass
class Probe:
    """One machine, measured by ``evencut-train probe`` (see :mod:`evencut.train.probe`).

    ``curve`` is the slowest GPU's (a data-parallel step waits for it). ``fixed_ms`` per optimizer
    step = optimizer step + all-reduce of the fp32 gradients, assumed not to overlap with compute
    (DDP overlaps part of it, so this errs on the slow side)."""

    machine: dict
    model: dict
    curve: TimeCurve
    optimizer_ms: float
    allreduce: Optional[AllReduce] = None
    watts: Optional[list] = None
    tflops: list = field(default_factory=list)
    gpu_speed: list = field(default_factory=list)
    max_tokens_tested: int = 0
    max_tokens_fit: Optional[int] = None
    price_per_hour: Optional[float] = None
    launch_ms: Optional[float] = None          # fastest time of a tiny batch: the kernel-launch floor
    overlap: Optional[float] = None            # older probes: proportional overlap (see Sync)
    overlap_points: list = field(default_factory=list)
    sync_points: list = field(default_factory=list)   # [[micro-batch ms, exposed share of the all-reduce], ...]
    sync_raw: list = field(default_factory=list)      # [[n, L, no_sync ms, synced ms, all-reduce ms], ...]
    date: str = ""
    evencut: str = ""
    notes: list = field(default_factory=list)
    source: str = ""

    @property
    def gpu(self) -> str:
        return str(self.machine.get("gpu", "?"))

    @property
    def gpus(self) -> int:
        return int(self.machine.get("gpus", 1))

    @property
    def grad_bytes(self) -> int:
        return 4 * int(self.model["params"])          # DDP reduces fp32 gradients

    def allreduce_ms(self, gpus: Optional[int] = None) -> float:
        g = self.gpus if gpus is None else gpus
        if g <= 1:
            return 0.0
        if self.allreduce is None:
            raise EvencutError(f"evencut: the probe of {self.gpu} ran on one GPU; it cannot tell the gradient "
                               f"sync time of {g} GPUs (run it with torchrun on all GPUs)")
        return self.allreduce.ms(self.grad_bytes, g)

    def fixed_ms(self, gpus: Optional[int] = None) -> float:
        """Optimizer plus the whole all-reduce: the most a step can spend outside compute."""
        return self.optimizer_ms + self.allreduce_ms(gpus)

    def sync(self, gpus: Optional[int] = None) -> Sync:
        """The exposed-sync model for ``gpus`` GPUs (all of it exposed if the probe did not measure it)."""
        return Sync(self.allreduce_ms(gpus), self.overlap or 0.0, tuple(tuple(p) for p in self.sync_points))

    @property
    def measured_sync(self) -> bool:
        return bool(self.sync_points) or self.overlap is not None

    def mean_watts(self) -> Optional[float]:
        ok = [w for w in (self.watts or []) if w]
        return sum(ok) / len(ok) if ok else None

    def label(self) -> str:
        return f"{self.gpus}x {self.gpu}" + (f" ({self.source})" if self.source else "")

    def to_dict(self) -> dict:
        return {"format": FORMAT, "evencut": self.evencut, "date": self.date, "machine": self.machine,
             "model": self.model, "curve": self.curve.to_dict(), "optimizer_ms": self.optimizer_ms,
             "allreduce": self.allreduce.to_dict() if self.allreduce else None, "watts": self.watts,
             "tflops": self.tflops, "gpu_speed": self.gpu_speed, "max_tokens_tested": self.max_tokens_tested,
             "max_tokens_fit": self.max_tokens_fit, "price_per_hour": self.price_per_hour,
             "launch_ms": self.launch_ms, "overlap": self.overlap, "overlap_points": self.overlap_points,
             "sync_points": self.sync_points, "sync_raw": self.sync_raw,
             "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict, source: str = "") -> "Probe":
        if d.get("format") != FORMAT:
            raise EvencutError(f"evencut: not an Evencut probe file (format {d.get('format')!r}, expected {FORMAT})")
        try:
            return cls(machine=dict(d["machine"]), model=dict(d["model"]), curve=TimeCurve.from_dict(d["curve"]),
                       optimizer_ms=float(d["optimizer_ms"]),
                       allreduce=AllReduce.from_dict(d["allreduce"]) if d.get("allreduce") else None,
                       watts=d.get("watts"), tflops=list(d.get("tflops") or []),
                       gpu_speed=list(d.get("gpu_speed") or []), max_tokens_tested=int(d.get("max_tokens_tested") or 0),
                       max_tokens_fit=d.get("max_tokens_fit"), price_per_hour=d.get("price_per_hour"),
                       launch_ms=d.get("launch_ms"), overlap=d.get("overlap"),
                       overlap_points=list(d.get("overlap_points") or []),
                       sync_points=[list(p) for p in d.get("sync_points") or []],
                       sync_raw=list(d.get("sync_raw") or []),
                       date=str(d.get("date", "")), evencut=str(d.get("evencut", "")), notes=list(d.get("notes") or []),
                       source=source)
        except (KeyError, TypeError, ValueError) as e:
            raise EvencutError(f"evencut: damaged probe file: {e}") from None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Probe":
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            raise EvencutError(f"evencut: {p} is not JSON: {e}") from None
        return cls.from_dict(data, source=p.name)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")


def same_model(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in ("params", "layers", "d_model"))


__all__ = ["AllReduce", "FORMAT", "Probe", "Sync", "TimeCurve", "isotonic", "ring_factor", "same_model"]
