"""``evencut-train pick``: which machine finishes your training job for the least money?

The job is your sequence lengths, the sequences per optimizer step (fixed by your training recipe),
the largest per-GPU batch that fits, the epochs, and the model shape (from the probe files). Each
candidate machine runs the job with its own GPU count: per-GPU batch and gradient accumulation are
chosen so every optimizer step keeps the same number of sequences. Its step time is replayed on
your lengths with the machine's measured time curve, once with the default DistributedSampler
split and once with Evencut, plus gradient sync and the optimizer.

Candidates:
* probe files (``evencut-train probe``): measured, the reliable rows;
* vast.ai offers (the JSON of ``vastai search offers --raw``): *estimated* from a probe of another
  GPU, its compute scaled by the two GPUs' bf16 tensor TFLOPS (datasheet figures, :data:`GPU_SPECS`)
  and its gradient sync by the offer's link bandwidth. Estimates rank offers for a closer look;
  probe the best two or three (a minute each) before renting for the whole job. An estimate across
  GPU architectures or vendors is flagged and given a wider error band: the datasheet ratio held
  for big models within one family (RTX 4060 to 4090: 1-7% at 310M and a 7B shape) but not across
  families (4060 to A100: 18-40% off).

How good is it (bench/results/backtest_probe.md)? On six rented machines (three 2x A100 SXM4, three
2x RTX 4090) a probe of the machine itself predicted real DDP step times within 1-5% (median per
machine with the measured sync overlap, worst run 8%; 2-12% when the sync is counted in full) and
named the cheapest of the six for 7 of 8 jobs, the miss a 3% tie. Estimated from an RTX 4060 probe and
datasheets instead, the same runs were off by 7-46% and the cheapest was named for 2 of 8 jobs: a
91M model reaches far less of an A100's datasheet speed than of a 4060's, and rented hosts differ in
links (two of three "A100 SXM4" pairs had no NVLink between them) and CPUs.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from .. import EvencutError
from . import _plans as P
from .timing import AllReduce, Probe, Sync, TimeCurve, ring_factor, same_model

# Datasheet dense bf16 tensor TFLOPS (fp32 accumulate; no sparsity) and the GPU-to-GPU link a
# multi-GPU box of that card usually has: "nvlink", "pcie" (peer-to-peer over PCIe) or "host"
# (no peer-to-peer: GeForce cards copy through host memory, which made an 8x RTX 4090 box spend
# 64 ms per all-reduce of 120 MB in our run). Matched by substring, longest key first.
GPU_SPECS = {
    "b200": (2250, "nvlink"), "h200": (989, "nvlink"), "gh200": (989, "nvlink"), "h100 sxm": (989, "nvlink"),
    "h100 80gb hbm3": (989, "nvlink"), "h100 nvl": (835, "pcie"), "h100 pcie": (756, "pcie"), "h100": (989, "nvlink"),
    "a100 sxm4": (312, "nvlink"), "a100 sxm": (312, "nvlink"), "a100 pcie": (312, "pcie"),
    "a100 80gb pcie": (312, "pcie"), "a100": (312, "nvlink"),
    "l40s": (362, "pcie"), "l40": (181, "pcie"), "l4": (121, "pcie"), "a40": (150, "pcie"), "a10": (125, "pcie"),
    "rtx 6000 ada": (364, "pcie"), "rtx 6000ada": (364, "pcie"), "a6000": (155, "pcie"),
    "rtx 5090": (210, "host"), "rtx 4090": (165, "host"), "rtx 4080": (97, "host"), "rtx 3090": (71, "host"),
    "rtx 4060": (30, "host"),
    # AMD Instinct (datasheet; not measured by us). Their Infinity Fabric (xGMI) links between GPUs
    # are counted in the "nvlink" class.
    "mi300x": (1307, "nvlink"), "mi250x": (383, "nvlink"), "mi250": (362, "nvlink"), "mi210": (181, "pcie"),
}

# Architecture families, matched on the normalised name in this order. The datasheet ratio between
# two GPUs of one family held for big models in our tests; across families it did not.
GPU_FAMILIES = (
    ("amd cdna", r"\bmi\d{3}"), ("turing", r"\b(rtx 20\d\d|t4\b|quadro rtx|q rtx|titan rtx)"),
    ("blackwell", r"\b(g?b200|b100|rtx 50\d\d|rtx pro)"),
    ("hopper", r"\b(gh200|h100|h200|h800)"), ("ada", r"\b(rtx 40\d\d|l4\b|l40|rtx \d000 ?ada)"),
    ("ampere", r"\b(rtx 30\d\d|a100|a800|a40\b|a30\b|a10g?\b|a16\b|a2\b|rtx a\d000|a\d000)"),
    ("volta", r"\bv100"),
)

# Cards missing above: bf16 tensor TFLOPS from the offer's fp32 figure (vast's total_flops / num_gpus).
# On the cards in the table that vast lists, bf16/fp32 was 1.9-2.0 for GeForce (RTX 3090, 4080S, 4090,
# 5090) and 4.5-5.0 for workstation and data-centre Ada cards (RTX 6000 Ada, L40S, L4): rough.
FP32_TO_BF16 = {"geforce": 2.0, "pro": 4.5}

# Bus bandwidth of a ring all-reduce as a share of the offer's quoted link bandwidth. NVLink:
# 4x A100 SXM4 measured 176 GB/s against vast's bw_nvlink of 300. Host copies: 8x RTX 4090 measured
# 3.3 GB/s against a PCIe bandwidth of ~21 GB/s. PCIe peer-to-peer: not measured by us (a guess).
LINK_EFFICIENCY = {"nvlink": 0.59, "pcie": 0.5, "host": 0.16}
LINK_LATENCY_MS = 0.1


def _short(name: str) -> str:
    return name.replace("NVIDIA ", "").replace("GeForce ", "")


def _key(name: str) -> str:
    n = name.lower().replace("nvidia", " ").replace("geforce", " ").replace("tesla", " ")
    return " ".join(n.replace("-", " ").replace("_", " ").split())


def _match(name: str) -> Optional[str]:
    """The GPU_SPECS key a GPU name stands for. Whole words only, and never a "Ti" or "Super" variant
    of a listed card (an RTX 4060 Ti is not an RTX 4060)."""
    key = _key(name)
    for k in sorted(GPU_SPECS, key=len, reverse=True):
        if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])(?!\s*(?:ti|super)\b)", key):
            return k
    return None


def gpu_spec(name: str) -> Optional[tuple]:
    k = _match(name)
    return GPU_SPECS[k] if k else None


def gpu_model(name: str) -> str:
    """A name that two spellings of the same card share ("A100 SXM4" and "NVIDIA A100-SXM4-80GB")."""
    return _match(name) or _key(name)


def gpu_family(name: str) -> Optional[str]:
    """The architecture family of a GPU name ("ada", "ampere", "hopper", "amd cdna", ...), or None."""
    key = _key(name)
    return next((fam for fam, pat in GPU_FAMILIES if re.search(pat, key)), None)


def gpu_vendor(name: str) -> str:
    key = _key(name)
    return "amd" if re.search(r"\b(amd|radeon|instinct|mi\d{3})", key) else "nvidia"


def offer_spec(offer: dict) -> Optional[tuple]:
    """(bf16 TFLOPS, link class, rough?) of an offer's GPU: the datasheet table, else its fp32 figure
    times FP32_TO_BF16 for cards with bf16 (compute capability 8.0 or newer), else None."""
    name = str(offer.get("gpu_name", ""))
    spec = gpu_spec(name)
    if spec:
        return spec[0], spec[1], False
    cc, flops, n = offer.get("compute_cap"), offer.get("total_flops"), offer.get("num_gpus") or 1
    if not cc or not flops or float(cc) < 80 or (float(cc) >= 100 and float(cc) < 800):
        return None                     # vast writes 8.6 as 860; below 8.0 there is no bf16
    fp32 = float(flops) / int(n)
    k = _key(name)
    if re.match(r"(rtx|gtx|titan) ", k) and not re.match(r"rtx (pro|a\d)", k):
        return FP32_TO_BF16["geforce"] * fp32, "host", True
    if re.match(r"(rtx pro|rtx a\d|a\d{1,2}\b|l\d)", k):
        return FP32_TO_BF16["pro"] * fp32, "pcie", True
    return None


@dataclass(frozen=True)
class Job:
    """The training job, independent of the machine."""

    lengths: Sequence[int]
    sequences_per_step: int
    per_device_batch: int
    max_len: Optional[int] = None
    epochs: float = 1.0
    pad_multiple: int = 1
    seed: int = 0
    sample_steps: int = 200

    def validate(self) -> None:
        for name in ("sequences_per_step", "per_device_batch", "sample_steps", "pad_multiple"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise EvencutError(f"evencut: {name} must be an integer >= 1")
        if not self.epochs > 0:
            raise EvencutError("evencut: epochs must be > 0")
        if len(self.lengths) < self.sequences_per_step:
            raise EvencutError("evencut: fewer sequences than one optimizer step")


@dataclass
class Candidate:
    name: str
    gpu: str
    gpus: int
    price_per_hour: Optional[float]
    curve: TimeCurve
    optimizer_ms: float
    allreduce: Optional[AllReduce] = None      # measured
    link_gbps: Optional[float] = None          # estimated bus bandwidth (offers)
    watts: Optional[float] = None              # per GPU
    measured: bool = True
    memory_fit: Optional[int] = None           # padded tokens per GPU that fit (probe with all layers)
    notes: list = field(default_factory=list)
    error: float = 0.08                        # expected relative error of this row's prediction
    overlap: float = 0.0                       # older probes: proportional overlap (see timing.Sync)
    sync_points: tuple = ()                    # probe-measured exposed share of the sync by batch time

    def sync_ms(self, grad_bytes: int) -> float:
        if self.gpus <= 1:
            return 0.0
        if self.allreduce is not None:
            return self.allreduce.ms(grad_bytes, self.gpus)
        if self.link_gbps:
            return LINK_LATENCY_MS + grad_bytes / 1e6 * ring_factor(self.gpus) / self.link_gbps
        raise EvencutError(f"evencut: no gradient sync figure for {self.name}")


def from_probe(p: Probe, price: Optional[float] = None) -> Candidate:
    fit = p.max_tokens_fit if p.model.get("probe_layers") == p.model.get("layers") else None
    notes = []
    if p.gpus > 1 and p.machine.get("p2p") == "none":
        notes.append("no peer-to-peer")
    if len(p.gpu_speed) > 1 and max(p.gpu_speed) > 1.05:
        notes.append(f"slowest GPU {100 * (max(p.gpu_speed) - 1):.0f}% behind")
    watts = p.mean_watts()
    if watts is None:
        from .diagnose import gpu_watts             # board power: an upper bound, as for the offers
        watts = gpu_watts(p.gpu)
    return Candidate(_short(p.label()), p.gpu, p.gpus, price if price is not None else p.price_per_hour, p.curve,
                     p.optimizer_ms, allreduce=p.allreduce, watts=watts, measured=True, memory_fit=fit, notes=notes,
                     overlap=p.overlap or 0.0, sync_points=tuple(tuple(x) for x in p.sync_points))


def measured_link_class(p: Probe) -> str:
    """The link class a probe measured: "nvlink" (NV# links), "host" (no peer-to-peer) or "pcie"."""
    if str(p.machine.get("worst_link") or "").startswith("NV"):
        return "nvlink"
    return "host" if p.machine.get("p2p") == "none" else "pcie"


def estimate_curve(ref: Probe, ratio: float) -> TimeCurve:
    """Another GPU's curve from a probe: ``max(launch floor, ratio x best rate x cost)``. The compute
    rate scales with tensor throughput; the floor (launching the kernels of a tiny batch, a matter
    of the host CPU and the model's layer count) is taken as the reference's fastest tiny batch."""
    floor = ref.launch_ms or ref.curve.points[0][1]
    rate = ref.curve.best * ratio
    knee = floor / rate
    hi = max(ref.curve.points[-1][0], 4 * knee)
    return TimeCurve(ref.curve.a, ((knee / 1000, floor), (knee, floor), (hi, rate * hi)))


def from_offer(offer: dict, ref: Probe, probes: Sequence[Probe] = ()) -> Candidate:
    """Estimate a vast.ai offer (see the module docstring). A probe of the same GPU model among
    ``probes`` gives its measured curve; that is still an estimate, because hosts differ in CPU (the
    kernel-launch floor) and links: a small model's step time moved 5-62% between rented boxes of one
    card. Otherwise ``ref`` is scaled by tensor TFLOPS, flagged when the two GPUs are of different
    architectures or vendors."""
    name = str(offer.get("gpu_name", "?"))
    spec = offer_spec(offer)
    if spec is None:
        raise EvencutError(f"no bf16 figures for {name} (not in the table, or older than compute capability 8.0)")
    gpus = int(offer.get("num_gpus") or 1)
    if (offer.get("bw_nvlink") or 0) > 0:
        link = "nvlink"
    elif "bw_nvlink" in offer and spec[1] == "nvlink":
        link = "pcie"       # vast measured no NVLink: e.g. 2 of a board's SXM GPUs joined over PCIe (seen: 2x A100 SXM4)
    else:
        link = spec[1]
    raw = (float(offer.get("bw_nvlink") or 0) or 300.0) if link == "nvlink" else (float(offer.get("pcie_bw") or 0) or 12.0)
    link_gbps = raw * LINK_EFFICIENCY[link]
    same = [p for p in probes if gpu_model(p.gpu) == gpu_model(name)]
    allreduce, overlap, sync_points = None, 0.0, ()
    error = ERROR_ESTIMATED
    if same:
        twin = max(same, key=lambda p: p.gpus)
        curve, opt_ms = twin.curve, twin.optimizer_ms
        notes = [f"estimate from a probe of the same GPU ({twin.source or twin.gpu})"]
        if twin.allreduce is not None and twin.machine.get("worst_link") and link == measured_link_class(twin):
            allreduce = twin.allreduce                  # same card, same kind of link: its measured sync
            overlap = twin.overlap or 0.0
            sync_points = tuple(tuple(x) for x in twin.sync_points)
    else:
        ref_spec = gpu_spec(ref.gpu)
        ref_tf = ref_spec[0] if ref_spec else max((t for t in ref.tflops if t), default=None)
        if not ref_tf:
            raise EvencutError(f"no TFLOPS figure for the reference {ref.gpu}")
        ratio = ref_tf / spec[0]
        curve, opt_ms = estimate_curve(ref, ratio), ref.optimizer_ms * ratio
        notes = ["estimate" + (", rough (bf16 from the fp32 figure)" if spec[2] else "")]
        ref_fam, fam = gpu_family(ref.gpu), gpu_family(name)
        if gpu_vendor(ref.gpu) != gpu_vendor(name):
            notes.append(f"different vendor than the probed {_short(ref.gpu)}: datasheet estimate unreliable")
            error = ERROR_CROSS_ARCH
        elif ref_fam is None or fam is None or ref_fam != fam:
            notes.append(f"different architecture ({ref_fam or '?'} -> {fam or '?'}): datasheet estimate unreliable")
            error = ERROR_CROSS_ARCH
    if link == "host" and gpus > 1:
        notes.append("no peer-to-peer (GeForce)")
    where = offer.get("geolocation")
    label = f"{gpus}x {name}" + (f", {where}" if where else "") + f" (vast {offer.get('id', '?')})"
    return Candidate(label, name, gpus, float(offer["dph_total"]) if offer.get("dph_total") is not None else None,
                     curve, opt_ms, allreduce=allreduce, link_gbps=link_gbps,
                     watts=float(offer["gpu_max_power"]) if offer.get("gpu_max_power") else None,
                     measured=False, notes=notes, overlap=overlap, sync_points=sync_points, error=error)


def load_offers(path: Union[str, Path]) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("offers", [])
    if not isinstance(data, list):
        raise EvencutError("evencut: an offers file must hold a JSON list (vastai search offers --raw)")
    return data


def layout(gpus: int, sequences_per_step: int, per_device_max: int) -> Optional[tuple]:
    """(per-GPU batch, accumulation) that keeps exactly ``sequences_per_step`` per optimizer step,
    with the largest per-GPU batch that fits; None when the GPUs cannot share the step evenly."""
    if sequences_per_step % gpus:
        return None
    q = sequences_per_step // gpus
    b = max(d for d in range(1, min(q, per_device_max) + 1) if q % d == 0)
    return b, q // b


@dataclass
class Result:
    candidate: Candidate
    per_device: int
    accum: int
    mode: str
    step_ms: float
    default_step_ms: float
    hours: float
    cost: Optional[float]
    kwh: Optional[float]
    peak_padded: int
    over_memory: bool

    @property
    def speedup(self) -> float:
        return self.default_step_ms / self.step_ms


class PlanCache:
    """The job's batch plans by (GPUs, per-GPU batch, accumulation), reduced to what timing needs:
    per optimizer step, per GPU, the (sequences, padded length) of every micro-batch in order (None
    for an empty one). A plan does not depend on the machine, so offers with the same layout share it."""

    def __init__(self, job: Job, a: int) -> None:
        self.job, self.cost = job, P.AttentionCost(a)
        self.lens = [min(x, job.max_len) for x in job.lengths] if job.max_len else list(job.lengths)
        n = len(self.lens)
        steps = max(1, min(job.sample_steps, n // job.sequences_per_step))
        order = list(range(n))
        random.Random(P.epoch_seed(job.seed, 0)).shuffle(order)
        self.order = order[: steps * job.sequences_per_step]
        self._plans: dict = {}

    def _shapes(self, plan, gpus: int, accum: int) -> tuple:
        steps, peak = [], 0
        for w in range(0, len(plan), accum):
            per_gpu = [[] for _ in range(gpus)]
            for micro in plan[w:w + accum]:
                for r, batch in enumerate(micro):
                    if batch:
                        plen = P.pad_length(max(self.lens[i] for i in batch), None, self.job.pad_multiple)
                        per_gpu[r].append((len(batch), plen))
                        peak = max(peak, len(batch) * plen)
                    else:
                        per_gpu[r].append(None)
            steps.append(per_gpu)
        return steps, peak

    def plans(self, gpus: int, per_device: int, accum: int) -> dict:
        key = (gpus, per_device, accum)
        if key not in self._plans:
            g, o, s = gpus * per_device, self.order, self.lens
            raw = {"default": P.per_step(o, s, gpus, g, "strided"),
                   "step": P.per_step(o, s, gpus, g, "padded", grouped=True, cost=self.cost)}
            if accum > 1:
                raw["window"] = P.window(o, s, gpus, g, accum, cost=self.cost)
            self._plans[key] = {m: self._shapes(pl, gpus, accum) for m, pl in raw.items()}
        return self._plans[key]


def plan_ms(shapes: list, curve: TimeCurve, exposed: Optional[Sync] = None) -> float:
    """Mean time of an optimizer step (DDP: the GPUs wait once per step), with the same rule as
    _plans.evaluate: the slowest GPU's micro-batches plus the sync its last one cannot hide."""
    total = 0.0
    for step in shapes:
        times = [[curve.batch_ms(*b) if b else 0.0 for b in gpu] for gpu in step]
        compute, sync = P.window_step_ms(times, exposed)
        total += compute + sync
    return total / len(shapes)


def evaluate(job: Job, cand: Candidate, grad_bytes: int, cache: Optional[PlanCache] = None) -> Optional[Result]:
    lay = layout(cand.gpus, job.sequences_per_step, job.per_device_batch)
    if lay is None:
        return None
    b, accum = lay
    cache = cache or PlanCache(job, cand.curve.a)
    fixed = cand.optimizer_ms
    exposed = Sync(cand.sync_ms(grad_bytes), cand.overlap, cand.sync_points) if cand.gpus > 1 else None
    plans = cache.plans(cand.gpus, b, accum)
    base = plan_ms(plans["default"][0], cand.curve, exposed) + fixed
    mode, best, peak = min(((m, plan_ms(sh, cand.curve, exposed) + fixed, pk) for m, (sh, pk) in plans.items()
                            if m != "default"), key=lambda x: x[1])
    total_steps = len(job.lengths) * job.epochs / job.sequences_per_step
    hours = best * total_steps / 3.6e6
    return Result(cand, b, accum, mode, best, base, hours,
                  hours * cand.price_per_hour if cand.price_per_hour is not None else None,
                  hours * cand.gpus * cand.watts / 1000 if cand.watts else None, peak,
                  bool(cand.memory_fit and peak > cand.memory_fit))


@dataclass
class Pick:
    job: Job
    results: list
    skipped: list
    model: dict
    by: str

    def report(self) -> str:
        return format_pick(self)


def pick(job: Job, probes: Sequence[Probe], offers: Sequence[dict] = (), *, prices: Optional[dict] = None,
         by: str = "cost", reference: Optional[Probe] = None) -> Pick:
    """Ranks the probed machines and the offers for ``job``; ``by`` is "cost" or "time"."""
    job.validate()
    if not probes:
        raise EvencutError("evencut: pick needs at least one probe file (it sets the model and anchors estimates)")
    if by not in ("cost", "time"):
        raise EvencutError("evencut: by must be 'cost' or 'time'")
    model = probes[0].model
    for p in probes[1:]:
        if not same_model(model, p.model):
            raise EvencutError(f"evencut: {p.source or p.gpu} measured another model; probe every machine with the "
                               "same model shape")
    if reference is not None and not same_model(model, reference.model):
        raise EvencutError(f"evencut: the reference {reference.source or reference.gpu} measured another model; "
                           "probe it with the same model shape")
    ref = reference or max(probes, key=lambda p: p.max_tokens_tested)
    grad_bytes = 4 * int(model["params"])
    prices = prices or {}
    cands, skipped = [], []
    for p in probes:
        cands.append(from_probe(p, prices.get(p.source)))
    for o in offers:
        try:
            cands.append(from_offer(o, ref, probes))
        except (EvencutError, KeyError, TypeError, ValueError) as e:
            skipped.append(f"vast {o.get('id', '?')} {o.get('num_gpus', '?')}x {o.get('gpu_name', '?')}: {e}")
    results = []
    cache = PlanCache(job, ref.curve.a)          # every probe measured the same model, so the same cost
    for c in cands:
        r = evaluate(job, c, grad_bytes, cache)
        if r is None:
            skipped.append(f"{c.name}: {job.sequences_per_step} sequences per step do not split evenly over {c.gpus} GPUs")
        else:
            results.append(r)

    if by == "cost":
        results.sort(key=lambda r: (r.cost is None, r.cost or 0.0, r.hours))
    else:
        results.sort(key=lambda r: r.hours)
    return Pick(job, results, skipped, model, by)


# expected errors (bench/results/backtest_probe.md): probes 1-5% (worst 8%); estimates within one
# architecture family ~22%; across families 18-40% for big models and worse for small ones
ERROR_MEASURED, ERROR_ESTIMATED, ERROR_CROSS_ARCH = 0.08, 0.25, 0.4


def close_to_first(pk: "Pick", r: "Result") -> bool:
    """True when ``r`` trails the first row by less than the prediction error of the two, so the
    order between them is not to be trusted."""
    first = pk.results[0]
    margin = max(first.candidate.error, r.candidate.error)
    if pk.by == "cost" and first.cost and r.cost is not None:
        return r.cost <= first.cost * (1 + margin)
    return r.hours <= first.hours * (1 + margin)


def _hours(h: float) -> str:
    return f"{h:,.1f} h" if h >= 1 else f"{60 * h:,.0f} min"


def format_pick(pk: Pick) -> str:
    j, m = pk.job, pk.model
    lines = ["Evencut pick",
             f"  job      {len(j.lengths):,} sequences x {j.epochs:g} epoch(s), {j.sequences_per_step} sequences per "
             f"optimizer step, at most {j.per_device_batch} per GPU per micro-step; model {m['layers']} layers x "
             f"{m['d_model']} wide ({int(m['params']) / 1e6:,.0f}M weights)",
             f"  ranked   by {'total cost' if pk.by == 'cost' else 'wall time'}; Evencut batching (the default "
             f"DistributedSampler split in the 'default' column)", ""]
    width = max([7] + [len(r.candidate.name) for r in pk.results])     # never cut a name: it says which file
    head = (f"  {'#':>2} {'machine':{width}} {'$/h':>7} {'batch':>9} {'step ms':>8} {'default':>8} {'time':>9} "
            f"{'cost':>9} {'kWh':>7}  notes")
    lines.append(head)
    for i, r in enumerate(pk.results, 1):
        c = r.candidate
        notes = list(c.notes)
        if i > 1 and close_to_first(pk, r):
            notes.insert(0, "~#1 (within the error)")
        if r.over_memory:
            notes.append(f"largest batch {r.peak_padded:,} tokens > {c.memory_fit:,} that fit")
        price = f"{c.price_per_hour:.2f}" if c.price_per_hour is not None else "?"
        cost = f"${r.cost:,.2f}" if r.cost is not None else "?"
        kwh = f"{r.kwh:,.1f}" if r.kwh is not None else "?"
        lines.append(f"  {i:2} {c.name:{width}} {price:>7} {f'{r.per_device}x{r.accum}':>9} {r.step_ms:8.1f} "
                     f"{r.default_step_ms:8.1f} {_hours(r.hours):>9} {cost:>9} {kwh:>7}  {', '.join(notes)}")
    lines += ["", "  batch = sequences per GPU per micro-step x accumulation; default = step ms with DistributedSampler",
              "  kWh: measured power where the probe could read it, else the board's power limit (an upper bound)"]
    est = [r for r in pk.results if not r.candidate.measured]
    if est:
        top = est[:3]
        mins = 2
        lines.append("  estimate = not measured: compute scaled from a probe of another GPU by datasheet TFLOPS, sync "
                     "from the quoted link bandwidth; in our backtest such estimates were off by 7-46% and named the "
                     "cheapest of six rented machines for 2 of 8 jobs (probes: 7 of 8, the miss a 3% tie). Before "
                     "renting for the whole job, probe "
                     f"{', '.join(r.candidate.name for r in top)} for ~{mins} minutes each (about "
                     f"${sum((r.candidate.price_per_hour or 0) * mins / 60 for r in top):.2f} in all).")
    if pk.results:
        best = pk.results[0]
        cheapest_h = min((r for r in pk.results if r.candidate.price_per_hour is not None),
                         key=lambda r: r.candidate.price_per_hour, default=None)
        if cheapest_h is not None and cheapest_h is not best and best.cost is not None and cheapest_h.cost:
            lines.append(f"  the cheapest per hour ({cheapest_h.candidate.name}) would cost ${cheapest_h.cost:,.2f}, "
                         f"{cheapest_h.cost / best.cost:.1f}x the best pick")
    for s in pk.skipped[:10]:
        lines.append(f"  skipped: {s}")
    if len(pk.skipped) > 10:
        lines.append(f"  skipped: {len(pk.skipped) - 10} more")
    return "\n".join(lines)


__all__ = ["Candidate", "GPU_SPECS", "Job", "Pick", "PlanCache", "Result", "evaluate", "plan_ms", "from_offer", "from_probe", "gpu_spec",
           "layout", "load_offers", "pick"]
