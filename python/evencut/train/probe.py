"""``evencut-train probe``: measure one machine for one model shape in a minute or two.

Run it on the machine you rent, on all its GPUs, with your model's shape:

    torchrun --standalone --nproc-per-node=gpu -m evencut.train probe \\
        --layers 32 --d-model 4096 --vocab 32000 --max-len 4096 --max-tokens 16384 \\
        --probe-layers 4 --price-per-hour 14.2 -o h100x8.json

(``python -m evencut.train probe ...`` measures a single GPU.) Every GPU times, at the same time,
forward + backward of a synthetic decoder (:mod:`evencut.train._model`) on ~20 batch shapes from
one short sequence up to ``--max-tokens`` padded tokens. The result is a time curve: milliseconds
per micro-batch against its cost. The slowest GPU's curve is kept, because a data-parallel step
waits for it. Also measured: all-reduce of the fp32 gradients at several sizes, one AdamW step,
bf16 matmul TFLOPS per GPU, board power under load (if nvidia-smi can read it), the GPU links
(``nvidia-smi topo -m``) and peer-to-peer access. ``--probe-layers k`` builds only k of the
model's layers and scales the times by the FLOP ratio (for models that do not fit one GPU).

Hardware: tested on NVIDIA GPUs. On AMD GPUs PyTorch's ROCm build offers the same ``torch.cuda``
interface and RCCL behind the "nccl" backend, so the timings should work; power is then read with
``amd-smi`` or ``rocm-smi`` and links are not read. That path has not been tested on AMD hardware.
Other accelerators (Intel XPU, Apple MPS, TPU) are not supported: the probe would time the CPU.

Feed the file to ``evencut-train diagnose --probe`` (absolute step times, hours, money) and to
``evencut-train pick`` (which machine finishes the job for the least money). Needs torch.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import random
import re
import statistics
import subprocess
import threading
import time
from typing import Optional

import torch
import torch.distributed as dist

from .. import EvencutError, __version__
from . import _model as M
from ._plans import attention_constant
from .timing import AllReduce, Probe, TimeCurve

LINK_ORDER = ("NV", "PIX", "PXB", "PHB", "NODE", "SYS")     # best to worst


def shapes(max_len: int, max_tokens: int) -> list[tuple[int, int]]:
    """(sequences, length) pairs from one short sequence up to ``max_tokens`` padded tokens,
    smallest first (so running out of memory ends the sweep)."""
    lengths, L = [], 32
    while L < max_len:
        lengths.append(L)
        L *= 2
    lengths.append(max_len)
    out = set()
    for L in lengths:
        most = max(1, max_tokens // L)
        out.update((n, L) for n in (1, max(1, most // 16), max(1, most // 4), max(1, most // 2), most))
    return sorted(out, key=lambda s: (s[0] * s[1], s[1]))


def _ms_samples(fn, reps: int, warmup: int, dev: torch.device) -> list[float]:
    times = []
    for i in range(warmup + reps):
        _sync(dev)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        if i >= warmup:
            times.append(1000 * (time.perf_counter() - t0))
    return times


def _ms_median(fn, reps: int, warmup: int, dev: torch.device) -> float:
    return statistics.median(_ms_samples(fn, reps, warmup, dev))


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _barrier(dev: torch.device) -> None:
    if dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[dev.index])
        else:
            dist.barrier()


def is_oom(e: BaseException) -> bool:
    """Out of GPU memory. With expandable segments the driver's failure arrives as a plain
    RuntimeError ("CUDA driver error: out of memory"), not as torch.cuda.OutOfMemoryError."""
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return (bool(oom) and isinstance(e, oom)) or (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def batch_samples(model: torch.nn.Module, n: int, L: int, vocab: int, dev: torch.device, reps: int,
                  warmup: int = 1) -> list[float]:
    """Milliseconds of forward + backward on ``n`` sequences of ``L`` tokens, ``reps`` times."""
    idx = torch.randint(0, vocab, (n, L), device=dev)
    tgt = torch.randint(0, vocab, (n, L), device=dev)

    def step() -> None:
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = model(idx, tgt)
        loss.backward()
        model.zero_grad(set_to_none=True)
    return _ms_samples(step, reps, warmup, dev)


def time_batch(model: torch.nn.Module, n: int, L: int, vocab: int, dev: torch.device, reps: int,
               warmup: int = 1) -> float:
    """Median milliseconds of forward + backward on ``n`` sequences of ``L`` tokens."""
    return statistics.median(batch_samples(model, n, L, vocab, dev, reps, warmup))


def exposed_share(no_sync_ms: float, synced_ms: float, allreduce_ms: float) -> float:
    """Share of the all-reduce a DDP step could not hide: the same batch took ``synced_ms`` with
    gradient sync and ``no_sync_ms`` inside ``ddp.no_sync()``, so ``synced - no_sync`` of the
    ``allreduce_ms`` stayed exposed. Clamped to [0, 1]."""
    if allreduce_ms <= 0:
        return 0.0
    return min(1.0, max(0.0, (synced_ms - no_sync_ms) / allreduce_ms))


def sync_shapes(fits: list, count: int = 5) -> list:
    """Up to ``count`` batch shapes of distinct sizes from ``fits``, the sizes training batches have
    (1k padded tokens and up; smaller ones only to fill up), each the middle length of its size."""
    levels = sorted({n * L for n, L in fits})
    big = [lv for lv in levels if lv >= min(1024, levels[-1])]
    small = [lv for lv in levels if lv not in big]
    pool = sorted(small[len(small) - max(0, count - len(big)):] + big) if len(big) < count else big
    chosen = sorted({pool[round(i * (len(pool) - 1) / max(1, count - 1))] for i in range(count)})
    out = []
    for tok in chosen:
        same = sorted((sh for sh in fits if sh[0] * sh[1] == tok), key=lambda sh: sh[1])
        out.append(same[len(same) // 2])
    return out


def ddp_pair_samples(ddp, n: int, L: int, vocab: int, dev: torch.device, reps: int,
                     warmup: int = 2) -> tuple[list[float], list[float]]:
    """Times the same batch inside ``ddp.no_sync()`` and with gradient sync, alternating, so both
    see the same moment, the same DDP overhead and the same host: (no_sync ms, synced ms) lists."""
    import contextlib

    idx = torch.randint(0, vocab, (n, L), device=dev)
    tgt = torch.randint(0, vocab, (n, L), device=dev)

    def step(sync: bool) -> None:
        with contextlib.nullcontext() if sync else ddp.no_sync():
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = ddp(idx, tgt)
            loss.backward()
        ddp.zero_grad(set_to_none=True)
    out: tuple[list[float], list[float]] = ([], [])
    for i in range(warmup + reps):
        for k, sync in enumerate((False, True)):
            _barrier(dev)                     # both ranks start together; a synced step is collective
            _sync(dev)
            t0 = time.perf_counter()
            step(sync)
            _sync(dev)
            if i >= warmup:
                out[k].append(1000 * (time.perf_counter() - t0))
    return out


def matmul_tflops(dev: torch.device, n: int = 4096, reps: int = 20) -> Optional[float]:
    if dev.type != "cuda":
        return None
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    _sync(dev)
    return 2 * n ** 3 * reps / (time.perf_counter() - t0) / 1e12


def optimizer_ms(model: torch.nn.Module, dev: torch.device) -> float:
    """One fused AdamW step over the model's weights. If its state does not fit next to the model
    (a few layers of a 7B shape on an 8 GB card), the step is timed on one block and scaled by
    weight count: an optimizer step reads and writes every weight once, so it is linear in them."""
    fused = {"fused": True} if dev.type == "cuda" else {}

    def time_on(params: list) -> float:
        opt = None
        try:
            for p in params:
                p.grad = torch.zeros_like(p)
            opt = torch.optim.AdamW(params, lr=1e-4, **fused)
            return _ms_median(opt.step, 5, 2, dev)
        finally:
            del opt
            for p in params:
                p.grad = None
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    params = list(model.parameters())
    try:
        return time_on(params)
    except RuntimeError as e:
        if not is_oom(e):
            raise
    block = list(model.blocks[0].parameters())
    return time_on(block) * sum(p.numel() for p in params) / sum(p.numel() for p in block)


def allreduce_ms(nbytes: int, dev: torch.device, reps: int) -> float:
    x = torch.ones(max(1, nbytes // 4), dtype=torch.float32, device=dev)
    for _ in range(3):
        dist.all_reduce(x)
    times = []
    for _ in range(reps):
        _sync(dev)
        _barrier(dev)
        t0 = time.perf_counter()
        dist.all_reduce(x)
        _sync(dev)
        times.append(1000 * (time.perf_counter() - t0))
    del x
    return statistics.median(times)


# ---------------------------------------------------------------------------------------------
# machine facts (rank 0)

def _run(cmd: list[str]) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _smi(args: list[str]) -> Optional[str]:
    return _run(["nvidia-smi", *args])


def vendor() -> str:
    """"amd" on a ROCm build of PyTorch, else "nvidia" (CUDA) or "cpu"."""
    if getattr(torch.version, "hip", None):
        return "amd"
    return "nvidia" if torch.cuda.is_available() else "cpu"


def _watts_in(node, out: list) -> None:
    """Collects power readings in watts from amd-smi / rocm-smi JSON, whose layout differs between
    versions: {"value": 312, "unit": "W"} objects, or "... Power (W)": "312.0" entries."""
    if isinstance(node, dict):
        unit = str(node.get("unit", "")).strip().upper()
        if unit == "W" and _num(str(node.get("value", ""))) is not None:
            out.append(_num(str(node["value"])))
            return
        for k, v in node.items():
            if "power" in str(k).lower() and "(w)" in str(k).lower() and _num(str(v)) is not None:
                out.append(_num(str(v)))
            else:
                _watts_in(v, out)
    elif isinstance(node, list):
        for v in node:
            _watts_in(v, out)


def amd_power(index: int) -> Optional[float]:
    """Board power of AMD GPU ``index`` from amd-smi, else rocm-smi (untested on AMD hardware)."""
    for cmd in (["amd-smi", "metric", "--power", "--json", "-g", str(index)],
                ["rocm-smi", "--showpower", "--json", "-d", str(index)]):
        text = _run(cmd)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        found: list = []
        _watts_in(data, found)
        if found:
            return found[0]
    return None


def parse_topology(text: str, gpus: int) -> dict:
    """Link type between every pair of the first ``gpus`` GPUs in ``nvidia-smi topo -m`` output,
    and the worst of them (a ring all-reduce runs at the speed of its slowest hop)."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text or "")
    rows = {}
    for line in clean.splitlines():
        cells = line.split()
        if cells and re.fullmatch(r"GPU\d+", cells[0]):
            rows[int(cells[0][3:])] = cells[1:]
    pairs = []
    for i in range(gpus):
        for j in range(i + 1, gpus):
            if i in rows and j < len(rows[i]):
                pairs.append(rows[i][j])
    if not pairs:
        return {"links": None, "worst": None}

    def rank(link: str) -> int:
        return next((k for k, p in enumerate(LINK_ORDER) if link.startswith(p)), len(LINK_ORDER))
    worst = max(pairs, key=rank)
    return {"links": sorted(set(pairs), key=rank), "worst": worst}


def _p2p(count: int) -> Optional[str]:
    if count < 2:
        return None
    ok = [torch.cuda.can_device_access_peer(i, j) for i in range(count) for j in range(count) if i != j]
    return "all" if all(ok) else "none" if not any(ok) else "partial"


def _visible_indices(world: int) -> list[int]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis and all(x.strip().isdigit() for x in vis.split(",")):
        return [int(x) for x in vis.split(",")][:world]
    return list(range(world))


_SAMPLES_AVG = re.compile(r"Power Samples.*?Avg\s*:\s*([\d.]+)\s*W", re.S)


def samples_average(text: str) -> Optional[float]:
    """The "Power Samples" average of ``nvidia-smi -q -d POWER`` output (the driver's own samples of
    the last few seconds). Some cards report N/A for power.draw but fill in these samples, e.g. an
    RTX 4060 with driver 616.92."""
    m = _SAMPLES_AVG.search(text or "")
    return float(m.group(1)) if m else None


def samples_per_gpu(text: str) -> list:
    """The "Power Samples" average of every GPU block in ``nvidia-smi -q -d POWER`` output, in the
    order nvidia-smi lists them (the order of the -i ids)."""
    blocks = re.split(r"(?m)^GPU [0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d", text or "")[1:]
    return [samples_average(b) for b in blocks]


class PowerSampler:
    """Board power of the given GPUs from nvidia-smi, sampled in a thread while work runs. Reads
    power.draw; for a GPU that reports it as N/A, falls back to the driver's power samples."""

    def __init__(self, indices: list[int], every: float = 0.25, amd: bool = False) -> None:
        self.indices, self.every, self.amd = indices, every, amd
        self.samples: dict[int, list[float]] = {i: [] for i in indices}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.readable = True

    def _run(self) -> None:
        if self.amd:
            return self._run_amd()
        while not self._stop.is_set():
            out = _smi(["--query-gpu=index,power.draw", "--format=csv,noheader,nounits"])
            if out is None:
                self.readable = False
                return
            missing = set(self.indices)
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in self.samples:
                    try:
                        self.samples[int(parts[0])].append(float(parts[1]))
                        missing.discard(int(parts[0]))
                    except ValueError:                      # "[N/A]", "Not Supported"
                        pass
            if missing:
                ids = sorted(missing)
                text = _smi(["-q", "-d", "POWER", "-i", ",".join(map(str, ids))]) or ""
                for i, w in zip(ids, samples_per_gpu(text)):
                    if w is not None:
                        self.samples[i].append(w)
            self._stop.wait(self.every)

    def _run_amd(self) -> None:
        while not self._stop.is_set():
            got = {i: amd_power(i) for i in self.indices}
            if not any(w is not None for w in got.values()):
                self.readable = False
                return
            for i, w in got.items():
                if w is not None:
                    self.samples[i].append(w)
            self._stop.wait(self.every)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def watts(self) -> Optional[list]:
        if not self.readable or not any(self.samples.values()):
            return None
        return [statistics.fmean(self.samples[i]) if self.samples[i] else None for i in self.indices]


def machine_facts(dev: torch.device, world: int) -> dict:
    facts = {"gpus": world, "platform": platform.platform(), "torch": torch.__version__}
    if dev.type != "cuda":
        facts.update(gpu="CPU", memory_gb=None)
        return facts
    props = torch.cuda.get_device_properties(dev)
    facts.update(gpu=torch.cuda.get_device_name(dev), memory_gb=round(props.total_memory / 2 ** 30, 1),
                 gpus_present=torch.cuda.device_count(), cuda=torch.version.cuda, vendor=vendor(),
                 rocm=getattr(torch.version, "hip", None))
    try:
        v = torch.cuda.nccl.version()
        facts["nccl"] = ".".join(map(str, v)) if isinstance(v, tuple) else str(v)
    except Exception:  # noqa: BLE001  (no NCCL on this build)
        facts["nccl"] = None
    q = _smi(["--query-gpu=driver_version,power.limit,pcie.link.gen.max,pcie.link.width.max",
              "--format=csv,noheader,nounits"])
    if q:
        first = [p.strip() for p in q.splitlines()[0].split(",")]
        if len(first) == 4:
            facts.update(driver=first[0], power_limit_w=_num(first[1]), pcie_gen=_num(first[2]),
                         pcie_width=_num(first[3]))
    topo = _smi(["topo", "-m"])
    facts["topology"] = re.sub(r"\x1b\[[0-9;]*m", "", topo) if topo else None
    t = parse_topology(topo or "", world)
    facts["links"], facts["worst_link"] = t["links"], t["worst"]
    facts["p2p"] = _p2p(min(world, torch.cuda.device_count()))
    return facts


def _num(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------


def run_probe(*, layers: int, d_model: int, vocab: int, max_len: int, max_tokens: int,
              heads: Optional[int] = None, probe_layers: Optional[int] = None, reps: int = 3, rounds: int = 3,
              power_seconds: float = 4.0, price_per_hour: Optional[float] = None,
              device: Optional[str] = None, out: Optional[str] = None, quiet: bool = False,
              backend: Optional[str] = None) -> Optional[Probe]:
    """Measures this machine (all ranks of a torchrun job, or one GPU). Returns the
    :class:`~evencut.train.timing.Probe` on rank 0 (and writes it to ``out``), None elsewhere."""
    heads = heads or max(1, d_model // 64)
    pl = min(layers, probe_layers or layers)
    for name, v in (("layers", layers), ("d_model", d_model), ("vocab", vocab), ("max_len", max_len),
                    ("max_tokens", max_tokens), ("reps", reps), ("rounds", rounds), ("probe_layers", pl)):
        if v < 1:
            raise EvencutError(f"evencut: {name} must be >= 1")
    if max_tokens < max_len:
        raise EvencutError("evencut: max_tokens must be at least max_len (one full-length sequence)")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device is None:
        device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    own_group = False
    if world > 1 and not dist.is_initialized():
        if backend is None:
            nccl = dev.type == "cuda" and dist.is_nccl_available() and platform.system() == "Linux"
            backend = "nccl" if nccl else "gloo"
        dist.init_process_group(backend)
        own_group = True
    try:
        return _measure(rank, world, dev, layers, d_model, vocab, heads, pl, max_len, max_tokens, reps, rounds,
                        power_seconds, price_per_hour, out, quiet)
    finally:
        if own_group:
            dist.destroy_process_group()


def _measure(rank, world, dev, layers, d_model, vocab, heads, pl, max_len, max_tokens, reps, rounds,
             power_seconds, price_per_hour, out, quiet) -> Optional[Probe]:
    say = (lambda *a: print(*a, flush=True)) if rank == 0 and not quiet else (lambda *a: None)
    torch.manual_seed(1234 + rank)
    full_w = M.weights(vocab, d_model, layers)
    probe_w = M.weights(vocab, d_model, pl)
    a_full = attention_constant(full_w, layers, d_model)
    a_probe = attention_constant(probe_w, pl, d_model)
    notes: list[str] = []
    if pl < layers:
        notes.append(f"measured with {pl} of {layers} layers; times scaled by the FLOP ratio")

    say(f"evencut probe: {world} rank(s) on {dev.type}; model {layers}x{d_model}, vocab {vocab}"
        + (f" (timing {pl} layers)" if pl < layers else ""))
    tflops = matmul_tflops(dev)
    model = M.GPT(vocab, max_len, layers=pl, d=d_model, heads=heads).to(dev)
    for n, L in shapes(min(max_len, 128), min(max_tokens, 2048))[:3]:      # warm-up: kernels, allocator
        time_batch(model, n, L, vocab, dev, 1)

    # Small batches are bound by launching kernels, which drifts with the host's state (on an RTX
    # 4060 under Windows the same tiny batch took ~22 ms in one run and ~40 ms in the next). So
    # every shape is timed in several rounds, in a different order each round, and the median of
    # all its samples is kept. The first round goes smallest first, to find the memory limit.
    # Memory: Windows does not fail an allocation beyond the card's memory, it moves it to system
    # RAM and runs several times slower. So a batch whose peak memory exceeds what was free at the
    # start counts as not fitting, like an out-of-memory error elsewhere.
    grid = shapes(max_len, max_tokens)
    samples: dict[tuple[int, int], list[float]] = {s: [] for s in grid}
    peaks: dict[tuple[int, int], int] = {}
    fit_limit: Optional[int] = None
    budget = None
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        budget = torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev)
    for rnd in range(rounds):
        seq = list(grid)
        if rnd:
            random.Random(rnd).shuffle(seq)               # same order on every rank
        for n, L in seq:
            _barrier(dev)                                 # all GPUs work at the same time, as in training
            if fit_limit is not None and n * L >= fit_limit:
                continue
            try:
                if rnd == 0 and budget is not None:       # one trial run, then check its peak memory
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(dev)
                    batch_samples(model, n, L, vocab, dev, 1, warmup=0)
                    peaks[(n, L)] = torch.cuda.max_memory_reserved(dev)
                    if peaks[(n, L)] > budget:
                        raise torch.cuda.OutOfMemoryError("peak memory above the free memory (would spill to RAM)")
                samples[(n, L)] += batch_samples(model, n, L, vocab, dev, reps)
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                fit_limit = n * L
                samples[(n, L)] = []
                model.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    measured: dict[tuple[int, int], Optional[float]] = {
        sh: (statistics.median(v) if v and (fit_limit is None or sh[0] * sh[1] < fit_limit) else None)
        for sh, v in samples.items()}
    say(f"  timed {sum(v is not None for v in measured.values())} batch shapes, {rounds} rounds")

    # power under sustained load: the largest batch that fit, repeated on every GPU
    watts = None
    biggest = max((s for s, v in measured.items() if v is not None), key=lambda s: s[0] * s[1], default=None)
    if dev.type == "cuda" and biggest is not None and power_seconds > 0:
        _barrier(dev)
        sampler = PowerSampler(_visible_indices(world), amd=vendor() == "amd") if rank == 0 else None
        if sampler:
            sampler.start()
        end = time.perf_counter() + power_seconds
        while time.perf_counter() < end:
            time_batch(model, biggest[0], biggest[1], vocab, dev, 1, warmup=0)
        _barrier(dev)
        if sampler:
            sampler.stop()
            watts = sampler.watts()

    # DDP hides part of the gradient all-reduce behind the backward pass, more for small batches
    # (the GPU idles between kernel launches) than for large ones. At five batch sizes, time the same
    # batch inside ddp.no_sync() and with sync, alternating; the difference is the exposed sync.
    ddp_raw: list = []
    if world > 1:
        # a synced step that runs out of memory on one rank would leave the others waiting in the
        # all-reduce, so only shapes that fit with room for DDP's buffers on every rank are used
        spare = 4 * sum(p.numel() for p in model.parameters())
        mine = [sh for sh in grid if measured.get(sh) is not None
                and (budget is None or peaks.get(sh, 0) + spare <= 0.9 * budget)]
        everyone: list = [None] * world
        dist.all_gather_object(everyone, mine)
        fits = [sh for sh in grid if all(sh in e for e in everyone)]
        if fits:
            picks = sync_shapes(fits)
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp = DDP(model, device_ids=[dev.index] if dev.type == "cuda" else None, gradient_as_bucket_view=True)
            for n, L in picks:
                free, synced = ddp_pair_samples(ddp, n, L, vocab, dev, max(5, reps))
                ddp_raw.append((n, L, statistics.median(free), statistics.median(synced)))
            del ddp
    probe_params = sum(p.numel() for p in model.parameters())

    # one optimizer step (AdamW), scaled from the probe's weights to the model's
    opt_ms = optimizer_ms(model, dev) * full_w / probe_w
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # gradient all-reduce at several sizes, up to the gradients' size (at most 1 GB)
    ar_points = None
    if world > 1:
        grad_mb = 4 * full_w / 1e6
        sizes = sorted({mb for mb in (1, 8, 64, 256, 1024) if mb < grad_mb} | {min(grad_mb, 1024.0)})
        mine = [allreduce_ms(int(mb * 1e6), dev, reps=10) for mb in sizes]
        ar_points = list(zip(sizes, mine))

    # the fastest tiny batch, scaled to the full model like the curve (more layers, more kernels)
    tiny = [min(v) * layers * (a_full + L) / (pl * (a_probe + L)) for (n, L), v in samples.items() if v and n * L <= 512]
    payload = {"measured": measured, "tflops": tflops, "allreduce": ar_points, "fit_limit": fit_limit,
               "launch": min(tiny) if tiny else None, "ddp": ddp_raw}
    gathered = [payload]
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, payload)
    if rank != 0:
        return None

    # the slowest GPU at each shape every GPU managed
    common = [s for s in measured if all(g["measured"].get(s) is not None for g in gathered)]
    if not common:
        raise EvencutError("evencut: no batch shape could be measured on every GPU (out of memory at the smallest?)")
    scale = {s: layers * (a_full + s[1]) / (pl * (a_probe + s[1])) for s in common}
    slowest = {s: max(g["measured"][s] for g in gathered) * scale[s] for s in common}
    curve = TimeCurve.fit(a_full, [(s[0] * s[1] * (a_full + s[1]), ms) for s, ms in slowest.items()])
    fastest = {s: min(g["measured"][s] for g in gathered) for s in common}
    speed = [statistics.fmean(g["measured"][s] / fastest[s] for s in common) for g in gathered]
    allreduce = None
    sync_points, sync_raw = [], []
    if world > 1:
        pts = [(mb, max(g["allreduce"][k][1] for g in gathered)) for k, (mb, _) in enumerate(ar_points)]
        allreduce = AllReduce(world, tuple(pts))
        ar_probe = allreduce.ms(4 * probe_params)            # the probed layers' fp32 gradients
        for k, (n, L, _, _) in enumerate(ddp_raw):
            free = max(g["ddp"][k][2] for g in gathered)     # slowest GPU, no sync
            synced = max(g["ddp"][k][3] for g in gathered)   # the step ends when the last GPU is done
            sync_raw.append([n, L, round(free, 3), round(synced, 3), round(ar_probe, 3)])
            x = free * layers * (a_full + L) / (pl * (a_probe + L))     # in the full model's time
            sync_points.append([round(x, 3), round(exposed_share(free, synced, ar_probe), 4)])
        sync_points.sort()
    limits = [g["fit_limit"] for g in gathered if g["fit_limit"] is not None]
    facts = machine_facts(dev, world)
    if facts.get("vendor") == "amd":
        notes.append("AMD GPU: links are not read, power comes from amd-smi/rocm-smi; untested on AMD hardware")
    if budget is not None:
        facts["free_memory_gb"] = round(budget / 2 ** 30, 1)
    probe = Probe(
        machine=facts,
        model={"params": full_w, "layers": layers, "d_model": d_model, "heads": heads, "vocab": vocab,
               "probe_layers": pl, "max_len": max_len, "attention_constant": a_full},
        curve=curve, optimizer_ms=opt_ms, allreduce=allreduce, watts=watts,
        tflops=[g["tflops"] for g in gathered], gpu_speed=speed,
        max_tokens_tested=max(s[0] * s[1] for s in measured),
        max_tokens_fit=max(s[0] * s[1] for s in common) if limits else None,
        price_per_hour=price_per_hour, date=dt.date.today().isoformat(), evencut=__version__, notes=notes,
        launch_ms=min((g["launch"] for g in gathered if g["launch"]), default=None),
        sync_points=sync_points, sync_raw=sync_raw)
    probe.machine["shapes"] = [[s[0], s[1], round(slowest[s], 3)] for s in sorted(common, key=lambda s: s[0] * s[1])]
    if out:
        probe.save(out)
    say(format_probe(probe))
    if out:
        say(f"\nwritten to {out}")
    return probe


# ---------------------------------------------------------------------------------------------


def format_probe(p: Probe) -> str:
    m, md = p.machine, p.model
    lines = [f"Evencut probe: {p.gpus}x {p.gpu}"
             + (f", {m.get('memory_gb')} GB" if m.get("memory_gb") else "")
             + (f" ({m['free_memory_gb']} GB free)" if m.get("free_memory_gb") else "")
             + (f", ${p.price_per_hour:g}/h" if p.price_per_hour else "")]
    link = m.get("worst_link")
    if p.gpus > 1:
        lines.append(f"  links      {', '.join(m.get('links') or ['?'])} (slowest {link}); peer-to-peer: {m.get('p2p')}")
    lines.append(f"  model      {md['layers']} layers x {md['d_model']} wide, vocab {md['vocab']}: "
                 f"{md['params'] / 1e6:,.1f}M weights" + (f" (timed {md['probe_layers']} layers, scaled)"
                                                          if md["probe_layers"] < md["layers"] else ""))
    tf = [t for t in p.tflops if t]
    if tf:
        lines.append(f"  compute    bf16 matmul {statistics.median(tf):.1f} TFLOPS per GPU"
                     + (f" (range {min(tf):.1f}-{max(tf):.1f})" if len(tf) > 1 else ""))
    if len(p.gpu_speed) > 1:
        slow = max(p.gpu_speed)
        lines.append(f"  stragglers slowest GPU {100 * (slow - 1):.1f}% behind the fastest"
                     + ("  <- a slow card: every step waits for it" if slow > 1.05 else ""))
    lines.append("  one micro-batch, forward + backward (slowest GPU):")
    lines.append(f"    {'tokens':>8} {'length':>7} {'ms':>9} {'efficiency':>11}")
    best = p.curve.best
    for n, L, ms in m.get("shapes", []):
        cost = p.curve.cost(n, L)
        lines.append(f"    {n * L:8,} {L:7} {ms:9.2f} {best * cost / p.curve.ms(cost):10.0%}")
    lines.append("    efficiency = share of the GPU's best measured speed; small batches waste the rest")
    if p.allreduce:
        lines.append(f"  grad sync  all-reduce of {p.grad_bytes / 1e6:,.0f} MB fp32 gradients: "
                     f"{p.allreduce_ms():.1f} ms (bus {p.allreduce.bus_gbps():.1f} GB/s)")
        if p.sync_points:
            parts = ", ".join(f"{100 * f:.0f}% after a {t:.0f} ms batch" for t, f in p.sync_points)
            lines.append(f"             exposed after DDP's overlap with the backward pass: {parts}")
        elif p.overlap is not None:
            lines.append(f"             DDP hides {p.overlap:.0%} of the last micro-batch's time worth of it")
    else:
        lines.append("  grad sync  one GPU: none measured (run with torchrun on all GPUs for multi-GPU)")
    lines.append(f"  optimizer  AdamW step {p.optimizer_ms:.1f} ms")
    w = p.mean_watts()
    lines.append(f"  power      {w:.0f} W per GPU under load" if w else "  power      not readable here (nvidia-smi)")
    if p.max_tokens_fit:
        lines.append(f"  memory     out of memory above {p.max_tokens_fit:,} padded tokens per GPU"
                     + (" (of the timed layers only)" if md["probe_layers"] < md["layers"] else ""))
    else:
        lines.append(f"  memory     every batch fit, up to {p.max_tokens_tested:,} padded tokens per GPU")
    for note in p.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


__all__ = ["format_probe", "machine_facts", "parse_topology", "run_probe", "shapes", "time_batch"]
