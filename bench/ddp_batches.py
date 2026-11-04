"""Test 4, on a real multi-GPU machine: the data-parallel comparison of gpu_batches.py (part B)
with real distributed training. Every GPU trains on its own slice, gradients are all-reduced
over NCCL (PyTorch DistributedDataParallel), and each step's wall time is measured, so the
slowest GPU and the communication are both included, nothing is modelled.

Input: a file with one sequence length per line, already shuffled; gpu_batches.py writes it
with --save-lengths. Every rank builds the same plans from it (bench/gpu_batches.py, step_plans).

Run (8 GPUs):
    torchrun --standalone --nproc_per_node=8 bench/ddp_batches.py \
        --lengths tinystories_tr.txt --offset 20480 --name TinyStories-Turkish
    torchrun --standalone --nproc_per_node=8 bench/ddp_batches.py \
        --lengths dolly15k_gpt2.txt --seq-len 1024 --global-batch 128 --name Dolly-15k \
        --out-name ddp_batches_dolly
Needs: PyTorch with CUDA (NCCL on Linux; gloo is used where NCCL is missing, for testing only).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

# gpu_batches.py turns on expandable segments for an 8 GB Windows card that otherwise spills to
# system RAM. Linux has no such spill and WSL2 does not support the setting, so this script uses
# PyTorch's default allocator unless PYTORCH_CUDA_ALLOC_CONF is set explicitly.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "")

import gpu_batches as gb  # noqa: E402  (imports torch)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

import evencut  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = {"small": dict(layers=8, d=512, heads=8), "base": dict(layers=12, d=768, heads=12),
          "large": dict(layers=24, d=1024, heads=16)}


def barrier(dev: torch.device) -> None:
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[dev.index])
    else:
        dist.barrier()


def allreduce_ms(numel: int, dev: torch.device, reps: int = 10) -> float:
    """Median time of one all-reduce as large as the gradients (fp32, as DDP sends them)."""
    x = torch.ones(numel, dtype=torch.float32, device=dev)
    for _ in range(3):
        dist.all_reduce(x)
    times = []
    for _ in range(reps):
        torch.cuda.synchronize(dev)
        barrier(dev)
        t0 = time.perf_counter()
        dist.all_reduce(x)
        torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times)


def run_plan(ddp, opt, plan, rank: int, vocab: int, dev: torch.device, accum: int = 1) -> float:
    """Trains on plan, ``accum`` entries per optimizer step (this rank takes its own slice of
    each); gradients are all-reduced once per optimizer step. Returns total seconds."""
    total = 0.0
    for w in range(0, len(plan), accum):
        window = plan[w:w + accum]
        batches = [gb.make_batch(slices[rank], vocab, dev) for slices in window]
        torch.cuda.synchronize(dev)
        barrier(dev)
        t0 = time.perf_counter()
        for j, (idx, tgt) in enumerate(batches):
            sync = j == len(batches) - 1
            with (contextlib.nullcontext() if sync else ddp.no_sync()):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = ddp(idx, tgt)
                loss.backward()              # the last one returns once the all-reduce has finished
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        barrier(dev)                         # the step ends when the slowest GPU is done
        total += time.perf_counter() - t0
    return total


def topology() -> str:
    try:
        return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi not available"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=Path, required=True, help="one sequence length per line, shuffled")
    ap.add_argument("--offset", type=int, default=0, help="skip this many lengths (20480 matches gpu_batches.py part A)")
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--model", choices=sorted(MODELS), default="small")
    ap.add_argument("--global-batch", type=int, default=256, help="sequences per step over all GPUs")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--accum", type=int, default=1, help="micro-steps per optimizer step (gradient accumulation)")
    ap.add_argument("--extended", action="store_true",
                    help="also run window/pooled balancing (through the real BalancedBatchSampler) and the Hugging "
                         "Face samplers")
    ap.add_argument("--repeats", type=int, default=2, help="timed passes over every plan (median is kept)")
    ap.add_argument("--only", help="run only the plans whose 'order | split' contains one of these texts, separated "
                                   "by semicolons (to save time on a rented machine)")
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--out-name", default="ddp_batches")
    args = ap.parse_args()

    # NCCL is the real thing; EVENCUT_DDP_BACKEND=gloo lets two test ranks share one GPU (NCCL refuses that)
    backend = os.environ.get("EVENCUT_DDP_BACKEND") or ("nccl" if dist.is_nccl_available() else "gloo")
    init = os.environ.get("EVENCUT_DDP_INIT")  # e.g. file:///C:/tmp/ddp_init, for local tests without torchrun
    if init:
        dist.init_process_group(backend, init_method=init, rank=int(os.environ["RANK"]),
                                world_size=int(os.environ["WORLD_SIZE"]))
    else:
        dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device("cuda", local % torch.cuda.device_count())
    torch.cuda.set_device(dev)
    torch.manual_seed(args.seed)

    lens = [min(int(x), args.seq_len) for x in args.lengths.read_text(encoding="ascii").split()]
    stream = lens[args.offset: args.offset + args.steps * args.global_batch]
    if args.steps % args.accum:
        raise SystemExit("--steps must be a multiple of --accum")
    if len(stream) < args.steps * args.global_batch:
        raise SystemExit("not enough lengths for --steps x --global-batch after --offset")
    plans = gb.step_plans(stream, world, args.global_batch, args.steps, args.seed, accum=args.accum,
                          extended=args.extended, dataset_size=len(lens))
    sampler_checked = False
    if args.extended:
        # the window and pooled rows are what the real BalancedBatchSampler yields on this rank (it
        # reads the rank and world size from torch.distributed); stop if it differs from the plan code
        from evencut.train import BalancedBatchSampler
        checks = [("pooled", (gb.POOLED, "Evencut, length-grouped steps + balanced"))]
        if args.accum > 1:
            checks.append(("window", ("random", "Evencut, whole optimizer step balanced (window)")))
        for mode, key in checks:
            sampler = BalancedBatchSampler(stream, global_batch=args.global_batch, accum=args.accum, mode=mode,
                                           shuffle=False, seed=args.seed)
            assert (sampler.rank, sampler.world_size) == (rank, world)
            whole = sampler.plan()
            assert list(sampler) == [m[rank] for m in whole], f"{mode}: sampler yields other batches than its plan"
            as_lengths = [[[stream[i] for i in b] for b in m] for m in whole]
            assert as_lengths == plans[key], f"{mode}: BalancedBatchSampler differs from the benchmark's plan code"
            plans[key] = as_lengths
        sampler_checked = [mode for mode, _ in checks]

    if args.only:
        wanted = [w.strip() for w in args.only.split(";") if w.strip()]
        plans = {k: v for k, v in plans.items() if any(w in f"{k[0]} | {k[1]}" for w in wanted)}
        if not plans:
            raise SystemExit("--only matched no plan")
    model = gb.GPT(args.vocab, args.seq_len, **MODELS[args.model]).to(dev)
    params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[dev.index], gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4, fused=True)
    comm = allreduce_ms(params, dev)
    if rank == 0:
        print(f"{args.name}: {world} x {torch.cuda.get_device_name(dev)}, backend {backend}, model {args.model} "
              f"({params / 1e6:.1f}M params), gradient all-reduce {comm:.1f} ms")

    rows = []
    for (order, how), plan in plans.items():
        warm = (plan * (args.warmup // len(plan) + 1))[: max(args.warmup, args.accum)]
        run_plan(ddp, opt, warm, rank, args.vocab, dev, args.accum)
        secs = statistics.median(run_plan(ddp, opt, plan, rank, args.vocab, dev, args.accum)
                                 for _ in range(args.repeats))
        tokens = sum(sum(s) for step in plan for s in step)
        pad = sum(gb.padded(s) for step in plan for s in step)
        row = {"order": order, "split": how, "step_ms": 1000 * secs * args.accum / len(plan), "tokens_per_s": tokens / secs,
               "wasted": 1 - tokens / pad}
        rows.append(row)
        if rank == 0:
            print(f"{order:15} {how:45} step {row['step_ms']:7.1f} ms  {row['tokens_per_s']:10,.0f} tok/s  "
                  f"padding {row['wasted']:6.1%}", flush=True)

    if rank == 0:
        report = {"evencut": evencut.__version__, "date": dt.date.today().isoformat(), "dataset": args.name,
                  "gpus": world, "gpu": torch.cuda.get_device_name(dev), "backend": backend,
                  "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version()))
                  if backend == "nccl" else None, "platform": platform.platform(), "model": args.model,
                  "params": params, "allreduce_ms": comm, "seq_len": args.seq_len,
                  "global_batch": args.global_batch, "steps": args.steps, "accum": args.accum, "topology": topology(),
                  "sampler_checked": sampler_checked,
                  "args": gb.recorded_args(args), "rows": rows}
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / args.out_name
        out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        write_markdown(report, out.with_suffix(".md"))
        print(f"Written to {out.with_suffix('.md')}")
    dist.destroy_process_group()


def write_markdown(r: dict, path: Path) -> None:
    lines = [
        f"# Test 4, real multi-GPU run ({r['dataset']})",
        "",
        f"evencut {r['evencut']}, {r['date']}, {r['gpus']} x {r['gpu']}, {r['backend']} "
        f"{r['nccl'] or ''}, PyTorch {r['torch']}, {r['platform']}. Model {r['model']} "
        f"({r['params'] / 1e6:.1f}M parameters), {r['global_batch']} sequences per micro-step, "
        f"{r.get('accum', 1)} micro-steps per optimizer step (gradient accumulation), {r['steps']} micro-steps per plan. "
        f"One all-reduce of the fp32 gradients ({r['params'] * 4 / 1e6:.0f} MB) takes {r['allreduce_ms']:.1f} ms. "
        "Produced by `bench/ddp_batches.py`; wall time per step, communication included.",
        "",
        "| data order | split | step time | real tokens / s | padding | vs the order's first row | "
        "vs default (random, equal counts) |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    base: dict[str, dict] = {}
    default = next((x for x in r["rows"] if x["order"] == "random"), None)
    for x in r["rows"]:
        if x["order"] != "noise floor":
            base.setdefault(x["order"], x)
        rel = "" if x["order"] == "noise floor" else f"{x['tokens_per_s'] / base[x['order']]['tokens_per_s']:.2f}x"
        dflt = "" if x["order"] == "noise floor" or default is None else f"{x['tokens_per_s'] / default['tokens_per_s']:.2f}x"
        lines.append(f"| {x['order']} | {x['split']} | {x['step_ms']:.1f} ms | {x['tokens_per_s']:,.0f} | "
                     f"{x['wasted']:.1%} | {rel} | {dflt} |")
    lines += ["", "GPU topology (`nvidia-smi topo -m`):", "", "```", r["topology"].rstrip(), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
