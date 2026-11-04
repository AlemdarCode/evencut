"""Test 4: real stories, real GPU.

Data: a tokenized corpus where every story ends with an end-of-text token (uint16 ids). Our run
uses the 359,954 training stories of TinyStories-Turkish (esat-krky/TinyStories_Turkish),
tokenized by the minimasal project; any such file works. Stories longer than --seq-len are
truncated. The model is a 29.9M-parameter decoder (8 layers, width 512, 8 heads, vocab 8192,
RoPE, SwiGLU, bf16), the same shape as minimasal's "standart" model. Token values do not affect
speed, so random ids of the real story lengths are fed in.

Part A, one GPU: whole stories are padded to the longest story in their batch. Four ways to
form batches, each a full training step (forward, backward, AdamW), all within the same memory
budget of --budget padded tokens per step:
  1. random order, fixed batch size;
  2. sorted by length, fixed batch size;
  3. sorted by length, filled greedily up to the budget (the usual "max tokens" batching);
  4. sorted by length, evencut.ai.token_budget_batches, which balances real tokens only, so its
     budget has to be lowered until the padded batches fit;
  5. sorted by length, evencut.ai.padded_batches: fewest steps under the padded budget, then
     evened out.

Part B, eight data-parallel GPUs: each step takes --global-batch stories and gives every GPU a
contiguous slice, padded to its own longest story. Every slice is run (forward + backward) on
this one GPU and timed; a real 8-GPU step waits for the slowest slice, so step time = the
largest of the eight. Splits: equal story counts (the default), Evencut balancing real tokens
(evencut.partition) and Evencut balancing padded size (evencut.ai.padded_slices). Two data
orders: random, and length-grouped (the global batch sorted by length, which data loaders do to
reduce padding). A noise-floor row gives every GPU the same slice.

Run:   python bench/gpu_batches.py --tokens train.bin --eot 0
       python bench/gpu_batches.py --jsonl databricks-dolly-15k.jsonl \
           --fields instruction,context,response --tokenizer gpt2 --seq-len 1024 --fixed 16 \
           --global-batch 128 --skip-a --name Dolly-15k --out-name gpu_batches_dolly
Needs: PyTorch with CUDA, evencut (and the tokenizers package for --jsonl).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Optional

# Batches of changing shape fragment PyTorch's CUDA memory pool. On an 8 GB card under Windows
# the driver then spills to system RAM and steps run ~3x slower. Growable segments avoid that;
# every method below runs with the same setting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402  (must come after the allocator setting)

import evencut  # noqa: E402
from evencut import ai  # noqa: E402
from evencut.train import _plans as plans  # noqa: E402
from evencut.train._model import GPT, Block, weights as model_weights  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def recorded_args(args: argparse.Namespace) -> dict:
    """The run's arguments for the result file, with portable paths: relative to the repo when
    inside it, else ``~/...`` under the home folder. A result file says how it was made, not
    whose computer made it."""
    home = Path.home().resolve()

    def portable(v):
        if not isinstance(v, Path):
            return str(v)
        p = v.expanduser().resolve()
        for base, prefix in ((ROOT, ""), (home, "~/")):
            try:
                return prefix + p.relative_to(base).as_posix()
            except ValueError:
                pass
        return v.as_posix()

    return {k: portable(v) for k, v in vars(args).items()}


# ---------------------------------------------------------------------------------------------
# model: decoder-only transformer, the shape of minimasal's "standart" preset

# (moved to evencut.train._model so `evencut-train probe` times exactly this model)


# ---------------------------------------------------------------------------------------------
# data

def story_lengths(path: Path, eot: int, seq_len: int) -> list[int]:
    data = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.int16)  # ids < 32768
    ends = (data == eot).nonzero().flatten()
    lengths = torch.diff(ends, prepend=torch.tensor([-1])).tolist()
    return [min(x, seq_len) for x in lengths]


def padded(batch: list[int]) -> int:
    return len(batch) * max(batch)


def split(lens: list[int], ends) -> list[list[int]]:
    out, prev = [], 0
    for e in ends:
        out.append(lens[prev:e])
        prev = e
    return out


def greedy_padded(s: list[int], budget: int) -> list[list[int]]:
    """Length-sorted (ascending) sequences cut greedily into batches of at most budget padded
    tokens: the usual "max tokens" batching (fairseq, many trainers)."""
    return split(s, plans.greedy_ends(s, budget))


def budget_batches(s: list[int], count: int) -> list[list[int]]:
    """Greedy max-token batching with the smallest budget that yields at most ``count`` batches,
    topped up to exactly ``count`` by halving the largest batches (which only helps it)."""
    return split(s, plans.budget_ends(s, count))


SORTED_ROWS = (("sorted dataset", "token budget per GPU (fairseq-style)"),
               ("sorted dataset", "Evencut, padded_slices over the sorted data"))


def attention_constant(vocab: int = 8192, d: int = 512, layers: int = 8) -> int:
    """A in cost(members, longest) = members * longest * (A + longest) for the GPT above:
    per padded sequence of length L the dense layers cost L * F (F = 2 x weights, forward) and
    causal attention costs about layers * 2 * d * L^2, so A = F / (2 * layers * d)."""
    return plans.attention_constant(model_weights(vocab, d, layers), layers, d)


POOLED = "pooled (like group_by_length)"


def step_plans(stream: list[int], gpus: int, global_batch: int, steps: int, seed: int,
               attention: Optional[int] = None, accum: int = 1, extended: bool = False,
               dataset_size: Optional[int] = None) -> dict[tuple[str, str], list[list[list[int]]]]:
    """Every data-parallel plan compared in part B: (data order, split) -> steps -> one slice of
    sequence lengths per GPU. All plans cover exactly the same sequences in the same number of steps.
    With ``attention`` (see attention_constant) the plans balanced on padded size plus attention
    are added. The plans come from evencut.train._plans (the code the sampler uses), built over
    dataset indices and mapped back to lengths here. ``steps`` counts micro-steps; with ``accum``
    > 1 an optimizer step is ``accum`` of them. ``extended`` adds the phase-2 strategies: Evencut
    window and pooled balancing (what BalancedBatchSampler does), Hugging Face group_by_length and,
    if transformers is installed, Hugging Face batch_rebalance."""
    stream = stream[: steps * global_batch]
    order = range(len(stream))
    attn = plans.AttentionCost(attention) if attention is not None else None
    splitters = [("equal counts", "equal", None), ("Evencut, balanced tokens", "tokens", None),
                 ("Evencut, balanced padded size", "padded", None)]
    if attn is not None:
        splitters.append(("Evencut, padded size + attention", "padded", attn))
    out: dict[tuple[str, str], list[list[list[int]]]] = {}
    one = stream[: global_batch // gpus]
    out[("noise floor", "identical slices")] = [[one] * gpus for _ in range(steps)]
    for order_name, grouped in (("random", False), ("length-grouped", True)):
        for how, kind, cost in splitters:
            out[(order_name, how)] = plans.per_step(order, stream, gpus, global_batch, kind, grouped=grouped, cost=cost)
    out[("any order", "LPT on padded size (not contiguous)")] = plans.lpt_any_order(order, stream, gpus, global_batch)
    sorted_rows = [("token budget per GPU (fairseq-style)", dict(method="fairseq")),
                   ("Evencut, padded_slices over the sorted data", dict(method="evencut"))]
    if attn is not None:
        sorted_rows.append(("Evencut, padded + attention over the sorted data", dict(method="evencut", cost=attn)))
    for how, kw in sorted_rows:
        out[("sorted dataset", how)] = plans.sorted_dataset(order, stream, gpus, steps, seed, **kw)
    if extended:
        if steps % accum:
            raise SystemExit("--steps-b must be a multiple of --accum")
        per_device = global_batch // gpus
        mult = plans.hf_mega_batch_mult(dataset_size or len(stream), per_device * accum)
        if accum > 1:
            out[("random", "Evencut, whole optimizer step balanced (window)")] = plans.window(
                order, stream, gpus, global_batch, accum, cost=attn)
        try:
            out[("random", "Hugging Face batch_rebalance")] = plans.hf_batch_rebalance(
                order, stream, gpus, global_batch, accum)
        except ImportError:
            print("transformers not installed: no Hugging Face batch_rebalance row")
        out[(POOLED, "Hugging Face group_by_length")] = plans.hf_group_by_length(
            order, stream, gpus, per_device, accum, mega_batch_mult=mult)
        out[(POOLED, "Evencut, length-grouped steps + balanced")] = plans.pooled(
            order, stream, gpus, global_batch, accum, pool_steps=plans.default_pool_steps(gpus),
            seed=plans.epoch_seed(seed, 0), balance="window" if accum > 1 else "step", cost=attn)
    for key, plan in out.items():
        if key[0] != "noise floor":
            out[key] = [[[stream[i] for i in b] for b in step] for step in plan]
    return out


def plans_part_a(lens: list[int], fixed: int, budget: int) -> dict[str, list[list[int]]]:
    s = sorted(lens)
    greedy = greedy_padded(s, budget)
    # token_budget_batches balances real tokens only; lower its budget until every padded batch fits
    b = budget
    while True:
        ev = split(s, ai.token_budget_batches(s, b).ends)
        worst = max(padded(x) for x in ev)
        if worst <= budget:
            break
        b = b * budget // worst
    return {
        "random order, fixed size": [lens[i:i + fixed] for i in range(0, len(lens), fixed)],
        "sorted, fixed size": [s[i:i + fixed] for i in range(0, len(s), fixed)],
        "sorted, greedy budget": greedy,
        "sorted, Evencut token budget (ignores padding)": ev,
        "sorted, Evencut padded budget": split(s, ai.padded_batches(s, budget).ends),
    }


# ---------------------------------------------------------------------------------------------
# GPU work

def make_batch(batch: list[int], vocab: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    b, t = len(batch), max(batch)
    idx = torch.randint(0, vocab, (b, t), device=dev)
    tgt = torch.randint(0, vocab, (b, t), device=dev)
    pad = torch.arange(t, device=dev)[None, :] >= torch.tensor(batch, device=dev)[:, None]
    return idx, tgt.masked_fill(pad, -100)


def train_steps(model: GPT, opt, batches: list[list[int]], vocab: int, dev) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch in batches:
        idx, tgt = make_batch(batch, vocab, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(idx, tgt)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def slice_time(model: GPT, batch: list[int], vocab: int, dev, reps: int) -> float:
    idx, tgt = make_batch(batch, vocab, dev)
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(idx, tgt)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        model.zero_grad(set_to_none=True)
    return statistics.median(times)


def jsonl_lengths(path: Path, fields: list[str], tokenizer: str, seq_len: int) -> list[int]:
    """Token count (+1 end-of-text) of every record of a JSON-lines file, fields joined by newlines."""
    from tokenizers import Tokenizer  # only needed for this input

    tok = Tokenizer.from_file(tokenizer) if Path(tokenizer).is_file() else Tokenizer.from_pretrained(tokenizer)
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                texts.append("\n".join(str(rec[k]) for k in fields if rec.get(k)))
    return [min(len(e.ids) + 1, seq_len) for e in tok.encode_batch(texts)]


COMM_MS = (("none", 0.0), ("NVLink-class, ~0.5 ms", 0.5), ("PCIe 4.0, ~7 ms", 7.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens", type=Path, help="uint16 token file, stories ended by --eot")
    src.add_argument("--jsonl", type=Path, help="JSON-lines text file, tokenized with --tokenizer")
    src.add_argument("--lengths", type=Path, help="ready-made sequence lengths, one per line")
    ap.add_argument("--eot", type=int, help="end-of-text id for --tokens")
    ap.add_argument("--fields", default="text", help="comma-separated JSON fields for --jsonl")
    ap.add_argument("--tokenizer", default="gpt2", help="tokenizer.json path or Hugging Face name for --jsonl")
    ap.add_argument("--name", default="TinyStories-Turkish", help="dataset name for the report")
    ap.add_argument("--out-name", default="gpu_batches", help="result file name (without extension)")
    ap.add_argument("--skip-a", action="store_true", help="only run part B")
    ap.add_argument("--attention-cost", action="store_true",
                    help="also balance on padded size plus the attention term (long sequences)")
    ap.add_argument("--save-lengths", type=Path, help="also write the shuffled lengths (one per line) for ddp_batches.py")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--stories-a", type=int, default=20_480, help="stories used in part A")
    ap.add_argument("--fixed", type=int, default=32, help="fixed batch size in part A")
    ap.add_argument("--budget", type=int, default=16_384, help="padded tokens per step in part A")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--steps-b", type=int, default=40, help="micro-steps in part B")
    ap.add_argument("--accum", type=int, default=1, help="micro-steps per optimizer step in part B")
    ap.add_argument("--extended", action="store_true",
                    help="also time window/pooled balancing and the Hugging Face samplers")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    assert args.fixed * args.seq_len <= args.budget, "the fixed batch must fit the budget in the worst case"

    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    if args.tokens:
        if args.eot is None:
            raise SystemExit("--tokens needs --eot")
        lens_all = story_lengths(args.tokens, args.eot, args.seq_len)
    elif args.jsonl:
        lens_all = jsonl_lengths(args.jsonl, args.fields.split(","), args.tokenizer, args.seq_len)
    else:
        lens_all = [min(int(x), args.seq_len) for x in args.lengths.read_text(encoding="ascii").split()]
    rng = random.Random(args.seed)
    rng.shuffle(lens_all)
    print(f"{args.name}: {len(lens_all):,} sequences, mean {statistics.mean(lens_all):.1f} tokens, "
          f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, evencut {evencut.__version__}")

    model = GPT(args.vocab, args.seq_len).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    # warm-up on stories from the end of the shuffled list (never measured), same budget as part A
    for batches in plans_part_a(lens_all[-4096:], args.fixed, args.budget).values():
        train_steps(model, opt, batches, args.vocab, dev)
    s_len = sorted(lens_all)
    report = {"evencut": evencut.__version__, "date": dt.date.today().isoformat(), "dataset": args.name,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "params": sum(p.numel() for p in model.parameters()), "stories": len(lens_all),
              "mean_len": statistics.mean(lens_all), "median_len": s_len[len(s_len) // 2],
              "p90_len": s_len[int(0.9 * (len(s_len) - 1))], "seq_len": args.seq_len,
              "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
              "args": recorded_args(args)}

    # ---- part A
    sample = lens_all[: args.stories_a]
    real = sum(sample)
    part_a = []
    for name, batches in ({} if args.skip_a else plans_part_a(sample, args.fixed, args.budget)).items():
        secs = statistics.median(train_steps(model, opt, batches, args.vocab, dev) for _ in range(3))
        pad = sum(padded(b) for b in batches)
        part_a.append({"method": name, "steps": len(batches), "padded_tokens": pad, "real_tokens": real,
                       "wasted": 1 - real / pad, "seconds": secs, "tokens_per_s": real / secs,
                       "largest_step": max(padded(b) for b in batches)})
        print(f"A {name:47} steps {len(batches):5}  wasted {1 - real / pad:6.1%}  {secs:6.2f} s  "
              f"{real / secs:9,.0f} real tok/s")

    # ---- part B
    first = 0 if args.skip_a else args.stories_a
    stream = lens_all[first: first + args.steps_b * args.global_batch]
    if len(stream) < args.steps_b * args.global_batch:
        raise SystemExit("not enough sequences for part B; lower --steps-b or --global-batch")
    if args.save_lengths:
        args.save_lengths.write_text("\n".join(map(str, lens_all)) + "\n", encoding="ascii")
        print(f"lengths written to {args.save_lengths}")
    part_b = []
    attention = attention_constant(args.vocab) if args.attention_cost else None
    plans = step_plans(stream, args.gpus, args.global_batch, args.steps_b, args.seed, attention,
                       accum=args.accum, extended=args.extended, dataset_size=len(lens_all))
    report["attention_constant"] = attention
    report["accum"] = args.accum
    report["sorted_rows_identical"] = plans[SORTED_ROWS[0]] == plans[SORTED_ROWS[1]]
    for (order, how), plan in plans.items():
        step_total = mean_total = 0.0
        tokens = pad_total = 0
        # GPUs wait for each other once per optimizer step (DDP with no_sync while accumulating):
        # each GPU's time is summed over the step's micro-steps, the step waits for the slowest
        for w in range(0, len(plan), args.accum):
            per_gpu = [0.0] * args.gpus
            for slices in plan[w:w + args.accum]:
                for g, sl in enumerate(slices):
                    per_gpu[g] += slice_time(model, sl, args.vocab, dev, args.reps)
                tokens += sum(sum(sl) for sl in slices)
                pad_total += sum(padded(sl) for sl in slices)
            step_total += max(per_gpu)
            mean_total += statistics.mean(per_gpu)
        row = {"order": order, "split": how, "step_ms": 1000 * step_total / (args.steps_b // args.accum),
               "idle": 1 - mean_total / step_total, "tokens_per_s": tokens / step_total,
               "wasted": 1 - tokens / pad_total}
        part_b.append(row)
        print(f"B {order:15} {how:30} step {row['step_ms']:6.1f} ms  idle {row['idle']:6.1%}  "
              f"{row['tokens_per_s']:9,.0f} tok/s (8 GPUs)  padding {row['wasted']:6.1%}")
    report["part_a"], report["part_b"] = part_a, part_b
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.out_name
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, out.with_suffix(".md"))
    print(f"\nWritten to {out.with_suffix('.md')}")


def write_markdown(r: dict, path: Path) -> None:
    lines = [
        f"# Test 4: real data on a real GPU ({r['dataset']})",
        "",
        f"evencut {r['evencut']}, {r['date']}, {r['gpu']}, PyTorch {r['torch']}. Produced by `bench/gpu_batches.py`. "
        f"{r['stories']:,} sequences (mean {r['mean_len']:.0f} tokens, median {r['median_len']}, "
        f"90th percentile {r['p90_len']}, cut at {r['seq_len']}), {r['params'] / 1e6:.1f}M-parameter decoder, bf16, "
        f"PYTORCH_CUDA_ALLOC_CONF={r['alloc_conf']}.",
    ]
    if r["part_a"]:
        a0 = r["part_a"][0]["tokens_per_s"]
        lines += [
            "",
            "## A. One GPU: forming padded batches (full training steps)",
            "",
            "| method | steps | padding (wasted share) | time | real tokens / s | vs random order |",
            "|---|--:|--:|--:|--:|--:|",
        ]
        for x in r["part_a"]:
            lines.append(f"| {x['method']} | {x['steps']:,} | {x['wasted']:.1%} | {x['seconds']:.1f} s | "
                         f"{x['tokens_per_s']:,.0f} | {x['tokens_per_s'] / a0:.2f}x |")
    lines += [
        "",
        "## B. Eight data-parallel GPUs: one step waits for the slowest GPU",
        "",
        "Every GPU's slice was run and timed on this GPU; step time = the slowest of the eight. The noise floor "
        "gives all eight GPUs the same slice, so its idle share is measurement noise alone."
        + (f" Gradient accumulation {r['accum']}: GPUs wait for each other once per optimizer step (DDP with "
           f"no_sync), so each GPU's time is summed over the {r['accum']} micro-steps and the step waits for the "
           f"slowest sum; times are per optimizer step." if r.get("accum", 1) > 1 else ""),
        "",
        "| data order | split | step time | GPUs idle (average) | real tokens / s (8 GPUs) | padding | "
        "vs the order's first row | vs default (random, equal counts) |",
        "|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    base = {}
    default = next((x for x in r["part_b"] if x["order"] == "random"), None)
    for x in r["part_b"]:
        if x["order"] != "noise floor":
            base.setdefault(x["order"], x)
        rel = "" if x["order"] == "noise floor" else f"{x['tokens_per_s'] / base[x['order']]['tokens_per_s']:.2f}x"
        dflt = "" if x["order"] == "noise floor" or default is None else f"{x['tokens_per_s'] / default['tokens_per_s']:.2f}x"
        lines.append(f"| {x['order']} | {x['split']} | {x['step_ms']:.1f} ms | {x['idle']:.1%} | "
                     f"{x['tokens_per_s']:,.0f} | {x['wasted']:.1%} | {rel} | {dflt} |")
    lines += [
        "",
        "\"sorted dataset\" sorts all sequences of the run, cuts them into one batch per GPU and step, and shuffles "
        "the batches; its first row is greedy max-token batching (fairseq-style), the strongest common baseline."
        + (" Here its first two rows hand out exactly the same batches (checked by the script): with the whole "
           "dataset sorted, the best max-token budget already gives the optimal split, and the gap between the two "
           "rows is run-to-run timing variation." if r.get("sorted_rows_identical") else ""),
        "",
        "### Adding gradient communication (modelled, not measured)",
        "",
        f"Each step also all-reduces the {r['params'] / 1e6:.1f}M bf16 gradients (~{r['params'] * 2 / 1e6:.0f} MB) "
        "across the eight GPUs, the same for every split. Rough ring all-reduce times are added below; real "
        "frameworks overlap part of it with the backward pass, so these are upper bounds.",
        "",
        "| data order | best Evencut split vs the order's first row: " + " | ".join(name for name, _ in COMM_MS) + " |",
        "|---|" + "--:|" * len(COMM_MS),
    ]
    for order in [o for o in base]:
        rows = [x for x in r["part_b"] if x["order"] == order]
        if len(rows) < 2:                        # nothing to compare within this order
            continue
        best = min(rows[1:], key=lambda x: x["step_ms"])
        cells = [f"{(base[order]['step_ms'] + c) / (best['step_ms'] + c):.2f}x" for _, c in COMM_MS]
        lines.append(f"| {order} ({best['split']} vs {base[order]['split']}) | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
