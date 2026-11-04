"""Test 5: when the order does not matter, is Evencut still the right tool?

Evencut splits a list into contiguous parts: it keeps the order. When any item may go to any
part, the classic method is LPT (longest processing time first): take the items from the
largest down and put each one on the currently lightest part. This test compares the two.

Part 1, plain work items (additive cost): the ten real datasets of tests 1-2 (bench_data.py,
SNAP IDs compacted). Score: largest part / average part, next to the lower bound
max(total / k, largest item) that no assignment of any kind can beat.

Part 2, data-parallel AI training (padded cost): every step's sequences are shared among GPUs
and each GPU pads its sequences to its longest one. Compared:
  * Evencut, contiguous in length order: evencut.ai.padded_slices on the sorted batch;
  * LPT on padded cost, order ignored: sequences from the longest down, each one to the GPU
    whose padded size (members x longest) ends up smallest.
Score: the predicted share of time GPUs wait, 1 - mean / max of the padded sizes per step
(test 4 showed that this predicts the measured idle share closely).

Run:   python bench/order_free.py [--data-dir DIR] [--lengths-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import heapq
import json
import random
import time
from pathlib import Path

import evencut
from evencut import ai

from bench_data import DATASETS, load

HERE = Path(__file__).resolve().parent
K_VALUES = [2, 8, 64, 1024, 4096]


# ---------------------------------------------------------------------------------------------
# part 1: additive cost

def lpt(weights: list[int], k: int) -> int:
    """Largest part of the LPT assignment (order ignored)."""
    heap = [(0, p) for p in range(k)]
    for w in sorted(weights, reverse=True):
        load, p = heapq.heappop(heap)
        heapq.heappush(heap, (load + w, p))
    return max(load for load, _ in heap)


# ---------------------------------------------------------------------------------------------
# part 2: padded cost

def lpt_padded(lengths: list[int], k: int) -> list[list[int]]:
    """Sequences from the longest down, each one to the GPU whose padded size ends up smallest
    (the LPT rule, with members x longest as the load)."""
    groups: list[list[int]] = [[] for _ in range(k)]
    for x in sorted(lengths, reverse=True):
        # a group's longest member is its first one, because the input is descending
        best = min(range(k), key=lambda g: ((len(groups[g]) + 1) * (groups[g][0] if groups[g] else x), g))
        groups[best].append(x)
    return groups


def padded(group: list[int]) -> int:
    return len(group) * max(group) if group else 0


def predicted_idle(steps: list[list[list[int]]]) -> tuple[float, float]:
    """(1 - mean/max over the run, average largest padded size per step)."""
    mx = sum(max(padded(g) for g in step) for step in steps)
    mean = sum(sum(padded(g) for g in step) / len(step) for step in steps)
    return 1 - mean / mx, mx / len(steps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".cache" / "evencut-bench")
    ap.add_argument("--lengths-dir", type=Path, help="tinystories_tr.txt, dolly15k_gpt2.txt, wikipedia_tr.txt")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    report: dict = {"evencut": evencut.__version__, "date": dt.date.today().isoformat(), "part1": [], "part2": []}

    print("Part 1: additive cost, order ignored (LPT) vs contiguous (Evencut)")
    for name, url, digest, kind, _field in DATASETS:
        w, _ = load(name, url, digest, kind, args.data_dir, compact=True)
        S, biggest = sum(w), max(w)
        for k in [k for k in K_VALUES if k <= len(w) // 2]:
            mean = S / k
            ev = evencut.partition(w, k).max_load
            t0 = time.perf_counter()
            lp = lpt(w, k)
            lpt_s = time.perf_counter() - t0
            bound = max(-(-S // k), biggest)
            row = {"dataset": name, "n": len(w), "k": k, "bound": bound / mean, "evencut": ev / mean, "lpt": lp / mean,
                   "lpt_seconds": round(lpt_s, 2)}
            report["part1"].append(row)
            print(f"  {name:12} k={k:5}: bound {row['bound']:.4f}  Evencut (contiguous) {row['evencut']:.4f}  "
                  f"LPT (any order) {row['lpt']:.4f}")

    if args.lengths_dir:
        print("\nPart 2: data-parallel steps with padding, 8 GPUs")
        for label, fname, seq, gbatch, offset in (("TinyStories-Turkish", "tinystories_tr.txt", 512, 256, 20480),
                                                  ("Dolly-15k", "dolly15k_gpt2.txt", 1024, 128, 0),
                                                  ("Wikipedia-TR", "wikipedia_tr.txt", 2048, 48, 0)):
            lens = [min(int(x), seq) for x in (args.lengths_dir / fname).read_text(encoding="ascii").split()]
            if label == "Wikipedia-TR":                      # gpu_batches.py shuffles its input once more
                random.Random(20261101).shuffle(lens)
            for gpus in (8,):
                steps_contig, steps_lpt = [], []
                for g in range(40):
                    glob = sorted(lens[offset + g * gbatch: offset + (g + 1) * gbatch])
                    ends = ai.padded_slices(glob, gpus).ends
                    steps_contig.append([glob[a:b] for a, b in zip((0, *ends[:-1]), ends)])
                    steps_lpt.append(lpt_padded(glob, gpus))
                (ic, mc), (il, ml) = predicted_idle(steps_contig), predicted_idle(steps_lpt)
                row = {"dataset": label, "gpus": gpus, "global_batch": gbatch, "seq_len": seq,
                       "contiguous_idle": ic, "lpt_idle": il, "contiguous_max_padded": mc, "lpt_max_padded": ml}
                report["part2"].append(row)
                print(f"  {label:20}: Evencut contiguous idle {ic:6.1%} (max padded {mc:8.0f})   "
                      f"LPT any order idle {il:6.1%} (max padded {ml:8.0f})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "order_free.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, args.out_dir / "order_free.md")
    print(f"\nWritten to {args.out_dir / 'order_free.md'}")


def write_markdown(r: dict, path: Path) -> None:
    lines = [
        "# Test 5: when the order does not matter",
        "",
        f"evencut {r['evencut']}, {r['date']}. Produced by `bench/order_free.py`.",
        "",
        "## Part 1: plain work items, largest part / average part (1.0000 = perfect)",
        "",
        "Evencut keeps the order (contiguous parts); LPT may put any item anywhere. The bound is "
        "max(total / k, largest item): no assignment of any kind beats it.",
        "",
        "| dataset | items | k | bound | Evencut (contiguous) | LPT (any order) |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for x in r["part1"]:
        lines.append(f"| {x['dataset']} | {x['n']:,} | {x['k']} | {x['bound']:.4f} | {x['evencut']:.4f} | {x['lpt']:.4f} |")
    if r["part2"]:
        lines += [
            "",
            "## Part 2: data-parallel steps with padding, predicted share of time GPUs wait",
            "",
            "| dataset | GPUs | sequences per step | Evencut, contiguous in length order: idle / largest padded slice | "
            "LPT on padded size, any order: idle / largest padded slice |",
            "|---|--:|--:|--:|--:|",
        ]
        for x in r["part2"]:
            lines.append(f"| {x['dataset']} | {x['gpus']} | {x['global_batch']} | {x['contiguous_idle']:.1%} / "
                         f"{x['contiguous_max_padded']:,.0f} | {x['lpt_idle']:.1%} / {x['lpt_max_padded']:,.0f} |")
        lines += ["", "Step time follows the largest padded slice, not the idle share: LPT mixes lengths on every "
                  "GPU, so each GPU pads more and the step gets slower even though the GPUs wait less."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
