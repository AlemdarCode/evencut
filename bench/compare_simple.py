"""Test 2: how much better than the simple methods people actually use?

Same ten public datasets as test 1 (bench_data.py). Each one is split into k contiguous parts
by Evencut and by four common simple methods; the score is the heaviest part, because in a
parallel step every worker waits for the slowest one.

Simple methods (all keep the order and give exactly k non-empty parts):
* equal count:          n/k items per part (np.array_split, "rows per process");
* nearest-target cut:   cut j at the boundary closest to j * total / k ("direct cut");
* greedy fill:          fill a part until it reaches the average, then start the next one;
                        the last part takes what is left;
* recursive bisection:  halve the work at the boundary closest to the middle, recurse.

Reported per method: heaviest part / average part (1.00 = perfect), and how many times longer
the slowest worker runs than with Evencut. "Best possible" is the lower bound
max(total / k, largest single item): no split of any kind can beat it.

SNAP networks are used with their node IDs compacted (nodes in ID order, gaps removed), so a
gap in the numbering cannot make a method look worse than it is.

Run:   python bench/compare_simple.py [--data-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
from bisect import bisect_left
from pathlib import Path

import evencut

from bench_data import DATASETS, load

HERE = Path(__file__).resolve().parent
K_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096]
DETAIL_K = (8, 64, 1024)
METHODS = ("equal count", "nearest-target cut", "greedy fill", "recursive bisection")


def prefix_sums(w: list[int]) -> list[int]:
    P = [0] * (len(w) + 1)
    s = 0
    for i, x in enumerate(w):
        s += x
        P[i + 1] = s
    return P


def _closest(P: list[int], k: int, t: int, lo: int, hi: int) -> int:
    """Boundary c in [lo, hi] whose P[c] * k is closest to t (ties: the earlier one)."""
    e = bisect_left(P, -(-t // k), lo, hi + 1)
    return min((c for c in (e - 1, e) if lo <= c <= hi), key=lambda c: (abs(P[c] * k - t), c))


def equal_count(P: list[int], k: int) -> list[int]:
    n = len(P) - 1
    return [n * (j + 1) // k for j in range(k)]


def nearest_target(P: list[int], k: int) -> list[int]:
    n, S = len(P) - 1, P[-1]
    ends, prev = [], 0
    for j in range(1, k):
        prev = _closest(P, k, j * S, prev + 1, n - (k - j))
        ends.append(prev)
    return ends + [n]


def greedy_fill(P: list[int], k: int) -> list[int]:
    n, S = len(P) - 1, P[-1]
    avg = -(-S // k)
    ends, i = [], 0
    for j in range(1, k):
        last = n - (k - j)
        e = bisect_left(P, P[i] + avg, i + 1, last + 1)   # first boundary where the part reaches avg
        i = max(i + 1, min(e, last))
        ends.append(i)
    return ends + [n]


def recursive_bisection(P: list[int], k: int) -> list[int]:
    ends: list[int] = []

    def rec(lo: int, hi: int, parts: int) -> None:
        if parts == 1:
            ends.append(hi)
            return
        left = parts // 2
        m = _closest(P, parts, P[lo] * parts + (P[hi] - P[lo]) * left, lo + left, hi - (parts - left))
        rec(lo, m, left)
        rec(m, hi, parts - left)

    rec(0, len(P) - 1, k)
    return ends


SPLITTERS = {"equal count": equal_count, "nearest-target cut": nearest_target,
             "greedy fill": greedy_fill, "recursive bisection": recursive_bisection}


def max_load(P: list[int], ends: list[int], k: int) -> int:
    assert len(ends) == k and ends[-1] == len(P) - 1, "a split must have exactly k parts"
    worst, prev = 0, 0
    for e in ends:
        assert e > prev, "parts must be non-empty"
        worst = max(worst, P[e] - P[prev])
        prev = e
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".cache" / "evencut-bench")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    print(f"evencut {evencut.__version__}, backend {evencut.backend}")

    report = {"evencut": evencut.__version__, "date": dt.date.today().isoformat(), "datasets": []}
    for name, url, digest, kind, field in DATASETS:
        w, _ = load(name, url, digest, kind, args.data_dir, compact=True)
        P = prefix_sums(w)
        n, S, biggest = len(w), P[-1], max(w)
        print(f"\n{name}: n={n:,} total={S:,} largest item={biggest:,}")
        rows = []
        for k in [k for k in K_VALUES if k <= n // 2]:
            t0 = time.perf_counter()
            best = evencut.partition(w, k).max_load
            ms = (time.perf_counter() - t0) * 1000
            mean = S / k
            bound = max(-(-S // k), biggest)
            row = {"k": k, "mean": mean, "best_possible": bound, "evencut": best, "evencut_ms": round(ms, 1),
                   "methods": {}}
            for m in METHODS:
                worst = max_load(P, SPLITTERS[m](P, k), k)
                assert worst >= best, f"{m} beat the optimum: impossible"
                row["methods"][m] = worst
            rows.append(row)
            print(f"  k={k:5}: best possible {bound / mean:.3f}  evencut {best / mean:.3f}  "
                  + "  ".join(f"{m} {v / mean:.3f}" for m, v in row["methods"].items()))
        report["datasets"].append({"name": name, "field": field, "n": n, "total": S, "largest_item": biggest,
                                   "results": rows})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "compare_simple.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, args.out_dir / "compare_simple.md")
    print(f"\nWritten to {args.out_dir / 'compare_simple.md'}")


def _ratio(x: float) -> str:
    return f"{x:.2f}x" if x < 10 else f"{x:.0f}x"


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Test 2: Evencut against simple methods",
        "",
        f"evencut {report['evencut']}, {report['date']}. Produced by `bench/compare_simple.py`. "
        "Score: how many times longer the slowest worker runs than with Evencut (1.00x = as good as "
        "Evencut). Ten real datasets; SNAP node IDs compacted.",
        "",
        "## Summary over the ten datasets",
        "",
        "| k | " + " | ".join(f"{m}: median / worst" for m in METHODS) + " |",
        "|--:|" + "--:|" * len(METHODS),
    ]
    all_k = sorted({r["k"] for d in report["datasets"] for r in d["results"]})
    for k in all_k:
        cells = []
        for m in METHODS:
            ratios = [r["methods"][m] / r["evencut"] for d in report["datasets"] for r in d["results"] if r["k"] == k]
            cells.append(f"{_ratio(statistics.median(ratios))} / {_ratio(max(ratios))}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    for k in DETAIL_K:
        lines += ["", f"## k = {k}: heaviest part / average part (1.00 = perfect)", "",
                  "| dataset | field | items | best possible | Evencut | " + " | ".join(METHODS) + " |",
                  "|---|---|--:|--:|--:|" + "--:|" * len(METHODS)]
        for d in report["datasets"]:
            r = next((r for r in d["results"] if r["k"] == k), None)
            if r is None:
                continue
            mean = r["mean"]
            lines.append(f"| {d['name']} | {d['field']} | {d['n']:,} | {r['best_possible'] / mean:.3f} | "
                         f"**{r['evencut'] / mean:.3f}** | "
                         + " | ".join(f"{r['methods'][m] / mean:.3f}" for m in METHODS) + " |")
    lines += [
        "",
        "\"Best possible\" is max(total / k, largest item): no split of any kind can beat it. When it is above "
        "1.00, a single item is heavier than a fair share and no method can be perfect.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
