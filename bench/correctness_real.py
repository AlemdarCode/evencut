"""Test 1: are Evencut's answers really optimal on real data?

Downloads ten public datasets, turns each one into an ordered list of work items, partitions it
with Evencut and checks every answer with code that shares nothing with Evencut.

The workload is the classic one for this problem ("chains-on-chains partitioning"):
* sparse matrices (NIST Matrix Market): row i costs its number of non-zeros; contiguous row
  blocks go to k processors for a sparse matrix-vector product;
* networks (Stanford SNAP): node i costs its out-degree; contiguous node-ID ranges go to k
  workers for one PageRank-style sweep. Gaps in the ID numbering are kept as zero-cost items,
  which also exercises zero weights (test 2 removes them).

Checks:
* every result is a valid split (k non-empty contiguous parts covering all items) and the
  reported loads match a recount;
* min_max, proof of optimality: if the largest part is M, the greedy split with bound M - 1
  needs more than k parts. Greedy needs the fewest parts for any bound, so nothing beats M;
* sum_squares, proof of optimality: a lower bound from Lagrangian duality, computed exactly
  by an independent C program (bench/verify/ssq_check.c, "dual" mode). Evencut's cost must
  equal the bound;
* sum_squares, second opinion: the same C program also runs a textbook dynamic program
  where that is affordable, and on the smallest dataset that DP is checked with the plain
  O(k n^2) DP, which assumes nothing;
* the pure-Python backend must give the same splits as the native one (--pure small|all).

Run:   python bench/correctness_real.py [--data-dir DIR] [--out-dir DIR] [--quick] [--pure all]
Needs: a C compiler for the sum_squares checker (cc, gcc or clang; or set CC). About 165 MB
of downloads, cached in --data-dir.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from bisect import bisect_right
from pathlib import Path

import evencut
from evencut import _pure

from bench_data import DATASETS, load

HERE = Path(__file__).resolve().parent

K_VALUES = [2, 3, 4, 8, 16, 32, 64, 128, 256, 1024, 4096]
SMALL_N = 40_000          # "--pure small" runs the pure-Python backend up to this size
FULL_DP_N = 5_000         # plain O(k n^2) DP only up to this size
DC_BUDGET = 6e9           # k * n * log2(n) budget for the textbook DP
FULL_BUDGET = 2e9         # k * n^2 budget for the plain DP


# ---------------------------------------------------------------------------------------------
# independent checks

def prefix_sums(w: list[int]) -> list[int]:
    P = [0] * (len(w) + 1)
    s = 0
    for i, x in enumerate(w):
        s += x
        P[i + 1] = s
    return P


def check_split(P: list[int], k: int, r: evencut.Partition) -> str | None:
    """None if r is a valid k-part split whose reported numbers match a recount."""
    n = len(P) - 1
    ends = list(r.ends)
    if len(ends) != min(k, n):
        return f"{len(ends)} parts instead of {min(k, n)}"
    prev = 0
    loads = []
    for e in ends:
        if not prev < e <= n:
            return f"bad end {e}"
        loads.append(P[e] - P[prev])
        prev = e
    if prev != n:
        return "items left over"
    if tuple(loads) != tuple(r.loads) or max(loads) != r.max_load:
        return "loads do not match a recount"
    if sum(x * x for x in loads) != r.sum_of_squares:
        return "sum of squares does not match a recount"
    return None


def greedy_parts(P: list[int], bound: int, stop_after: int) -> float:
    """Fewest parts with every part <= bound (inf if one item alone exceeds it)."""
    n = len(P) - 1
    i = parts = 0
    while i < n:
        j = bisect_right(P, P[i] + bound) - 1   # furthest end with load <= bound
        if j <= i:
            return math.inf
        i = j
        parts += 1
        if parts > stop_after:
            return parts
    return parts


def min_max_is_optimal(P: list[int], k: int, max_load: int) -> bool:
    return max_load == 0 or greedy_parts(P, max_load - 1, k) > k


def find_cc() -> str | None:
    for c in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if c and shutil.which(c):
            return c
    return None


def build_checker(data_dir: Path) -> Path | None:
    cc = find_cc()
    if cc is None:
        return None
    exe = data_dir / "bin" / ("ssq_check.exe" if os.name == "nt" else "ssq_check")
    src = HERE / "verify" / "ssq_check.c"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        exe.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([cc, "-O2", "-o", str(exe), str(src)], check=True)
    return exe


def run_checker(exe: Path, weights_file: Path, mode: str, arg: str) -> tuple[dict[int, int], float]:
    t0 = time.perf_counter()
    out = subprocess.run([str(exe), str(weights_file), mode, arg], check=True, capture_output=True, text=True).stdout
    secs = time.perf_counter() - t0
    values = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            values[int(fields[0])] = int(fields[1])
    return values, secs


# ---------------------------------------------------------------------------------------------
# main

def machine() -> str:
    cpu = platform.processor()
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return f"{cpu}, {platform.system()} {platform.release()}, Python {platform.python_version()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".cache" / "evencut-bench")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--quick", action="store_true", help="only the five small datasets")
    ap.add_argument("--pure", choices=("none", "small", "all"), default="small",
                    help="where to compare the pure-Python backend (all: slow, about half an hour)")
    args = ap.parse_args()

    checker = build_checker(args.data_dir)
    if checker is None:
        print("warning: no C compiler found, sum_squares answers are only checked for validity")
    sets = DATASETS[:5] if args.quick else DATASETS
    print(f"evencut {evencut.__version__}, backend {evencut.backend}; {machine()}")

    report = {"evencut": evencut.__version__, "backend": evencut.backend, "machine": machine(),
              "date": dt.date.today().isoformat(), "pure_mode": args.pure, "datasets": []}
    for name, url, digest, kind, field in sets:
        w, wfile = load(name, url, digest, kind, args.data_dir)
        P = prefix_sums(w)
        n, total = len(w), P[-1]
        ks = [k for k in K_VALUES if k <= n // 2]
        run_pure = args.pure == "all" or (args.pure == "small" and n <= SMALL_N)
        logn = math.log2(n)
        print(f"\n{name}: n={n:,} total={total:,} max item={max(w):,} zero items={w.count(0):,}")

        dual: dict[int, int] = {}
        dp: dict[int, int] = {}
        dual_secs = dp_secs = 0.0
        K_dp = K_full = 0
        full_agrees = None
        if checker:
            dual, dual_secs = run_checker(checker, wfile, "dual", ",".join(map(str, ks)))
            K_dp = max([k for k in ks if k * n * logn <= DC_BUDGET], default=0)
            if K_dp:
                dp, dp_secs = run_checker(checker, wfile, "dp", str(K_dp))
            if n <= FULL_DP_N:
                K_full = max([k for k in ks if k * n * n <= FULL_BUDGET], default=0)
                full, _ = run_checker(checker, wfile, "full", str(K_full))
                full_agrees = all(full[k] == dp[k] for k in full)
                print(f"  plain DP agrees with the textbook DP for k <= {K_full}: {full_agrees}")

        rows, problems = [], []
        for k in ks:
            for objective in ("min_max", "sum_squares"):
                t0 = time.perf_counter()
                r = evencut.partition(w, k, objective=objective)
                ms = (time.perf_counter() - t0) * 1000
                err = check_split(P, k, r)
                dp_same = None
                if objective == "min_max":
                    optimal = None if err else min_max_is_optimal(P, k, r.max_load)
                    proof = "greedy certificate"
                else:
                    optimal = None if err or k not in dual else r.sum_of_squares == dual[k]
                    proof = "duality certificate" if k in dual else "validity only"
                    if k in dp:
                        dp_same = r.sum_of_squares == dp[k]
                pure_same = pure_ms = None
                if run_pure:
                    t0 = time.perf_counter()
                    pure_ends = _pure.partition(w, k, objective, [], None, None, True)
                    pure_ms = (time.perf_counter() - t0) * 1000
                    pure_same = tuple(pure_ends) == r.ends
                row = {"k": k, "objective": objective, "valid": err is None, "error": err,
                       "optimal": optimal, "proof": proof, "matches_dp": dp_same,
                       "pure_backend_same": pure_same, "max_load": r.max_load,
                       "sum_of_squares": r.sum_of_squares, "imbalance": round(r.imbalance, 6),
                       "evencut_ms": round(ms, 2), "pure_ms": None if pure_ms is None else round(pure_ms, 1)}
                rows.append(row)
                if err or optimal is not True or dp_same is False or pure_same is False:
                    problems.append(row)
                    print(f"  PROBLEM k={k} {objective}: {row}")
        mm = sum(1 for r in rows if r["objective"] == "min_max" and r["optimal"])
        ss = sum(1 for r in rows if r["objective"] == "sum_squares" and r["optimal"])
        dpn = sum(1 for r in rows if r["matches_dp"])
        pure_n = sum(1 for r in rows if r["pure_backend_same"])
        print(f"  k={ks[0]}..{ks[-1]}: min_max proven {mm}/{len(ks)}, sum_squares proven {ss}/{len(ks)} "
              f"({dual_secs:.1f}s), DP agrees {dpn}/{len(dp and [k for k in ks if k in dp])} (k <= {K_dp}), "
              f"pure backend same {pure_n}/{2 * len(ks) if run_pure else 0}, problems: {len(problems)}")
        report["datasets"].append({
            "name": name, "source": url, "sha256": digest, "field": field,
            "items": "matrix rows (cost = non-zeros)" if kind == "mtx" else "node IDs (cost = out-degree)",
            "n": n, "total": total, "max_item": max(w), "zero_items": w.count(0), "k_values": ks,
            "dual_seconds": round(dual_secs, 2), "dp_up_to_k": K_dp, "dp_seconds": round(dp_secs, 2),
            "plain_dp_up_to_k": K_full, "plain_dp_agrees": full_agrees, "pure_backend_checked": run_pure,
            "problems": len(problems), "results": rows,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "correctness_real.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, args.out_dir / "correctness_real.md")
    total_checks = sum(len(d["results"]) for d in report["datasets"])
    total_problems = sum(d["problems"] for d in report["datasets"])
    print(f"\n{total_checks} results checked, {total_problems} problems. "
          f"Written to {args.out_dir / 'correctness_real.md'}")
    sys.exit(1 if total_problems else 0)


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Test 1: optimality on real data",
        "",
        f"evencut {report['evencut']} ({report['backend']} backend), {report['date']}, {report['machine']}.",
        "Produced by `bench/correctness_real.py`. Matrices: one item per row, cost = non-zeros. "
        "Networks: one item per node ID, cost = out-degree.",
        "",
        "| dataset | field | items | k | min_max proven optimal | sum_squares proven optimal | "
        "sum_squares = textbook DP | pure-Python backend: same split | pure-Python: slowest call |",
        "|---|---|--:|---|--:|--:|--:|--:|--:|",
    ]
    for d in report["datasets"]:
        rows = d["results"]
        mm = [r for r in rows if r["objective"] == "min_max"]
        ss = [r for r in rows if r["objective"] == "sum_squares"]
        dp = [r for r in ss if r["matches_dp"] is not None]
        pure = [r for r in rows if r["pure_backend_same"] is not None]
        slowest = max((r["pure_ms"] for r in pure), default=None)
        lines.append(
            f"| {d['name']} | {d['field']} | {d['n']:,} | {d['k_values'][0]}..{d['k_values'][-1]} | "
            f"{sum(1 for r in mm if r['optimal'])}/{len(mm)} | {sum(1 for r in ss if r['optimal'])}/{len(ss)} | "
            f"{sum(1 for r in dp if r['matches_dp'])}/{len(dp)} (k <= {d['dp_up_to_k']}) | "
            + (f"{sum(1 for r in pure if r['pure_backend_same'])}/{len(pure)} | {slowest / 1000:.1f} s |"
               if pure else "not run | |"))
    lines += [
        "",
        "Proofs: min_max by the greedy certificate (bound M - 1 needs more than k parts); sum_squares by a "
        "Lagrangian-duality lower bound computed exactly by `bench/verify/ssq_check.c`, which shares no code "
        "with Evencut. The textbook DP column is a second, independent opinion where it is affordable.",
        "",
        "Sources: NIST Matrix Market (math.nist.gov/MatrixMarket) and the Stanford Large Network "
        "Dataset Collection (snap.stanford.edu/data). Checksums are in the JSON file.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
