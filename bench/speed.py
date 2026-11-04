"""Test 3: how fast is Evencut, and how does it compare with other ways to get the same answer?

Datasets: the ten real ones from bench_data.py (SNAP IDs compacted) plus two synthetic inputs of
10 million items. Measured (median of several runs unless noted):

* Evencut, C++ only:        bench/speed_core.cpp, no Python involved;
* Evencut from Python:      evencut.partition(list_of_ints), everything included (input check,
                            conversion, result object);
* pure-Python backend:      the fallback used when the C++ library cannot be loaded (one run);
* exact alternatives, same optimum guaranteed:
    - min_max: binary search on the bound + greedy check, written in plain Python with bisect
      (the usual hand-written exact method);
    - sum_squares: the textbook dynamic program with divide-and-conquer optimisation, compiled C
      (bench/verify/ssq_check.c "dp"), and the plain O(k n^2) DP on the small datasets.

Run:   python bench/speed.py [--data-dir DIR] [--out-dir DIR]
Needs: a C and a C++ compiler (cc/gcc/clang and c++/g++/clang++, or set CC / CXX).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path

import evencut
from evencut import _pure

import bench_data
from bench_data import DATASETS, load

HERE = Path(__file__).resolve().parent
KS = (64, 4096)
SYNTHETIC_N = 10_000_000
PLAIN_DP_N = 12_000        # plain O(k n^2) DP only up to this size
DP_BUDGET = 6e9            # k * n * log2(n) budget for the textbook DP


def find(env: str, names: tuple[str, ...]) -> str | None:
    for c in (os.environ.get(env), *names):
        if c and shutil.which(c):
            return c
    return None


def build(data_dir: Path) -> tuple[Path, Path]:
    cc, cxx = find("CC", ("cc", "gcc", "clang")), find("CXX", ("c++", "g++", "clang++"))
    if cc is None or cxx is None:
        raise SystemExit("needs a C and a C++ compiler (set CC / CXX)")
    exe = ".exe" if os.name == "nt" else ""
    bindir = data_dir / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    core, check = bindir / f"speed_core{exe}", bindir / f"ssq_check{exe}"
    core_src, check_src = HERE / "speed_core.cpp", HERE / "verify" / "ssq_check.c"
    header = HERE.parent / "include" / "evencut" / "evencut.hpp"
    if not core.exists() or core.stat().st_mtime < max(core_src.stat().st_mtime, header.stat().st_mtime):
        subprocess.run([cxx, "-std=c++20", "-O2", f"-I{HERE.parent / 'include'}", "-o", str(core), str(core_src)],
                       check=True)
    if not check.exists() or check.stat().st_mtime < check_src.stat().st_mtime:
        subprocess.run([cc, "-O2", "-o", str(check), str(check_src)], check=True)
    return core, check


def synthetic(data_dir: Path) -> list[tuple[str, str, Path]]:
    out = []
    for name, field in (("uniform-10M", "synthetic: uniform 1..1000"),
                        ("heavy-tail-10M", "synthetic: log-normal, sigma 1.5")):
        path = data_dir / "weights" / f"{name}.u64"
        if not path.exists():
            rng = random.Random(20261101)
            if name.startswith("uniform"):
                w = [rng.randint(1, 1000) for _ in range(SYNTHETIC_N)]
            else:
                w = [int(rng.lognormvariate(0.0, 1.5) * 1000) + 1 for _ in range(SYNTHETIC_N)]
            path.parent.mkdir(parents=True, exist_ok=True)
            bench_data._write_u64(path, w)
        out.append((name, field, path))
    return out


def timed(fn, reps: int) -> tuple[float, object]:
    times, result = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000, result


def min_max_by_bisection(w: list[int], k: int) -> int:
    """Exact min_max the usual hand-written way: binary search on the bound, greedy check."""
    P = list(accumulate(w, initial=0))
    n, S = len(w), P[-1]
    lo, hi = max(max(w), -(-S // k)), S

    def fits(bound: int) -> bool:
        i = parts = 0
        while i < n:
            i = bisect_right(P, P[i] + bound) - 1
            parts += 1
            if parts > k:
                return False
        return True

    while lo < hi:
        mid = (lo + hi) // 2
        if fits(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def run_core(core: Path, path: Path, reps: int) -> dict[tuple[str, int], float]:
    out = subprocess.run([str(core), str(path), str(reps), *map(str, KS)], check=True, capture_output=True,
                         text=True).stdout
    res = {}
    for line in out.splitlines():
        obj, k, ms, _ = line.split()
        res[(obj, int(k))] = float(ms)
    return res


def run_dp(check: Path, path: Path, mode: str, k: int) -> tuple[float, int]:
    t0 = time.perf_counter()
    out = subprocess.run([str(check), str(path), mode, str(k)], check=True, capture_output=True, text=True).stdout
    ms = (time.perf_counter() - t0) * 1000
    last = out.splitlines()[-1].split()
    return ms, int(last[1])


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
    args = ap.parse_args()
    core, check = build(args.data_dir)
    print(f"evencut {evencut.__version__}, backend {evencut.backend}; {machine()}")

    inputs = []
    for name, url, digest, kind, field in DATASETS:
        _, path = load(name, url, digest, kind, args.data_dir, compact=True)
        inputs.append((name, field, path))
    inputs += synthetic(args.data_dir)

    report = {"evencut": evencut.__version__, "machine": machine(), "date": dt.date.today().isoformat(),
              "ks": list(KS), "datasets": []}
    for name, field, path in inputs:
        w = bench_data._read_u64(path)
        n = len(w)
        big = n >= SYNTHETIC_N
        print(f"\n{name}: n={n:,}")
        row: dict = {"name": name, "field": field, "n": n}
        c = run_core(core, path, 3 if big else 5)
        row["cpp_ms"] = {f"{o} k={k}": c[(o, k)] for (o, k) in c}
        py = {}
        for k in KS:
            for o in ("min_max", "sum_squares"):
                py[f"{o} k={k}"], _ = timed(lambda w=w, k=k, o=o: evencut.partition(w, k, objective=o), 1 if big else 3)
        row["python_ms"] = py
        if not big:
            row["pure_ms"] = {f"{o} k={KS[0]}": timed(lambda w=w, o=o: _pure.partition(w, KS[0], o, [], None, None, True), 1)[0]
                              for o in ("min_max", "sum_squares")}
        alt = {}
        for k in KS:
            ms, best = timed(lambda w=w, k=k: min_max_by_bisection(w, k), 1 if big else 3)
            assert best == evencut.partition(w, k).max_load, "binary search disagrees with Evencut"
            alt[f"min_max bisection k={k}"] = ms
        if KS[0] * n * max(1, n.bit_length()) <= DP_BUDGET:
            ms, value = run_dp(check, path, "dp", KS[0])
            assert value == evencut.partition(w, KS[0], objective="sum_squares").sum_of_squares
            alt[f"sum_squares textbook DP k={KS[0]}"] = ms
        if n <= PLAIN_DP_N:
            ms, value = run_dp(check, path, "full", KS[0])
            assert value == evencut.partition(w, KS[0], objective="sum_squares").sum_of_squares
            alt[f"sum_squares plain DP k={KS[0]}"] = ms
        row["alternatives_ms"] = alt
        report["datasets"].append(row)
        print("  C++ " + ", ".join(f"{key} {v:.1f} ms" for key, v in row["cpp_ms"].items()))
        print("  Python " + ", ".join(f"{key} {v:.1f} ms" for key, v in py.items()))
        if "pure_ms" in row:
            print("  pure " + ", ".join(f"{key} {v:.0f} ms" for key, v in row["pure_ms"].items()))
        print("  alternatives " + ", ".join(f"{key} {v:.0f} ms" for key, v in alt.items()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "speed.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, args.out_dir / "speed.md")
    print(f"\nWritten to {args.out_dir / 'speed.md'}")


def fmt(ms: float | None) -> str:
    if ms is None:
        return "-"
    if ms < 1:
        return f"{ms:.2f} ms"
    if ms < 1000:
        return f"{ms:.0f} ms"
    if ms < 120_000:
        return f"{ms / 1000:.1f} s"
    return f"{ms / 60000:.0f} min"


def write_markdown(report: dict, path: Path) -> None:
    k1, k2 = report["ks"]
    lines = [
        "# Test 3: speed",
        "",
        f"evencut {report['evencut']}, {report['date']}, {report['machine']}. Produced by `bench/speed.py`. "
        "Median of 5 runs (3 for 10M items; Python and alternatives: 3, or 1 for 10M; pure Python: 1).",
        "",
        "## Evencut",
        "",
        f"| dataset | items | C++ min_max k={k1} | k={k2} | C++ sum_squares k={k1} | k={k2} | "
        f"from Python min_max k={k1} | sum_squares k={k1} | pure Python min_max k={k1} | sum_squares k={k1} |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for d in report["datasets"]:
        c, p, q = d["cpp_ms"], d["python_ms"], d.get("pure_ms", {})
        big_k = (lambda o, d=d, c=c: "k > items" if k2 > d["n"] else fmt(c[f"{o} k={k2}"]))
        lines.append(
            f"| {d['name']} | {d['n']:,} | {fmt(c[f'min_max k={k1}'])} | {big_k('min_max')} | "
            f"{fmt(c[f'sum_squares k={k1}'])} | {big_k('sum_squares')} | {fmt(p[f'min_max k={k1}'])} | "
            f"{fmt(p[f'sum_squares k={k1}'])} | {fmt(q.get(f'min_max k={k1}'))} | {fmt(q.get(f'sum_squares k={k1}'))} |")
    lines += [
        "",
        "## Other ways to get the same exact answer",
        "",
        f"| dataset | items | min_max k={k1}: Evencut from Python | binary search + greedy (Python) | "
        f"min_max k={k2}: Evencut from Python | binary search + greedy (Python) | "
        f"sum_squares k={k1}: Evencut C++ | textbook DP (C) | plain DP (C) |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for d in report["datasets"]:
        c, p, a = d["cpp_ms"], d["python_ms"], d["alternatives_ms"]
        lines.append(
            f"| {d['name']} | {d['n']:,} | {fmt(p[f'min_max k={k1}'])} | {fmt(a[f'min_max bisection k={k1}'])} | "
            f"{fmt(p[f'min_max k={k2}'])} | {fmt(a[f'min_max bisection k={k2}'])} | "
            f"{fmt(c[f'sum_squares k={k1}'])} | {fmt(a.get(f'sum_squares textbook DP k={k1}'))} | "
            f"{fmt(a.get(f'sum_squares plain DP k={k1}'))} |")
    lines += [
        "",
        "All alternatives were checked to return the same optimum as Evencut. \"-\" = not run (too slow for "
        "that size). The textbook DP computes every k up to the requested one, which is how it works.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
