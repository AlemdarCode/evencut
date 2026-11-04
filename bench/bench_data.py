"""Real datasets shared by the benchmark scripts: download, verify, parse and cache.

* Sparse matrices (NIST Matrix Market): one item per row, cost = non-zeros in that row (the
  work of that row in a sparse matrix-vector product).
* Networks (Stanford SNAP): one item per node, cost = out-degree (the work of that node in one
  PageRank-style sweep). SNAP node IDs can have gaps (cit-Patents uses patent numbers, which
  start near 3 million). ``compact=False`` keeps one item per ID, gaps included as zero-cost
  items; ``compact=True`` keeps only the nodes that occur in the file, in ID order.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import sys
import urllib.request
from array import array
from pathlib import Path

import evencut

NIST = "https://math.nist.gov/pub/MatrixMarket2/"
SNAP = "https://snap.stanford.edu/data/"

# name, url, sha256, kind, field
DATASETS = [
    ("psmigr_1", NIST + "Harwell-Boeing/psmigr/psmigr_1.mtx.gz",
     "5f38479a7bf87d9150f076558e09dcab5f1cbfe2eaff013bfcfddc83a360a59b", "mtx", "economics (inter-county migration)"),
    ("bcsstk17", NIST + "Harwell-Boeing/bcsstruc2/bcsstk17.mtx.gz",
     "105425dc922f5aaa4f2a7b52d707e946cbc3cc309accea11f5365a30528f49a5", "mtx", "structural engineering"),
    ("memplus", NIST + "misc/hamm/memplus.mtx.gz",
     "35a850faf9eb84b8b84653a96ef587868059398ae61b9cfc2ad2a80c55926115", "mtx", "circuit simulation (memory chip)"),
    ("af23560", NIST + "NEP/airfoil/af23560.mtx.gz",
     "e47251e445aabfe7086b598068c070a55a02b83dcb90645f1814d11f6c84819f", "mtx", "fluid dynamics (airfoil)"),
    ("bcsstk30", NIST + "Harwell-Boeing/bcsstruc5/bcsstk30.mtx.gz",
     "02c69627a79ff7375200497dc90cc55ec28e74bccc75c5d4426c6dc174d75963", "mtx", "structural engineering"),
    ("amazon0601", SNAP + "amazon0601.txt.gz",
     "aa6dea3bac74bf8e396f23a42a44eb1a913f9837cb388a688a70b2cae8944a7b", "snap", "co-purchase network"),
    ("web-Google", SNAP + "web-Google.txt.gz",
     "bcac0af0471d749f4a8c010bca92b61cf2868a0570741de06892fc062f265ea6", "snap", "web graph"),
    ("roadNet-CA", SNAP + "roadNet-CA.txt.gz",
     "383f7b14424530a9e25c392969a9c19dace32ebcfd141f0967a1433498c2acdd", "snap", "road network"),
    ("wiki-Talk", SNAP + "wiki-Talk.txt.gz",
     "0f3fd7bb8435ad0d46bd3cd06d2fb947c72d298a9310c8ce82b4d29ef973fb9f", "snap", "communication network"),
    ("cit-Patents", SNAP + "cit-Patents.txt.gz",
     "9d05955aa997fbc84953c9eb7786675af80336a356404e75dc18aaa90d3553b0", "snap", "citation network"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected: str) -> None:
    if not dest.exists():
        print(f"  downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": f"evencut-bench/{evencut.__version__}"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
    got = sha256(dest)
    if got != expected:
        raise SystemExit(f"checksum mismatch for {dest.name}: {got} (expected {expected}); delete it and retry")


def parse_mtx(path: Path) -> list[int]:
    """Non-zeros per row of a Matrix Market coordinate file (symmetric storage expanded)."""
    with gzip.open(path, "rt", encoding="ascii") as f:
        header = f.readline().lower().split()
        if len(header) < 5 or header[0] != "%%matrixmarket" or header[2] != "coordinate":
            raise SystemExit(f"{path.name}: not a Matrix Market coordinate file")
        symmetric = header[4] in ("symmetric", "skew-symmetric", "hermitian")
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        rows, _cols, _entries = (int(x) for x in line.split())
        nnz = [0] * rows
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            nnz[i] += 1
            if symmetric and i != j:
                nnz[j] += 1
    return nnz


def parse_snap(path: Path) -> tuple[list[int], bytearray]:
    """Out-degree of every node ID 0..max_id, and which IDs occur in the file at all."""
    deg: list[int] = []
    present = bytearray()
    with gzip.open(path, "rt", encoding="ascii") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = int(parts[0]), int(parts[1])
            top = a if a > b else b
            if top >= len(deg):
                grow = top + 1 - len(deg)
                deg.extend([0] * grow)
                present.extend(b"\0" * grow)
            deg[a] += 1
            present[a] = 1
            present[b] = 1
    return deg, present


def _read_u64(path: Path) -> list[int]:
    a = array("Q")
    a.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        a.byteswap()
    return a.tolist()


def _write_u64(path: Path, values: list[int]) -> None:
    a = array("Q", values)
    if sys.byteorder != "little":
        a.byteswap()
    path.write_bytes(a.tobytes())


def load(name: str, url: str, digest: str, kind: str, data_dir: Path, compact: bool = False
         ) -> tuple[list[int], Path]:
    """Returns the item costs and a cached little-endian uint64 file holding them."""
    raw = data_dir / "raw" / url.rsplit("/", 1)[1]
    raw.parent.mkdir(parents=True, exist_ok=True)
    download(url, raw, digest)
    cache_dir = data_dir / "weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    full = cache_dir / f"{name}-{digest[:12]}.u64"
    packed = cache_dir / f"{name}-{digest[:12]}-compact.u64"
    want = packed if compact and kind == "snap" else full
    if want.exists():
        return _read_u64(want), want
    if kind == "mtx":
        _write_u64(full, parse_mtx(raw))
    else:
        deg, present = parse_snap(raw)
        _write_u64(full, deg)
        _write_u64(packed, [d for d, p in zip(deg, present) if p])
    return _read_u64(want), want
