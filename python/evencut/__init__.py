"""Evencut: exact, balanced partitioning of ordered workloads.

    >>> import evencut
    >>> r = evencut.partition([5, 3, 8, 2, 7, 4, 6, 1, 9, 5], 3)
    >>> r.ends, r.loads, r.max_load
    ((4, 7, 10), (18, 17, 15), 18)

Objectives (both give exact optima):

* ``"min_max"``      the largest part is as small as possible (bottleneck / makespan)
* ``"sum_squares"``  the sum of squared part loads is as small as possible (overall evenness)

The native C++ library is used when it is installed next to this package; otherwise a
pure-Python implementation with identical results takes over. ``evencut.backend`` tells which
one is active; set ``EVENCUT_FORCE_PURE=1`` to force the Python one.

See :mod:`evencut.ai` for helpers aimed at AI training (pipeline stages, token-balanced
shards, token-budget batches, causal-attention chunks).
"""

from __future__ import annotations

import array as _array
import numbers
import os
from typing import Any, Iterable, Optional, Sequence

from . import _pure
from ._types import U64_MAX, EvencutError, Partition, TreePartition

__version__ = "1.0.0"
__all__ = [
    "EvencutError",
    "OBJECTIVES",
    "Partition",
    "TreePartition",
    "backend",
    "evaluate",
    "partition",
    "partition_tree",
    "quantize",
]

OBJECTIVES = ("min_max", "sum_squares")

_lib: Any = None
if os.environ.get("EVENCUT_FORCE_PURE", "").lower() not in ("1", "true", "yes"):
    try:
        from . import _native

        _lib = _native.load()
    except OSError:  # present but not loadable (wrong architecture, missing runtime)
        _lib = None

backend: str = "native" if _lib is not None else "pure"
"""``"native"`` (C++ shared library) or ``"pure"`` (Python fallback)."""


def _as_list(values: Iterable[Any], name: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise EvencutError(f"evencut: {name} must be a sequence of integers, not a string")
    if hasattr(values, "tolist"):  # numpy arrays and similar
        values = values.tolist()
    try:
        return list(values)
    except TypeError:
        raise EvencutError(f"evencut: {name} must be a sequence of integers") from None


def _fits_u64_fast(values: list[Any]) -> bool:
    """True when every entry is a plain int in [0, 2**64 - 1], checked at C speed.

    A per-item Python loop costs ~0.4 µs per item, which dominated calls on millions of items.
    Anything unusual (bool, float, other Integral types, out-of-range values) returns False, and
    the caller's item-by-item loop then converts it or reports the exact offending entry.
    """
    if set(map(type, values)) - {int}:
        return False
    try:
        _array.array("Q", values)  # raises OverflowError for negative or too large values
    except OverflowError:
        return False
    return True


def _u64_list(values: Iterable[Any], name: str) -> list[int]:
    out = _as_list(values, name)
    if _fits_u64_fast(out):
        return out
    for i, v in enumerate(out):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise EvencutError(
                f"evencut: {name}[{i}] = {v!r} is not an integer; use evencut.quantize() for real numbers")
        v = int(v)
        if v < 0 or v > U64_MAX:
            raise EvencutError(f"evencut: {name}[{i}] = {v} is outside [0, 2**64 - 1]")
        out[i] = v
    return out


def _index_list(values: Iterable[Any], name: str) -> list[int]:
    out = _as_list(values, name)
    if _fits_u64_fast(out):
        return out
    for i, v in enumerate(out):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or int(v) < 0:
            raise EvencutError(f"evencut: {name}[{i}] = {v!r} must be a non-negative integer")
        out[i] = int(v)
    return out


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
        raise EvencutError(f"evencut: {name} must be an integer >= 1, got {value!r}")
    return int(value)


def partition(
    weights: Sequence[int],
    k: int,
    *,
    objective: str = "min_max",
    forbidden_cuts: Iterable[int] = (),
    secondary: Optional[Sequence[int]] = None,
    secondary_limit: Optional[int] = None,
    polish: bool = True,
) -> Partition:
    """Splits ``weights`` into ``k`` contiguous parts as evenly as possible.

    Args:
        weights: non-negative integers (use :func:`quantize` for real numbers).
        k: number of parts. If fewer cuts are possible (``k > len(weights)``, or cuts are
            forbidden), every allowed block becomes its own part.
        objective: ``"min_max"`` or ``"sum_squares"``.
        forbidden_cuts: cut positions that are not allowed, numbered like ``ends``: ``b``
            keeps items ``b-1`` and ``b`` together, so ``b`` never appears in the result's
            ``ends``. Valid values are ``1 .. len(weights) - 1``.
        secondary: a second value per item (for example memory); requires ``secondary_limit``
            and ``objective="min_max"``. Every part then satisfies sum(secondary) <= limit.
        polish: Whether to even out the non-maximal parts (``min_max`` only). The maximum
            is optimal either way.

    Returns:
        A :class:`Partition` with ``min(k, allowed blocks)`` parts, the cut positions and the
        exact objective values.

    Raises:
        EvencutError: Invalid input, or no split satisfies ``secondary_limit``.
    """
    w = _u64_list(weights, "weights")
    if not w:
        raise EvencutError("evencut: weights must not be empty")
    k = _positive_int(k, "k")
    if objective not in OBJECTIVES:
        raise EvencutError(f"evencut: objective must be one of {OBJECTIVES}, got {objective!r}")
    forbidden = _index_list(forbidden_cuts, "forbidden_cuts")
    for b in forbidden:  # checked here so every backend fails the same way, even beyond 64 bits
        if not 1 <= b <= len(w) - 1:
            raise EvencutError("evencut: forbidden_cuts entries must be in [1, n-1] (numbered like ends)")
    sec: Optional[list[int]] = None
    limit: Optional[int] = None
    if secondary is not None or secondary_limit is not None:
        if objective != "min_max":
            raise EvencutError("evencut: secondary limits are only supported with objective='min_max'")
        if secondary is None or secondary_limit is None:
            raise EvencutError("evencut: secondary and secondary_limit must be given together")
        sec = _u64_list(secondary, "secondary")
        if len(sec) != len(w):
            raise EvencutError("evencut: secondary must have exactly one value per weight")
        limit = _u64_list([secondary_limit], "secondary_limit")[0]
    # never more than len(w) parts are possible; clamping also keeps k inside size_t, where
    # ctypes would otherwise silently keep only the low 64 bits
    k = min(k, len(w))
    if _lib is not None:
        ends = _native.partition(_lib, w, k, objective, forbidden, sec, 0 if limit is None else limit, polish)
    else:
        ends = _pure.partition(w, k, objective, forbidden, sec, limit, polish)
    return Partition.from_ends(w, ends, objective)


def evaluate(weights: Sequence[int], ends: Sequence[int]) -> Partition:
    """Measures an existing split, for example a framework's default, against the optimum.

    Args:
        weights: Non-negative integers.
        ends: Exclusive end index of each part, strictly increasing, the last one equal to
            ``len(weights)``.

    Returns:
        A :class:`Partition` describing the given split.

    Raises:
        EvencutError: Invalid weights or ends.
    """
    w = _u64_list(weights, "weights")
    if not w:
        raise EvencutError("evencut: weights must not be empty")
    e = _index_list(ends, "ends")
    if not e or e[-1] != len(w):
        raise EvencutError("evencut: ends must be non-empty and its last entry must equal len(weights)")
    prev = 0
    for x in e:
        if x <= prev:
            raise EvencutError("evencut: ends must be strictly increasing and start above 0")
        prev = x
    if sum(w) > U64_MAX:
        raise EvencutError("evencut: total weight exceeds 2^64-1; scale the weights down")
    return Partition.from_ends(w, e, "evaluate")


def partition_tree(parents: Sequence[Optional[int]], weights: Sequence[int], k: int) -> TreePartition:
    """Splits a tree into ``k`` connected parts, minimising the largest part.

    Args:
        parents: ``parents[v]`` is the parent of node ``v``; the root has ``None`` (or ``-1``).
        weights: non-negative integer weight of each node.
        k: number of parts (at most ``len(weights)`` are possible).

    Returns:
        A :class:`TreePartition`; every part is connected and the largest one is as small as
        possible.

    Raises:
        EvencutError: Invalid input, or ``parents`` does not describe a single tree.
    """
    raw = _as_list(parents, "parents")
    par: list[Optional[int]] = []
    for i, p in enumerate(raw):
        if p is None or (isinstance(p, numbers.Integral) and not isinstance(p, bool) and int(p) == -1):
            par.append(None)
        elif isinstance(p, numbers.Integral) and not isinstance(p, bool) and int(p) >= 0:
            par.append(int(p))
        else:
            raise EvencutError(f"evencut: parents[{i}] = {p!r} must be a node index, None or -1")
    w = _u64_list(weights, "weights")
    if not par:
        raise EvencutError("evencut: the tree must have at least one node")
    if len(w) != len(par):
        raise EvencutError("evencut: weights must have exactly one value per node")
    k = _positive_int(k, "k")
    if _lib is not None:
        native_parents = [_native.NO_PARENT if p is None else p for p in par]
        if any(p is not None and p >= len(par) for p in par):
            raise EvencutError("evencut: parent indices must be in [0, n) and differ from the node")
        part_of = _native.partition_tree(_lib, native_parents, w, min(k, len(w)))
        cut = [v for v, p in enumerate(par) if p is not None and part_of[v] != part_of[p]]
    else:
        part_of, cut = _pure.partition_tree(par, w, k)
    return TreePartition.from_assignment(w, part_of, cut)


def quantize(values: Iterable[float], scale: float) -> list[int]:
    """Turns non-negative real numbers into integer weights: ``round(value * scale)``.

    Pick ``scale`` for the precision you need, e.g. ``1e6`` turns seconds into microseconds.
    Rounds half away from zero, exactly like the C++ library.

    Raises:
        EvencutError: A value is negative, NaN or infinite, ``scale`` is not positive, or a
            result exceeds 2**64 - 1.
    """
    return _pure.quantize(_as_list(values, "values"), scale)
