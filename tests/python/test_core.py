"""Property-based tests for evencut (run once per backend; set EVENCUT_FORCE_PURE=1 for the
pure-Python one). Optima are checked against brute force written independently of the library."""

from __future__ import annotations

import array
import itertools
import random
import re

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import evencut
from evencut import EvencutError, _pure

U64 = (1 << 64) - 1
PROPS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------------------------
def brute(weights, parts, forbidden=(), secondary=None, limit=None):
    """Best (max_load, sum_of_squares) over every split into exactly `parts` parts."""
    n = len(weights)
    banned = set(forbidden)
    allowed = [pos for pos in range(1, n) if pos not in banned]
    best_max = best_sq = None
    for cuts in itertools.combinations(allowed, parts - 1):
        prev, loads, fits = 0, [], True
        for end in list(cuts) + [n]:
            loads.append(sum(weights[prev:end]))
            if secondary is not None and sum(secondary[prev:end]) > limit:
                fits = False
            prev = end
        sq = sum(x * x for x in loads)
        best_sq = sq if best_sq is None else min(best_sq, sq)
        if fits:
            best_max = max(loads) if best_max is None else min(best_max, max(loads))
    return best_max, best_sq


def check_invariants(weights, r, parts, forbidden=(), secondary=None, limit=None):
    n = len(weights)
    assert r.parts == parts == len(r.loads)
    assert r.ends[-1] == n
    assert all(a < b for a, b in zip((0,) + r.ends, r.ends))
    assert all(e not in set(forbidden) for e in r.ends[:-1]), "a forbidden cut was used"
    segs = r.segments()
    assert [sum(weights[a:b]) for a, b in segs] == list(r.loads)
    assert r.total == sum(weights) and r.max_load == max(r.loads)
    assert r.sum_of_squares == sum(x * x for x in r.loads)
    assert r.assignment() == [p for p, (a, b) in enumerate(segs) for _ in range(a, b)]
    if secondary is not None:
        assert all(sum(secondary[a:b]) <= limit for a, b in segs)


# ---------------------------------------------------------------------------------------------
# strategies: valid inputs are generated directly (no assume())
# ---------------------------------------------------------------------------------------------
@st.composite
def problems(draw, max_n=9, with_secondary=False):
    n = draw(st.integers(1, max_n))
    kind = draw(st.sampled_from(["small", "zeros", "equal", "big", "huge"]))
    if kind == "small":
        w = draw(st.lists(st.integers(0, 9), min_size=n, max_size=n))
    elif kind == "zeros":
        w = draw(st.lists(st.sampled_from([0, 0, 0, 1, 2]), min_size=n, max_size=n))
    elif kind == "equal":
        w = [draw(st.integers(0, 3))] * n
    elif kind == "big":   # totals around the 2^40 fast-path limit
        w = draw(st.lists(st.integers(0, (1 << 41) // n), min_size=n, max_size=n))
    else:                 # totals up to 2^64 - 1 (wide arithmetic)
        w = draw(st.lists(st.integers(0, U64 // n), min_size=n, max_size=n))
    forbidden = draw(st.lists(st.integers(1, n - 1), max_size=n // 2, unique=True)) if n >= 2 else []
    k = draw(st.integers(1, n + 2))
    sec = limit = None
    if with_secondary and draw(st.booleans()):
        sec = draw(st.lists(st.integers(0, 9), min_size=n, max_size=n))
        limit = draw(st.integers(0, sum(sec) + 1))
    return w, k, forbidden, sec, limit


# ---------------------------------------------------------------------------------------------
# optimality (oracle) + invariants
# ---------------------------------------------------------------------------------------------
@PROPS
@given(problems(with_secondary=True))
@example(([0], 1, [], None, None))
@example(([5], 3, [], None, None))
@example(([7, 7, 7, 7], 3, [], None, None))
@example(([0, 0, 0], 2, [], None, None))
@example(([1 << 63, (1 << 63) - 1], 2, [], None, None))  # total exactly 2^64 - 1
@example(([1, 2, 3], 2, [], [1, 5, 1], 4))                 # an item alone exceeds the limit
def test_min_max_is_optimal(prob):
    w, k, forbidden, sec, limit = prob
    parts = min(k, len(w) - len(forbidden))
    best_max, _ = brute(w, parts, forbidden, sec, limit)
    if best_max is None:
        with pytest.raises(EvencutError):
            evencut.partition(w, k, forbidden_cuts=forbidden, secondary=sec, secondary_limit=limit)
        return
    for polish in (True, False):
        r = evencut.partition(w, k, forbidden_cuts=forbidden, secondary=sec, secondary_limit=limit, polish=polish)
        check_invariants(w, r, parts, forbidden, sec, limit)
        assert r.max_load == best_max


@PROPS
@given(problems())
@example(([7, 7, 7, 7, 7, 7], 4, [], None, None))  # straight piece of the cost curve
@example(([0, 0, 0, 0, 5], 3, [], None, None))
@example(([U64 // 3] * 3, 2, [], None, None))
def test_sum_squares_is_optimal(prob):
    w, k, forbidden, _, _ = prob
    parts = min(k, len(w) - len(forbidden))
    _, best_sq = brute(w, parts, forbidden)
    r = evencut.partition(w, k, objective="sum_squares", forbidden_cuts=forbidden)
    check_invariants(w, r, parts, forbidden)
    assert r.sum_of_squares == best_sq


@PROPS
@given(problems())
def test_evaluate_roundtrip(prob):
    w, k, forbidden, _, _ = prob
    r = evencut.partition(w, k, forbidden_cuts=forbidden)
    e = evencut.evaluate(w, r.ends)
    assert (e.ends, e.loads, e.max_load, e.sum_of_squares) == (r.ends, r.loads, r.max_load, r.sum_of_squares)


@pytest.mark.skipif(evencut.backend != "native", reason="needs the native library")
def test_native_and_pure_agree_on_larger_inputs():
    from evencut import _native

    rng = random.Random(2026)
    for _ in range(150):
        n = rng.randint(1, 250)
        top = rng.choice([10, 1000, 1 << 36, U64 // n])
        w = [rng.randint(0, top) for _ in range(n)]
        forbidden = sorted(rng.sample(range(1, n), rng.randint(0, (n - 1) // 3))) if n > 1 else []
        k = rng.randint(1, n + 2)
        for objective in ("min_max", "sum_squares"):
            a = _native.partition(evencut._lib, w, k, objective, forbidden, None, U64, True)
            b = _pure.partition(w, k, objective, forbidden, None, None, True)
            ra = evencut.Partition.from_ends(w, a, objective)
            rb = evencut.Partition.from_ends(w, b, objective)
            if objective == "min_max":
                assert ra.max_load == rb.max_load
            else:
                assert ra.sum_of_squares == rb.sum_of_squares
            assert ra.parts == rb.parts


# ---------------------------------------------------------------------------------------------
# trees
# ---------------------------------------------------------------------------------------------
@st.composite
def trees(draw, max_n=8):
    n = draw(st.integers(1, max_n))
    labels = draw(st.permutations(list(range(n))))
    parents = [None] * n
    for i in range(1, n):
        parents[labels[i]] = labels[draw(st.integers(0, i - 1))]
    weights = draw(st.lists(st.integers(0, 20), min_size=n, max_size=n))
    k = draw(st.integers(1, n + 1))
    return parents, weights, k


def tree_brute(parents, weights, parts):
    nonroot = [v for v, p in enumerate(parents) if p is not None]
    best = None
    for cut in itertools.combinations(nonroot, parts - 1):
        cut = set(cut)
        load = {}
        for v in range(len(parents)):
            u = v
            while parents[u] is not None and u not in cut:
                u = parents[u]
            load[u] = load.get(u, 0) + weights[v]
        best = max(load.values()) if best is None else min(best, max(load.values()))
    return best


@PROPS
@given(trees())
def test_tree_is_optimal_and_connected(t):
    parents, weights, k = t
    parts = min(k, len(weights))
    r = evencut.partition_tree(parents, weights, k)
    assert r.parts == parts
    assert r.max_load == tree_brute(parents, weights, parts)
    assert len(r.cut_nodes) == parts - 1
    for v, p in enumerate(parents):
        if p is not None:
            assert (r.part_of[v] != r.part_of[p]) == (v in r.cut_nodes)
    assert [sum(w for w, p in zip(weights, r.part_of) if p == q) for q in range(parts)] == list(r.loads)


def test_tree_accepts_minus_one_root_and_rejects_bad_trees():
    assert evencut.partition_tree([-1, 0, 0, 1], [4, 3, 5, 2], 2).max_load == 9
    with pytest.raises(EvencutError):
        evencut.partition_tree([None, None], [1, 1], 1)       # two roots
    with pytest.raises(EvencutError):
        evencut.partition_tree([None, 2, 1], [1, 1, 1], 2)    # cycle
    with pytest.raises(EvencutError):
        evencut.partition_tree([None, 5], [1, 1], 2)          # out of range


# ---------------------------------------------------------------------------------------------
# input validation and helpers
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("call", [
    lambda: evencut.partition([], 1),
    lambda: evencut.partition([1, 2], 0),
    lambda: evencut.partition([1, -2], 1),
    lambda: evencut.partition([1.5, 2], 1),
    lambda: evencut.partition([True, 2], 1),
    lambda: evencut.partition("12", 1),
    lambda: evencut.partition([U64, 1], 1),
    lambda: evencut.partition([1 << 64], 1),
    lambda: evencut.partition([1, 2, 3], 2, forbidden_cuts=[0]),   # positions are 1 .. n-1
    lambda: evencut.partition([1, 2, 3], 2, forbidden_cuts=[3]),
    lambda: evencut.partition([1, 2, 3], 2, secondary_limit=5),     # limit without values
    lambda: evencut.partition([1, 2, 3], 2, objective="median"),
    lambda: evencut.partition([1, 2, 3], 2, secondary=[1, 1, 1]),
    lambda: evencut.partition([1, 2, 3], 2, secondary=[1, 1], secondary_limit=5),
    lambda: evencut.partition([1, 2, 3], 2, objective="sum_squares", secondary=[1, 1, 1], secondary_limit=5),
    lambda: evencut.evaluate([1, 2, 3], [1, 1, 3]),
    lambda: evencut.evaluate([1, 2, 3], [1, 2]),
    lambda: evencut.quantize([1.0, float("nan")], 10),
    lambda: evencut.quantize([-1.0], 10),
    lambda: evencut.quantize([1.0], 0),
])
def test_invalid_input_raises(call):
    with pytest.raises(EvencutError):
        call()


def test_quantize_rounds_half_away_from_zero():
    assert evencut.quantize([0.25, 1.5, 2.5, 0.0], 100) == [25, 150, 250, 0]
    assert evencut.quantize([0.5, 1.5, 2.5], 1) == [1, 2, 3]      # not banker's rounding
    assert evencut.quantize([0.49999999999999994], 1) == [0]


def test_forbidden_cuts_use_the_numbering_of_ends():
    r = evencut.partition([1, 2, 3], 2, forbidden_cuts=[2])
    assert r.ends == (1, 3)  # 2 is forbidden, so it never appears in ends


def test_more_parts_than_items_and_k_one():
    w = [4, 0, 9]
    assert evencut.partition(w, 10).ends == (1, 2, 3)
    assert evencut.partition(w, 1).ends == (3,)
    assert evencut.partition(w, 1).to_dict()["sum_of_squares"] == "169"


def test_backend_is_reported():
    assert evencut.backend in ("native", "pure")


def test_code_review_regressions():
    w = list(range(1, 11))
    # k beyond 64 bits is clamped to the possible number of parts, never truncated by ctypes
    assert evencut.partition(w, 2**64 + 3).parts == len(w)
    assert evencut.partition_tree([None, 0, 0], [1, 2, 3], 2**64 + 1).parts == 3
    # 2**64 - 1 is a valid explicit secondary limit on every backend
    assert evencut.partition(w, 2, secondary=[1] * 10, secondary_limit=U64).parts == 2
    # cut positions beyond 64 bits and overflowing quantize products raise EvencutError
    with pytest.raises(EvencutError):
        evencut.partition(w, 2, forbidden_cuts=[2**64])
    with pytest.raises(EvencutError):
        evencut.quantize([1e300], 1e10)


def test_fast_input_path_keeps_the_same_rules():
    # plain ints take a C-speed check; everything else still goes through the item-by-item rules
    n = 50_000
    base = [7] * n
    for bad, needle in ((-1, f"weights[{n - 1}] = -1"), (1 << 64, f"weights[{n - 1}] = {1 << 64}"),
                        (True, f"weights[{n - 1}] = True"), (2.0, f"weights[{n - 1}] = 2.0")):
        with pytest.raises(EvencutError, match=re.escape(needle)):
            evencut.partition(base[:-1] + [bad], 4)
    with pytest.raises(EvencutError, match=re.escape(f"secondary[{n - 1}]")):
        evencut.partition(base, 4, secondary=base[:-1] + [-5], secondary_limit=10**9)

    class Count(int):  # an Integral that is not exactly int is still accepted and converted
        pass

    assert evencut.partition([Count(3), 4, Count(5)], 2).ends == evencut.partition([3, 4, 5], 2).ends
    assert evencut.partition(array.array("Q", [3, 4, 5]), 2).ends == (2, 3)
    assert evencut.partition([U64, 0], 2).max_load == U64
