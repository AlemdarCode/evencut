"""Pure-Python implementation of the Evencut algorithms.

Exact (Python integers never overflow) but slower than the native library. It is used when the
shared library is not available and, in the test-suite, as a second implementation to compare
against. The algorithms mirror include/evencut/evencut.hpp step by step.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
from typing import Optional

from ._types import U64_MAX, EvencutError

# ---------------------------------------------------------------------------------------------
# blocks (forbidden cuts glue neighbouring items together)
# ---------------------------------------------------------------------------------------------


class _Blocks:
    __slots__ = ("prefix", "sec_prefix", "item_end", "max_block", "max_sec_block")

    def __init__(self, weights: list[int], forbidden: list[int], secondary: Optional[list[int]]):
        n = len(weights)
        joined = bytearray(n)  # joined[i]: no cut between item i and i+1
        for b in forbidden:
            if not 1 <= b <= n - 1:
                raise EvencutError("evencut: forbidden_cuts entries must be in [1, n-1] (numbered like ends)")
            joined[b - 1] = 1
        self.prefix = [0]
        self.sec_prefix: Optional[list[int]] = [0] if secondary is not None else None
        self.item_end: list[int] = []
        self.max_block = 0
        self.max_sec_block = 0
        w = s = total = sec_total = 0
        for i in range(n):
            w += weights[i]
            total += weights[i]
            if secondary is not None:
                s += secondary[i]
                sec_total += secondary[i]
            if i + 1 == n or not joined[i]:
                self.prefix.append(total)
                if self.sec_prefix is not None:
                    self.sec_prefix.append(sec_total)
                self.item_end.append(i + 1)
                self.max_block = max(self.max_block, w)
                self.max_sec_block = max(self.max_sec_block, s)
                w = s = 0
        if total > U64_MAX:
            raise EvencutError("evencut: total weight exceeds 2^64-1; scale the weights down")
        if sec_total > U64_MAX:
            raise EvencutError("evencut: total secondary value exceeds 2^64-1; scale it down")

    @property
    def count(self) -> int:
        return len(self.item_end)

    def item_ends(self, bounds: list[int]) -> list[int]:
        return [self.item_end[b - 1] for b in bounds[1:]]


# ---------------------------------------------------------------------------------------------
# min_max
# ---------------------------------------------------------------------------------------------


def _furthest_end(bv: _Blocks, start: int, bound: int, sec_limit: Optional[int]) -> int:
    P = bv.prefix
    e = bisect_right(P, P[start] + bound, start + 1) - 1
    if bv.sec_prefix is not None and sec_limit is not None:
        Q = bv.sec_prefix
        e = min(e, bisect_right(Q, Q[start] + sec_limit, start + 1) - 1)
    return e


def _greedy_parts(bv: _Blocks, bound: int, sec_limit: Optional[int], stop_after: int) -> int:
    start, parts, m = 0, 0, bv.count
    while start < m:
        e = _furthest_end(bv, start, bound, sec_limit)
        if e == start:
            return U64_MAX
        parts += 1
        if parts > stop_after:
            return parts
        start = e
    return parts


def _balanced_split(P: list[int], s: int, e: int) -> int:
    half = P[s] + (P[e] - P[s]) // 2
    c = bisect_left(P, half, s + 1, e)
    c = min(max(c, s + 1), e - 1)
    worst = lambda x: max(P[x] - P[s], P[e] - P[x])  # noqa: E731
    if c > s + 1 and worst(c - 1) <= worst(c):
        c -= 1
    return c


def _expand(bv: _Blocks, bounds: list[int], target: int) -> list[int]:
    P = bv.prefix
    parts = len(bounds) - 1
    if parts >= target:
        return bounds
    heap = [(-(P[b] - P[a]), a, b) for a, b in zip(bounds, bounds[1:])]
    heapq.heapify(heap)
    done = []
    while parts < target and heap:
        neg, s, e = heapq.heappop(heap)
        if e - s < 2:
            done.append((neg, s, e))
            continue
        c = _balanced_split(P, s, e)
        heapq.heappush(heap, (-(P[c] - P[s]), s, c))
        heapq.heappush(heap, (-(P[e] - P[c]), c, e))
        parts += 1
    starts = sorted([s for _, s, _ in heap] + [s for _, s, _ in done])
    return starts + [bv.count]


def _polish(bv: _Blocks, bounds: list[int], bound: int, sec_limit: Optional[int]) -> None:
    P, Q = bv.prefix, bv.sec_prefix
    k = len(bounds) - 1
    if k < 2:
        return
    for _ in range(64):
        moved = False
        for s in range(1, k):
            lo_b, hi_b = bounds[s - 1], bounds[s + 1]
            cmax = min(_furthest_end(bv, lo_b, bound, sec_limit), hi_b - 1)
            cmin = bisect_left(P, max(P[hi_b] - bound, 0))
            if Q is not None and sec_limit is not None:
                cmin = max(cmin, bisect_left(Q, max(Q[hi_b] - sec_limit, 0)))
            cmin = max(cmin, lo_b + 1)
            if cmin > cmax:
                continue
            gap = lambda c: abs((P[c] - P[lo_b]) - (P[hi_b] - P[c]))  # noqa: E731
            half = P[lo_b] + (P[hi_b] - P[lo_b]) // 2
            best = bisect_left(P, half, cmin, cmax + 1)
            best = min(max(best, cmin), cmax)
            if best > cmin and gap(best - 1) <= gap(best):
                best -= 1
            if gap(best) < gap(bounds[s]):
                bounds[s] = best
                moved = True
        if not moved:
            break


def _min_max_blocks(bv: _Blocks, target: int, sec_limit: Optional[int], polish: bool) -> list[int]:
    if bv.sec_prefix is not None and sec_limit is not None and bv.max_sec_block > sec_limit:
        raise EvencutError(
            "evencut: an item (or a group of items joined by forbidden_cuts) exceeds secondary_limit on its own")
    total = bv.prefix[-1]
    lo = max(bv.max_block, -(-total // target))
    hi = total
    lo = min(lo, hi)
    if _greedy_parts(bv, hi, sec_limit, target) > target:
        raise EvencutError("evencut: no partition into the requested number of parts satisfies secondary_limit")
    while lo < hi:
        mid = (lo + hi) // 2
        if _greedy_parts(bv, mid, sec_limit, target) <= target:
            hi = mid
        else:
            lo = mid + 1
    bounds, start = [0], 0
    while start < bv.count:
        start = _furthest_end(bv, start, lo, sec_limit)
        bounds.append(start)
    bounds = _expand(bv, bounds, target)
    if polish:
        _polish(bv, bounds, lo, sec_limit)
    return bounds


# ---------------------------------------------------------------------------------------------
# sum_squares: odd penalties + convex hull trick + chord search + Monge splice
# ---------------------------------------------------------------------------------------------


def _run(P: list[int], mu: int) -> tuple[int, list[int]]:
    """Optimal partition for 2*sum(load^2) + mu*parts; returns (parts, boundaries)."""
    m = len(P) - 1
    slope, icpt, line = [0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    count, frm = [0] * (m + 1), [0] * (m + 1)
    head, tail = 0, 1
    for i in range(1, m + 1):
        x = P[i]
        while tail - head >= 2 and (slope[head + 1] - slope[head]) * x <= icpt[head] - icpt[head + 1]:
            head += 1
        j = line[head]
        two_x2 = 2 * x * x
        dp = slope[head] * x + icpt[head] + two_x2 + mu
        count[i] = count[j] + 1
        frm[i] = j
        if i == m:
            break
        s, c = -4 * x, dp + two_x2
        if tail > head and slope[tail - 1] == s:
            if icpt[tail - 1] <= c:
                continue
            tail -= 1
        while tail - head >= 2 and (c - icpt[tail - 2]) * (slope[tail - 2] - slope[tail - 1]) <= \
                (icpt[tail - 1] - icpt[tail - 2]) * (slope[tail - 2] - s):
            tail -= 1
        slope[tail], icpt[tail], line[tail] = s, c, i
        tail += 1
    bounds, i = [], m
    while True:
        bounds.append(i)
        if i == 0:
            break
        i = frm[i]
    bounds.reverse()
    return count[m], bounds


def _splice(a: list[int], b: list[int], k: int) -> list[int]:
    q = len(b) - 1
    for t in range(1, q + 1):
        L = bisect_left(a, b[t - 1])
        R = bisect_right(a, b[t]) - 1
        if R <= L:
            continue
        if L + 1 + q - t <= k <= R + q - t:
            s = k - 1 - q + t
            return a[: s + 1] + b[t:]
    raise RuntimeError("evencut: internal error while assembling an exact-k partition")


def _sum_squares_blocks(P: list[int], k: int) -> list[int]:
    m = len(P) - 1
    if k >= m:
        return list(range(m + 1))
    if k == 1:
        return [0, m]

    def cost(bounds: list[int]) -> int:
        return sum((P[b] - P[a]) ** 2 for a, b in zip(bounds, bounds[1:]))

    S = P[m]
    lo_b = list(range(m + 1))
    lo = [-1, m, cost(lo_b), lo_b]               # mu, parts, cost, bounds
    hi = [2 * S * S + 1, 1, S * S, [0, m]]
    bisect = False
    while hi[0] - lo[0] > 2:
        mu, chord, probe = None, False, False
        if not bisect:
            dc = lo[1] - hi[1]
            diff = hi[2] - lo[2]
            s = diff // dc
            exact = s * dc == diff
            below, above = 2 * s - 1, 2 * s + 1
            if exact and lo[0] < below < hi[0]:
                mu, chord, probe = below, True, True
            elif lo[0] < above < hi[0]:
                mu, chord, probe = above, True, exact
        if not chord:
            t = ((lo[0] - 1) // 2 + (hi[0] - 1) // 2) // 2
            mu = 2 * t + 1
        c, bounds = _run(P, mu)
        if c == k:
            return bounds
        new_count = c != lo[1] and c != hi[1]
        side = [mu, c, cost(bounds), bounds]
        if c > k:
            lo = side
        else:
            hi = side
        bisect = chord and not probe and not new_count
    return _splice(lo[3], hi[3], k)


# ---------------------------------------------------------------------------------------------
# public entry points (inputs are already validated by evencut.__init__)
# ---------------------------------------------------------------------------------------------


def partition(weights: list[int], k: int, objective: str, forbidden: list[int],
              secondary: Optional[list[int]], secondary_limit: Optional[int], polish: bool) -> list[int]:
    bv = _Blocks(weights, forbidden, secondary)
    target = min(k, bv.count)
    if objective == "min_max":
        bounds = _min_max_blocks(bv, target, secondary_limit if secondary is not None else None, polish)
    else:
        bounds = _sum_squares_blocks(bv.prefix, target)
    return bv.item_ends(bounds)


def partition_tree(parents: list[Optional[int]], weights: list[int], k: int) -> tuple[list[int], list[int]]:
    n = len(parents)
    root = None
    children: list[list[int]] = [[] for _ in range(n)]
    for v, p in enumerate(parents):
        if p is None:
            if root is not None:
                raise EvencutError("evencut: the tree has more than one root")
            root = v
        else:
            if not 0 <= p < n or p == v:
                raise EvencutError("evencut: parent indices must be in [0, n) and differ from the node")
            children[p].append(v)
    if root is None:
        raise EvencutError("evencut: the tree has no root (exactly one parent must be None)")
    order = [root]
    for v in order:
        order.extend(children[v])
    if len(order) != n:
        raise EvencutError("evencut: parent array does not form a single tree (cycle or detached nodes)")
    total = sum(weights)
    if total > U64_MAX:
        raise EvencutError("evencut: total weight exceeds 2^64-1; scale the weights down")
    target = min(k, n)
    resid = [0] * n

    def cuts_needed(bound: int, record: Optional[set]) -> int:
        cuts = 0
        for v in reversed(order):
            ch = children[v]
            total_v = weights[v] + sum(resid[c] for c in ch)
            if total_v > bound:
                heap = [(-resid[c], c) for c in ch]  # largest residual first, smaller index on ties
                heapq.heapify(heap)
                while total_v > bound:
                    neg, c = heapq.heappop(heap)
                    total_v += neg
                    cuts += 1
                    if record is not None:
                        record.add(c)
            resid[v] = total_v
        return cuts

    lo = max(max(weights), -(-total // target))
    hi = total
    lo = min(lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if cuts_needed(mid, None) + 1 <= target:
            hi = mid
        else:
            lo = mid + 1
    cut: set = set()
    parts = cuts_needed(lo, cut) + 1
    if parts < target:
        cand = sorted((v for v in range(n) if v != root and v not in cut), key=lambda v: (-resid[v], v))
        cut.update(cand[: target - parts])
    part_of = [0] * n
    next_id = 1
    for v in order:
        if v == root:
            part_of[v] = 0
        elif v in cut:
            part_of[v] = next_id
            next_id += 1
        else:
            part_of[v] = part_of[parents[v]]
    return part_of, sorted(cut)


def quantize(values: list[float], scale: float) -> list[int]:
    if not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
        raise EvencutError("evencut: scale must be a positive finite number")
    out = []
    for i, v in enumerate(values):
        v = float(v)
        if not math.isfinite(v) or v < 0:
            raise EvencutError(f"evencut: value at index {i} must be finite and non-negative")
        x = v * scale
        if not math.isfinite(x):
            raise EvencutError(f"evencut: value at index {i} does not fit in 64 bits after scaling")
        q = math.floor(x)
        if x - q >= 0.5:  # round half away from zero, exactly like std::round
            q += 1
        if q > U64_MAX:
            raise EvencutError(f"evencut: value at index {i} does not fit in 64 bits after scaling")
        out.append(int(q))
    return out
