"""Tests for evencut.ai (pipeline stages, shards, batches, attention chunks)."""

from __future__ import annotations

import itertools
import math

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from evencut import EvencutError
from evencut import ai

PROPS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
costs_st = st.lists(st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=9)


def best_bottleneck(costs, stages, emb=0.0, head=0.0, mem=None, limit=None, emb_mem=0.0, head_mem=0.0):
    n = len(costs)
    best = None
    for cuts in itertools.combinations(range(1, n), stages - 1):
        bounds = [0, *cuts, n]
        loads, fits = [], True
        for s, (a, b) in enumerate(zip(bounds, bounds[1:])):
            c = math.fsum(costs[a:b]) + (emb if s == 0 else 0) + (head if s == stages - 1 else 0)
            loads.append(c)
            if mem is not None:
                m = math.fsum(mem[a:b]) + (emb_mem if s == 0 else 0) + (head_mem if s == stages - 1 else 0)
                fits = fits and m <= limit
        if fits:
            best = max(loads) if best is None else min(best, max(loads))
    return best


def check_plan(plan, n, stages):
    assert plan.num_stages == stages
    assert plan.stages[0][0] == 0 and plan.stages[-1][1] == n
    assert all(a < b for a, b in plan.stages)
    assert all(prev[1] == nxt[0] for prev, nxt in zip(plan.stages, plan.stages[1:]))
    assert sum(plan.layers_per_stage()) == n


@PROPS
@given(costs_st, st.integers(1, 5), st.floats(0, 50), st.floats(0, 50))
@example([1.1125369292536007e-308], 1, 0.0, 0.0)  # found by Hypothesis: scale overflowed to inf
def test_pipeline_stages_is_optimal(costs, stages, emb, head):
    stages_used = min(stages, len(costs))
    plan = ai.pipeline_stages(costs, stages, embedding_cost=emb, head_cost=head)
    check_plan(plan, len(costs), stages_used)
    best = best_bottleneck(costs, stages_used, emb, head)
    # costs are quantised to ~1e-12 relative precision before the exact solver runs
    assert plan.bottleneck <= best * (1 + 1e-9) + 1e-9
    assert math.isclose(math.fsum(plan.stage_costs), math.fsum(costs) + emb + head, rel_tol=1e-12, abs_tol=1e-12)
    assert plan.bottleneck <= ai.uniform_stages(costs, stages_used, embedding_cost=emb, head_cost=head).bottleneck * (1 + 1e-9) + 1e-9


@PROPS
@given(costs_st, st.integers(1, 4), st.data())
def test_pipeline_memory_limit_is_respected(costs, stages, data):
    n = len(costs)
    mem = data.draw(st.lists(st.floats(0.0, 10.0), min_size=n, max_size=n))
    limit = data.draw(st.floats(0.0, 40.0))
    stages_used = min(stages, n)
    # Memory is enforced conservatively (each layer rounded up, the limit down, at ~1e-12 of the
    # scale), so the promise is: never over the limit, and at least as good as the optimum under
    # a limit tightened by that margin.
    scale = max(math.fsum(mem), limit)
    tight = limit - (n + 3) * scale * 1e-12
    best_tight = best_bottleneck(costs, stages_used, mem=mem, limit=tight)
    try:
        plan = ai.pipeline_stages(costs, stages, layer_memory=mem, memory_limit=limit)
    except EvencutError:
        assert best_tight is None
        return
    check_plan(plan, n, stages_used)
    assert all(m <= limit * (1 + 1e-12) + 1e-12 for m in plan.stage_memory)
    if best_tight is not None:
        assert plan.bottleneck <= best_tight * (1 + 1e-9) + 1e-9


def test_subnormal_memory_limit_regression():
    # found by Hypothesis: TARGET / 2.2e-308 overflowed and 0 * inf gave NaN
    plan = ai.pipeline_stages([0.0], 1, layer_memory=[0.0], memory_limit=2.2250738585072014e-308)
    assert plan.stage_memory == (0.0,)


def test_framework_formats():
    costs = [1.0] * 12
    plan = ai.pipeline_stages(costs, 3, embedding_cost=2.0, head_cost=3.0)
    starts = [a for a, _ in plan.stages]
    assert plan.torchtitan_split_points() == [f"layers.{s}" for s in starts[1:]]
    layout = plan.megatron_layout()
    assert layout.startswith("E") and layout.endswith("L") and layout.count("|") == 2 and layout.count("t") == 12
    parts = plan.deepspeed_parts()
    assert parts[0] == 0 and parts[-1] == 12 and len(parts) == 4
    # embedding (2) and head (3) make the end stages heavier: together they get fewer layers
    # than the even 4 + 4, and the bottleneck is 6 (e.g. 2+4 | 5 | 3+3; a bound of 5 needs >3 stages)
    assert plan.layers_per_stage()[0] + plan.layers_per_stage()[-1] < 8
    assert plan.bottleneck == pytest.approx(6.0)
    assert plan.step_time(8) == pytest.approx((8 + 3 - 1) * plan.bottleneck)
    assert "stage 0" in plan.summary()


def test_pipeline_rejects_bad_input():
    with pytest.raises(EvencutError):
        ai.pipeline_stages([], 2)
    with pytest.raises(EvencutError):
        ai.pipeline_stages([1.0, -1.0], 2)
    with pytest.raises(EvencutError):
        ai.pipeline_stages([1.0, 2.0], 2, memory_limit=5.0)          # memory without per-layer values
    with pytest.raises(EvencutError):
        ai.pipeline_stages([1.0, 2.0], 2, layer_memory=[5.0, 5.0], memory_limit=4.0)  # a layer alone too big


@PROPS
@given(st.lists(st.integers(0, 500), min_size=1, max_size=40), st.integers(500, 2000))
def test_token_budget_batches(tokens, budget):
    r = ai.token_budget_batches(tokens, budget)
    assert r.max_load <= budget
    # the greedy count is the minimum possible number of batches
    greedy, cur = 1, 0
    for t in tokens:
        if cur + t > budget:
            greedy, cur = greedy + 1, t
        else:
            cur += t
    assert r.parts == min(greedy, len(tokens))
    with pytest.raises(EvencutError):
        ai.token_budget_batches(tokens + [budget + 1], budget)


def _padded(group):
    return len(group) * max(group)


def _best_padded(lengths, k):
    """Brute force: smallest possible 'largest padded group' over all k-part contiguous splits."""
    n = len(lengths)
    return min(max(_padded(lengths[a:b]) for a, b in zip((0, *cuts), (*cuts, n)))
               for cuts in itertools.combinations(range(1, n), k - 1))


def _check_padded_split(lengths, r):
    prev = 0
    for e, pad, tok in zip(r.ends, r.padded, r.tokens):
        group = lengths[prev:e]
        assert e > prev and pad == _padded(group) and tok == sum(group)
        prev = e
    assert prev == len(lengths) and r.max_padded == max(r.padded)


@PROPS
@given(st.lists(st.integers(0, 60), min_size=1, max_size=10), st.integers(1, 6))
def test_padded_slices_is_optimal(lengths, k):
    r = ai.padded_slices(lengths, k)
    _check_padded_split(lengths, r)
    assert r.parts == min(k, len(lengths))
    assert r.max_padded == _best_padded(lengths, r.parts)


@PROPS
@given(st.lists(st.integers(1, 60), min_size=1, max_size=10), st.integers(60, 300))
def test_padded_batches_fewest_then_even(lengths, budget):
    r = ai.padded_batches(lengths, budget)
    _check_padded_split(lengths, r)
    assert r.max_padded <= budget
    fewest = min(k for k in range(1, len(lengths) + 1) if _best_padded(lengths, k) <= budget)
    assert r.parts == fewest
    assert r.max_padded == _best_padded(lengths, fewest)   # evened out: no lower maximum with that count


def _attention_cost(count, longest):
    return count * longest * (50 + longest)


@PROPS
@given(st.lists(st.integers(0, 60), min_size=1, max_size=9), st.integers(1, 5))
def test_padded_slices_with_a_cost_function_is_optimal(lengths, k):
    r = ai.padded_slices(lengths, k, cost=_attention_cost)
    n = len(lengths)
    best = min(max(_attention_cost(b - a, max(lengths[a:b])) for a, b in zip((0, *cuts), (*cuts, n)))
               for cuts in itertools.combinations(range(1, n), r.parts - 1))
    assert r.parts == min(k, n)
    assert r.max_cost == best
    assert r.costs == tuple(_attention_cost(b - a, max(lengths[a:b])) for a, b in r.segments())
    assert ai.padded_slices(lengths, k).costs == ai.padded_slices(lengths, k).padded   # default cost


def test_padded_slices_rejects_a_bad_cost():
    for bad in (lambda c, m: -1, lambda c, m: 1.5, lambda c, m: True):
        with pytest.raises(EvencutError):
            ai.padded_slices([3, 4, 5], 2, cost=bad)


def test_padded_helpers_reject_bad_input():
    for call in (lambda: ai.padded_slices([], 2), lambda: ai.padded_slices([3, -1], 2),
                 lambda: ai.padded_slices([3, 1.5], 2), lambda: ai.padded_slices([3, True], 2),
                 lambda: ai.padded_slices([3, 4], 0), lambda: ai.padded_batches([3, 9], 8),
                 lambda: ai.padded_batches([3, 4], 0)):
        with pytest.raises(EvencutError):
            call()
    r = ai.padded_slices([5, 1, 1, 1, 1], 2)        # one long item: the short ones form their own group
    assert r.ends == (1, 5) and r.padded == (5, 4) and abs(r.waste) < 1e-12


def test_causal_attention_chunks_shrink_towards_the_end():
    r = ai.causal_attention_chunks(32768, 4)
    lengths = [b - a for a, b in r.segments()]
    assert sum(lengths) == 32768
    assert lengths == sorted(lengths, reverse=True)   # later tokens attend to more keys
    assert r.imbalance < 1.001
    with pytest.raises(EvencutError):
        ai.causal_attention_chunks(0, 2)


def test_balanced_shards():
    r = ai.balanced_shards([3, 1, 4, 1, 5, 9, 2, 6], 3)
    assert r.parts == 3 and r.total == 31


def test_code_review_regressions():
    with pytest.raises(EvencutError):
        ai.pipeline_stages([1e308, 1e308], 2)                  # float sum overflows
    with pytest.raises(EvencutError):
        ai.pipeline_stages([1.0, 1.0], 2, layer_memory=[1e308, 1e308], memory_limit=1.0)
    with pytest.raises(EvencutError):
        ai.uniform_stages([1e308, 1e308], 2)
    with pytest.raises(EvencutError):
        ai.pipeline_stages([1.0, 1.0], 2, embedding_memory=1.0)  # memory extra without a limit
    plan = ai.pipeline_stages([1.0], 1, layer_memory=[1.0], memory_limit=3.0, embedding_memory="1")
    assert plan.stage_memory == (2.0,)                           # validated value, not the raw string
