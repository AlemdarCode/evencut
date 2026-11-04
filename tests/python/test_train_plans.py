"""Batch plans (evencut.train._plans): every plan hands out each sequence exactly once, in the
promised shape, and its costs decompose exactly. Optimality of the splits themselves is tested
against brute force in test_ai.py."""

import itertools
import random

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from evencut import EvencutError
from evencut.train import _plans as P

PROPS = settings(max_examples=120, deadline=None)


@st.composite
def setups(draw, max_gpus=5, max_accum=3, max_steps=4):
    """lengths, gpus, global batch, accum, steps: a run that fills whole optimizer steps."""
    gpus = draw(st.integers(1, max_gpus))
    global_batch = gpus * draw(st.integers(1, 4)) + draw(st.integers(0, gpus - 1))
    accum = draw(st.integers(1, max_accum))
    steps = draw(st.integers(1, max_steps))
    n = global_batch * accum * steps
    lengths = draw(st.lists(st.integers(0, 300), min_size=n, max_size=n))
    return lengths, gpus, global_batch, accum, steps


def covers_each_once(plan, order):
    seen = [i for step in plan for batch in step for i in batch]
    return sorted(seen) == sorted(order)


def check_shape(plan, gpus, accum, global_batch=None):
    assert len(plan) % accum == 0
    for w in range(0, len(plan), accum):
        window = plan[w:w + accum]
        assert all(len(step) == gpus for step in window)
        if global_batch is not None:
            assert sum(len(b) for step in window for b in step) == accum * global_batch


@PROPS
@given(setups(), st.sampled_from(["equal", "strided", "tokens", "padded"]), st.booleans())
def test_per_step_hands_out_every_sequence_once(s, how, grouped):
    lengths, gpus, gb, accum, steps = s
    order = list(range(len(lengths)))
    random.Random(len(lengths)).shuffle(order)
    plan = P.per_step(order, lengths, gpus, gb, how, grouped=grouped)
    assert len(plan) == accum * steps
    check_shape(plan, gpus, accum, gb)
    assert covers_each_once(plan, order)
    for k, step in enumerate(plan):          # a micro-step only uses its own global batch
        assert sorted(i for b in step for i in b) == sorted(order[k * gb:(k + 1) * gb])
    if how in ("equal", "strided"):
        for step in plan:
            sizes = [len(b) for b in step]
            assert max(sizes) - min(sizes) <= 1


@PROPS
@given(setups())
def test_padded_split_is_the_best_contiguous_split(s):
    """Oracle: no contiguous split of the sorted global batch has a smaller costliest slice."""
    lengths, gpus, gb, _, _ = s
    order = list(range(gb))
    step = P.per_step(order, lengths[:gb], gpus, gb, "padded", grouped=True)[0]
    got = max(P.padded_cost(len(b), max(lengths[i] for i in b)) if b else 0 for b in step)
    srt = sorted(lengths[:gb])
    best = None
    for cuts in itertools.combinations(range(1, gb), min(gpus, gb) - 1):
        bounds = (0, *cuts, gb)
        c = max(P.padded_cost(bounds[j + 1] - bounds[j], srt[bounds[j + 1] - 1]) for j in range(len(bounds) - 1))
        best = c if best is None else min(best, c)
    assert got == best


@PROPS
@given(setups())
@example(([5] * 12, 2, 3, 2, 2))
def test_window_gives_every_gpu_exactly_accum_batches(s):
    lengths, gpus, gb, accum, steps = s
    order = list(range(len(lengths)))
    random.Random(7).shuffle(order)
    plan = P.window(order, lengths, gpus, gb, accum)
    assert len(plan) == accum * steps
    check_shape(plan, gpus, accum, gb)
    assert covers_each_once(plan, order)
    pool = accum * gb
    for w in range(steps):                   # a window only uses its own optimizer step's sequences
        got = sorted(i for step in plan[w * accum:(w + 1) * accum] for b in step for i in b)
        assert got == sorted(order[w * pool:(w + 1) * pool])
        assert all(b for step in plan[w * accum:(w + 1) * accum] for b in step)


@PROPS
@given(setups(), st.integers(1, 5), st.sampled_from(["window", "step"]), st.integers(0, 3))
def test_pooled_keeps_step_sizes_and_stays_inside_its_pool(s, pool_steps, balance, seed):
    lengths, gpus, gb, accum, steps = s
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    plan = P.pooled(order, lengths, gpus, gb, accum, pool_steps=pool_steps, seed=seed, balance=balance)
    assert len(plan) == accum * steps
    check_shape(plan, gpus, accum, gb)
    assert covers_each_once(plan, order)
    size = accum * gb
    for w in range(steps):                   # a step only takes sequences from its own pool
        p = w // pool_steps
        pool = set(order[p * pool_steps * size:(p + 1) * pool_steps * size])
        assert {i for step in plan[w * accum:(w + 1) * accum] for b in step for i in b} <= pool


@PROPS
@given(setups(), st.integers(0, 3))
def test_pooled_with_one_step_pools_is_the_window_split(s, seed):
    """Oracle: a pool of one optimizer step regroups nothing, so the window split must come out."""
    lengths, gpus, gb, accum, _ = s
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    assert P.pooled(order, lengths, gpus, gb, accum, pool_steps=1, seed=seed) == P.window(order, lengths, gpus, gb,
                                                                                            accum)
    if accum == 1:
        assert P.pooled(order, lengths, gpus, gb, 1, pool_steps=1, seed=seed, balance="step") == \
            P.per_step(order, lengths, gpus, gb, "padded", grouped=True)


def test_pooling_cuts_padding_on_mixed_lengths():
    rng = random.Random(4)
    lengths = [rng.choice([20, 40, 900, 1000]) for _ in range(4 * 16 * 24)]
    order = list(range(len(lengths)))
    same = P.evaluate(P.window(order, lengths, 4, 16, 1), lengths, 4, 1)
    pooled = P.evaluate(P.pooled(order, lengths, 4, 16, 1, pool_steps=12, seed=0), lengths, 4, 1)
    assert pooled.padded_fraction < same.padded_fraction and pooled.step_cost < same.step_cost
    with pytest.raises(EvencutError):
        P.pooled(order, lengths, 4, 16, 1, pool_steps=0, seed=0)


@PROPS
@given(setups())
def test_plans_are_deterministic_and_order_free_of_input_types(s):
    """Every rank builds the plan on its own, so the same inputs must give the same plan, also when
    the order arrives as a range or a tuple."""
    lengths, gpus, gb, accum, _ = s
    order = list(range(len(lengths)))
    a = P.window(order, lengths, gpus, gb, accum)
    assert a == P.window(range(len(lengths)), tuple(lengths), gpus, gb, accum)
    assert P.per_step(order, lengths, gpus, gb, "padded", grouped=True) == \
        P.per_step(tuple(order), lengths, gpus, gb, "padded", grouped=True)


@PROPS
@given(setups(), st.sampled_from(["evencut", "fairseq"]), st.integers(0, 3))
def test_sorted_dataset_covers_everything(s, method, seed):
    lengths, gpus, gb, accum, steps = s
    order = list(range(len(lengths)))
    micro = accum * steps
    plan = P.sorted_dataset(order, lengths, gpus, micro, seed, method=method)
    assert len(plan) == micro and all(len(step) == gpus for step in plan)
    assert all(b for step in plan for b in step)
    assert covers_each_once(plan, order)


@PROPS
@given(st.lists(st.integers(1, 200), min_size=1, max_size=40), st.data())
def test_budget_ends_gives_exactly_count_contiguous_batches(lens, data):
    s = sorted(lens)
    count = data.draw(st.integers(1, len(s)))
    ends = P.budget_ends(s, count)
    assert len(ends) == count and ends[-1] == len(s)
    assert all(a < b for a, b in zip((0, *ends[:-1]), ends))


@PROPS
@given(st.lists(st.integers(0, 1000), min_size=1, max_size=30), st.data())
def test_lpt_cardinality_deals_exact_counts(costs, data):
    gpus = data.draw(st.integers(1, 6))
    per = data.draw(st.integers(1, 5))
    costs = (costs * (gpus * per))[: gpus * per]
    taken = P.lpt_cardinality(costs, gpus, per)
    assert all(len(t) == per for t in taken)
    assert sorted(g for t in taken for g in t) == list(range(gpus * per))
    with pytest.raises(EvencutError):
        P.lpt_cardinality(costs + [1], gpus, per)


@PROPS
@given(setups(), st.sampled_from([None, 0, 50, 7296]), st.sampled_from([1, 8]))
def test_evaluate_decomposes_gpu_time(s, a, multiple):
    """step = useful + padding + waiting; waiting at every micro-step can only cost more."""
    lengths, gpus, gb, accum, _ = s
    cost = None if a is None else P.AttentionCost(a)
    plan = P.per_step(range(len(lengths)), lengths, gpus, gb, "strided" if gb % gpus == 0 else "equal")
    w = P.evaluate(plan, lengths, gpus, accum, cost=cost, pad_multiple=multiple)
    m = P.evaluate(plan, lengths, gpus, accum, cost=cost, pad_multiple=multiple, sync="microstep")
    eps = 1e-9 * max(1.0, w.step_cost)          # busy and useful are sums of averages: allow rounding
    assert w.step_cost >= w.busy_cost - eps and w.busy_cost >= w.useful_cost - eps and w.useful_cost >= 0
    assert w.useful_share + w.padding_share + w.idle_share == pytest.approx(1.0)
    assert w.real_tokens == sum(lengths) and w.padded_tokens >= w.real_tokens
    assert w.sequences == len(lengths)
    assert m.step_cost >= w.step_cost and m.busy_cost == w.busy_cost


def test_pad_to_max_length_charges_the_full_length():
    lengths = [1, 2, 3, 4]
    plan = P.per_step(range(4), lengths, 2, 4, "equal")
    st_ = P.evaluate(plan, lengths, 2, 1, pad_to=10)
    assert st_.padded_tokens == 40 and st_.peak_padded == 20 and st_.peak_longest == 10


def test_attention_constant_matches_the_benchmark_model():
    # 8 layers, width 512, vocab 8192, SwiGLU 1408: 29,884,416 weights (bench/gpu_batches.py)
    assert P.attention_constant(29_884_416, 8, 512) == 7296
    with pytest.raises(EvencutError):
        P.attention_constant(0, 8, 512)
    with pytest.raises(EvencutError):
        P.AttentionCost(-1)


def test_window_beats_per_step_on_heavy_tailed_lengths():
    """Example, not a law: on long-tailed data, balancing whole optimizer steps leaves less waiting
    than balancing each micro-step (the case window balancing exists for)."""
    rng = random.Random(3)
    lengths = [min(2048, int(rng.paretovariate(1.2) * 60)) for _ in range(8 * 32 * 4 * 6)]
    order = list(range(len(lengths)))
    step = P.evaluate(P.per_step(order, lengths, 8, 32, "padded", grouped=True), lengths, 8, 4)
    win = P.evaluate(P.window(order, lengths, 8, 32, 4), lengths, 8, 4)
    assert win.step_cost < step.step_cost
    assert win.idle_share < step.idle_share


def test_bad_shapes_are_refused():
    with pytest.raises(EvencutError):
        P.per_step(range(10), [1] * 10, 4, 3, "equal")          # global batch smaller than gpus
    with pytest.raises(EvencutError):
        P.per_step(range(10), [1] * 10, 2, 4, "equal")          # 10 does not fill batches of 4
    with pytest.raises(EvencutError):
        P.per_step(range(8), [1] * 8, 2, 4, "nonsense")
    with pytest.raises(EvencutError):
        P.evaluate([[[0], [1]]], [1, 1], 2, 1, sync="sometimes")


hf = pytest.importorskip  # alias for readability below


def test_group_by_length_matches_transformers():
    tpu = hf("transformers.trainer_pt_utils")
    torch = hf("torch")
    gpus, per_device, accum = 4, 8, 2
    for seed in range(6):
        # distinct lengths, so the longest sequence is unique and usually not in the first
        # mega-batch: that exercises Hugging Face's "longest batch first" swap
        lengths = list(range(1, 4 * 8 * 2 * 30 + 1))
        random.Random(seed).shuffle(lengths)
        lengths = [x % 700 + (1000 if x == len(lengths) else 0) for x in lengths]
        g1, g2 = torch.Generator(), torch.Generator()
        g1.manual_seed(seed)
        g2.manual_seed(seed)
        theirs = tpu.get_length_grouped_indices(lengths, per_device * accum, generator=g1)
        order = torch.randperm(len(lengths), generator=g2).tolist()
        ours = P.hf_group_by_length(order, lengths, gpus, per_device, accum)
        flat = [i for step in ours for b in step for i in b]
        assert flat == theirs


def test_batch_rebalance_plan_is_well_formed():
    hf("transformers.trainer_pt_utils")
    rng = random.Random(6)
    lengths = [rng.randint(1, 900) for _ in range(4 * 16 * 2 * 3)]
    order = list(range(len(lengths)))
    plan = P.hf_batch_rebalance(order, lengths, 4, 16, 2)
    check_shape(plan, 4, 2, 16)
    assert covers_each_once(plan, order)
