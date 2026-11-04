"""BalancedBatchSampler without torch: what every rank yields, and that it is what the diagnosis
predicted."""

import random

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from evencut import EvencutError
from evencut.train import BalancedBatchSampler
from evencut.train import _plans as P
from evencut.train.diagnose import Setup, diagnose

PROPS = settings(max_examples=80, deadline=None)


@st.composite
def configs(draw):
    world = draw(st.integers(1, 4))
    per_device = draw(st.integers(1, 4))
    accum = draw(st.integers(1, 3))
    steps = draw(st.integers(1, 5))
    extra = draw(st.integers(0, world * per_device * accum - 1))     # a partial last step
    n = world * per_device * accum * steps + extra
    lengths = draw(st.lists(st.integers(0, 400), min_size=n, max_size=n))
    mode = draw(st.sampled_from(["pooled", "window", "step", "sorted"]))
    return dict(lengths=lengths, world=world, per_device=per_device, accum=accum, mode=mode,
                pool_steps=draw(st.integers(1, 4)), seed=draw(st.integers(0, 3)),
                fsdp=draw(st.booleans()), drop_last=draw(st.booleans()))


def samplers(c, **kw):
    return [BalancedBatchSampler(c["lengths"], per_device_batch=c["per_device"], accum=c["accum"], mode=c["mode"],
                                 pool_steps=c["pool_steps"], seed=c["seed"], sync_every_microstep=c["fsdp"],
                                 drop_last=c["drop_last"], rank=r, world_size=c["world"], **kw)
            for r in range(c["world"])]


@PROPS
@given(configs(), st.integers(0, 2))
@example(dict(lengths=[3] * 10, world=2, per_device=2, accum=1, mode="pooled", pool_steps=1, seed=0, fsdp=False,
              drop_last=False), 0)
def test_every_rank_yields_its_share_of_one_epoch(c, epoch):
    ranks = samplers(c)
    for s in ranks:
        s.set_epoch(epoch)
    yielded = [list(s) for s in ranks]
    n, per = len(c["lengths"]), c["world"] * c["per_device"] * c["accum"]
    steps = n // per if c["drop_last"] else -(-n // per)
    # same number of micro-batches everywhere (else DDP hangs), none empty
    assert all(len(y) == len(s) == steps * c["accum"] for y, s in zip(yielded, ranks))
    assert all(b for y in yielded for b in y)
    seen = [i for y in yielded for b in y for i in b]
    if c["drop_last"]:
        assert len(seen) == len(set(seen)) == steps * per           # each sequence at most once
    else:
        assert set(seen) == set(range(n)) and len(seen) == steps * per  # all of them, padded by repeats
    if c["mode"] != "sorted":                                        # fixed sequences per optimizer step
        for w in range(steps):
            assert sum(len(y[w * c["accum"] + j]) for y in yielded for j in range(c["accum"])) == per


@PROPS
@given(configs())
def test_all_ranks_build_the_same_plan_and_epochs_differ(c):
    ranks = samplers(c)
    plans = [s.plan(0) for s in ranks]
    assert all(p == plans[0] for p in plans)
    # yielded batches are the plan's column for the rank
    for r, s in enumerate(ranks):
        assert list(s) == [micro[r] for micro in plans[0]]
    again = samplers(c)[0]
    assert again.plan(0) == plans[0]                                # deterministic across instances


def test_epochs_reshuffle_and_seeds_matter():
    lengths = [random.Random(1).randint(1, 900) for _ in range(4 * 8 * 2 * 10)]
    s = BalancedBatchSampler(lengths, per_device_batch=8, accum=2, rank=0, world_size=4)
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first
    s.set_epoch(0)
    assert list(s) == first
    other = BalancedBatchSampler(lengths, per_device_batch=8, accum=2, rank=0, world_size=4, seed=1)
    assert list(other) != first
    unshuffled = BalancedBatchSampler(lengths, per_device_batch=8, accum=2, rank=0, world_size=4, shuffle=False)
    assert unshuffled.order(0) == list(range(len(lengths)))


@PROPS
@given(configs())
def test_global_tokens_counts_every_gpu_of_the_step(c):
    ranks = samplers(c)
    yielded = [list(s) for s in ranks]
    s = ranks[0]
    for w in range(s.steps):
        batches = [b for y in yielded for b in y[w * s.accum:(w + 1) * s.accum]]
        assert s.global_tokens(w) == sum(c["lengths"][i] for b in batches for i in b)
        assert s.global_sequences(w) == sum(len(b) for b in batches)
    with pytest.raises(EvencutError):
        s.global_tokens(s.steps)


def test_the_diagnosis_predicts_the_sampler():
    """Oracle: the diagnosis replays the sampler's epoch 0 for the same seed, so their plans (and
    costs) agree on every whole pool the diagnosis replays."""
    rng = random.Random(9)
    lengths = [max(1, min(2048, int(rng.paretovariate(1.1) * 50))) for _ in range(20_000)]
    for mode, key, accum in (("pooled", "evencut_pooled", 2), ("window", "evencut_window", 4),
                             ("step", "evencut_step", 1)):
        setup = Setup(gpus=4, global_batch=32, accum=accum, max_len=2048, seed=5, pool_steps=6)
        steps = 60                                              # a multiple of pool_steps
        d = diagnose(lengths, setup, sample_steps=steps, strategies=[key])
        s = BalancedBatchSampler(lengths, global_batch=32, accum=accum, mode=mode, pool_steps=6, seed=5,
                                 max_len=2048, rank=0, world_size=4)
        predicted = d.row(key).stats
        actual = P.evaluate(s.plan()[: steps * accum], s.lengths, 4, accum)
        assert actual == predicted, mode


def test_sampler_takes_numpy_like_lengths_and_cuts_at_max_len():
    class Num:                      # stands in for numpy integers: int() works, not an int subclass
        def __init__(self, v): self.v = v
        def __int__(self): return self.v
        def __eq__(self, o): return self.v == o
        def __index__(self): return self.v
    s = BalancedBatchSampler([Num(5), Num(9000), Num(3), Num(7)], global_batch=2, max_len=1000, rank=0, world_size=2)
    assert s.lengths == [5, 1000, 3, 7]


def test_bad_arguments_are_refused():
    with pytest.raises(EvencutError):
        BalancedBatchSampler([1] * 10, rank=0, world_size=1)                       # no batch size
    with pytest.raises(EvencutError):
        BalancedBatchSampler([1] * 10, 4, per_device_batch=2, rank=0, world_size=1)  # both
    with pytest.raises(EvencutError):
        BalancedBatchSampler([1] * 10, global_batch=2, rank=0, world_size=4)        # < 1 per GPU
    with pytest.raises(EvencutError):
        BalancedBatchSampler([1] * 3, global_batch=4, rank=0, world_size=1)         # not one step
    with pytest.raises(EvencutError):
        BalancedBatchSampler([1] * 10, global_batch=2, rank=2, world_size=2)
    with pytest.raises(EvencutError):
        BalancedBatchSampler([1, -1] * 5, global_batch=2, rank=0, world_size=1)
    with pytest.raises(EvencutError):
        BalancedBatchSampler([1.5] * 10, global_batch=2, rank=0, world_size=1)
    with pytest.raises(EvencutError):
        BalancedBatchSampler([1] * 10, global_batch=2, mode="magic", rank=0, world_size=1)


def test_rank_and_world_come_from_torchrun_environment(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    s = BalancedBatchSampler([5] * 64, per_device_batch=4)
    assert (s.rank, s.world_size, s.global_batch) == (2, 4, 16)
