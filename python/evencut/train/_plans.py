"""Batch plans for data-parallel training: which sequences every GPU gets at every micro-step.

A *plan* is a list of micro-steps. Each micro-step holds one batch per GPU (rank), and each batch
is a list of dataset indices. With gradient accumulation ``accum``, micro-steps
``[w * accum, (w + 1) * accum)`` make up optimizer step ``w``.

Everything here is pure Python and deterministic. The training sampler and the diagnosis build
their plans with these same functions, so a diagnosis predicts exactly what the sampler hands out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .. import EvencutError, partition
from .. import ai

CostFn = Callable[[int, int], int]
Batch = List[int]
Step = List[Batch]
Plan = List[Step]


def epoch_seed(seed: int, epoch: int) -> int:
    """The random seed of one epoch; the sampler and the diagnosis both use it, so a diagnosis
    with seed s replays exactly what the sampler hands out in epoch 0 with seed s."""
    return seed * 1_000_003 + epoch


def padded_cost(count: int, longest: int) -> int:
    """What a GPU computes for ``count`` sequences padded to ``longest`` tokens."""
    return count * longest


class AttentionCost:
    """``cost(members, longest) = members * longest * (a + longest)``: dense layers plus causal
    attention. ``a`` comes from :func:`attention_constant`. A class (not a lambda) so it pickles."""

    def __init__(self, a: int) -> None:
        if isinstance(a, bool) or not isinstance(a, int) or a < 0:
            raise EvencutError("evencut: the attention constant must be a non-negative integer")
        self.a = a

    def __call__(self, count: int, longest: int) -> int:
        return count * longest * (self.a + longest)

    def __repr__(self) -> str:
        return f"AttentionCost({self.a})"


def attention_constant(params: int, layers: int, d_model: int) -> int:
    """``a`` in ``members * longest * (a + longest)`` for a decoder with ``params`` weights.

    Per token, the dense layers cost about 2 x params FLOPs (forward) and causal attention about
    2 x layers x d_model x length, so ``a = params / (layers * d_model)``. Count the output head
    once (tied or not). For example 29.9M weights, 8 layers, width 512 give a = 7296.
    """
    for name, v in (("params", params), ("layers", layers), ("d_model", d_model)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise EvencutError(f"evencut: {name} must be a positive integer")
    return round(params / (layers * d_model))


def _cut(items: Sequence[int], ends: Sequence[int]) -> list[list[int]]:
    out, prev = [], 0
    for e in ends:
        out.append(list(items[prev:e]))
        prev = e
    return out


def _by_length(indices: Sequence[int], lengths: Sequence[int]) -> list[int]:
    """Ascending length; ties keep index order, so every rank sorts identically."""
    return sorted(indices, key=lambda i: (lengths[i], i))


def _chunks(order: Sequence[int], size: int, what: str) -> list[list[int]]:
    if size < 1:
        raise EvencutError(f"evencut: {what} must be >= 1")
    if len(order) % size:
        raise EvencutError(f"evencut: {len(order)} sequences do not fill whole {what}s of {size}")
    return [list(order[k:k + size]) for k in range(0, len(order), size)]


def _check_gpus(gpus: int, global_batch: int) -> None:
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 1:
        raise EvencutError("evencut: gpus must be an integer >= 1")
    if global_batch < gpus:
        raise EvencutError("evencut: the global batch must hold at least one sequence per GPU")


# ---------------------------------------------------------------------------------------------
# max-token batching over sorted data (fairseq-style), used as a baseline


def greedy_ends(s: Sequence[int], budget: int) -> list[int]:
    """Length-sorted (ascending) sequences cut greedily into batches of at most ``budget`` padded
    tokens: the usual "max tokens" batching. Returns the batch ends."""
    ends: list[int] = []
    count = 0
    for i, x in enumerate(s):                    # ascending, so x is the new longest one
        if count and (count + 1) * x > budget:
            ends.append(i)
            count = 0
        count += 1
    ends.append(len(s))
    return ends


def budget_ends(s: Sequence[int], count: int) -> list[int]:
    """Greedy max-token batching with the smallest budget that yields at most ``count`` batches,
    topped up to exactly ``count`` by halving the largest batches (which only helps it)."""
    lo, hi = max(s), len(s) * max(s)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(greedy_ends(s, mid)) <= count:
            hi = mid
        else:
            lo = mid + 1
    ends = greedy_ends(s, lo)
    while len(ends) < count:
        starts = [0, *ends[:-1]]
        g = max((g for g in range(len(ends)) if ends[g] - starts[g] > 1),
                key=lambda g: (ends[g] - starts[g]) * s[ends[g] - 1])
        ends.insert(g, starts[g] + (ends[g] - starts[g]) // 2)
    return ends


# ---------------------------------------------------------------------------------------------
# plans


def per_step(order: Sequence[int], lengths: Sequence[int], gpus: int, global_batch: int, how: str, *,
             grouped: bool = False, cost: Optional[CostFn] = None) -> Plan:
    """Every micro-step takes the next ``global_batch`` sequences of ``order`` and splits them
    over the GPUs on its own.

    how: ``"equal"`` equal counts in contiguous slices; ``"strided"`` equal counts, GPU r taking
    every gpus-th sequence (what DistributedSampler does); ``"tokens"`` contiguous slices with even
    real tokens (:func:`evencut.partition`); ``"padded"`` contiguous slices with the smallest
    largest padded cost (:func:`evencut.ai.padded_slices`, exact). ``grouped`` sorts every global
    batch by length first.
    """
    _check_gpus(gpus, global_batch)
    plan: Plan = []
    for glob in _chunks(order, global_batch, "global batch"):
        if grouped:
            glob = _by_length(glob, lengths)
        lens = [lengths[i] for i in glob]
        if how == "strided":
            plan.append([glob[r::gpus] for r in range(gpus)])
            continue
        if how == "equal":
            ends = [len(glob) * (r + 1) // gpus for r in range(gpus)]
        elif how == "tokens":
            ends = list(partition(lens, gpus).ends)
        elif how == "padded":
            ends = list(ai.padded_slices(lens, gpus, cost=cost).ends)
        else:
            raise EvencutError(f"evencut: unknown split {how!r}")
        plan.append(_cut(glob, ends))
    return plan


def lpt_cardinality(costs: Sequence[int], gpus: int, per_gpu: int) -> list[list[int]]:
    """Longest-processing-time assignment with a cap: the costliest batch goes first, each to the
    least-loaded GPU that still has room, so every GPU ends up with exactly ``per_gpu`` batches.
    Returns, per GPU, the batch numbers in the order they were assigned (costliest first)."""
    if len(costs) != gpus * per_gpu:
        raise EvencutError("evencut: need exactly gpus x per_gpu batches")
    load, taken = [0] * gpus, [[] for _ in range(gpus)]
    for g in sorted(range(len(costs)), key=lambda g: (-costs[g], g)):
        r = min((r for r in range(gpus) if len(taken[r]) < per_gpu), key=lambda r: (load[r], r))
        load[r] += costs[g]
        taken[r].append(g)
    return taken


def window(order: Sequence[int], lengths: Sequence[int], gpus: int, global_batch: int, accum: int, *,
           cost: Optional[CostFn] = None) -> Plan:
    """Balances whole optimizer steps instead of single micro-steps.

    With gradient accumulation, DDP synchronises the GPUs once per optimizer step (the other
    micro-steps run under ``no_sync``), so what must be even is each GPU's *total* over the
    ``accum`` micro-steps. The step's ``accum x global_batch`` sequences are sorted, cut into
    ``accum x gpus`` contiguous batches with the smallest largest cost (exact), and dealt out by
    :func:`lpt_cardinality` so every GPU gets exactly ``accum`` batches. The optimizer still sees
    the same sequences per step; individual micro-batches vary in size.
    """
    _check_gpus(gpus, global_batch)
    fn = cost or padded_cost
    plan: Plan = []
    for pool in _chunks(order, accum * global_batch, "optimizer step"):
        pool = _by_length(pool, lengths)
        ends = ai.padded_slices([lengths[i] for i in pool], accum * gpus, cost=cost).ends
        batches = _cut(pool, ends)
        costs = [fn(len(b), lengths[b[-1]]) for b in batches]     # sorted: the last is the longest
        taken = lpt_cardinality(costs, gpus, accum)
        for j in range(accum):
            plan.append([batches[taken[r][j]] for r in range(gpus)])
    return plan


def default_pool_steps(gpus: int, mega_batch_mult: int = 50) -> int:
    """Optimizer steps per pool that match Hugging Face's ``group_by_length`` mega-batch
    (50 per-device batches), so both mix sequences across steps equally little."""
    return max(1, mega_batch_mult // gpus)


def pooled(order: Sequence[int], lengths: Sequence[int], gpus: int, global_batch: int, accum: int, *,
           pool_steps: int, seed: int, balance: str = "window", cost: Optional[CostFn] = None) -> Plan:
    """Length-grouped steps, then balanced GPUs: the order is cut into pools of ``pool_steps``
    optimizer steps; each pool is sorted by length and cut into whole optimizer steps of
    ``accum x global_batch`` similar-length sequences; the steps of a pool are shuffled
    (``seed``); every step is then split over the GPUs by :func:`window` (``balance="window"``)
    or micro-step by micro-step (``"step"``).

    Every optimizer step keeps exactly ``accum x global_batch`` sequences. Grouping lengths
    across steps is what ``group_by_length`` does too; ``pool_steps=1`` keeps every step's
    sequences exactly as the order has them.
    """
    if isinstance(pool_steps, bool) or not isinstance(pool_steps, int) or pool_steps < 1:
        raise EvencutError("evencut: pool_steps must be an integer >= 1")
    if balance not in ("window", "step"):
        raise EvencutError("evencut: balance must be 'window' or 'step'")
    _check_gpus(gpus, global_batch)
    size = accum * global_batch
    steps = _chunks(order, size, "optimizer step")
    rng = random.Random(seed)
    plan: Plan = []
    for p in range(0, len(steps), pool_steps):
        pool = _by_length([i for s in steps[p:p + pool_steps] for i in s], lengths)
        grouped = [pool[k:k + size] for k in range(0, len(pool), size)]
        rng.shuffle(grouped)
        for step in grouped:
            if balance == "window":
                plan += window(step, lengths, gpus, global_batch, accum, cost=cost)
            else:
                plan += per_step(step, lengths, gpus, global_batch, "padded", cost=cost)
    return plan


def sorted_dataset(order: Sequence[int], lengths: Sequence[int], gpus: int, steps: int, seed: int, *,
                   method: str = "evencut", cost: Optional[CostFn] = None) -> Plan:
    """Sorts all of ``order``, cuts it into ``steps x gpus`` batches, shuffles the batches with
    ``seed`` and hands the next ``gpus`` of them out at every micro-step.

    ``method="evencut"`` cuts with :func:`evencut.ai.padded_slices` (exact); ``"fairseq"`` with
    greedy max-token batching at the smallest feasible budget. The number of sequences per step
    is not fixed here: short-sequence steps hold many, long-sequence steps few.
    """
    if steps < 1 or len(order) < steps * gpus:
        raise EvencutError("evencut: need at least one sequence per GPU and step")
    s = _by_length(order, lengths)
    lens = [lengths[i] for i in s]
    perm = list(range(steps * gpus))
    random.Random(seed).shuffle(perm)
    if method == "evencut":
        ends = list(ai.padded_slices(lens, steps * gpus, cost=cost).ends)
    elif method == "fairseq":
        ends = budget_ends(lens, steps * gpus)
    else:
        raise EvencutError(f"evencut: unknown method {method!r}")
    batches = _cut(s, ends)
    shuffled = [batches[i] for i in perm]
    return [shuffled[k * gpus:(k + 1) * gpus] for k in range(steps)]


def lpt_any_order(order: Sequence[int], lengths: Sequence[int], gpus: int, global_batch: int) -> Plan:
    """Per global batch, every sequence (longest first) goes to the GPU whose padded size would end
    up smallest. Not contiguous in any order, so it mixes lengths; a baseline, not an Evencut method."""
    _check_gpus(gpus, global_batch)
    plan: Plan = []
    for glob in _chunks(order, global_batch, "global batch"):
        groups: list[list[int]] = [[] for _ in range(gpus)]
        for i in sorted(glob, key=lambda i: (-lengths[i], i)):
            x = lengths[i]
            best = min(range(gpus),
                       key=lambda q: ((len(groups[q]) + 1) * (lengths[groups[q][0]] if groups[q] else x), q))
            groups[best].append(i)
        plan.append(groups)
    return plan


def hf_mega_batch_mult(dataset_size: int, batch_size: int) -> int:
    """Hugging Face's default for ``get_length_grouped_indices``: 50, or fewer so the dataset holds
    at least 4 mega-batches."""
    return max(min(dataset_size // (batch_size * 4), 50), 1)


def hf_group_by_length(order: Sequence[int], lengths: Sequence[int], gpus: int, per_device_batch: int,
                       accum: int, *, mega_batch_mult: Optional[int] = None) -> Plan:
    """What ``Trainer(group_by_length=True)`` hands out (transformers 5.x with Accelerate).

    Mirrors ``get_length_grouped_indices``: the shuffled order is cut into mega-batches of
    ``mega_batch_mult x per_device_batch x accum`` sequences, each sorted longest first, and the
    longest sequence overall is moved to the very front. Accelerate then deals the
    ``per_device_batch``-sized batches round-robin: batch k goes to GPU ``k % gpus``.
    """
    bs = per_device_batch * accum
    mult = mega_batch_mult or hf_mega_batch_mult(len(order), bs)
    size = mult * bs
    megas = [sorted(order[k:k + size], key=lambda i: lengths[i], reverse=True)
             for k in range(0, len(order), size)]
    maxima = [lengths[m[0]] for m in megas]
    top = maxima.index(max(maxima))
    megas[0][0], megas[top][0] = megas[top][0], megas[0][0]
    flat = [i for m in megas for i in m]
    batches = [flat[k:k + per_device_batch] for k in range(0, len(flat), per_device_batch)]
    whole = len(batches) // gpus
    return [batches[k * gpus:(k + 1) * gpus] for k in range(whole)
            if all(len(b) == per_device_batch for b in batches[k * gpus:(k + 1) * gpus])]


def hf_batch_rebalance(order: Sequence[int], lengths: Sequence[int], gpus: int, global_batch: int,
                       accum: int) -> Plan:
    """What ``Trainer(train_sampling_strategy="batch_rebalance")`` hands out, computed by the
    installed transformers itself (``BatchRebalanceSampler``, transformers >= 5.17; needs torch).

    Raises ImportError when transformers (or that sampler) is not installed.
    """
    from transformers.trainer_pt_utils import BatchRebalanceSampler  # optional, heavy

    hf = BatchRebalanceSampler(lengths=list(lengths), effective_batch_size=accum * global_batch, dp_size=gpus,
                               grad_accum=accum, shuffle=False, rank=0)
    plan: Plan = []
    for pool in _chunks(order, accum * global_batch, "optimizer step"):
        rank_mbs = hf._assign(pool, [lengths[i] for i in pool])
        for j in range(accum):
            plan.append([list(rank_mbs[r][j]) for r in range(gpus)])
    return plan


# ---------------------------------------------------------------------------------------------
# evaluating a plan


@dataclass(frozen=True)
class PlanStats:
    """What a plan costs, in cost units (padded tokens unless a cost function was given).

    ``step_cost`` sums, over optimizer steps, the slowest GPU's cost: the time the run takes when
    compute-bound. ``busy_cost`` sums the average GPU's cost, ``useful_cost`` what the same
    sequences would cost with no padding and perfect balance (packing). So per GPU,
    ``step_cost = useful + padding + idle``. With a time curve, likewise in milliseconds:
    ``step_ms = useful + padding + small-batch overhead + waiting``.
    """

    gpus: int
    steps: int
    microsteps: int
    step_cost: float
    busy_cost: float
    useful_cost: float
    real_tokens: int
    padded_tokens: int
    sequences: int
    peak_padded: int
    peak_longest: int
    # with a measured time curve (evaluate(..., timing=...)), in milliseconds summed over the
    # optimizer steps: the slowest GPU (the step), the average GPU's time, the same work at the
    # GPU's best measured efficiency, and the real tokens' share of that
    step_ms: float = 0.0
    busy_ms: float = 0.0
    linear_ms: float = 0.0
    useful_ms: float = 0.0
    sync_ms: float = 0.0          # exposed gradient sync (evaluate(..., exposed=...)), summed over steps

    @property
    def idle_share(self) -> float:
        return 1.0 - self.busy_cost / self.step_cost if self.step_cost else 0.0

    @property
    def padding_share(self) -> float:
        return (self.busy_cost - self.useful_cost) / self.step_cost if self.step_cost else 0.0

    @property
    def useful_share(self) -> float:
        return self.useful_cost / self.step_cost if self.step_cost else 1.0

    @property
    def padded_fraction(self) -> float:
        """Share of the computed tokens that are padding (ignores the cost function)."""
        return 1.0 - self.real_tokens / self.padded_tokens if self.padded_tokens else 0.0


def pad_length(longest: int, pad_to: Optional[int], multiple: int) -> int:
    if pad_to is not None:
        return pad_to
    return -(-longest // multiple) * multiple


def window_step_ms(times: Sequence[Sequence[float]], exposed=None, totals: Optional[Sequence[float]] = None) -> tuple:
    """(compute ms, exposed sync ms) of one optimizer step when the GPUs wait once per step (DDP):
    the slowest GPU's summed micro-batch times, and the gradient sync its last micro-batch cannot
    hide. ``times[r]`` lists GPU r's micro-batch times in order; ``totals`` (optional) their sums.
    The one rule both :func:`evaluate` and ``pick`` use."""
    loads = list(totals) if totals is not None else [sum(t) for t in times]
    slowest = max(range(len(loads)), key=lambda r: loads[r])
    last = times[slowest][-1] if len(times[slowest]) else 0.0
    return loads[slowest], (exposed(last) if exposed is not None else 0.0)


def evaluate(plan: Plan, lengths: Sequence[int], gpus: int, accum: int, *, cost: Optional[CostFn] = None,
             pad_to: Optional[int] = None, pad_multiple: int = 1, sync: str = "window",
             timing=None, exposed=None) -> PlanStats:
    """Costs ``plan``. ``sync="window"``: GPUs wait for each other once per optimizer step (DDP
    with ``no_sync`` during accumulation, ZeRO-1/2), so a GPU's load is its sum over the step's
    micro-steps. ``sync="microstep"``: they wait at every micro-step (FSDP, ZeRO-3, or DDP without
    ``no_sync``). ``pad_to`` pads every batch to a fixed length (``padding="max_length"``);
    otherwise batches pad to their longest sequence, rounded up to ``pad_multiple``. ``timing``
    (a :class:`~evencut.train.timing.TimeCurve`) adds measured milliseconds to the result; ``exposed``
    (a :class:`~evencut.train.timing.Sync`, needs ``timing``) the gradient sync each step cannot hide,
    from the slowest GPU's last micro-batch (with ``sync="microstep"``: at every micro-step)."""
    if sync not in ("window", "microstep"):
        raise EvencutError("evencut: sync must be 'window' or 'microstep'")
    if accum < 1 or len(plan) % accum:
        raise EvencutError("evencut: the plan must hold whole optimizer steps")
    fn = cost or padded_cost
    step_cost = busy = 0.0
    useful = 0
    real = padded_tok = seqs = peak = peak_long = 0
    t_step = t_busy = t_linear = t_useful = t_sync = 0.0
    for w in range(0, len(plan), accum):
        loads = [0] * gpus
        tloads = [0.0] * gpus
        for micro in plan[w:w + accum]:
            if len(micro) != gpus:
                raise EvencutError("evencut: every micro-step needs one batch per GPU")
            mloads = []
            mtimes = []
            for r, b in enumerate(micro):
                if not b:
                    mloads.append(0)
                    mtimes.append(0.0)
                    continue
                longest = max(lengths[i] for i in b)
                plen = pad_length(longest, pad_to, pad_multiple)
                c = fn(len(b), plen)
                loads[r] += c
                mloads.append(c)
                for i in b:
                    useful += fn(1, lengths[i])
                    real += lengths[i]
                padded_tok += len(b) * plen
                seqs += len(b)
                if len(b) * plen > peak:
                    peak = len(b) * plen
                peak_long = max(peak_long, plen)
                if timing is not None:
                    t = timing.batch_ms(len(b), plen)
                    tloads[r] += t
                    mtimes.append(t)
                    t_linear += timing.linear_ms(len(b), plen)
                    t_useful += sum(timing.linear_ms(1, lengths[i]) for i in b)
            if sync == "microstep":
                step_cost += max(mloads)
                if timing is not None:
                    t_step += max(mtimes)
                    if exposed is not None:
                        t_sync += exposed(max(mtimes))
        if sync == "window":
            step_cost += max(loads)
            if timing is not None:
                # mtimes: every GPU's last micro-batch; tloads: its sum over the step
                compute, exposed_ms = window_step_ms([[mtimes[r]] for r in range(gpus)], exposed, totals=tloads)
                t_step += compute
                t_sync += exposed_ms
        busy += sum(loads) / gpus
        t_busy += sum(tloads) / gpus
    return PlanStats(gpus, len(plan) // accum, len(plan), step_cost, busy, useful / gpus, real, padded_tok,
                     seqs, peak, peak_long, t_step, t_busy, t_linear / gpus, t_useful / gpus, t_sync)


__all__ = [
    "AttentionCost", "CostFn", "Plan", "PlanStats", "attention_constant", "budget_ends", "epoch_seed", "evaluate",
    "greedy_ends", "hf_batch_rebalance", "hf_group_by_length", "hf_mega_batch_mult", "lpt_any_order",
    "lpt_cardinality", "pad_length", "padded_cost", "per_step", "sorted_dataset", "window",
]
