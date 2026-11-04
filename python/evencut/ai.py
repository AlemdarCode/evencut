"""Helpers for AI training workloads, built on :func:`evencut.partition`.

Everything here balances *work* (time, tokens, FLOPs) across devices, so a training step
finishes sooner because fewer GPUs sit idle waiting for the slowest one. It does not change
what or how well the model learns.

* :func:`pipeline_stages`       layers -> pipeline-parallel stages (with memory limits), plus
                                 ready-made settings for torchtitan, Megatron-Core and DeepSpeed
* :func:`balanced_shards`       an ordered dataset -> data-parallel ranks with even token counts
* :func:`token_budget_batches`  fewest batches under a token budget, then evened out
* :func:`padded_slices`         sequences padded to their group's longest -> k groups (for
                                 example one per GPU) with the smallest largest padded size
* :func:`padded_batches`        fewest padded batches under a memory budget, then evened out
* :func:`causal_attention_chunks`  a long sequence -> context-parallel chunks with even
                                 causal-attention cost
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from . import EvencutError, Partition, partition, quantize

# Quantised totals stay just below 2^40, which keeps the native solver on its fast exact path
# while giving ~12 significant digits of precision.
_TARGET_TOTAL = 1_000_000_000_000


def _ratios(values: list[float], denom: float) -> list[float]:
    """values / denom, each in [0, 1]. Dividing first (instead of multiplying by
    TARGET / denom) cannot overflow, even for subnormal totals."""
    if denom <= 0.0:
        return [0.0] * len(values)
    return [min(v / denom, 1.0) for v in values]


def _total(values: list[float], name: str) -> float:
    """Returns the exact float sum, or raises EvencutError when it does not fit in a float."""
    try:
        total = math.fsum(values)
    except OverflowError:  # fsum reports an intermediate overflow this way
        total = math.inf
    if not math.isfinite(total):
        raise EvencutError(f"evencut: the sum of {name} is too large for floating point; scale the values down")
    return total


def _real_list(values: Sequence[float], name: str) -> list[float]:
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        raise EvencutError(f"evencut: {name} must contain numbers") from None
    for i, v in enumerate(out):
        if not math.isfinite(v) or v < 0:
            raise EvencutError(f"evencut: {name}[{i}] = {v!r} must be finite and non-negative")
    return out


@dataclass(frozen=True)
class PipelinePlan:
    """Stage assignment for pipeline parallelism.

    ``stages[s] = (first_layer, end_layer)`` (end exclusive) indexes the transformer layers
    passed to :func:`pipeline_stages`; the embedding belongs to stage 0 and the output head to
    the last stage. ``stage_costs`` include those extras.
    """

    stages: tuple[tuple[int, int], ...]
    stage_costs: tuple[float, ...]
    stage_memory: Optional[tuple[float, ...]]
    num_layers: int

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def bottleneck(self) -> float:
        """Cost of the slowest stage: every micro-batch waits for it."""
        return max(self.stage_costs)

    @property
    def balance(self) -> float:
        """Mean stage cost / slowest stage cost; 1.0 means perfectly even."""
        return (sum(self.stage_costs) / len(self.stage_costs)) / self.bottleneck if self.bottleneck else 1.0

    def layers_per_stage(self) -> list[int]:
        return [end - start for start, end in self.stages]

    def step_time(self, microbatches: int) -> float:
        """Rough step time of a GPipe/1F1B schedule: (microbatches + stages - 1) * bottleneck."""
        if microbatches < 1:
            raise EvencutError("evencut: microbatches must be >= 1")
        return (microbatches + self.num_stages - 1) * self.bottleneck

    def torchtitan_split_points(self, prefix: str = "layers.") -> list[str]:
        """Value for torchtitan's ``pipeline_parallel_split_points``: the first layer of every
        stage after the first, e.g. ``["layers.9", "layers.17"]``."""
        return [f"{prefix}{start}" for start, _ in self.stages[1:]]

    def megatron_layout(self) -> str:
        """Value for Megatron-Core's ``--pipeline-model-parallel-layout``: ``E`` (embedding),
        ``t`` (transformer layer), ``L`` (loss/head) and ``|`` between stages, e.g.
        ``"Etttttttt|tttttttt|ttttttttL"``. Check the syntax against your Megatron version."""
        parts = []
        for s, (start, end) in enumerate(self.stages):
            text = ("E" if s == 0 else "") + "t" * (end - start) + ("L" if s == self.num_stages - 1 else "")
            parts.append(text)
        return "|".join(parts)

    def deepspeed_parts(self) -> list[int]:
        """Stage boundaries in the format of DeepSpeed ``PipelineModule.parts``
        (``num_stages + 1`` entries; stage s owns layers ``[parts[s], parts[s+1])``).
        DeepSpeed has no public option for explicit boundaries; overriding
        ``PipelineModule._partition_layers`` to set them relies on internal API."""
        return [start for start, _ in self.stages] + [self.num_layers]

    def summary(self) -> str:
        lines = [f"{self.num_stages} stages, bottleneck {self.bottleneck:.6g}, balance {self.balance:.3f}"]
        for s, ((start, end), cost) in enumerate(zip(self.stages, self.stage_costs)):
            mem = f", memory {self.stage_memory[s]:.6g}" if self.stage_memory is not None else ""
            lines.append(f"  stage {s}: layers {start}-{end - 1} ({end - start}), cost {cost:.6g}{mem}")
        return "\n".join(lines)


def _plan_from_bounds(costs: list[float], memory: Optional[list[float]], starts_ends: list[tuple[int, int]],
                      embedding_cost: float, head_cost: float, embedding_memory: float, head_memory: float,
                      ) -> PipelinePlan:
    stage_costs, stage_mem = [], []
    last = len(starts_ends) - 1
    for s, (a, b) in enumerate(starts_ends):
        c = math.fsum(costs[a:b]) + (embedding_cost if s == 0 else 0.0) + (head_cost if s == last else 0.0)
        stage_costs.append(c)
        if memory is not None:
            stage_mem.append(math.fsum(memory[a:b]) + (embedding_memory if s == 0 else 0.0)
                             + (head_memory if s == last else 0.0))
    return PipelinePlan(tuple(starts_ends), tuple(stage_costs), tuple(stage_mem) if memory is not None else None,
                        len(costs))


def pipeline_stages(
    layer_costs: Sequence[float],
    num_stages: int,
    *,
    embedding_cost: float = 0.0,
    head_cost: float = 0.0,
    layer_memory: Optional[Sequence[float]] = None,
    embedding_memory: float = 0.0,
    head_memory: float = 0.0,
    memory_limit: Optional[float] = None,
) -> PipelinePlan:
    """Assigns transformer layers to ``num_stages`` pipeline stages so the slowest stage is as
    fast as possible (exact optimum for the given costs).

    Args:
        layer_costs: measured time (or FLOPs) of each transformer layer, forward + backward.
        num_stages: number of pipeline stages (GPUs in the pipeline dimension).
        embedding_cost, head_cost: extra cost on the first / last stage (embedding, output
            head + loss). Uneven first/last stages are exactly what these extras produce.
        layer_memory, embedding_memory, head_memory: memory of each part (any unit).
        memory_limit: per-stage memory budget in the same unit; requires ``layer_memory``.
            Enforced conservatively: values are rounded to ~1e-12 of their total (layers up,
            the limit down), so a stage within that margin of the limit may be refused, but no
            returned stage ever exceeds it.

    Raises:
        EvencutError: invalid input, or no assignment fits ``memory_limit``.
    """
    costs = _real_list(layer_costs, "layer_costs")
    if not costs:
        raise EvencutError("evencut: layer_costs must not be empty")
    extras = _real_list([embedding_cost, head_cost], "embedding_cost/head_cost")
    n = len(costs)
    # items: [embedding] + layers + [head]; the extras are glued to the first / last layer
    items = [extras[0]] + costs + [extras[1]]
    weights = quantize(_ratios(items, _total(items, "layer_costs")), _TARGET_TOTAL)
    forbidden = [1, n + 1]  # no cut embedding|layer0 (position 1) or layer(n-1)|head (position n+1)
    secondary = None
    limit = None
    memory: Optional[list[float]] = None
    mem_extras = [0.0, 0.0]
    if memory_limit is None and layer_memory is None and (embedding_memory or head_memory):
        raise EvencutError("evencut: embedding_memory/head_memory need layer_memory and memory_limit")
    if memory_limit is not None or layer_memory is not None:
        if memory_limit is None or layer_memory is None:
            raise EvencutError("evencut: layer_memory and memory_limit must be given together")
        memory = _real_list(layer_memory, "layer_memory")
        if len(memory) != n:
            raise EvencutError("evencut: layer_memory must have one value per layer")
        mem_extras = _real_list([embedding_memory, head_memory], "embedding_memory/head_memory")
        mem_items = [mem_extras[0]] + memory + [mem_extras[1]]
        limit_f = _real_list([memory_limit], "memory_limit")[0]
        denom = max(_total(mem_items, "layer_memory"), limit_f)
        # conservative rounding: items up, the limit down, so no stage can exceed the budget
        secondary = [math.ceil(r * _TARGET_TOTAL) for r in _ratios(mem_items, denom)]
        limit = math.floor(_ratios([limit_f], denom)[0] * _TARGET_TOTAL)
    result = partition(weights, num_stages, objective="min_max", forbidden_cuts=forbidden,
                       secondary=secondary, secondary_limit=limit)
    stages = []
    for begin, end in result.segments():
        first = max(begin - 1, 0)       # item i+1 is layer i
        last = min(end - 1, n)          # exclusive, head item excluded
        stages.append((first, last))
    return _plan_from_bounds(costs, memory, stages, extras[0], extras[1], mem_extras[0], mem_extras[1])


def uniform_stages(layer_costs: Sequence[float], num_stages: int, *, embedding_cost: float = 0.0,
                   head_cost: float = 0.0) -> PipelinePlan:
    """The common default for comparison: equal layer counts per stage (the first
    ``num_layers % num_stages`` stages get one extra layer)."""
    costs = _real_list(layer_costs, "layer_costs")
    extras = _real_list([embedding_cost, head_cost], "embedding_cost/head_cost")
    _total(costs + extras, "layer_costs")
    n, p = len(costs), num_stages
    if isinstance(p, bool) or not isinstance(p, numbers.Integral) or not 1 <= p <= n:
        raise EvencutError("evencut: num_stages must be an integer between 1 and the number of layers")
    base, extra = divmod(n, p)
    stages, start = [], 0
    for s in range(p):
        size = base + (1 if s < extra else 0)
        stages.append((start, start + size))
        start += size
    return _plan_from_bounds(costs, None, stages, extras[0], extras[1], 0.0, 0.0)


def balanced_shards(sample_tokens: Sequence[int], num_ranks: int, *, objective: str = "sum_squares") -> Partition:
    """Splits an ordered dataset into ``num_ranks`` contiguous shards with even token counts.

    Sort the samples by length first if you also want each shard to hold similar lengths
    (less padding); the shards then become contiguous length ranges.
    """
    return partition(sample_tokens, num_ranks, objective=objective)


def token_budget_batches(sample_tokens: Sequence[int], max_tokens: int) -> Partition:
    """Fewest consecutive batches whose token sum stays within ``max_tokens``, then evened out
    so no batch is much fuller than the others (the largest batch is as small as possible).
    """
    tokens = [int(t) if isinstance(t, numbers.Integral) and not isinstance(t, bool) else t for t in sample_tokens]
    if not tokens:
        raise EvencutError("evencut: sample_tokens must not be empty")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, numbers.Integral) or max_tokens < 1:
        raise EvencutError("evencut: max_tokens must be an integer >= 1")
    batches, current = 1, 0
    for i, t in enumerate(tokens):
        if isinstance(t, bool) or not isinstance(t, numbers.Integral) or t < 0:
            raise EvencutError(f"evencut: sample_tokens[{i}] must be a non-negative integer")
        if t > max_tokens:
            raise EvencutError(f"evencut: sample {i} has {t} tokens, more than max_tokens={max_tokens}")
        if current + t > max_tokens:
            batches += 1
            current = t
        else:
            current += t
    result = partition(tokens, batches, objective="min_max")
    assert result.max_load <= max_tokens  # the greedy count is feasible, so the optimum is too
    return result


@dataclass(frozen=True)
class PaddedSplit:
    """Contiguous groups of sequences, each group padded to its longest member.

    ``ends[g]`` is the exclusive end of group ``g`` (like :attr:`Partition.ends`), ``padded[g]``
    = members x longest member (what the GPU actually computes), ``tokens[g]`` = real tokens,
    ``costs[g]`` = the cost the split was balanced on (``padded`` unless a cost function was given).
    """

    ends: tuple[int, ...]
    padded: tuple[int, ...]
    tokens: tuple[int, ...]
    costs: tuple[int, ...]

    @property
    def parts(self) -> int:
        return len(self.ends)

    @property
    def max_padded(self) -> int:
        """The largest padded group: the biggest memory peak."""
        return max(self.padded)

    @property
    def max_cost(self) -> int:
        """The most expensive group under the cost the split was made with (the slowest GPU)."""
        return max(self.costs)

    @property
    def waste(self) -> float:
        """Share of the padded tokens that are padding (0.0 = none)."""
        total = sum(self.padded)
        return 1.0 - sum(self.tokens) / total if total else 0.0

    def segments(self) -> list[tuple[int, int]]:
        return list(zip((0, *self.ends[:-1]), self.ends))


def _length_list(lengths: Sequence[int]) -> list[int]:
    try:
        out = list(lengths)
    except TypeError:
        raise EvencutError("evencut: lengths must be a sequence of integers") from None
    if not out:
        raise EvencutError("evencut: lengths must not be empty")
    for i, v in enumerate(out):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
            raise EvencutError(f"evencut: lengths[{i}] = {v!r} must be a non-negative integer")
        out[i] = int(v)
    return out


CostFn = Callable[[int, int], int]


def _padded_cost(count: int, longest: int) -> int:
    return count * longest


def _checked(cost: CostFn) -> CostFn:
    def run(count: int, longest: int) -> int:
        v = cost(count, longest)
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
            raise EvencutError(f"evencut: cost({count}, {longest}) returned {v!r}; it must be a non-negative integer")
        return int(v)
    return run


def _padded_greedy(lengths: list[int], bound: int, stop_after: int, cost: CostFn = _padded_cost
                   ) -> Optional[list[int]]:
    """Ends of the fewest groups whose cost stays <= bound, or None if more than stop_after are
    needed. Extending a group never lowers its cost (members and longest member only grow, and
    the cost is non-decreasing in both), so taking every group as far as it goes needs the
    fewest groups (standard exchange argument)."""
    ends: list[int] = []
    count = longest = 0
    for i, x in enumerate(lengths):
        top = x if x > longest else longest
        if count and cost(count + 1, top) > bound:
            ends.append(i)
            if len(ends) >= stop_after:
                return None
            count, top = 0, x
        if cost(1, x) > bound:
            return None
        count += 1
        longest = top
    ends.append(len(lengths))
    return ends


def _make_split(lengths: list[int], ends: list[int], cost: CostFn = _padded_cost) -> PaddedSplit:
    padded, tokens, costs, prev = [], [], [], 0
    for e in ends:
        group = lengths[prev:e]
        padded.append(len(group) * max(group))
        tokens.append(sum(group))
        costs.append(cost(len(group), max(group)))
        prev = e
    return PaddedSplit(tuple(ends), tuple(padded), tuple(tokens), tuple(costs))


def padded_slices(lengths: Sequence[int], num_parts: int, *, cost: Optional[CostFn] = None) -> PaddedSplit:
    """Splits sequences, in order, into ``num_parts`` contiguous groups that are each padded to
    their longest member, so that the most expensive group is as cheap as possible (exact).

    Typical use: one group per data-parallel GPU; the step waits for the slowest. By default a
    group costs members x longest, the padded tokens it computes. Balancing real tokens
    (``partition``) ignores the padding; this does not. Sort the sequences by length first
    (length-grouped batching) to keep the padding itself small.

    Args:
        cost: optional ``cost(members, longest) -> int`` used instead, for example to add the
            attention term of long sequences: ``lambda n, L: n * L * (A + L)`` with
            ``A = (forward FLOPs per token without attention) / (2 * layers * width)``. It
            must return non-negative integers and never decrease when either argument grows.
    """
    w = _length_list(lengths)
    if isinstance(num_parts, bool) or not isinstance(num_parts, numbers.Integral) or num_parts < 1:
        raise EvencutError("evencut: num_parts must be an integer >= 1")
    if cost is not None and not callable(cost):
        raise EvencutError("evencut: cost must be a function cost(members, longest) -> int")
    fn = _padded_cost if cost is None else _checked(cost)
    k = min(int(num_parts), len(w))
    lo = max(fn(1, x) for x in w)                # every item sits in some group
    if cost is None:
        lo = max(lo, -(-sum(w) // k))            # padded tokens cover the real ones
    hi = fn(len(w), max(w))                      # everything in one group
    while lo < hi:                               # smallest bound that k groups can meet
        mid = (lo + hi) // 2
        if _padded_greedy(w, mid, k, fn) is not None:
            hi = mid
        else:
            lo = mid + 1
    ends = _padded_greedy(w, lo, k, fn)
    assert ends is not None and len(ends) <= k
    # fewer groups than asked: split the costliest ones (splitting never raises a group's cost)
    while len(ends) < k:
        split = _make_split(w, ends, fn)
        g = max((g for g in range(split.parts) if split.ends[g] - (split.ends[g - 1] if g else 0) > 1),
                key=lambda g: split.costs[g])
        start = split.ends[g - 1] if g else 0
        ends.insert(g, (start + split.ends[g] + 1) // 2)
    return _make_split(w, ends, fn)


def padded_batches(lengths: Sequence[int], max_padded_tokens: int) -> PaddedSplit:
    """Fewest consecutive batches whose padded size (members x longest) stays within
    ``max_padded_tokens``, then evened out so the largest batch is as small as possible.

    Sort by length first to keep the padding small. Unlike :func:`token_budget_batches`, the
    budget applies to what the GPU actually holds, padding included.
    """
    w = _length_list(lengths)
    if isinstance(max_padded_tokens, bool) or not isinstance(max_padded_tokens, numbers.Integral) \
            or max_padded_tokens < 1:
        raise EvencutError("evencut: max_padded_tokens must be an integer >= 1")
    for i, x in enumerate(w):
        if x > max_padded_tokens:
            raise EvencutError(f"evencut: sequence {i} has {x} tokens, more than max_padded_tokens")
    fewest = _padded_greedy(w, int(max_padded_tokens), len(w) + 1)
    assert fewest is not None
    result = padded_slices(w, len(fewest))
    assert result.max_padded <= max_padded_tokens
    return result


def causal_attention_chunks(seq_len: int, num_ranks: int, *, per_key_cost: int = 1,
                            per_token_cost: int = 0) -> Partition:
    """Splits a sequence of ``seq_len`` tokens into ``num_ranks`` contiguous chunks with even
    causal-attention cost.

    Token ``i`` (0-based) attends to ``i + 1`` keys, so its cost is modelled as
    ``per_token_cost + per_key_cost * (i + 1)``: later tokens are dearer and later chunks come
    out shorter. Communication is not modelled.
    """
    if isinstance(seq_len, bool) or not isinstance(seq_len, numbers.Integral) or seq_len < 1:
        raise EvencutError("evencut: seq_len must be an integer >= 1")
    for name, v in (("per_key_cost", per_key_cost), ("per_token_cost", per_token_cost)):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
            raise EvencutError(f"evencut: {name} must be a non-negative integer")
    weights = [per_token_cost + per_key_cost * (i + 1) for i in range(seq_len)]
    return partition(weights, num_ranks, objective="min_max")
