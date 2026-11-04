"""A drop-in batch sampler for data-parallel training that wastes less GPU time.

Replace ``DataLoader(ds, sampler=DistributedSampler(ds), batch_size=8)`` with::

    from evencut.train import BalancedBatchSampler
    sampler = BalancedBatchSampler(lengths, per_device_batch=8, accum=4)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=pad_to_longest)

and call ``sampler.set_epoch(epoch)`` each epoch, as with DistributedSampler. ``lengths[i]`` is
the token count of ``ds[i]``; the collate function must pad each batch to its own longest
sequence (dynamic padding), otherwise there is nothing to save.

Every rank builds the same plan from the seed and the epoch, without communicating, and yields
its own batches. In the default ``"window"`` mode every optimizer step holds exactly the sequences
a shuffled DistributedSampler run would put in it; only which GPU gets which changes, so training
statistics stay the same. ``"pooled"`` also groups similar lengths into the same steps (as Hugging
Face's ``group_by_length`` does); it pads less, but its many short-sequence batches can run below
what the GPU needs to be efficient, and in our RTX 4060 measurements it was slower than
``"window"`` (bench/results/gpu_batches_wikipedia_v2.md, gpu_batches_dolly_accum8.md). Run
``evencut-train diagnose`` first to see what it saves on your data.

Per-GPU batches differ in size, so a loss averaged per GPU (``loss.mean()``) gives sequences on
small batches more weight than DDP's average should. Normalise by the step's global token count
instead; :meth:`BalancedBatchSampler.loss_scale` gives the factor without communication::

    loss = token_loss.sum() * sampler.loss_scale(i)          # i: micro-batch index in the epoch

The Hugging Face Trainer already does this (``num_items_in_batch``); see :mod:`evencut.train.hf`.
In PyTorch Lightning pass ``Trainer(use_distributed_sampler=False)``, and use
``sampler.loss_scale(batch_idx, accum_divided=True)``: Lightning divides the loss by
``accumulate_grad_batches`` itself. With Accelerate, prepare the loader with
:func:`accelerate_prepare` (plain ``accelerator.prepare`` would split the batches again) and use
``accum_divided=True`` too: ``accelerator.backward`` divides by the accumulation steps.
"""

from __future__ import annotations

import os
import random
from typing import Iterator, List, Optional, Sequence

from .. import EvencutError
from . import _plans as P

MODES = ("pooled", "window", "step", "sorted")


class _EpochForwarder:
    """``batch_sampler.sampler``: some trainers (PyTorch Lightning) call ``set_epoch`` only on a
    DataLoader's sampler and on its batch sampler's ``.sampler``, never on the batch sampler
    itself. This forwards it, so the epochs still reshuffle. Iterating gives the epoch order."""

    def __init__(self, owner: "BalancedBatchSampler") -> None:
        self._owner = owner

    def set_epoch(self, epoch: int) -> None:
        self._owner.set_epoch(epoch)

    def __iter__(self) -> Iterator[int]:
        return iter(self._owner.order())

    def __len__(self) -> int:
        return self._owner.steps * self._owner.accum * self._owner.global_batch


def _dist_info(rank: Optional[int], world_size: Optional[int]) -> tuple[int, int]:
    """rank / world size: as given, else from torch.distributed if initialised, else from the
    RANK / WORLD_SIZE variables torchrun sets, else a single process."""
    if rank is None or world_size is None:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank() if rank is None else rank
                world_size = dist.get_world_size() if world_size is None else world_size
        except ImportError:
            pass
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise EvencutError("evencut: world_size must be an integer >= 1")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise EvencutError(f"evencut: rank must be an integer in [0, {world_size})")
    return rank, world_size


def _length_list(lengths: Sequence[int], max_len: Optional[int]) -> list[int]:
    try:
        out = list(lengths)
    except TypeError:
        raise EvencutError("evencut: lengths must be a sequence of integers") from None
    for i, v in enumerate(out):
        if isinstance(v, bool):
            raise EvencutError(f"evencut: lengths[{i}] = {v!r} is not an integer")
        try:
            iv = int(v)
        except (TypeError, ValueError, OverflowError):
            raise EvencutError(f"evencut: lengths[{i}] = {v!r} is not an integer") from None
        if iv != v or iv < 0:
            raise EvencutError(f"evencut: lengths[{i}] = {v!r} must be a non-negative integer")
        out[i] = min(iv, max_len) if max_len is not None else iv
    return out


class BalancedBatchSampler:
    """Yields, for this rank, lists of dataset indices: one list per micro-step.

    Args:
        lengths: token count of every dataset item (cut at ``max_len`` if given).
        per_device_batch / global_batch: sequences per micro-step on one GPU / on all GPUs
            (give one; global = per-device x world size).
        accum: gradient accumulation steps; an optimizer step is ``accum`` micro-steps.
        mode: ``"window"`` (default): keep every optimizer step's sequences exactly as a random
            order has them and balance the GPUs over the whole step (each micro-step on its own
            when ``accum == 1`` or with ``sync_every_microstep``). ``"step"``: balance each
            micro-step on its own. ``"pooled"``: sort a pool of optimizer steps by length, form
            the steps from it, balance each step. ``"sorted"``: sort the whole epoch into
            max-token batches (fairseq-style; sequences per step vary).
        pool_steps: optimizer steps per pool in ``"pooled"`` mode (default ``50 // GPUs``, the
            mixing of Hugging Face's ``group_by_length``; 1 keeps every step's sequences).
        sync_every_microstep: True for FSDP / ZeRO-3 (GPUs wait for each other at every
            micro-step), so each micro-step is balanced on its own.
        cost: optional ``cost(members, longest) -> int`` to balance on, e.g.
            :class:`evencut.train.AttentionCost` for long sequences; default padded tokens.
        rank, world_size: default from torch.distributed / torchrun's environment.
        seed, shuffle, drop_last: as DistributedSampler. With ``drop_last=False`` the last
            optimizer step is filled up with sequences from the start of the epoch.
    """

    def __init__(self, lengths: Sequence[int], global_batch: Optional[int] = None, *,
                 per_device_batch: Optional[int] = None, accum: int = 1, mode: str = "window",
                 pool_steps: Optional[int] = None, sync_every_microstep: bool = False,
                 cost: Optional[P.CostFn] = None, max_len: Optional[int] = None,
                 rank: Optional[int] = None, world_size: Optional[int] = None, seed: int = 0,
                 shuffle: bool = True, drop_last: bool = True) -> None:
        self.rank, self.world_size = _dist_info(rank, world_size)
        if (global_batch is None) == (per_device_batch is None):
            raise EvencutError("evencut: give exactly one of global_batch and per_device_batch")
        for name, v in (("global_batch", global_batch), ("per_device_batch", per_device_batch), ("accum", accum),
                        ("pool_steps", pool_steps), ("max_len", max_len)):
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise EvencutError(f"evencut: {name} must be an integer >= 1")
        if mode not in MODES:
            raise EvencutError(f"evencut: mode must be one of {', '.join(MODES)}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvencutError("evencut: seed must be an integer")
        if cost is not None and not callable(cost):
            raise EvencutError("evencut: cost must be a function cost(members, longest) -> int")
        self.global_batch = global_batch if global_batch is not None else per_device_batch * self.world_size
        if self.global_batch < self.world_size:
            raise EvencutError("evencut: the global batch needs at least one sequence per GPU")
        self.lengths = _length_list(lengths, max_len)
        self.accum, self.mode, self.cost, self.seed = accum, mode, cost, seed
        self.sync_every_microstep, self.shuffle, self.drop_last = bool(sync_every_microstep), shuffle, drop_last
        self.pool_steps = pool_steps or P.default_pool_steps(self.world_size)
        n, per = len(self.lengths), self.accum * self.global_batch
        self.steps = n // per if drop_last else -(-n // per)
        if self.steps < 1:
            raise EvencutError(f"evencut: {n} sequences do not fill one optimizer step of {per}")
        self.epoch = 0
        self._cache: Optional[tuple[int, P.Plan]] = None
        self.sampler = _EpochForwarder(self)

    # -- the DistributedSampler-like interface --------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """Micro-batches per epoch on this rank; the same on every rank (DDP needs that)."""
        return self.steps * self.accum

    def __iter__(self) -> Iterator[List[int]]:
        for micro in self.plan():
            yield list(micro[self.rank])

    # -- inspection and loss normalisation --------------------------------------------------------

    def order(self, epoch: Optional[int] = None) -> list[int]:
        """The epoch's sequence order before batching (identical on every rank)."""
        e = self.epoch if epoch is None else int(epoch)
        n, per = len(self.lengths), self.accum * self.global_batch
        order = list(range(n))
        if self.shuffle:
            random.Random(P.epoch_seed(self.seed, e)).shuffle(order)
        need = self.steps * per
        if need <= n:
            return order[:need]
        return order + (order * (-(-(need - n) // n)))[: need - n]   # drop_last=False: wrap around

    def plan(self, epoch: Optional[int] = None) -> P.Plan:
        """Every rank's batches for the epoch: plan[micro_step][rank] -> dataset indices."""
        e = self.epoch if epoch is None else int(epoch)
        if self._cache is not None and self._cache[0] == e:
            return self._cache[1]
        order, lens, k, g, m = self.order(e), self.lengths, self.world_size, self.global_batch, self.accum
        whole_step = m > 1 and not self.sync_every_microstep
        if self.mode == "pooled":
            plan = P.pooled(order, lens, k, g, m, pool_steps=self.pool_steps, seed=P.epoch_seed(self.seed, e),
                            balance="window" if whole_step else "step", cost=self.cost)
        elif self.mode == "window" and whole_step:
            plan = P.window(order, lens, k, g, m, cost=self.cost)
        elif self.mode in ("window", "step"):
            plan = P.per_step(order, lens, k, g, "padded", grouped=True, cost=self.cost)
        else:
            plan = P.sorted_dataset(order, lens, k, self.steps * m, P.epoch_seed(self.seed, e), cost=self.cost)
        self._cache = (e, plan)
        return plan

    def global_tokens(self, step: int, epoch: Optional[int] = None) -> int:
        """Real tokens (lengths, cut at max_len) of optimizer step ``step`` over all GPUs: the
        denominator of a token-averaged loss, known on every rank without communication."""
        plan = self.plan(epoch)
        if not 0 <= step < self.steps:
            raise EvencutError(f"evencut: step must be in [0, {self.steps})")
        window = plan[step * self.accum:(step + 1) * self.accum]
        return sum(self.lengths[i] for micro in window for b in micro for i in b)

    def loss_scale(self, micro_step: int, *, accum_divided: bool = False, epoch: Optional[int] = None) -> float:
        """Multiply a token-summed loss by this to train on the token mean of the whole optimizer
        step, as one big batch would: ``world_size / global_tokens(micro_step // accum)``, since
        DDP averages gradients over the GPUs. ``accum_divided=True`` when the framework divides
        the loss by the accumulation steps itself (PyTorch Lightning does)."""
        tokens = self.global_tokens(micro_step // self.accum, epoch)
        if tokens == 0:
            return 0.0
        return self.world_size * (self.accum if accum_divided else 1) / tokens

    def global_sequences(self, step: int, epoch: Optional[int] = None) -> int:
        """Sequences in optimizer step ``step`` over all GPUs (constant except in "sorted" mode)."""
        plan = self.plan(epoch)
        if not 0 <= step < self.steps:
            raise EvencutError(f"evencut: step must be in [0, {self.steps})")
        return sum(len(b) for micro in plan[step * self.accum:(step + 1) * self.accum] for b in micro)

    def stats(self, epoch: Optional[int] = None) -> P.PlanStats:
        """What the epoch's plan costs (padding, waiting): see :class:`evencut.train.PlanStats`."""
        return P.evaluate(self.plan(epoch), self.lengths, self.world_size, self.accum, cost=self.cost,
                          sync="microstep" if self.sync_every_microstep else "window")

    def __repr__(self) -> str:
        return (f"BalancedBatchSampler(mode={self.mode!r}, rank={self.rank}, world_size={self.world_size}, "
                f"global_batch={self.global_batch}, accum={self.accum}, steps={self.steps})")


def global_count(local: float, device=None) -> float:
    """Sum of ``local`` over all ranks (one all-reduce). For loss denominators the sampler cannot
    know, such as label tokens after masking. Needs torch.distributed initialised."""
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return float(local)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else "cpu"
    t = torch.tensor([float(local)], device=device, dtype=torch.float64)
    dist.all_reduce(t)
    return float(t.item())


def accelerate_prepare(accelerator, loader):
    """``accelerator.prepare(loader)`` for a DataLoader whose batch_sampler is a
    :class:`BalancedBatchSampler`. Accelerate wraps batch samplers to split every batch over the
    processes; this sampler already gives each process its own batches, so the wrapper is made a
    pass-through (the Hugging Face Trainer does the same for its own rank-aware sampler)."""
    prepared = accelerator.prepare(loader)
    wrapper = getattr(prepared, "batch_sampler", None)
    if wrapper is not None and isinstance(getattr(wrapper, "batch_sampler", None), BalancedBatchSampler):
        wrapper.num_processes = 1
        wrapper.process_index = 0
    return prepared
