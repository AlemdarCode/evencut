"""Evencut batching for the Hugging Face Trainer (tested with transformers 5.22).

    from evencut.train.hf import EvencutTrainer
    trainer = EvencutTrainer(model=model, args=args, train_dataset=ds, data_collator=collator)  # was: Trainer(...)

Everything else stays: ``per_device_train_batch_size``, ``gradient_accumulation_steps``, the seed
and ``dataloader_drop_last`` come from the TrainingArguments. The collator must pad each batch
to its own longest sequence (``DataCollatorWithPadding``, ``DataCollatorForSeq2Seq``, or
``padding="longest"``), not to a fixed length.

Lengths come from the ``length_column_name`` column when the dataset has it (fastest), else
from the first model input (``input_ids``) of every row, or pass ``evencut_lengths=[...]``.

Why a subclass: Accelerate wraps a DataLoader's batch sampler to split batches over the GPUs.
This sampler already hands every GPU its own batches, so the wrapper is turned into a
pass-through, exactly as the Trainer does for its own ``batch_rebalance`` sampler. Variable
per-GPU batch sizes need a loss averaged over the step's tokens, which the Trainer already does
(``average_tokens_across_devices``, default on) when the model accepts ``num_items_in_batch``.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import seed_worker

from .. import EvencutError
from .sampler import BalancedBatchSampler

try:
    import datasets
except ImportError:  # the datasets package is optional
    datasets = None


class EvencutTrainer(Trainer):
    """``transformers.Trainer`` whose training batches come from :class:`BalancedBatchSampler`.

    Extra keyword arguments (all optional): ``evencut_mode`` ("window" (default), "step",
    "pooled", "sorted"), ``evencut_pool_steps``, ``evencut_cost``, ``evencut_max_len`` and
    ``evencut_lengths``. FSDP and DeepSpeed ZeRO-3 runs balance every micro-step on its own.
    """

    def __init__(self, *args, evencut_mode: str = "window", evencut_pool_steps: Optional[int] = None,
                 evencut_cost=None, evencut_max_len: Optional[int] = None,
                 evencut_lengths: Optional[Sequence[int]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evencut_mode = evencut_mode
        self.evencut_pool_steps = evencut_pool_steps
        self.evencut_cost = evencut_cost
        self.evencut_max_len = evencut_max_len
        self.evencut_lengths = None if evencut_lengths is None else list(evencut_lengths)
        self.evencut_sampler: Optional[BalancedBatchSampler] = None

    def _evencut_dataset_lengths(self, dataset) -> list:
        if self.evencut_lengths is not None:
            if len(self.evencut_lengths) != len(dataset):
                raise EvencutError("evencut: evencut_lengths needs one length per training example")
            return self.evencut_lengths
        column = self.args.length_column_name
        if datasets is not None and isinstance(dataset, datasets.Dataset) and column in dataset.column_names:
            return list(dataset[column])
        names = getattr(self.processing_class, "model_input_names", None) or ["input_ids"]
        first = dataset[0]
        if not hasattr(first, "keys") or names[0] not in first:
            raise EvencutError(f"evencut: cannot find lengths; add a {column!r} column or pass evencut_lengths")
        return [len(dataset[i][names[0]]) for i in range(len(dataset))]

    def _evencut_every_microstep(self) -> bool:
        if self.is_fsdp_enabled:
            return True
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        return plugin is not None and getattr(plugin, "zero_stage", 0) == 3

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.args.n_gpu > 1:
            raise EvencutError("evencut: one GPU per process (DDP / torchrun) is needed, not torch.nn.DataParallel")
        dataset = self.train_dataset
        lengths = self._evencut_dataset_lengths(dataset)
        collator = self.data_collator
        if datasets is not None and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description="Training")
        else:
            collator = self._get_collator_with_removed_columns(collator, description="Training")
        sampler = BalancedBatchSampler(
            lengths, per_device_batch=self.args.per_device_train_batch_size,
            accum=self.args.gradient_accumulation_steps, mode=self.evencut_mode, pool_steps=self.evencut_pool_steps,
            sync_every_microstep=self._evencut_every_microstep(), cost=self.evencut_cost,
            max_len=self.evencut_max_len, rank=self.args.process_index, world_size=max(1, self.args.world_size),
            seed=self.args.seed, drop_last=self.args.dataloader_drop_last)
        self.evencut_sampler = sampler
        params = dict(batch_sampler=sampler, collate_fn=collator, num_workers=self.args.dataloader_num_workers,
                      pin_memory=self.args.dataloader_pin_memory)
        if self.args.dataloader_num_workers > 0:
            params.update(persistent_workers=self.args.dataloader_persistent_workers,
                          prefetch_factor=self.args.dataloader_prefetch_factor,
                          worker_init_fn=partial(seed_worker, num_workers=self.args.dataloader_num_workers,
                                                 rank=self.args.process_index))
        loader = self.accelerator.prepare(DataLoader(dataset, **params))
        wrapper = getattr(loader, "batch_sampler", None)
        if wrapper is not None and getattr(wrapper, "batch_sampler", None) is sampler:
            wrapper.num_processes = 1                 # already per-GPU: do not split again
            wrapper.process_index = 0
        return loader
