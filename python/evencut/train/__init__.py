"""Find and fix wasted GPU time in data-parallel training.

* :mod:`evencut.train.diagnose`  how much GPU time padding and waiting cost (no GPU needed)
* :class:`BalancedBatchSampler`  the fix: a drop-in batch sampler for DistributedSampler
* :mod:`evencut.train.hf`  the same for the Hugging Face Trainer
* ``evencut-train`` / ``python -m evencut.train``  the command line

The batch plans themselves live in :mod:`evencut.train._plans`; the diagnosis and the training
sampler share them, so what the diagnosis predicts is what the sampler does.
"""

from __future__ import annotations

from ._plans import AttentionCost, PlanStats, attention_constant, evaluate, padded_cost
from .sampler import BalancedBatchSampler, accelerate_prepare, global_count

__all__ = ["AttentionCost", "BalancedBatchSampler", "PlanStats", "accelerate_prepare", "attention_constant", "evaluate",
           "global_count", "padded_cost"]
