"""Synthetic AI-training scenarios: what balancing buys compared with common defaults.

All workloads below are modelled, not measured on GPUs. The assumptions are written next to the
numbers so they can be replaced by real profiles. Run:  python bench/ai_scenarios.py
"""

from __future__ import annotations

import math
import random
import time

import evencut
from evencut import ai


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0


def scenario_pipeline() -> list[str]:
    # Llama-3-8B-like decoder: hidden 4096, FFN 14336, GQA (8 KV heads), vocab 128256, seq 8192.
    h, ffn, kv, vocab, seq = 4096, 14336, 1024, 128256, 8192
    layer_params = h * (h + kv + kv + h) + 3 * h * ffn             # ~218M
    layer_flops = 2 * layer_params + 2 * h * seq                     # per token, forward, causal attention
    head_flops = 2 * h * vocab                                       # logits
    layer_cost = 1.0                                                 # cost unit = one decoder layer
    head_cost = head_flops / layer_flops + 0.1                       # + softmax / cross-entropy (memory bound)
    embedding_cost = 0.01                                            # a gather: almost free
    emb_params = head_params = h * vocab                             # untied embedding and head (~525M each)
    rows = ["| stages | method | layers per stage | slowest stage | step time (32 micro-batches) | vs equal layers |",
            "|--:|---|---|--:|--:|--:|"]
    for p in (2, 4, 8):
        costs = [layer_cost] * 32
        uniform = ai.uniform_stages(costs, p, embedding_cost=embedding_cost, head_cost=head_cost)
        # DeepSpeed partition_method="parameters" balances parameter counts over the layer list
        # [embedding, 32 decoder layers, head]; evencut on the same counts stands in for it here.
        par = evencut.partition([emb_params] + [layer_params] * 32 + [head_params], p)
        stages = []
        for begin, end in par.segments():
            stages.append((max(begin - 1, 0), min(end - 1, 32)))
        params_plan = ai._plan_from_bounds(costs, None, stages, embedding_cost, head_cost, 0.0, 0.0)
        best, ms = timed(ai.pipeline_stages, costs, p, embedding_cost=embedding_cost, head_cost=head_cost)
        base = uniform.step_time(32)
        for name, plan in (("equal layers (Megatron default)", uniform),
                           ("parameter count (DeepSpeed default)", params_plan),
                           (f"evencut ({ms:.2f} ms)", best)):
            rows.append(f"| {p} | {name} | {plan.layers_per_stage()} | {plan.bottleneck:.2f} | "
                        f"{plan.step_time(32):.1f} | {base / plan.step_time(32):.3f}x |")
    note = (f"Cost unit: one decoder layer (fwd FLOPs per token {layer_flops / 1e6:.0f}M). Head = {head_cost:.2f} "
            f"units (logits {head_flops / 1e6:.0f}M FLOPs + loss), embedding = {embedding_cost} units. "
            "Step time = (micro-batches + stages - 1) x slowest stage (GPipe/1F1B fill-drain).")
    return ["### A. Pipeline stages (Llama-3-8B-like, 32 layers)", "", note, "", *rows]


def scenario_shards() -> list[str]:
    rng = random.Random(7)
    lengths = [min(32768, max(16, int(rng.lognormvariate(math.log(600), 1.0)))) for _ in range(100_000)]
    ranks = 8
    rows = ["| data order | split | max tokens per rank | max / mean | step-time gain |", "|---|---|--:|--:|--:|"]
    for order_name, data in (("sorted by length", sorted(lengths)), ("shuffled", lengths)):
        n = len(data)
        equal_count = evencut.evaluate(data, [n * (r + 1) // ranks for r in range(ranks)])
        balanced, ms = timed(ai.balanced_shards, data, ranks)
        for name, r in (("equal sample counts", equal_count), (f"evencut ({ms:.1f} ms)", balanced)):
            rows.append(f"| {order_name} | {name} | {r.max_load:,} | {r.imbalance:.3f} | "
                        f"{equal_count.max_load / r.max_load:.2f}x |")
    note = ("100,000 samples, lengths log-normal (median 600 tokens, capped at 32k), 8 data-parallel ranks. "
            "Every rank waits for the one with the most tokens, so step time follows the maximum.")
    return ["### B. Token-balanced data-parallel shards", "", note, "", *rows]


def scenario_attention() -> list[str]:
    seq, ranks = 131072, 8
    weights = [i + 1 for i in range(seq)]                     # token i attends to i + 1 keys
    equal = evencut.evaluate(weights, [seq * (r + 1) // ranks for r in range(ranks)])
    balanced, ms = timed(ai.causal_attention_chunks, seq, ranks)
    lengths = [b - a for a, b in balanced.segments()]
    rows = ["| split | chunk lengths | max / mean attention cost |", "|---|---|--:|",
            f"| equal lengths | {[seq // ranks] * ranks} | {equal.imbalance:.3f} |",
            f"| evencut ({ms:.1f} ms) | {lengths} | {balanced.imbalance:.4f} |"]
    note = ("A 128k-token sequence over 8 context-parallel ranks, causal attention cost only (token i "
            "attends to i+1 keys); communication not modelled. Production systems often use a zig-zag "
            "layout instead; this split is for cases that need contiguous chunks.")
    return ["### C. Causal-attention chunks (context parallelism)", "", note, "", *rows]


def main() -> None:
    print(f"evencut {evencut.__version__}, backend: {evencut.backend}\n")
    for block in (scenario_pipeline(), scenario_shards(), scenario_attention()):
        print("\n".join(block))
        print()


if __name__ == "__main__":
    main()
