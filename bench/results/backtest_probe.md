# Backtest: does a one-minute probe predict step times?

2026-11-02. Produced by `bench/backtest_probe.py` from the probe files small: NVIDIA GeForce RTX 4060 (2026-11-02), base: NVIDIA GeForce RTX 4060 (2026-11-02), large: NVIDIA GeForce RTX 4060 (2026-11-02), 7b: NVIDIA GeForce RTX 4060 (2026-11-02). *probe* = predicted from the probe's time curve with no measurement of the run; *cost model* = the no-probe model scaled so the default row is exact (it needs one measured step). Errors are medians over the rows of a run; *speed-up* = error of the predicted speed-up over the default row; *regret* = how much slower the predicted-best split measured than the measured-best one (same-steps splits / all splits).

## A. Same machine: RTX 4060 probe vs the RTX 4060 simulation runs

| run | GPUs | model | step time error (probe) | speed-up error: probe | cost model | regret: probe | cost model |
|---|--:|---|--:|--:|--:|--:|--:|
| gpu_batches | 8 x 4060 | small | 2% (worst 4%) | 2% | 3% | 0% / 0% | 0% / 0% |
| gpu_batches_dolly | 8 x 4060 | small | 2% (worst 9%) | 3% | 2% | 0% / 0% | 0% / 0% |
| gpu_batches_dolly_accum8 | 8 x 4060 | small | 6% (worst 13%) | 10% | 14% | 0% / 0% | 0% / 12% |
| gpu_batches_wikipedia | 8 x 4060 | small | 3% (worst 11%) | 3% | 6% | 0% / 0% | 0% / 0% |
| gpu_batches_wikipedia_v2 | 8 x 4060 | small | 3% (worst 14%) | 6% | 2% | 0% / 0% | 0% / 14% |

Median over runs: step time error 3%; speed-up error probe 3% vs cost model 3%.

## B. Not probed: the 4060 probe scaled by datasheet TFLOPS vs real DDP runs

| run | GPUs | model | step time error (probe) | speed-up error: probe | cost model | regret: probe | cost model |
|---|--:|---|--:|--:|--:|--:|--:|
| ddp_a100x4 | 4 x A100-SXM4-80GB | small | 6% (worst 20%) | 14% | 4% | 7% / 28% | 0% / 1% |
| ddp_a100x4_accum8 | 4 x A100-SXM4-80GB | small | 14% (worst 32%) | 13% | 1% | 4% / 30% | 0% / 1% |
| ddp_a100x4_base_accum4 | 4 x A100-SXM4-80GB | base | 47% (worst 47%) | 5% | 3% | 0% / 10% | 0% / 0% |
| ddp_a100x4_dolly | 4 x A100-SXM4-80GB | small | 20% (worst 42%) | 44% | 31% | 0% / 52% | 0% / 0% |
| ddp_a100x4_dolly_accum8 | 4 x A100-SXM4-80GB | small | 25% (worst 43%) | 35% | 18% | 0% / 58% | 0% / 1% |
| ddp_a100x4_dolly_base_accum4 | 4 x A100-SXM4-80GB | base | 43% (worst 48%) | 9% | 10% | 0% / 41% | 0% / 0% |
| ddp_a100x4_dolly_lpt | 4 x A100-SXM4-80GB | small | 23% (worst 42%) | 39% | 24% | 0% / 56% | 0% / 0% |
| ddp_a100x4_lpt | 4 x A100-SXM4-80GB | small | 6% (worst 21%) | 11% | 3% | 7% / 29% | 0% / 0% |
| ddp_a100x4_wikipedia | 4 x A100-SXM4-80GB | small | 23% (worst 38%) | 25% | 18% | 20% / 93% | 0% / 1% |
| ddp_a100x4_wikipedia_base_accum4 | 4 x A100-SXM4-80GB | base | 38% (worst 43%) | 7% | 7% | 0% / 21% | 0% / 0% |
| ddp_batches | 8 x RTX 4090 | small | 6% (worst 7%) | 0% | 5% | 0% / 1% | 0% / 1% |
| ddp_batches_accum8 | 8 x RTX 4090 | small | 37% (worst 38%) | 2% | 15% | 2% / 3% | 2% / 0% |
| ddp_batches_base_accum4 | 8 x RTX 4090 | base | 4% (worst 5%) | 1% | 1% | 0% / 0% | 0% / 0% |
| ddp_batches_dolly | 8 x RTX 4090 | small | 6% (worst 16%) | 9% | 13% | 0% / 0% | 0% / 1% |
| ddp_batches_dolly_accum8 | 8 x RTX 4090 | small | 37% (worst 38%) | 28% | 71% | 0% / 0% | 0% / 2% |
| ddp_batches_dolly_base_accum4 | 8 x RTX 4090 | base | 6% (worst 14%) | 4% | 2% | 0% / 1% | 0% / 1% |

Median over runs: step time error 22%; speed-up error probe 10% vs cost model 8%.

## C. Which machine is cheaper for the job?

Dollars per million training sequences at the prices paid (8x RTX 4090 $2.90/h, 4x A100 SXM $4.28/h); *estimated* = section B (4060 probe scaled by datasheet TFLOPS, measured all-reduce).

| job | split | 8x 4090 measured | estimated | 4x A100 measured | estimated | cheaper (measured) | cheaper (estimated) |
|---|---|--:|--:|--:|--:|---|---|
| Dolly-15k, base model, accumulation 4 | default | $0.800 | $0.749 | $1.429 | $0.803 | 4090 | 4090 |
| Dolly-15k, base model, accumulation 4 | Evencut | $0.674 | $0.646 | $1.242 | $0.690 | 4090 | 4090 |
| Dolly-15k, small model, accumulation 1 | default | $0.581 | $0.672 | $0.745 | $0.435 | 4090 | A100 |
| Dolly-15k, small model, accumulation 1 | Evencut | $0.529 | $0.574 | $0.541 | $0.428 | 4090 | A100 |
| Dolly-15k, small model, accumulation 8 | default | $0.309 | $0.250 | $0.570 | $0.409 | 4090 | 4090 |
| Dolly-15k, small model, accumulation 8 | Evencut | $0.304 | $0.211 | $0.508 | $0.405 | 4090 | 4090 |
| TinyStories-Turkish, base model, accumulation 4 | default | $0.265 | $0.251 | $0.428 | $0.229 | 4090 | A100 |
| TinyStories-Turkish, base model, accumulation 4 | Evencut | $0.247 | $0.238 | $0.406 | $0.214 | 4090 | A100 |
| TinyStories-Turkish, small model, accumulation 1 | default | $0.259 | $0.275 | $0.228 | $0.214 | A100 | A100 |
| TinyStories-Turkish, small model, accumulation 1 | Evencut | $0.257 | $0.272 | $0.212 | $0.214 | A100 | A100 |
| TinyStories-Turkish, small model, accumulation 8 | default | $0.150 | $0.096 | $0.199 | $0.203 | 4090 | 4090 |
| TinyStories-Turkish, small model, accumulation 8 | Evencut | $0.153 | $0.094 | $0.192 | $0.203 | 4090 | 4090 |

The estimate names the cheaper machine in 8 of 12 cases.

## D. Probed: 2 x NVIDIA A100-SXM4-40GB ($1.341/h, `vast_a100x2`)

Links ['SYS'], peer-to-peer all; bf16 matmul 209, 209 TFLOPS per GPU; launch floor 13.1 ms (small model); DDP hides an unmeasured part of the sync (older probe: sync counted in full); power 253 W, 235 W.

| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) | same, sync counted in full | speed-up error | regret | step time error without the probe (datasheet estimate) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ddp_dolly_accum8 | small | 11.7 ms | 11.6 ms | 2% (worst 7%) | (same) | 5% | 0% / 0% | 32% |
| ddp_dolly_base_accum4 | base | 35.1 ms | 35.4 ms | 4% (worst 8%) | (same) | 1% | 0% / 0% | 46% |
| ddp_tinystories | small | 11.8 ms | 11.6 ms | 8% (worst 11%) | (same) | 3% | 0% / 2% | 18% |
| ddp_wikipedia | small | 11.7 ms | 11.6 ms | 10% (worst 14%) | (same) | 2% | 0% / 1% | 27% |

Median over runs: step time error 6%.

#### ddp_dolly_accum8 (Dolly-15k, small, 32 sequences per micro-step, accumulation 8)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 281.2 | 268.9 | -4% | 185.3 | -34% |
| random | equal counts | 294.2 | 310.3 | +5% | 186.5 | -37% |
| random | Evencut, balanced tokens | 294.8 | 307.8 | +4% | 187.5 | -36% |
| random | Evencut, balanced padded size | 284.9 | 296.8 | +4% | 185.6 | -35% |
| length-grouped | equal counts | 309.9 | 333.1 | +7% | 186.9 | -40% |
| length-grouped | Evencut, balanced tokens | 270.2 | 272.4 | +1% | 185.3 | -31% |
| length-grouped | Evencut, balanced padded size | 267.5 | 268.9 | +1% | 185.3 | -31% |
| any order | LPT on padded size (not contiguous) | 285.2 | 303.7 | +6% | 185.4 | -35% |
| sorted dataset | token budget per GPU (fairseq-style) | 271.3 | 268.1 | -1% | 185.3 | -32% |
| sorted dataset | Evencut, padded_slices over the sorted data | 263.5 | 268.1 | +2% | 185.3 | -30% |
| random | Evencut, whole optimizer step balanced (window) | 265.5 | 268.3 | +1% | 185.3 | -30% |
| pooled (like group_by_length) | Hugging Face group_by_length | 269.6 | 269.4 | -0% | 185.4 | -31% |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 264.3 | 266.4 | +1% | 185.3 | -30% |

#### ddp_dolly_base_accum4 (Dolly-15k, base, 48 sequences per micro-step, accumulation 4)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 318.9 | 321.9 | +1% | 169.7 | -47% |
| random | equal counts | 503.0 | 522.4 | +4% | 273.9 | -46% |
| random | Evencut, balanced tokens | 473.1 | 491.2 | +4% | 257.1 | -46% |
| random | Evencut, balanced padded size | 452.5 | 471.0 | +4% | 244.6 | -46% |
| length-grouped | equal counts | 551.7 | 566.4 | +3% | 295.6 | -46% |
| length-grouped | Evencut, balanced tokens | 298.4 | 311.8 | +4% | 175.5 | -41% |
| length-grouped | Evencut, balanced padded size | 267.9 | 283.1 | +6% | 166.0 | -38% |
| any order | LPT on padded size (not contiguous) | 481.4 | 495.5 | +3% | 257.6 | -46% |
| sorted dataset | token budget per GPU (fairseq-style) | 221.4 | 239.7 | +8% | 164.3 | -26% |
| sorted dataset | Evencut, padded_slices over the sorted data | 221.2 | 239.7 | +8% | 164.3 | -26% |

#### ddp_tinystories (TinyStories-Turkish, small, 64 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 40.5 | 44.5 | +10% | 33.8 | -17% |
| random | equal counts | 42.9 | 44.9 | +5% | 33.8 | -21% |
| random | Evencut, balanced tokens | 42.8 | 44.9 | +5% | 33.8 | -21% |
| random | Evencut, balanced padded size | 41.6 | 44.5 | +7% | 33.8 | -19% |
| length-grouped | equal counts | 42.5 | 44.9 | +6% | 33.8 | -20% |
| length-grouped | Evencut, balanced tokens | 41.3 | 44.8 | +8% | 33.8 | -18% |
| length-grouped | Evencut, balanced padded size | 40.7 | 44.5 | +9% | 33.8 | -17% |
| any order | LPT on padded size (not contiguous) | 41.9 | 44.5 | +6% | 33.8 | -19% |
| sorted dataset | token budget per GPU (fairseq-style) | 40.0 | 44.5 | +11% | 33.8 | -15% |
| sorted dataset | Evencut, padded_slices over the sorted data | 40.0 | 44.5 | +11% | 33.8 | -15% |

#### ddp_wikipedia (Wikipedia-TR, small, 12 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 39.9 | 44.5 | +11% | 33.7 | -16% |
| random | equal counts | 49.1 | 54.6 | +11% | 33.7 | -31% |
| random | Evencut, balanced tokens | 47.6 | 52.3 | +10% | 34.1 | -28% |
| random | Evencut, balanced padded size | 45.9 | 50.1 | +9% | 33.7 | -27% |
| length-grouped | equal counts | 49.8 | 54.6 | +10% | 33.7 | -32% |
| length-grouped | Evencut, balanced tokens | 47.8 | 50.5 | +6% | 34.3 | -28% |
| length-grouped | Evencut, balanced padded size | 41.3 | 44.8 | +8% | 33.7 | -18% |
| any order | LPT on padded size (not contiguous) | 46.2 | 49.5 | +7% | 33.7 | -27% |
| sorted dataset | token budget per GPU (fairseq-style) | 39.2 | 44.5 | +13% | 33.7 | -14% |
| sorted dataset | Evencut, padded_slices over the sorted data | 39.0 | 44.5 | +14% | 33.7 | -13% |

## D. Probed: 2 x NVIDIA GeForce RTX 4090 ($0.951/h, `vast_rtx4090x2`)

Links ['PHB'], peer-to-peer none; bf16 matmul 149, 150 TFLOPS per GPU; launch floor 13.0 ms (small model); DDP hides an unmeasured part of the sync (older probe: sync counted in full); power 293 W, 296 W.

| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) | same, sync counted in full | speed-up error | regret | step time error without the probe (datasheet estimate) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ddp_dolly_accum8 | small | 47.2 ms | 47.3 ms | 3% (worst 7%) | (same) | 2% | 0% / 5% | 11% |
| ddp_dolly_base_accum4 | base | 146.8 ms | 146.6 ms | 8% (worst 8%) | (same) | 2% | 0% / 0% | 12% |
| ddp_tinystories | small | 47.2 ms | 47.3 ms | 9% (worst 10%) | (same) | 1% | 0% / 0% | 11% |
| ddp_wikipedia | small | 46.9 ms | 47.3 ms | 12% (worst 17%) | (same) | 5% | 0% / 0% | 7% |

Median over runs: step time error 9%.

#### ddp_dolly_accum8 (Dolly-15k, small, 32 sequences per micro-step, accumulation 8)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 210.8 | 221.4 | +5% | 221.1 | +5% |
| random | equal counts | 288.6 | 297.3 | +3% | 271.9 | -6% |
| random | Evencut, balanced tokens | 287.6 | 295.3 | +3% | 269.4 | -6% |
| random | Evencut, balanced padded size | 269.8 | 278.5 | +3% | 254.6 | -6% |
| length-grouped | equal counts | 337.0 | 343.8 | +2% | 298.8 | -11% |
| length-grouped | Evencut, balanced tokens | 204.0 | 209.6 | +3% | 225.7 | +11% |
| length-grouped | Evencut, balanced padded size | 177.1 | 188.9 | +7% | 221.1 | +25% |
| any order | LPT on padded size (not contiguous) | 283.3 | 293.4 | +4% | 262.4 | -7% |
| sorted dataset | token budget per GPU (fairseq-style) | 163.1 | 157.0 | -4% | 221.1 | +36% |
| sorted dataset | Evencut, padded_slices over the sorted data | 155.3 | 157.0 | +1% | 221.1 | +42% |
| random | Evencut, whole optimizer step balanced (window) | 169.0 | 157.1 | -7% | 221.1 | +31% |
| pooled (like group_by_length) | Hugging Face group_by_length | 176.1 | 171.9 | -2% | 225.4 | +28% |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 170.4 | 169.3 | -1% | 223.1 | +31% |

#### ddp_dolly_base_accum4 (Dolly-15k, base, 48 sequences per micro-step, accumulation 4)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 430.0 | 464.2 | +8% | 401.3 | -7% |
| random | equal counts | 678.1 | 713.7 | +5% | 591.6 | -13% |
| random | Evencut, balanced tokens | 637.2 | 678.4 | +6% | 560.5 | -12% |
| random | Evencut, balanced padded size | 605.8 | 653.9 | +8% | 537.1 | -11% |
| length-grouped | equal counts | 738.8 | 768.0 | +4% | 635.8 | -14% |
| length-grouped | Evencut, balanced tokens | 412.0 | 444.0 | +8% | 380.5 | -8% |
| length-grouped | Evencut, balanced padded size | 372.4 | 402.6 | +8% | 354.5 | -5% |
| any order | LPT on padded size (not contiguous) | 636.6 | 683.3 | +7% | 562.6 | -12% |
| sorted dataset | token budget per GPU (fairseq-style) | 247.9 | 268.2 | +8% | 276.8 | +12% |
| sorted dataset | Evencut, padded_slices over the sorted data | 248.0 | 268.2 | +8% | 276.8 | +12% |

#### ddp_tinystories (TinyStories-Turkish, small, 64 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 62.1 | 68.6 | +10% | 69.5 | +12% |
| random | equal counts | 64.2 | 70.5 | +10% | 70.0 | +9% |
| random | Evencut, balanced tokens | 64.2 | 70.5 | +10% | 69.9 | +9% |
| random | Evencut, balanced padded size | 62.9 | 69.2 | +10% | 69.5 | +11% |
| length-grouped | equal counts | 64.9 | 70.5 | +9% | 70.0 | +8% |
| length-grouped | Evencut, balanced tokens | 62.5 | 67.7 | +8% | 69.8 | +12% |
| length-grouped | Evencut, balanced padded size | 61.1 | 66.5 | +9% | 69.5 | +14% |
| any order | LPT on padded size (not contiguous) | 63.0 | 69.5 | +10% | 69.6 | +11% |
| sorted dataset | token budget per GPU (fairseq-style) | 59.4 | 63.6 | +7% | 69.5 | +17% |
| sorted dataset | Evencut, padded_slices over the sorted data | 59.5 | 63.6 | +7% | 69.5 | +17% |

#### ddp_wikipedia (Wikipedia-TR, small, 12 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 58.9 | 62.9 | +7% | 69.2 | +17% |
| random | equal counts | 76.1 | 88.7 | +17% | 79.6 | +5% |
| random | Evencut, balanced tokens | 74.5 | 84.0 | +13% | 78.5 | +5% |
| random | Evencut, balanced padded size | 70.5 | 80.3 | +14% | 75.2 | +7% |
| length-grouped | equal counts | 76.6 | 88.7 | +16% | 79.6 | +4% |
| length-grouped | Evencut, balanced tokens | 71.5 | 79.3 | +11% | 76.1 | +6% |
| length-grouped | Evencut, balanced padded size | 63.1 | 69.2 | +10% | 69.8 | +11% |
| any order | LPT on padded size (not contiguous) | 69.6 | 79.1 | +14% | 74.6 | +7% |
| sorted dataset | token budget per GPU (fairseq-style) | 58.6 | 62.6 | +7% | 69.2 | +18% |
| sorted dataset | Evencut, padded_slices over the sorted data | 58.5 | 62.6 | +7% | 69.2 | +18% |

## D. Probed: 2 x NVIDIA A100-SXM4-40GB ($1.424/h, `vast_a100x2_v2`)

Links ['SYS'], peer-to-peer all; bf16 matmul 210, 210 TFLOPS per GPU; launch floor 22.4 ms (small model); DDP hides 19% of the last micro-batch's time worth of sync (proportional model); power 293 W, 288 W.

| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) | same, sync counted in full | speed-up error | regret | step time error without the probe (datasheet estimate) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ddp_dolly_accum8 | small | 8.8 ms | 8.7 ms | 2% (worst 7%) | 2% | 4% | 0% / 1% | 39% |
| ddp_dolly_base_accum4 | base | 26.1 ms | 26.3 ms | 2% (worst 4%) | 2% | 1% | 0% / 0% | 43% |
| ddp_tinystories | small | 9.0 ms | 8.7 ms | 8% (worst 10%) | 9% | 2% | 0% / 1% | 26% |
| ddp_wikipedia | small | 9.1 ms | 8.7 ms | 8% (worst 9%) | 9% | 3% | 0% / 0% | 30% |

Median over runs: step time error 5%.

#### ddp_dolly_accum8 (Dolly-15k, small, 32 sequences per micro-step, accumulation 8)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 294.7 | 291.6 | -1% | 182.3 | -38% |
| random | equal counts | 310.7 | 315.9 | +2% | 183.5 | -41% |
| random | Evencut, balanced tokens | 307.8 | 315.2 | +2% | 184.5 | -40% |
| random | Evencut, balanced padded size | 308.3 | 307.8 | -0% | 182.6 | -41% |
| length-grouped | equal counts | 307.7 | 330.4 | +7% | 183.9 | -40% |
| length-grouped | Evencut, balanced tokens | 300.3 | 292.4 | -3% | 182.3 | -39% |
| length-grouped | Evencut, balanced padded size | 303.9 | 289.4 | -5% | 182.3 | -40% |
| any order | LPT on padded size (not contiguous) | 309.7 | 312.3 | +1% | 182.5 | -41% |
| sorted dataset | token budget per GPU (fairseq-style) | 295.0 | 282.6 | -4% | 182.3 | -38% |
| sorted dataset | Evencut, padded_slices over the sorted data | 296.2 | 282.6 | -5% | 182.3 | -38% |
| random | Evencut, whole optimizer step balanced (window) | 297.4 | 282.6 | -5% | 182.3 | -39% |
| pooled (like group_by_length) | Hugging Face group_by_length | 292.6 | 286.4 | -2% | 182.4 | -38% |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 291.0 | 284.6 | -2% | 182.3 | -37% |

#### ddp_dolly_base_accum4 (Dolly-15k, base, 48 sequences per micro-step, accumulation 4)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 293.4 | 281.1 | -4% | 160.6 | -45% |
| random | equal counts | 467.0 | 458.2 | -2% | 264.9 | -43% |
| random | Evencut, balanced tokens | 438.0 | 429.3 | -2% | 248.0 | -43% |
| random | Evencut, balanced padded size | 417.0 | 412.9 | -1% | 235.5 | -44% |
| length-grouped | equal counts | 507.1 | 496.7 | -2% | 286.5 | -43% |
| length-grouped | Evencut, balanced tokens | 274.0 | 272.8 | -0% | 166.5 | -39% |
| length-grouped | Evencut, balanced padded size | 241.5 | 246.9 | +2% | 156.9 | -35% |
| any order | LPT on padded size (not contiguous) | 441.1 | 434.6 | -1% | 248.6 | -44% |
| sorted dataset | token budget per GPU (fairseq-style) | 200.0 | 202.3 | +1% | 155.2 | -22% |
| sorted dataset | Evencut, padded_slices over the sorted data | 203.5 | 202.3 | -1% | 155.2 | -24% |

#### ddp_tinystories (TinyStories-Turkish, small, 64 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 40.3 | 38.7 | -4% | 31.0 | -23% |
| random | equal counts | 43.2 | 38.9 | -10% | 31.0 | -28% |
| random | Evencut, balanced tokens | 42.8 | 38.9 | -9% | 31.0 | -27% |
| random | Evencut, balanced padded size | 42.2 | 38.7 | -8% | 31.0 | -27% |
| length-grouped | equal counts | 42.1 | 38.9 | -7% | 31.0 | -26% |
| length-grouped | Evencut, balanced tokens | 41.6 | 38.8 | -7% | 31.0 | -25% |
| length-grouped | Evencut, balanced padded size | 42.0 | 38.6 | -8% | 31.0 | -26% |
| any order | LPT on padded size (not contiguous) | 42.0 | 38.7 | -8% | 31.0 | -26% |
| sorted dataset | token budget per GPU (fairseq-style) | 41.7 | 38.5 | -8% | 31.0 | -26% |
| sorted dataset | Evencut, padded_slices over the sorted data | 41.3 | 38.5 | -7% | 31.0 | -25% |

#### ddp_wikipedia (Wikipedia-TR, small, 12 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 40.6 | 38.5 | -5% | 31.1 | -24% |
| random | equal counts | 45.9 | 43.8 | -5% | 31.1 | -32% |
| random | Evencut, balanced tokens | 47.2 | 42.9 | -9% | 31.4 | -33% |
| random | Evencut, balanced padded size | 44.9 | 41.4 | -8% | 31.1 | -31% |
| length-grouped | equal counts | 46.7 | 43.8 | -6% | 31.1 | -33% |
| length-grouped | Evencut, balanced tokens | 45.4 | 42.0 | -8% | 31.6 | -30% |
| length-grouped | Evencut, balanced padded size | 42.3 | 38.7 | -8% | 31.1 | -27% |
| any order | LPT on padded size (not contiguous) | 44.5 | 41.1 | -8% | 31.1 | -30% |
| sorted dataset | token budget per GPU (fairseq-style) | 41.4 | 38.2 | -8% | 31.1 | -25% |
| sorted dataset | Evencut, padded_slices over the sorted data | 41.9 | 38.2 | -9% | 31.1 | -26% |

## D. Probed: 2 x NVIDIA GeForce RTX 4090 ($0.978/h, `vast_rtx4090x2_v2`)

Links ['PXB'], peer-to-peer none; bf16 matmul 146, 148 TFLOPS per GPU; launch floor 27.9 ms (small model); DDP hides 21% of the last micro-batch's time worth of sync (proportional model); power 358 W, 364 W.

| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) | same, sync counted in full | speed-up error | regret | step time error without the probe (datasheet estimate) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ddp_dolly_accum8 | small | 15.9 ms | 16.0 ms | 1% (worst 9%) | 4% | 4% | 0% / 1% | 44% |
| ddp_dolly_base_accum4 | base | 47.8 ms | 48.3 ms | 3% (worst 5%) | 9% | 2% | 0% / 0% | 12% |
| ddp_tinystories | small | 16.0 ms | 16.0 ms | 0% (worst 12%) | 17% | 12% | 11% / 12% | 25% |
| ddp_wikipedia | small | 16.0 ms | 16.0 ms | 1% (worst 6%) | 18% | 5% | 0% / 0% | 15% |

Median over runs: step time error 1%.

#### ddp_dolly_accum8 (Dolly-15k, small, 32 sequences per micro-step, accumulation 8)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 346.9 | 350.1 | +1% | 189.8 | -45% |
| random | equal counts | 351.3 | 369.8 | +5% | 240.6 | -32% |
| random | Evencut, balanced tokens | 353.0 | 369.3 | +5% | 238.0 | -33% |
| random | Evencut, balanced padded size | 352.2 | 363.6 | +3% | 223.3 | -37% |
| length-grouped | equal counts | 349.4 | 381.0 | +9% | 267.4 | -23% |
| length-grouped | Evencut, balanced tokens | 349.5 | 351.8 | +1% | 194.3 | -44% |
| length-grouped | Evencut, balanced padded size | 351.9 | 349.9 | -1% | 189.8 | -46% |
| any order | LPT on padded size (not contiguous) | 351.8 | 367.1 | +4% | 231.0 | -34% |
| sorted dataset | token budget per GPU (fairseq-style) | 346.6 | 348.9 | +1% | 189.8 | -45% |
| sorted dataset | Evencut, padded_slices over the sorted data | 344.1 | 348.9 | +1% | 189.8 | -45% |
| random | Evencut, whole optimizer step balanced (window) | 345.3 | 348.9 | +1% | 189.8 | -45% |
| pooled (like group_by_length) | Hugging Face group_by_length | 347.0 | 350.9 | +1% | 194.0 | -44% |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 347.6 | 349.6 | +1% | 191.8 | -45% |

#### ddp_dolly_base_accum4 (Dolly-15k, base, 48 sequences per micro-step, accumulation 4)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 318.1 | 311.1 | -2% | 302.3 | -5% |
| random | equal counts | 558.4 | 539.3 | -3% | 492.7 | -12% |
| random | Evencut, balanced tokens | 521.5 | 503.8 | -3% | 461.6 | -11% |
| random | Evencut, balanced padded size | 492.3 | 481.6 | -2% | 438.2 | -11% |
| length-grouped | equal counts | 617.8 | 587.3 | -5% | 536.9 | -13% |
| length-grouped | Evencut, balanced tokens | 302.9 | 309.0 | +2% | 281.5 | -7% |
| length-grouped | Evencut, balanced padded size | 265.7 | 278.3 | +5% | 255.5 | -4% |
| any order | LPT on padded size (not contiguous) | 527.8 | 509.1 | -4% | 463.6 | -12% |
| sorted dataset | token budget per GPU (fairseq-style) | 240.3 | 246.2 | +2% | 177.9 | -26% |
| sorted dataset | Evencut, padded_slices over the sorted data | 244.7 | 246.2 | +1% | 177.9 | -27% |

#### ddp_tinystories (TinyStories-Turkish, small, 64 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 51.4 | 50.9 | -1% | 38.3 | -25% |
| random | equal counts | 45.7 | 51.0 | +12% | 38.8 | -15% |
| random | Evencut, balanced tokens | 51.2 | 51.0 | -0% | 38.8 | -24% |
| random | Evencut, balanced padded size | 50.9 | 50.9 | -0% | 38.4 | -25% |
| length-grouped | equal counts | 52.0 | 51.0 | -2% | 38.8 | -25% |
| length-grouped | Evencut, balanced tokens | 51.2 | 51.0 | -0% | 38.6 | -24% |
| length-grouped | Evencut, balanced padded size | 51.2 | 50.9 | -1% | 38.3 | -25% |
| any order | LPT on padded size (not contiguous) | 51.2 | 50.9 | -1% | 38.4 | -25% |
| sorted dataset | token budget per GPU (fairseq-style) | 51.0 | 50.9 | -0% | 38.3 | -25% |
| sorted dataset | Evencut, padded_slices over the sorted data | 51.1 | 50.9 | -0% | 38.3 | -25% |

#### ddp_wikipedia (Wikipedia-TR, small, 12 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 52.2 | 50.9 | -3% | 38.3 | -27% |
| random | equal counts | 52.3 | 54.8 | +5% | 48.8 | -7% |
| random | Evencut, balanced tokens | 53.7 | 54.0 | +1% | 47.7 | -11% |
| random | Evencut, balanced padded size | 52.3 | 53.1 | +1% | 44.4 | -15% |
| length-grouped | equal counts | 51.6 | 54.8 | +6% | 48.8 | -5% |
| length-grouped | Evencut, balanced tokens | 53.5 | 53.5 | -0% | 45.3 | -15% |
| length-grouped | Evencut, balanced padded size | 51.5 | 51.0 | -1% | 39.0 | -24% |
| any order | LPT on padded size (not contiguous) | 51.9 | 52.8 | +2% | 43.8 | -16% |
| sorted dataset | token budget per GPU (fairseq-style) | 51.6 | 50.8 | -2% | 38.3 | -26% |
| sorted dataset | Evencut, padded_slices over the sorted data | 51.5 | 50.8 | -1% | 38.3 | -26% |

## D. Probed: 2 x NVIDIA A100-SXM4-80GB ($1.904/h, `vast_a100x2_v3`)

Links ['NV4'], peer-to-peer all; bf16 matmul 159, 159 TFLOPS per GPU; launch floor 16.3 ms (small model); DDP hides part of the sync; still exposed: 68% after a 17 ms batch, 66% after a 17 ms batch, 59% after a 17 ms batch, 64% after a 24 ms batch, 58% after a 44 ms batch; power 262 W, 263 W.

| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) | same, sync counted in full | speed-up error | regret | step time error without the probe (datasheet estimate) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ddp_dolly_accum8 | small | 2.3 ms | 1.8 ms | 3% (worst 9%) | 3% | 2% | 0% / 1% | 24% |
| ddp_dolly_base_accum4 | base | 6.6 ms | 5.3 ms | 2% (worst 3%) | 3% | 1% | 0% / 0% | 47% |
| ddp_tinystories | small | 2.4 ms | 1.8 ms | 2% (worst 4%) | 5% | 1% | 0% / 1% | 7% |
| ddp_wikipedia | small | 2.3 ms | 1.8 ms | 5% (worst 13%) | 8% | 8% | 0% / 1% | 26% |

Median over runs: step time error 3%.

#### ddp_dolly_accum8 (Dolly-15k, small, 32 sequences per micro-step, accumulation 8)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 176.2 | 182.4 | +4% | 175.9 | -0% |
| random | equal counts | 236.1 | 243.4 | +3% | 177.1 | -25% |
| random | Evencut, balanced tokens | 234.6 | 241.8 | +3% | 178.1 | -24% |
| random | Evencut, balanced padded size | 221.0 | 228.4 | +3% | 176.2 | -20% |
| length-grouped | equal counts | 275.2 | 280.8 | +2% | 177.5 | -36% |
| length-grouped | Evencut, balanced tokens | 163.2 | 172.4 | +6% | 175.9 | +8% |
| length-grouped | Evencut, balanced padded size | 143.8 | 156.1 | +9% | 175.9 | +22% |
| any order | LPT on padded size (not contiguous) | 232.1 | 240.0 | +3% | 176.1 | -24% |
| sorted dataset | token budget per GPU (fairseq-style) | 139.7 | 136.3 | -2% | 175.9 | +26% |
| sorted dataset | Evencut, padded_slices over the sorted data | 139.0 | 136.3 | -2% | 175.9 | +27% |
| random | Evencut, whole optimizer step balanced (window) | 139.5 | 136.5 | -2% | 175.9 | +26% |
| pooled (like group_by_length) | Hugging Face group_by_length | 149.4 | 147.0 | -2% | 176.0 | +18% |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 139.3 | 144.8 | +4% | 175.9 | +26% |

#### ddp_dolly_base_accum4 (Dolly-15k, base, 48 sequences per micro-step, accumulation 4)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 279.0 | 276.4 | -1% | 141.2 | -49% |
| random | equal counts | 457.5 | 468.9 | +2% | 245.4 | -46% |
| random | Evencut, balanced tokens | 428.5 | 438.2 | +2% | 228.5 | -47% |
| random | Evencut, balanced padded size | 405.9 | 417.9 | +3% | 216.1 | -47% |
| length-grouped | equal counts | 500.6 | 511.9 | +2% | 267.1 | -47% |
| length-grouped | Evencut, balanced tokens | 257.8 | 258.5 | +0% | 147.0 | -43% |
| length-grouped | Evencut, balanced padded size | 225.8 | 228.1 | +1% | 137.5 | -39% |
| any order | LPT on padded size (not contiguous) | 429.8 | 442.6 | +3% | 229.1 | -47% |
| sorted dataset | token budget per GPU (fairseq-style) | 117.6 | 118.4 | +1% | 135.8 | +15% |
| sorted dataset | Evencut, padded_slices over the sorted data | 117.7 | 118.4 | +1% | 135.8 | +15% |

#### ddp_tinystories (TinyStories-Turkish, small, 64 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 22.3 | 23.1 | +4% | 24.4 | +9% |
| random | equal counts | 24.2 | 24.8 | +2% | 24.4 | +1% |
| random | Evencut, balanced tokens | 24.4 | 24.8 | +2% | 24.4 | -0% |
| random | Evencut, balanced padded size | 23.1 | 23.7 | +3% | 24.4 | +6% |
| length-grouped | equal counts | 24.4 | 24.8 | +1% | 24.4 | -0% |
| length-grouped | Evencut, balanced tokens | 21.6 | 22.3 | +3% | 24.4 | +13% |
| length-grouped | Evencut, balanced padded size | 20.8 | 21.2 | +2% | 24.4 | +17% |
| any order | LPT on padded size (not contiguous) | 23.3 | 24.0 | +3% | 24.4 | +4% |
| sorted dataset | token budget per GPU (fairseq-style) | 19.2 | 19.0 | -1% | 24.4 | +27% |
| sorted dataset | Evencut, padded_slices over the sorted data | 19.1 | 19.0 | -0% | 24.4 | +28% |

#### ddp_wikipedia (Wikipedia-TR, small, 12 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 19.3 | 19.0 | -1% | 24.3 | +26% |
| random | equal counts | 34.8 | 39.5 | +13% | 24.3 | -30% |
| random | Evencut, balanced tokens | 33.7 | 35.7 | +6% | 24.7 | -27% |
| random | Evencut, balanced padded size | 30.2 | 32.8 | +9% | 24.3 | -19% |
| length-grouped | equal counts | 35.5 | 39.5 | +11% | 24.3 | -31% |
| length-grouped | Evencut, balanced tokens | 30.7 | 32.1 | +5% | 24.9 | -19% |
| length-grouped | Evencut, balanced padded size | 22.9 | 23.9 | +4% | 24.3 | +6% |
| any order | LPT on padded size (not contiguous) | 29.4 | 31.8 | +8% | 24.3 | -17% |
| sorted dataset | token budget per GPU (fairseq-style) | 19.3 | 18.9 | -2% | 24.3 | +26% |
| sorted dataset | Evencut, padded_slices over the sorted data | 19.1 | 18.9 | -1% | 24.3 | +28% |

## D. Probed: 2 x NVIDIA GeForce RTX 4090 ($0.891/h, `vast_rtx4090x2_v3`)

Links ['NODE'], peer-to-peer none; bf16 matmul 126, 126 TFLOPS per GPU; launch floor 15.9 ms (small model); DDP hides part of the sync; still exposed: 54% after a 33 ms batch, 66% after a 34 ms batch, 58% after a 35 ms batch, 76% after a 35 ms batch, 85% after a 54 ms batch; power 232 W, 237 W.

| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) | same, sync counted in full | speed-up error | regret | step time error without the probe (datasheet estimate) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ddp_dolly_accum8 | small | 33.8 ms | 34.0 ms | 5% (worst 8%) | 2% | 12% | 0% / 6% | 31% |
| ddp_dolly_base_accum4 | base | 99.2 ms | 108.8 ms | 1% (worst 4%) | 4% | 1% | 0% / 1% | 21% |
| ddp_tinystories | small | 36.7 ms | 34.0 ms | 2% (worst 4%) | 21% | 2% | 0% / 5% | 3% |
| ddp_wikipedia | small | 36.5 ms | 34.0 ms | 3% (worst 5%) | 17% | 3% | 0% / 0% | 3% |

Median over runs: step time error 2%.

#### ddp_dolly_accum8 (Dolly-15k, small, 32 sequences per micro-step, accumulation 8)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 304.7 | 292.5 | -4% | 207.7 | -32% |
| random | equal counts | 323.9 | 351.1 | +8% | 258.5 | -20% |
| random | Evencut, balanced tokens | 326.0 | 345.8 | +6% | 255.9 | -21% |
| random | Evencut, balanced padded size | 313.7 | 331.1 | +6% | 241.2 | -23% |
| length-grouped | equal counts | 363.1 | 381.6 | +5% | 285.4 | -21% |
| length-grouped | Evencut, balanced tokens | 309.6 | 297.1 | -4% | 212.2 | -31% |
| length-grouped | Evencut, balanced padded size | 316.0 | 292.5 | -7% | 207.7 | -34% |
| any order | LPT on padded size (not contiguous) | 324.7 | 339.9 | +5% | 249.0 | -23% |
| sorted dataset | token budget per GPU (fairseq-style) | 299.0 | 292.5 | -2% | 207.7 | -31% |
| sorted dataset | Evencut, padded_slices over the sorted data | 300.5 | 292.5 | -3% | 207.7 | -31% |
| random | Evencut, whole optimizer step balanced (window) | 308.5 | 292.5 | -5% | 207.7 | -33% |
| pooled (like group_by_length) | Hugging Face group_by_length | 308.5 | 297.3 | -4% | 211.9 | -31% |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 306.3 | 293.2 | -4% | 209.7 | -32% |

#### ddp_dolly_base_accum4 (Dolly-15k, base, 48 sequences per micro-step, accumulation 4)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 438.4 | 439.4 | +0% | 353.7 | -19% |
| random | equal counts | 710.4 | 700.6 | -1% | 544.1 | -23% |
| random | Evencut, balanced tokens | 666.6 | 659.3 | -1% | 513.0 | -23% |
| random | Evencut, balanced padded size | 634.7 | 633.0 | -0% | 489.6 | -23% |
| length-grouped | equal counts | 780.0 | 758.8 | -3% | 588.3 | -25% |
| length-grouped | Evencut, balanced tokens | 404.6 | 415.8 | +3% | 332.9 | -18% |
| length-grouped | Evencut, balanced padded size | 376.1 | 374.0 | -1% | 306.9 | -18% |
| any order | LPT on padded size (not contiguous) | 672.6 | 665.9 | -1% | 515.1 | -23% |
| sorted dataset | token budget per GPU (fairseq-style) | 243.0 | 252.5 | +4% | 229.3 | -6% |
| sorted dataset | Evencut, padded_slices over the sorted data | 241.7 | 252.5 | +4% | 229.3 | -5% |

#### ddp_tinystories (TinyStories-Turkish, small, 64 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 57.2 | 56.6 | -1% | 59.0 | +3% |
| random | equal counts | 58.6 | 57.3 | -2% | 59.4 | +1% |
| random | Evencut, balanced tokens | 58.3 | 57.3 | -2% | 59.4 | +2% |
| random | Evencut, balanced padded size | 57.2 | 56.6 | -1% | 59.0 | +3% |
| length-grouped | equal counts | 59.1 | 57.3 | -3% | 59.4 | +1% |
| length-grouped | Evencut, balanced tokens | 56.4 | 57.1 | +1% | 59.3 | +5% |
| length-grouped | Evencut, balanced padded size | 55.2 | 56.6 | +3% | 59.0 | +7% |
| any order | LPT on padded size (not contiguous) | 57.0 | 56.6 | -1% | 59.0 | +3% |
| sorted dataset | token budget per GPU (fairseq-style) | 54.9 | 56.6 | +3% | 59.0 | +7% |
| sorted dataset | Evencut, padded_slices over the sorted data | 54.5 | 56.6 | +4% | 59.0 | +8% |

#### ddp_wikipedia (Wikipedia-TR, small, 12 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 54.1 | 56.6 | +5% | 58.9 | +9% |
| random | equal counts | 71.8 | 75.0 | +4% | 69.3 | -4% |
| random | Evencut, balanced tokens | 70.7 | 70.6 | -0% | 68.2 | -3% |
| random | Evencut, balanced padded size | 66.2 | 66.9 | +1% | 64.9 | -2% |
| length-grouped | equal counts | 73.9 | 75.0 | +1% | 69.3 | -6% |
| length-grouped | Evencut, balanced tokens | 64.4 | 67.4 | +5% | 65.8 | +2% |
| length-grouped | Evencut, balanced padded size | 58.1 | 57.3 | -1% | 59.5 | +2% |
| any order | LPT on padded size (not contiguous) | 65.5 | 66.1 | +1% | 64.3 | -2% |
| sorted dataset | token budget per GPU (fairseq-style) | 53.9 | 56.6 | +5% | 58.9 | +9% |
| sorted dataset | Evencut, padded_slices over the sorted data | 53.8 | 56.6 | +5% | 58.9 | +9% |

## D. Probed: 1 x NVIDIA A100-SXM4-40GB ($0.607/h, `vast_a100x1_large`)

Links None, peer-to-peer None; bf16 matmul 210 TFLOPS per GPU; launch floor 32.4 ms (small model); DDP hides an unmeasured part of the sync (older probe: sync counted in full); power 378 W.

| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) | same, sync counted in full | speed-up error | regret | step time error without the probe (datasheet estimate) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ddp_tinystories_large_accum4 | large | 0.0 ms | 0.0 ms | 2% (worst 7%) | (same) | 4% | 0% / 0% | 29% |
| ddp_wikipedia_large | large | 0.0 ms | 0.0 ms | 2% (worst 2%) | (same) | 2% | 0% / 0% | 29% |

Median over runs: step time error 2%.

#### ddp_tinystories_large_accum4 (TinyStories-Turkish, large, 16 sequences per micro-step, accumulation 4)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 277.2 | 295.5 | +7% | 365.7 | +32% |
| random | equal counts | 288.4 | 295.0 | +2% | 365.7 | +27% |
| random | Evencut, balanced padded size | 288.5 | 295.0 | +2% | 365.7 | +27% |
| random | Evencut, whole optimizer step balanced (window) | 245.6 | 242.0 | -1% | 365.7 | +49% |

#### ddp_wikipedia_large (Wikipedia-TR, large, 6 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 97.9 | 96.6 | -1% | 94.1 | -4% |
| random | equal counts | 164.2 | 167.6 | +2% | 116.6 | -29% |
| random | Evencut, balanced padded size | 165.1 | 167.6 | +2% | 116.6 | -29% |

## D. Probed: 2 x NVIDIA GeForce RTX 4090 ($0.978/h, `vast_rtx4090x2_large`)

Links ['PXB'], peer-to-peer none; bf16 matmul 146, 148 TFLOPS per GPU; launch floor 75.1 ms (small model); DDP hides part of the sync; still exposed: 83% after a 70 ms batch, 100% after a 71 ms batch, 75% after a 83 ms batch, 40% after a 173 ms batch, 15% after a 369 ms batch; power 414 W, 409 W.

| run | model | all-reduce: measured by the run | predicted by the probe | step time error (probe) | same, sync counted in full | speed-up error | regret | step time error without the probe (datasheet estimate) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| ddp_tinystories_large_accum4 | large | 162.9 ms | 163.2 ms | 4% (worst 8%) | 5% | 4% | 7% / 7% | 8% |
| ddp_wikipedia_large | large | 200.8 ms | 163.2 ms | 3% (worst 9%) | 31% | 5% | 0% / 0% | 31% |

Median over runs: step time error 3%.

#### ddp_tinystories_large_accum4 (TinyStories-Turkish, large, 32 sequences per micro-step, accumulation 4)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 490.8 | 473.3 | -4% | 531.7 | +8% |
| random | equal counts | 513.3 | 474.5 | -8% | 531.7 | +4% |
| random | Evencut, balanced padded size | 458.7 | 473.3 | +3% | 531.7 | +16% |
| random | Evencut, whole optimizer step balanced (window) | 491.3 | 471.5 | -4% | 531.7 | +8% |

#### ddp_wikipedia_large (Wikipedia-TR, large, 12 sequences per micro-step, accumulation 1)

| data order | split | measured ms | probe ms | error | datasheet estimate ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 241.8 | 219.1 | -9% | 298.0 | +23% |
| random | equal counts | 345.6 | 335.3 | -3% | 454.2 | +31% |
| random | Evencut, balanced padded size | 302.9 | 303.7 | +0% | 419.7 | +39% |

## E. Which probed machine is cheaper for the job?

Dollars per million training sequences at the prices paid, for every job run on more than one probed machine; *probe* = predicted from each machine's own probe, *estimate* = from the RTX 4060 probe and datasheets, without probing the machine. Cheapest in bold.

**Dolly-15k, base model, accumulation 4**

| split | machine | measured | probe | estimate |
|---|---|--:|--:|--:|
| default | vast_rtx4090x2_v2 | **$0.790** | **$0.763** | $0.697 |
| default | vast_rtx4090x2_v3 | $0.916 | $0.903 | $0.701 |
| default | vast_rtx4090x2 | $0.933 | $0.982 | $0.814 |
| default | vast_a100x2_v2 | $0.962 | $0.944 | $0.546 |
| default | vast_a100x2 | $0.976 | $1.014 | **$0.531** |
| default | vast_a100x2_v3 | $1.260 | $1.292 | $0.676 |
| Evencut | vast_rtx4090x2_v2 | **$0.697** | **$0.681** | $0.620 |
| Evencut | vast_rtx4090x2_v3 | $0.818 | $0.816 | $0.631 |
| Evencut | vast_rtx4090x2 | $0.834 | $0.900 | $0.739 |
| Evencut | vast_a100x2_v2 | $0.859 | $0.851 | $0.485 |
| Evencut | vast_a100x2 | $0.878 | $0.914 | **$0.475** |
| Evencut | vast_a100x2_v3 | $1.118 | $1.151 | $0.595 |

**Dolly-15k, small model, accumulation 8**

| split | machine | measured | probe | estimate |
|---|---|--:|--:|--:|
| default | vast_rtx4090x2 | **$0.298** | **$0.307** | $0.281 |
| default | vast_rtx4090x2_v3 | $0.313 | $0.339 | **$0.250** |
| default | vast_rtx4090x2_v2 | $0.373 | $0.392 | $0.255 |
| default | vast_a100x2 | $0.428 | $0.452 | $0.271 |
| default | vast_a100x2_v2 | $0.480 | $0.488 | $0.284 |
| default | vast_a100x2_v3 | $0.488 | $0.503 | $0.366 |
| Evencut | vast_rtx4090x2 | **$0.278** | **$0.287** | $0.263 |
| Evencut | vast_rtx4090x2_v3 | $0.303 | $0.320 | **$0.233** |
| Evencut | vast_rtx4090x2_v2 | $0.374 | $0.386 | $0.237 |
| Evencut | vast_a100x2 | $0.415 | $0.432 | $0.270 |
| Evencut | vast_a100x2_v3 | $0.456 | $0.472 | $0.364 |
| Evencut | vast_a100x2_v2 | $0.476 | $0.476 | $0.282 |

**TinyStories-Turkish, large model, accumulation 4**

| split | machine | measured | probe | estimate |
|---|---|--:|--:|--:|
| default | vast_a100x1_large | **$0.760** | **$0.777** | **$0.963** |
| default | vast_rtx4090x2_large | $1.089 | $1.007 | $1.128 |
| Evencut | vast_a100x1_large | **$0.760** | **$0.777** | **$0.963** |
| Evencut | vast_rtx4090x2_large | $0.974 | $1.005 | $1.128 |

**TinyStories-Turkish, small model, accumulation 1**

| split | machine | measured | probe | estimate |
|---|---|--:|--:|--:|
| default | vast_rtx4090x2_v2 | **$0.194** | $0.217 | **$0.165** |
| default | vast_a100x2_v3 | $0.200 | **$0.205** | $0.201 |
| default | vast_rtx4090x2_v3 | $0.227 | $0.222 | $0.230 |
| default | vast_a100x2 | $0.250 | $0.261 | $0.197 |
| default | vast_rtx4090x2 | $0.265 | $0.291 | $0.289 |
| default | vast_a100x2_v2 | $0.267 | $0.240 | $0.192 |
| Evencut | vast_a100x2_v3 | **$0.191** | **$0.196** | $0.201 |
| Evencut | vast_rtx4090x2_v2 | $0.216 | $0.216 | **$0.163** |
| Evencut | vast_rtx4090x2_v3 | $0.221 | $0.219 | $0.228 |
| Evencut | vast_a100x2 | $0.242 | $0.259 | $0.197 |
| Evencut | vast_rtx4090x2 | $0.260 | $0.286 | $0.287 |
| Evencut | vast_a100x2_v2 | $0.261 | $0.239 | $0.192 |

**Wikipedia-TR, large model, accumulation 1**

| split | machine | measured | probe | estimate |
|---|---|--:|--:|--:|
| default | vast_a100x1_large | **$4.615** | **$4.709** | **$3.276** |
| default | vast_rtx4090x2_large | $7.823 | $7.592 | $10.282 |
| Evencut | vast_a100x1_large | **$4.639** | **$4.709** | **$3.276** |
| Evencut | vast_rtx4090x2_large | $6.858 | $6.876 | $9.502 |

**Wikipedia-TR, small model, accumulation 1**

| split | machine | measured | probe | estimate |
|---|---|--:|--:|--:|
| default | vast_rtx4090x2_v2 | **$1.185** | **$1.241** | $1.104 |
| default | vast_rtx4090x2_v3 | $1.481 | $1.546 | $1.429 |
| default | vast_a100x2_v2 | $1.513 | $1.443 | **$1.024** |
| default | vast_a100x2 | $1.524 | $1.695 | $1.047 |
| default | vast_a100x2_v3 | $1.535 | $1.740 | $1.073 |
| default | vast_rtx4090x2 | $1.674 | $1.953 | $1.752 |
| Evencut | vast_rtx4090x2_v2 | **$1.185** | **$1.201** | **$1.004** |
| Evencut | vast_a100x2_v3 | $1.330 | $1.444 | $1.073 |
| Evencut | vast_rtx4090x2_v3 | $1.366 | $1.379 | $1.338 |
| Evencut | vast_a100x2 | $1.426 | $1.554 | $1.047 |
| Evencut | vast_a100x2_v2 | $1.481 | $1.366 | $1.024 |
| Evencut | vast_rtx4090x2 | $1.553 | $1.769 | $1.655 |


Probes name the cheaper machine in 11 of 12 cases; datasheet estimates in 6.

## F. Compute alone, by model size: datasheet estimate vs measurement

One micro-batch of 512-token sequences, forward + backward, on the slowest GPU. *Speed-up* = how much faster than the RTX 4060 (same model shape): by datasheet bf16 TFLOPS, and as measured. *Error* = datasheet estimate (the 4060 probe scaled, with its launch floor) over the measurement.

| model | machine | GPU | datasheet speed-up | measured, 2k tokens | measured, 8k tokens | error, 2k tokens | error, 8k tokens |
|---|---|---|--:|--:|--:|--:|--:|
| small | vast_a100x2 | A100-SXM4-40GB | 10.4x | 1.0x | 3.5x | -32% | -32% |
| small | vast_rtx4090x2 | RTX 4090 | 5.5x | 2.4x | 4.8x | +60% | -8% |
| small | vast_a100x2_v2 | A100-SXM4-40GB | 10.4x | 0.9x | 3.1x | -38% | -40% |
| small | vast_rtx4090x2_v2 | RTX 4090 | 5.5x | 0.8x | 2.6x | -49% | -49% |
| small | vast_a100x2_v3 | A100-SXM4-80GB | 10.4x | 1.9x | 4.7x | +29% | -10% |
| small | vast_rtx4090x2_v3 | RTX 4090 | 5.5x | 1.0x | 3.3x | -36% | -36% |
| base | vast_a100x2 | A100-SXM4-40GB | 10.4x | 1.5x | 4.9x | -36% | -44% |
| base | vast_rtx4090x2 | RTX 4090 | 5.5x | 3.5x | 4.6x | +55% | -21% |
| base | vast_a100x2_v2 | A100-SXM4-40GB | 10.4x | 1.6x | 5.3x | -29% | -40% |
| base | vast_rtx4090x2_v2 | RTX 4090 | 5.5x | 1.3x | 4.6x | -43% | -21% |
| base | vast_a100x2_v3 | A100-SXM4-80GB | 10.4x | 3.0x | 5.3x | +32% | -40% |
| base | vast_rtx4090x2_v3 | RTX 4090 | 5.5x | 1.8x | 4.2x | -23% | -27% |
| large | vast_a100x1_large | A100-SXM4-40GB | 10.4x | 4.7x | 6.5x | +90% | -40% |
| large | vast_rtx4090x2_large | RTX 4090 | 5.5x | 2.6x | 5.6x | +7% | -7% |
| 7b | vast_a100x1_large | A100-SXM4-40GB | 10.4x | 6.7x | 7.8x | +17% | -25% |
| 7b | vast_rtx4090x2_large | RTX 4090 | 5.5x | 4.7x | 5.1x | -15% | -7% |

- small: median |error| of the datasheet estimate 36% (worst 60%, 12 cases)
- base: median |error| of the datasheet estimate 34% (worst 55%, 12 cases)
- large: median |error| of the datasheet estimate 23% (worst 90%, 4 cases)
- 7b: median |error| of the datasheet estimate 16% (worst 25%, 4 cases)

## Every row

### gpu_batches (4060 sim, TinyStories-Turkish, small model, 8 GPUs, 256 sequences per micro-step, accumulation 1)

Probe: measured (same GPU).

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 93.5 | 90.8 | -3% | 91.9 | -2% |
| random | equal counts | 114.5 | 113.2 | -1% | 114.5 | +0% |
| random | Evencut, balanced tokens | 113.8 | 111.7 | -2% | 112.7 | -1% |
| random | Evencut, balanced padded size | 97.2 | 96.1 | -1% | 97.2 | -0% |
| length-grouped | equal counts | 111.1 | 113.2 | +2% | 114.5 | +3% |
| length-grouped | Evencut, balanced tokens | 78.8 | 82.0 | +4% | 83.3 | +6% |
| length-grouped | Evencut, balanced padded size | 68.9 | 70.1 | +2% | 71.5 | +4% |
| sorted dataset | token budget per GPU (fairseq-style) | 65.0 | 65.4 | +1% | 67.0 | +3% |
| sorted dataset | Evencut, padded_slices over the sorted data | 64.8 | 65.4 | +1% | 67.0 | +3% |

### gpu_batches_dolly (4060 sim, Dolly-15k, small model, 8 GPUs, 128 sequences per micro-step, accumulation 1)

Probe: measured (same GPU).

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 93.3 | 98.1 | +5% | 98.2 | +5% |
| random | equal counts | 236.7 | 241.3 | +2% | 236.7 | +0% |
| random | Evencut, balanced tokens | 207.9 | 210.3 | +1% | 206.1 | -1% |
| random | Evencut, balanced padded size | 145.7 | 145.2 | -0% | 144.7 | -1% |
| length-grouped | equal counts | 238.3 | 241.3 | +1% | 236.7 | -1% |
| length-grouped | Evencut, balanced tokens | 64.3 | 65.4 | +2% | 66.5 | +3% |
| length-grouped | Evencut, balanced padded size | 48.0 | 47.3 | -2% | 48.0 | +0% |
| sorted dataset | token budget per GPU (fairseq-style) | 41.6 | 40.1 | -4% | 38.9 | -6% |
| sorted dataset | Evencut, padded_slices over the sorted data | 44.1 | 40.1 | -9% | 38.9 | -12% |

### gpu_batches_dolly_accum8 (4060 sim, Dolly-15k, small model, 8 GPUs, 128 sequences per micro-step, accumulation 8)

Probe: measured (same GPU).

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1909.5 | 1975.5 | +3% | 1904.7 | -0% |
| random | equal counts | 1365.0 | 1407.3 | +3% | 1365.0 | +0% |
| random | Evencut, balanced tokens | 1573.6 | 1375.5 | -13% | 1334.0 | -15% |
| random | Evencut, balanced padded size | 1308.3 | 1144.7 | -13% | 1121.1 | -14% |
| length-grouped | equal counts | 2040.6 | 1888.3 | -7% | 1820.4 | -11% |
| length-grouped | Evencut, balanced tokens | 567.4 | 535.5 | -6% | 534.3 | -6% |
| length-grouped | Evencut, balanced padded size | 407.7 | 362.2 | -11% | 357.5 | -12% |
| any order | LPT on padded size (not contiguous) | 1432.4 | 1240.1 | -13% | 1208.9 | -16% |
| sorted dataset | token budget per GPU (fairseq-style) | 330.9 | 310.2 | -6% | 292.1 | -12% |
| sorted dataset | Evencut, padded_slices over the sorted data | 331.7 | 310.2 | -6% | 292.1 | -12% |
| random | Evencut, whole optimizer step balanced (window) | 337.2 | 313.7 | -7% | 296.3 | -12% |
| random | Hugging Face batch_rebalance | 339.7 | 322.9 | -5% | 307.9 | -9% |
| pooled (like group_by_length) | Hugging Face group_by_length | 395.9 | 377.1 | -5% | 318.1 | -20% |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 369.8 | 346.9 | -6% | 287.3 | -22% |

### gpu_batches_wikipedia (4060 sim, Wikipedia-TR, small model, 8 GPUs, 48 sequences per micro-step, accumulation 1)

Probe: measured (same GPU).

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 68.8 | 61.0 | -11% | 64.0 | -7% |
| random | equal counts | 209.3 | 210.3 | +0% | 209.3 | +0% |
| random | Evencut, balanced tokens | 231.0 | 230.6 | -0% | 231.9 | +0% |
| random | Evencut, balanced padded size | 157.7 | 148.8 | -6% | 151.3 | -4% |
| random | Evencut, padded size + attention | 148.4 | 147.9 | -0% | 150.4 | +1% |
| length-grouped | equal counts | 204.0 | 210.3 | +3% | 209.3 | +3% |
| length-grouped | Evencut, balanced tokens | 97.1 | 100.6 | +4% | 103.7 | +7% |
| length-grouped | Evencut, balanced padded size | 65.7 | 67.8 | +3% | 70.2 | +7% |
| length-grouped | Evencut, padded size + attention | 65.0 | 66.8 | +3% | 69.3 | +7% |
| any order | LPT on padded size (not contiguous) | 152.9 | 158.0 | +3% | 160.3 | +5% |
| sorted dataset | token budget per GPU (fairseq-style) | 59.0 | 60.3 | +2% | 63.2 | +7% |
| sorted dataset | Evencut, padded_slices over the sorted data | 59.2 | 60.3 | +2% | 63.2 | +7% |
| sorted dataset | Evencut, padded + attention over the sorted data | 60.6 | 62.8 | +4% | 65.6 | +8% |

### gpu_batches_wikipedia_v2 (4060 sim, Wikipedia-TR, small model, 8 GPUs, 48 sequences per micro-step, accumulation 1)

Probe: measured (same GPU).

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 63.2 | 61.0 | -3% | 62.1 | -2% |
| random | equal counts | 203.2 | 210.3 | +3% | 203.2 | +0% |
| random | Evencut, balanced tokens | 226.1 | 230.6 | +2% | 225.2 | -0% |
| random | Evencut, balanced padded size | 153.7 | 148.8 | -3% | 146.9 | -4% |
| random | Evencut, padded size + attention | 151.6 | 147.9 | -2% | 146.0 | -4% |
| length-grouped | equal counts | 200.0 | 210.3 | +5% | 203.2 | +2% |
| length-grouped | Evencut, balanced tokens | 100.4 | 100.6 | +0% | 100.6 | +0% |
| length-grouped | Evencut, balanced padded size | 68.4 | 67.8 | -1% | 68.2 | -0% |
| length-grouped | Evencut, padded size + attention | 65.7 | 66.8 | +2% | 67.3 | +2% |
| any order | LPT on padded size (not contiguous) | 181.8 | 158.0 | -13% | 155.6 | -14% |
| sorted dataset | token budget per GPU (fairseq-style) | 61.9 | 60.3 | -3% | 61.4 | -1% |
| sorted dataset | Evencut, padded_slices over the sorted data | 67.0 | 60.3 | -10% | 61.4 | -8% |
| sorted dataset | Evencut, padded + attention over the sorted data | 64.0 | 62.8 | -2% | 63.6 | -1% |
| random | Hugging Face batch_rebalance | 76.4 | 76.6 | +0% | 77.0 | +1% |
| pooled (like group_by_length) | Hugging Face group_by_length | 89.5 | 81.2 | -9% | 75.0 | -16% |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 70.6 | 60.5 | -14% | 53.0 | -25% |

### ddp_a100x4 (4x A100-SXM4-80GB, TinyStories-Turkish, small model, 4 GPUs, 128 sequences per micro-step, accumulation 1)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 21.8 | 23.0 | +5% | 21.2 | -3% |
| random | equal counts | 24.6 | 23.0 | -6% | 24.6 | +0% |
| random | Evencut, balanced tokens | 24.4 | 23.0 | -6% | 24.4 | +0% |
| random | Evencut, balanced padded size | 22.9 | 23.0 | +1% | 22.3 | -2% |
| length-grouped | equal counts | 24.4 | 23.0 | -6% | 24.6 | +1% |
| length-grouped | Evencut, balanced tokens | 20.5 | 23.0 | +12% | 19.4 | -5% |
| length-grouped | Evencut, balanced padded size | 19.7 | 23.0 | +17% | 17.4 | -11% |
| sorted dataset | token budget per GPU (fairseq-style) | 19.3 | 23.0 | +19% | 15.7 | -19% |
| sorted dataset | Evencut, padded_slices over the sorted data | 19.2 | 23.0 | +20% | 15.7 | -18% |

### ddp_a100x4_accum8 (4x A100-SXM4-80GB, TinyStories-Turkish, small model, 4 GPUs, 128 sequences per micro-step, accumulation 8)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 153.6 | 174.6 | +14% | 156.0 | +2% |
| random | equal counts | 171.1 | 174.6 | +2% | 171.1 | +0% |
| random | Evencut, balanced tokens | 170.7 | 174.6 | +2% | 170.1 | -0% |
| random | Evencut, balanced padded size | 165.0 | 174.6 | +6% | 162.8 | -1% |
| length-grouped | equal counts | 181.7 | 174.6 | -4% | 181.6 | -0% |
| length-grouped | Evencut, balanced tokens | 143.5 | 174.6 | +22% | 142.4 | -1% |
| length-grouped | Evencut, balanced padded size | 135.9 | 174.6 | +28% | 126.1 | -7% |
| sorted dataset | token budget per GPU (fairseq-style) | 132.6 | 174.6 | +32% | 112.9 | -15% |
| sorted dataset | Evencut, padded_slices over the sorted data | 131.9 | 174.6 | +32% | 112.9 | -14% |

### ddp_a100x4_base_accum4 (4x A100-SXM4-80GB, TinyStories-Turkish, base model, 4 GPUs, 192 sequences per micro-step, accumulation 4)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 249.1 | 132.0 | -47% | 242.5 | -3% |
| random | equal counts | 276.8 | 147.9 | -47% | 276.8 | +0% |
| random | Evencut, balanced tokens | 277.1 | 147.7 | -47% | 277.4 | +0% |
| random | Evencut, balanced padded size | 262.3 | 138.2 | -47% | 259.8 | -1% |
| length-grouped | equal counts | 286.5 | 153.2 | -47% | 289.9 | +1% |
| length-grouped | Evencut, balanced tokens | 232.5 | 134.6 | -42% | 226.9 | -2% |
| length-grouped | Evencut, balanced padded size | 208.4 | 132.0 | -37% | 198.0 | -5% |
| sorted dataset | token budget per GPU (fairseq-style) | 190.3 | 132.0 | -31% | 176.4 | -7% |
| sorted dataset | Evencut, padded_slices over the sorted data | 190.2 | 132.0 | -31% | 176.4 | -7% |

### ddp_a100x4_dolly (4x A100-SXM4-80GB, Dolly-15k, small model, 4 GPUs, 64 sequences per micro-step, accumulation 1)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 23.8 | 23.0 | -3% | 19.3 | -19% |
| random | equal counts | 40.1 | 23.4 | -42% | 40.1 | +0% |
| random | Evencut, balanced tokens | 35.9 | 23.4 | -35% | 35.2 | -2% |
| random | Evencut, balanced padded size | 29.1 | 23.0 | -21% | 27.0 | -7% |
| length-grouped | equal counts | 40.0 | 23.4 | -42% | 40.1 | +0% |
| length-grouped | Evencut, balanced tokens | 20.3 | 23.0 | +13% | 14.6 | -28% |
| length-grouped | Evencut, balanced padded size | 19.3 | 23.0 | +19% | 11.5 | -41% |
| sorted dataset | token budget per GPU (fairseq-style) | 19.2 | 23.0 | +20% | 8.0 | -58% |
| sorted dataset | Evencut, padded_slices over the sorted data | 19.1 | 23.0 | +20% | 8.0 | -58% |

### ddp_a100x4_dolly_accum8 (4x A100-SXM4-80GB, Dolly-15k, small model, 4 GPUs, 64 sequences per micro-step, accumulation 8)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 173.5 | 174.6 | +1% | 151.5 | -13% |
| random | equal counts | 245.5 | 176.3 | -28% | 245.5 | +0% |
| random | Evencut, balanced tokens | 238.4 | 177.8 | -25% | 237.5 | -0% |
| random | Evencut, balanced padded size | 218.6 | 174.6 | -20% | 213.2 | -2% |
| length-grouped | equal counts | 312.7 | 177.7 | -43% | 327.4 | +5% |
| length-grouped | Evencut, balanced tokens | 142.8 | 174.6 | +22% | 117.9 | -17% |
| length-grouped | Evencut, balanced padded size | 139.5 | 174.6 | +25% | 85.3 | -39% |
| sorted dataset | token budget per GPU (fairseq-style) | 138.8 | 174.6 | +26% | 57.7 | -58% |
| sorted dataset | Evencut, padded_slices over the sorted data | 138.0 | 174.6 | +26% | 57.7 | -58% |

### ddp_a100x4_dolly_base_accum4 (4x A100-SXM4-80GB, Dolly-15k, base model, 4 GPUs, 96 sequences per micro-step, accumulation 4)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 263.1 | 137.4 | -48% | 244.7 | -7% |
| random | equal counts | 461.7 | 259.3 | -44% | 461.7 | +0% |
| random | Evencut, balanced tokens | 439.4 | 245.9 | -44% | 440.4 | +0% |
| random | Evencut, balanced padded size | 401.3 | 222.9 | -44% | 399.5 | -0% |
| length-grouped | equal counts | 534.1 | 305.7 | -43% | 549.5 | +3% |
| length-grouped | Evencut, balanced tokens | 211.1 | 133.1 | -37% | 188.0 | -11% |
| length-grouped | Evencut, balanced padded size | 163.5 | 132.0 | -19% | 140.7 | -14% |
| sorted dataset | token budget per GPU (fairseq-style) | 115.8 | 132.0 | +14% | 94.5 | -18% |
| sorted dataset | Evencut, padded_slices over the sorted data | 115.8 | 132.0 | +14% | 94.5 | -18% |

### ddp_a100x4_dolly_lpt (4x A100-SXM4-80GB, Dolly-15k, small model, 4 GPUs, 64 sequences per micro-step, accumulation 1)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 23.9 | 23.0 | -4% | 19.3 | -19% |
| random | equal counts | 40.1 | 23.4 | -42% | 40.1 | +0% |
| random | Evencut, balanced tokens | 35.8 | 23.4 | -35% | 35.2 | -2% |
| random | Evencut, balanced padded size | 29.0 | 23.0 | -21% | 27.0 | -7% |
| length-grouped | equal counts | 40.0 | 23.4 | -42% | 40.1 | +0% |
| length-grouped | Evencut, balanced tokens | 20.1 | 23.0 | +15% | 14.6 | -27% |
| length-grouped | Evencut, balanced padded size | 18.9 | 23.0 | +22% | 11.4 | -39% |
| any order | LPT on padded size (not contiguous) | 30.6 | 23.0 | -25% | 28.9 | -6% |
| sorted dataset | token budget per GPU (fairseq-style) | 18.7 | 23.0 | +23% | 8.0 | -57% |
| sorted dataset | Evencut, padded_slices over the sorted data | 18.7 | 23.0 | +23% | 8.0 | -57% |

### ddp_a100x4_lpt (4x A100-SXM4-80GB, TinyStories-Turkish, small model, 4 GPUs, 128 sequences per micro-step, accumulation 1)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 21.7 | 23.0 | +6% | 21.1 | -3% |
| random | equal counts | 24.4 | 23.0 | -6% | 24.4 | +0% |
| random | Evencut, balanced tokens | 24.2 | 23.0 | -5% | 24.3 | +0% |
| random | Evencut, balanced padded size | 22.7 | 23.0 | +1% | 22.2 | -2% |
| length-grouped | equal counts | 24.4 | 23.0 | -6% | 24.4 | +0% |
| length-grouped | Evencut, balanced tokens | 20.5 | 23.0 | +12% | 19.3 | -6% |
| length-grouped | Evencut, balanced padded size | 19.6 | 23.0 | +18% | 17.4 | -11% |
| any order | LPT on padded size (not contiguous) | 23.1 | 23.0 | -0% | 22.6 | -2% |
| sorted dataset | token budget per GPU (fairseq-style) | 19.1 | 23.0 | +21% | 15.6 | -18% |
| sorted dataset | Evencut, padded_slices over the sorted data | 19.0 | 23.0 | +21% | 15.6 | -18% |

### ddp_a100x4_wikipedia (4x A100-SXM4-80GB, Wikipedia-TR, small model, 4 GPUs, 24 sequences per micro-step, accumulation 1)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 19.3 | 23.0 | +19% | 12.1 | -37% |
| random | equal counts | 36.6 | 23.0 | -37% | 36.6 | +0% |
| random | Evencut, balanced tokens | 38.9 | 24.2 | -38% | 38.1 | -2% |
| random | Evencut, balanced padded size | 30.6 | 23.0 | -25% | 28.7 | -6% |
| length-grouped | equal counts | 36.5 | 23.0 | -37% | 36.6 | +0% |
| length-grouped | Evencut, balanced tokens | 27.8 | 23.3 | -16% | 23.5 | -15% |
| length-grouped | Evencut, balanced padded size | 20.1 | 23.0 | +14% | 15.0 | -25% |
| any order | LPT on padded size (not contiguous) | 31.0 | 23.0 | -26% | 29.2 | -6% |
| sorted dataset | token budget per GPU (fairseq-style) | 19.1 | 23.0 | +21% | 12.1 | -37% |
| sorted dataset | Evencut, padded_slices over the sorted data | 18.9 | 23.0 | +21% | 12.1 | -36% |

### ddp_a100x4_wikipedia_base_accum4 (4x A100-SXM4-80GB, Wikipedia-TR, base model, 4 GPUs, 24 sequences per micro-step, accumulation 4)

Probe: estimated as pick does for an offer (estimate, different architecture (ada -> ampere): datasheet estimate unreliable), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 122.8 | 132.0 | +7% | 105.4 | -14% |
| random | equal counts | 288.8 | 170.6 | -41% | 288.8 | +0% |
| random | Evencut, balanced tokens | 288.2 | 174.2 | -40% | 284.8 | -1% |
| random | Evencut, balanced padded size | 248.1 | 150.6 | -39% | 242.6 | -2% |
| length-grouped | equal counts | 313.2 | 178.4 | -43% | 320.9 | +2% |
| length-grouped | Evencut, balanced tokens | 225.9 | 143.8 | -36% | 211.6 | -6% |
| length-grouped | Evencut, balanced padded size | 140.5 | 132.0 | -6% | 120.5 | -14% |
| any order | LPT on padded size (not contiguous) | 255.8 | 154.2 | -40% | 251.3 | -2% |
| sorted dataset | token budget per GPU (fairseq-style) | 115.8 | 132.0 | +14% | 95.4 | -18% |
| sorted dataset | Evencut, padded_slices over the sorted data | 115.8 | 132.0 | +14% | 95.4 | -18% |

### ddp_batches (8x RTX 4090, TinyStories-Turkish, small model, 8 GPUs, 256 sequences per micro-step, accumulation 1)

Probe: estimated as pick does for an offer (estimate, no peer-to-peer (GeForce)), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 81.7 | 86.5 | +6% | 78.7 | -4% |
| random | equal counts | 82.2 | 87.3 | +6% | 82.2 | +0% |
| random | Evencut, balanced tokens | 81.9 | 87.1 | +6% | 81.9 | +0% |
| random | Evencut, balanced padded size | 81.7 | 86.5 | +6% | 79.5 | -3% |
| length-grouped | equal counts | 81.9 | 87.3 | +7% | 82.2 | +0% |
| length-grouped | Evencut, balanced tokens | 81.7 | 86.7 | +6% | 77.3 | -5% |
| length-grouped | Evencut, balanced padded size | 81.7 | 86.5 | +6% | 75.4 | -8% |
| sorted dataset | token budget per GPU (fairseq-style) | 81.7 | 86.5 | +6% | 74.7 | -9% |
| sorted dataset | Evencut, padded_slices over the sorted data | 80.9 | 86.5 | +7% | 74.7 | -8% |

### ddp_batches_accum8 (8x RTX 4090, TinyStories-Turkish, small model, 8 GPUs, 256 sequences per micro-step, accumulation 8)

Probe: estimated as pick does for an offer (estimate, no peer-to-peer (GeForce)), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 382.0 | 238.9 | -37% | 344.7 | -10% |
| random | equal counts | 380.4 | 243.5 | -36% | 380.4 | +0% |
| random | Evencut, balanced tokens | 386.3 | 243.0 | -37% | 377.4 | -2% |
| random | Evencut, balanced padded size | 388.1 | 238.9 | -38% | 356.4 | -8% |
| length-grouped | equal counts | 379.6 | 244.1 | -36% | 412.3 | +9% |
| length-grouped | Evencut, balanced tokens | 381.6 | 239.8 | -37% | 317.7 | -17% |
| length-grouped | Evencut, balanced padded size | 383.5 | 238.9 | -38% | 280.1 | -27% |
| sorted dataset | token budget per GPU (fairseq-style) | 378.2 | 238.9 | -37% | 266.8 | -29% |
| sorted dataset | Evencut, padded_slices over the sorted data | 378.0 | 238.9 | -37% | 266.8 | -29% |

### ddp_batches_base_accum4 (8x RTX 4090, TinyStories-Turkish, base model, 8 GPUs, 384 sequences per micro-step, accumulation 4)

Probe: estimated as pick does for an offer (estimate, no peer-to-peer (GeForce)), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 447.9 | 433.6 | -3% | 454.1 | +1% |
| random | equal counts | 504.4 | 479.5 | -5% | 504.4 | +0% |
| random | Evencut, balanced tokens | 498.3 | 475.5 | -5% | 500.0 | +0% |
| random | Evencut, balanced padded size | 470.6 | 453.2 | -4% | 475.6 | +1% |
| length-grouped | equal counts | 528.3 | 504.7 | -4% | 531.9 | +1% |
| length-grouped | Evencut, balanced tokens | 432.8 | 421.3 | -3% | 440.6 | +2% |
| length-grouped | Evencut, balanced padded size | 389.9 | 380.2 | -2% | 395.6 | +1% |
| sorted dataset | token budget per GPU (fairseq-style) | 381.8 | 368.3 | -4% | 382.6 | +0% |
| sorted dataset | Evencut, padded_slices over the sorted data | 384.0 | 368.3 | -4% | 382.6 | -0% |

### ddp_batches_dolly (8x RTX 4090, Dolly-15k, small model, 8 GPUs, 128 sequences per micro-step, accumulation 1)

Probe: estimated as pick does for an offer (estimate, no peer-to-peer (GeForce)), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 86.1 | 87.2 | +1% | 76.3 | -11% |
| random | equal counts | 92.4 | 106.8 | +16% | 92.4 | +0% |
| random | Evencut, balanced tokens | 89.4 | 101.5 | +14% | 88.8 | -1% |
| random | Evencut, balanced padded size | 84.1 | 91.2 | +9% | 81.7 | -3% |
| length-grouped | equal counts | 91.8 | 106.8 | +16% | 92.4 | +1% |
| length-grouped | Evencut, balanced tokens | 82.6 | 87.2 | +6% | 72.6 | -12% |
| length-grouped | Evencut, balanced padded size | 82.6 | 87.2 | +6% | 70.5 | -15% |
| sorted dataset | token budget per GPU (fairseq-style) | 83.0 | 87.2 | +5% | 69.4 | -16% |
| sorted dataset | Evencut, padded_slices over the sorted data | 82.6 | 87.2 | +6% | 69.4 | -16% |

### ddp_batches_dolly_accum8 (8x RTX 4090, Dolly-15k, small model, 8 GPUs, 128 sequences per micro-step, accumulation 8)

Probe: estimated as pick does for an offer (estimate, no peer-to-peer (GeForce)), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 385.3 | 238.7 | -38% | 257.4 | -33% |
| random | equal counts | 392.3 | 318.4 | -19% | 392.3 | +0% |
| random | Evencut, balanced tokens | 388.8 | 301.9 | -22% | 386.6 | -1% |
| random | Evencut, balanced padded size | 386.9 | 268.5 | -31% | 341.9 | -12% |
| length-grouped | equal counts | 429.8 | 392.5 | -9% | 525.2 | +22% |
| length-grouped | Evencut, balanced tokens | 376.9 | 238.7 | -37% | 197.0 | -48% |
| length-grouped | Evencut, balanced padded size | 378.6 | 238.7 | -37% | 154.5 | -59% |
| sorted dataset | token budget per GPU (fairseq-style) | 385.4 | 238.7 | -38% | 138.3 | -64% |
| sorted dataset | Evencut, padded_slices over the sorted data | 376.8 | 238.7 | -37% | 138.3 | -63% |

### ddp_batches_dolly_base_accum4 (8x RTX 4090, Dolly-15k, base model, 8 GPUs, 192 sequences per micro-step, accumulation 4)

Probe: estimated as pick does for an offer (estimate, no peer-to-peer (GeForce)), floor kept.

| data order | split | measured ms | probe ms | error | cost model ms | error |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 462.1 | 451.1 | -2% | 473.8 | +3% |
| random | equal counts | 762.7 | 714.3 | -6% | 762.7 | +0% |
| random | Evencut, balanced tokens | 746.1 | 692.1 | -7% | 738.3 | -1% |
| random | Evencut, balanced padded size | 642.3 | 615.5 | -4% | 654.2 | +2% |
| length-grouped | equal counts | 852.5 | 803.5 | -6% | 860.6 | +1% |
| length-grouped | Evencut, balanced tokens | 384.3 | 370.9 | -3% | 385.7 | +0% |
| length-grouped | Evencut, balanced padded size | 374.7 | 328.0 | -12% | 327.4 | -13% |
| sorted dataset | token budget per GPU (fairseq-style) | 378.0 | 326.7 | -14% | 301.5 | -20% |
| sorted dataset | Evencut, padded_slices over the sorted data | 376.7 | 326.7 | -13% | 301.5 | -20% |

