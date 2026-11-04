# Backtest: `evencut-train diagnose` against measured runs

Predicted vs measured speed-up over the default split (random order, equal counts), per data-parallel step. Produced by `bench/backtest_diagnose.py` from the runs in `bench/results` (the same lengths, seeds and plan code). Error = predicted / measured - 1.

* **padded**: cost = padded tokens, compute-bound (no model shape, no times given).
* **attention**: padded tokens plus causal attention for the run's model shape.
* **+comm**: attention cost, plus the run's measured all-reduce as a fixed cost per optimizer step, scaled by the measured default step time (what `--step-ms` and `--fixed-ms` add, with the all-reduce as the fixed part).
* **2 points**: attention cost, with step time = a + b x cost fitted to two measured rows, the default and the noise floor (all GPUs get the same slice), and the other rows predicted from it. A probe that times two batch sizes on the machine gives the same two points; `a` is the fixed time per step (kernel launches, optimizer, communication).

The noise floor and default rows are the calibration points and are left out of the summary.

## TinyStories-Turkish, 4060 sim, small model, accumulation 1 (`gpu_batches.json`)

8 GPUs, 256 sequences per micro-step, 40 micro-steps, cut at 512; attention constant 7296; all-reduce 0.0 ms; fitted fixed time a = 7.9 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.22x | 1.24x (+1%) | 1.25x (+2%) | 1.25x (+2%) | 1.22x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.01x | 1.02x (+1%) | 1.02x (+1%) | 1.02x (+1%) | 1.01x (+1%) |
| random | Evencut, balanced padded size | 1.18x | 1.17x (-1%) | 1.18x (+0%) | 1.18x (+0%) | 1.16x (-1%) |
| length-grouped | equal counts | 1.03x | 1.00x (-3%) | 1.00x (-3%) | 1.00x (-3%) | 1.00x (-3%) |
| length-grouped | Evencut, balanced tokens | 1.45x | 1.37x (-6%) | 1.37x (-5%) | 1.37x (-5%) | 1.34x (-8%) |
| length-grouped | Evencut, balanced padded size | 1.66x | 1.58x (-5%) | 1.60x (-4%) | 1.60x (-4%) | 1.54x (-7%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.76x | 1.68x (-4%) | 1.71x (-3%) | 1.71x (-3%) | 1.63x (-7%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.77x | 1.68x (-5%) | 1.71x (-3%) | 1.71x (-3%) | 1.63x (-8%) |

## Dolly-15k, 4060 sim, small model, accumulation 1 (`gpu_batches_dolly.json`)

8 GPUs, 128 sequences per micro-step, 40 micro-steps, cut at 1024; attention constant 7296; all-reduce 0.0 ms; fitted fixed time a = -8.4 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 2.54x | 2.25x (-11%) | 2.41x (-5%) | 2.41x (-5%) | 2.54x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.14x | 1.14x (-0%) | 1.15x (+1%) | 1.15x (+1%) | 1.15x (+1%) |
| random | Evencut, balanced padded size | 1.63x | 1.60x (-2%) | 1.64x (+1%) | 1.64x (+1%) | 1.67x (+3%) |
| length-grouped | equal counts | 0.99x | 1.00x (+1%) | 1.00x (+1%) | 1.00x (+1%) | 1.00x (+1%) |
| length-grouped | Evencut, balanced tokens | 3.68x | 3.17x (-14%) | 3.56x (-3%) | 3.56x (-3%) | 3.92x (+6%) |
| length-grouped | Evencut, balanced padded size | 4.93x | 4.65x (-6%) | 4.93x (-0%) | 4.93x (-0%) | 5.73x (+16%) |
| sorted dataset | token budget per GPU (fairseq-style) | 5.69x | 5.59x (-2%) | 6.08x (+7%) | 6.08x (+7%) | 7.43x (+31%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 5.37x | 5.59x (+4%) | 6.08x (+13%) | 6.08x (+13%) | 7.43x (+38%) |

## Dolly-15k, 4060 sim, small model, accumulation 8 (`gpu_batches_dolly_accum8.json`)

8 GPUs, 128 sequences per micro-step, 80 micro-steps, cut at 1024; attention constant 7296; all-reduce 0.0 ms.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 0.71x | 0.73x (+3%) | 0.72x (+0%) | 0.72x (+0%) |  |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |  |
| random | Evencut, balanced tokens | 0.87x | 1.02x (+17%) | 1.02x (+18%) | 1.02x (+18%) |  |
| random | Evencut, balanced padded size | 1.04x | 1.20x (+15%) | 1.22x (+17%) | 1.22x (+17%) |  |
| length-grouped | equal counts | 0.67x | 0.76x (+14%) | 0.75x (+12%) | 0.75x (+12%) |  |
| length-grouped | Evencut, balanced tokens | 2.41x | 2.32x (-4%) | 2.55x (+6%) | 2.55x (+6%) |  |
| length-grouped | Evencut, balanced padded size | 3.35x | 3.56x (+6%) | 3.82x (+14%) | 3.82x (+14%) |  |
| any order | LPT on padded size (not contiguous) | 0.95x | 1.12x (+18%) | 1.13x (+18%) | 1.13x (+18%) |  |
| sorted dataset | token budget per GPU (fairseq-style) | 4.12x | 4.36x (+6%) | 4.67x (+13%) | 4.67x (+13%) |  |
| sorted dataset | Evencut, padded_slices over the sorted data | 4.12x | 4.36x (+6%) | 4.67x (+14%) | 4.67x (+14%) |  |
| random | Evencut, whole optimizer step balanced (window) | 4.05x | 4.38x (+8%) | 4.61x (+14%) | 4.61x (+14%) |  |
| random | Hugging Face batch_rebalance | 4.02x | 4.13x (+3%) | 4.43x (+10%) | 4.43x (+10%) |  |
| pooled (like group_by_length) | Hugging Face group_by_length | 3.45x | 4.08x (+18%) | 4.29x (+24%) | 4.29x (+24%) |  |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 3.69x | 4.49x (+22%) | 4.75x (+29%) | 4.75x (+29%) |  |

## Wikipedia-TR, 4060 sim, small model, accumulation 1 (`gpu_batches_wikipedia.json`)

8 GPUs, 48 sequences per micro-step, 40 micro-steps, cut at 2048; attention constant 7296; all-reduce 0.0 ms; fitted fixed time a = 6.9 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 3.04x | 2.81x (-8%) | 3.27x (+8%) | 3.27x (+8%) | 3.04x (-0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 0.91x | 0.88x (-3%) | 0.90x (-0%) | 0.90x (-0%) | 0.91x (-0%) |
| random | Evencut, balanced padded size | 1.33x | 1.35x (+2%) | 1.38x (+4%) | 1.38x (+4%) | 1.37x (+3%) |
| random | Evencut, padded size + attention | 1.41x | 1.30x (-8%) | 1.39x (-1%) | 1.39x (-1%) | 1.37x (-3%) |
| length-grouped | equal counts | 1.03x | 1.00x (-3%) | 1.00x (-3%) | 1.00x (-3%) | 1.00x (-3%) |
| length-grouped | Evencut, balanced tokens | 2.16x | 1.64x (-24%) | 2.02x (-6%) | 2.02x (-6%) | 1.95x (-9%) |
| length-grouped | Evencut, balanced padded size | 3.18x | 2.85x (-10%) | 2.98x (-6%) | 2.98x (-6%) | 2.80x (-12%) |
| length-grouped | Evencut, padded size + attention | 3.22x | 2.56x (-20%) | 3.02x (-6%) | 3.02x (-6%) | 2.83x (-12%) |
| any order | LPT on padded size (not contiguous) | 1.37x | 1.26x (-8%) | 1.31x (-5%) | 1.31x (-5%) | 1.29x (-6%) |
| sorted dataset | token budget per GPU (fairseq-style) | 3.55x | 3.06x (-14%) | 3.31x (-7%) | 3.31x (-7%) | 3.08x (-13%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 3.54x | 3.06x (-13%) | 3.31x (-6%) | 3.31x (-6%) | 3.08x (-13%) |
| sorted dataset | Evencut, padded + attention over the sorted data | 3.46x | 2.63x (-24%) | 3.19x (-8%) | 3.19x (-8%) | 2.98x (-14%) |

## Wikipedia-TR, 4060 sim, small model, accumulation 1 (`gpu_batches_wikipedia_v2.json`)

8 GPUs, 48 sequences per micro-step, 40 micro-steps, cut at 2048; attention constant 7296; all-reduce 0.0 ms; fitted fixed time a = 1.5 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 3.22x | 2.81x (-13%) | 3.27x (+2%) | 3.27x (+2%) | 3.22x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 0.90x | 0.88x (-2%) | 0.90x (+0%) | 0.90x (+0%) | 0.90x (+0%) |
| random | Evencut, balanced padded size | 1.32x | 1.35x (+2%) | 1.38x (+5%) | 1.38x (+5%) | 1.38x (+4%) |
| random | Evencut, padded size + attention | 1.34x | 1.30x (-3%) | 1.39x (+4%) | 1.39x (+4%) | 1.39x (+4%) |
| length-grouped | equal counts | 1.02x | 1.00x (-2%) | 1.00x (-2%) | 1.00x (-2%) | 1.00x (-2%) |
| length-grouped | Evencut, balanced tokens | 2.02x | 1.64x (-19%) | 2.02x (-0%) | 2.02x (-0%) | 2.00x (-1%) |
| length-grouped | Evencut, balanced padded size | 2.97x | 2.85x (-4%) | 2.98x (+0%) | 2.98x (+0%) | 2.94x (-1%) |
| length-grouped | Evencut, padded size + attention | 3.09x | 2.56x (-17%) | 3.02x (-2%) | 3.02x (-2%) | 2.98x (-4%) |
| any order | LPT on padded size (not contiguous) | 1.12x | 1.26x (+13%) | 1.31x (+17%) | 1.31x (+17%) | 1.30x (+17%) |
| sorted dataset | token budget per GPU (fairseq-style) | 3.28x | 3.06x (-7%) | 3.31x (+1%) | 3.31x (+1%) | 3.26x (-1%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 3.03x | 3.06x (+1%) | 3.31x (+9%) | 3.31x (+9%) | 3.26x (+7%) |
| sorted dataset | Evencut, padded + attention over the sorted data | 3.17x | 2.63x (-17%) | 3.19x (+1%) | 3.19x (+1%) | 3.14x (-1%) |
| random | Hugging Face batch_rebalance | 2.66x | 2.19x (-18%) | 2.64x (-1%) | 2.64x (-1%) | 2.61x (-2%) |
| pooled (like group_by_length) | Hugging Face group_by_length | 2.27x | 2.55x (+12%) | 2.71x (+19%) | 2.71x (+19%) | 2.68x (+18%) |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 2.88x | 3.42x (+19%) | 3.84x (+33%) | 3.84x (+33%) | 3.76x (+30%) |

## TinyStories-Turkish, 4x A100-SXM4-80GB, small model, accumulation 1 (`ddp_a100x4.json`)

4 GPUs, 128 sequences per micro-step, 40 micro-steps, cut at 512; attention constant 7296; all-reduce 1.0 ms; fitted fixed time a = 5.5 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.12x | 1.16x (+3%) | 1.17x (+4%) | 1.16x (+3%) | 1.12x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.01x | 1.00x (-0%) | 1.01x (-0%) | 1.00x (-0%) | 1.00x (-0%) |
| random | Evencut, balanced padded size | 1.07x | 1.10x (+3%) | 1.11x (+3%) | 1.10x (+2%) | 1.08x (+1%) |
| length-grouped | equal counts | 1.01x | 1.00x (-1%) | 1.00x (-1%) | 1.00x (-1%) | 1.00x (-1%) |
| length-grouped | Evencut, balanced tokens | 1.20x | 1.28x (+7%) | 1.28x (+7%) | 1.26x (+6%) | 1.20x (+1%) |
| length-grouped | Evencut, balanced padded size | 1.25x | 1.42x (+14%) | 1.43x (+15%) | 1.41x (+13%) | 1.31x (+5%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.27x | 1.58x (+24%) | 1.60x (+26%) | 1.56x (+23%) | 1.41x (+11%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.28x | 1.58x (+24%) | 1.60x (+25%) | 1.56x (+22%) | 1.41x (+10%) |

## TinyStories-Turkish, 4x A100-SXM4-80GB, small model, accumulation 8 (`ddp_a100x4_accum8.json`)

4 GPUs, 128 sequences per micro-step, 80 micro-steps, cut at 512; attention constant 7296; all-reduce 1.0 ms.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.11x | 1.09x (-2%) | 1.10x (-1%) | 1.10x (-2%) |  |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |  |
| random | Evencut, balanced tokens | 1.00x | 1.01x (+0%) | 1.01x (+0%) | 1.01x (+0%) |  |
| random | Evencut, balanced padded size | 1.04x | 1.05x (+1%) | 1.05x (+1%) | 1.05x (+1%) |  |
| length-grouped | equal counts | 0.94x | 0.94x (+0%) | 0.94x (+0%) | 0.94x (+0%) |  |
| length-grouped | Evencut, balanced tokens | 1.19x | 1.20x (+1%) | 1.20x (+1%) | 1.20x (+1%) |  |
| length-grouped | Evencut, balanced padded size | 1.26x | 1.35x (+7%) | 1.36x (+8%) | 1.36x (+8%) |  |
| sorted dataset | token budget per GPU (fairseq-style) | 1.29x | 1.50x (+17%) | 1.52x (+18%) | 1.52x (+17%) |  |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.30x | 1.50x (+16%) | 1.52x (+17%) | 1.52x (+17%) |  |

## TinyStories-Turkish, 4x A100-SXM4-80GB, base model, accumulation 4 (`ddp_a100x4_base_accum4.json`)

4 GPUs, 192 sequences per micro-step, 40 micro-steps, cut at 512; attention constant 9899; all-reduce 2.8 ms; fitted fixed time a = 55.3 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.11x | 1.14x (+2%) | 1.14x (+3%) | 1.14x (+3%) | 1.11x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.00x | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) |
| random | Evencut, balanced padded size | 1.06x | 1.06x (+1%) | 1.07x (+1%) | 1.07x (+1%) | 1.05x (-0%) |
| length-grouped | equal counts | 0.97x | 0.96x (-1%) | 0.95x (-1%) | 0.95x (-1%) | 0.96x (-0%) |
| length-grouped | Evencut, balanced tokens | 1.19x | 1.22x (+3%) | 1.22x (+3%) | 1.22x (+2%) | 1.17x (-2%) |
| length-grouped | Evencut, balanced padded size | 1.33x | 1.40x (+5%) | 1.40x (+6%) | 1.40x (+5%) | 1.30x (-2%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.45x | 1.56x (+7%) | 1.58x (+8%) | 1.57x (+8%) | 1.41x (-3%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.46x | 1.56x (+7%) | 1.58x (+8%) | 1.57x (+8%) | 1.41x (-3%) |

## Dolly-15k, 4x A100-SXM4-80GB, small model, accumulation 1 (`ddp_a100x4_dolly.json`)

4 GPUs, 64 sequences per micro-step, 40 micro-steps, cut at 1024; attention constant 7296; all-reduce 1.0 ms; fitted fixed time a = 9.4 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.68x | 2.01x (+19%) | 2.13x (+27%) | 2.07x (+23%) | 1.68x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.12x | 1.13x (+1%) | 1.14x (+2%) | 1.14x (+2%) | 1.11x (-1%) |
| random | Evencut, balanced padded size | 1.38x | 1.47x (+7%) | 1.50x (+9%) | 1.48x (+8%) | 1.34x (-2%) |
| length-grouped | equal counts | 1.00x | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) |
| length-grouped | Evencut, balanced tokens | 1.97x | 2.59x (+32%) | 2.87x (+46%) | 2.74x (+39%) | 1.99x (+1%) |
| length-grouped | Evencut, balanced padded size | 2.08x | 3.60x (+73%) | 3.74x (+80%) | 3.50x (+68%) | 2.28x (+10%) |
| sorted dataset | token budget per GPU (fairseq-style) | 2.09x | 5.12x (+145%) | 5.58x (+168%) | 4.99x (+139%) | 2.69x (+29%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 2.09x | 5.12x (+144%) | 5.58x (+166%) | 4.99x (+138%) | 2.69x (+28%) |

## Dolly-15k, 4x A100-SXM4-80GB, small model, accumulation 8 (`ddp_a100x4_dolly_accum8.json`)

4 GPUs, 64 sequences per micro-step, 80 micro-steps, cut at 1024; attention constant 7296; all-reduce 1.0 ms; fitted fixed time a = 58.2 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.41x | 1.55x (+10%) | 1.62x (+15%) | 1.62x (+15%) | 1.41x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.03x | 1.03x (-0%) | 1.03x (+0%) | 1.03x (+0%) | 1.03x (-0%) |
| random | Evencut, balanced padded size | 1.12x | 1.14x (+1%) | 1.15x (+3%) | 1.15x (+3%) | 1.11x (-1%) |
| length-grouped | equal counts | 0.79x | 0.76x (-3%) | 0.75x (-5%) | 0.75x (-5%) | 0.80x (+1%) |
| length-grouped | Evencut, balanced tokens | 1.72x | 1.92x (+12%) | 2.09x (+22%) | 2.08x (+21%) | 1.66x (-3%) |
| length-grouped | Evencut, balanced padded size | 1.76x | 2.81x (+60%) | 2.90x (+65%) | 2.88x (+64%) | 2.00x (+14%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.77x | 4.04x (+128%) | 4.31x (+144%) | 4.25x (+140%) | 2.42x (+37%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.78x | 4.04x (+127%) | 4.31x (+143%) | 4.25x (+139%) | 2.42x (+36%) |

## Dolly-15k, 4x A100-SXM4-80GB, base model, accumulation 4 (`ddp_a100x4_dolly_base_accum4.json`)

4 GPUs, 96 sequences per micro-step, 40 micro-steps, cut at 1024; attention constant 9899; all-reduce 2.8 ms; fitted fixed time a = 41.8 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.75x | 1.82x (+4%) | 1.90x (+8%) | 1.89x (+8%) | 1.75x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.05x | 1.05x (-0%) | 1.05x (-0%) | 1.05x (-0%) | 1.04x (-1%) |
| random | Evencut, balanced padded size | 1.15x | 1.15x (-0%) | 1.16x (+1%) | 1.16x (+0%) | 1.14x (-1%) |
| length-grouped | equal counts | 0.86x | 0.85x (-2%) | 0.84x (-3%) | 0.84x (-3%) | 0.85x (-1%) |
| length-grouped | Evencut, balanced tokens | 2.19x | 2.31x (+6%) | 2.48x (+13%) | 2.46x (+12%) | 2.19x (-0%) |
| length-grouped | Evencut, balanced padded size | 2.82x | 3.22x (+14%) | 3.33x (+18%) | 3.28x (+16%) | 2.75x (-3%) |
| sorted dataset | token budget per GPU (fairseq-style) | 3.99x | 4.79x (+20%) | 5.01x (+26%) | 4.89x (+23%) | 3.67x (-8%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 3.99x | 4.79x (+20%) | 5.01x (+26%) | 4.89x (+23%) | 3.67x (-8%) |

## Dolly-15k, 4x A100-SXM4-80GB, small model, accumulation 1 (`ddp_a100x4_dolly_lpt.json`)

4 GPUs, 64 sequences per micro-step, 40 micro-steps, cut at 1024; attention constant 7296; all-reduce 1.0 ms; fitted fixed time a = 9.6 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.68x | 2.01x (+20%) | 2.13x (+27%) | 2.07x (+24%) | 1.68x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.12x | 1.13x (+1%) | 1.14x (+2%) | 1.14x (+2%) | 1.11x (-1%) |
| random | Evencut, balanced padded size | 1.38x | 1.47x (+7%) | 1.50x (+9%) | 1.48x (+8%) | 1.34x (-3%) |
| length-grouped | equal counts | 1.00x | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) |
| length-grouped | Evencut, balanced tokens | 1.99x | 2.59x (+30%) | 2.87x (+44%) | 2.74x (+37%) | 1.98x (-1%) |
| length-grouped | Evencut, balanced padded size | 2.12x | 3.60x (+69%) | 3.74x (+76%) | 3.50x (+65%) | 2.25x (+6%) |
| any order | LPT on padded size (not contiguous) | 1.31x | 1.37x (+5%) | 1.40x (+7%) | 1.38x (+6%) | 1.28x (-2%) |
| sorted dataset | token budget per GPU (fairseq-style) | 2.14x | 5.12x (+139%) | 5.58x (+161%) | 5.00x (+134%) | 2.65x (+24%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 2.15x | 5.12x (+138%) | 5.58x (+160%) | 5.00x (+133%) | 2.65x (+24%) |

## TinyStories-Turkish, 4x A100-SXM4-80GB, small model, accumulation 1 (`ddp_a100x4_lpt.json`)

4 GPUs, 128 sequences per micro-step, 40 micro-steps, cut at 512; attention constant 7296; all-reduce 1.0 ms; fitted fixed time a = 5.5 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.12x | 1.16x (+3%) | 1.17x (+4%) | 1.16x (+3%) | 1.12x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.01x | 1.00x (-0%) | 1.01x (-0%) | 1.00x (-0%) | 1.00x (-0%) |
| random | Evencut, balanced padded size | 1.07x | 1.10x (+2%) | 1.11x (+3%) | 1.10x (+2%) | 1.08x (+1%) |
| length-grouped | equal counts | 1.00x | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) |
| length-grouped | Evencut, balanced tokens | 1.19x | 1.28x (+7%) | 1.28x (+7%) | 1.26x (+6%) | 1.20x (+1%) |
| length-grouped | Evencut, balanced padded size | 1.25x | 1.42x (+14%) | 1.43x (+15%) | 1.41x (+13%) | 1.31x (+4%) |
| any order | LPT on padded size (not contiguous) | 1.06x | 1.08x (+2%) | 1.09x (+2%) | 1.08x (+2%) | 1.07x (+1%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.28x | 1.58x (+23%) | 1.60x (+25%) | 1.56x (+22%) | 1.41x (+10%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.29x | 1.58x (+23%) | 1.60x (+24%) | 1.56x (+21%) | 1.41x (+10%) |

## Wikipedia-TR, 4x A100-SXM4-80GB, small model, accumulation 1 (`ddp_a100x4_wikipedia.json`)

4 GPUs, 24 sequences per micro-step, 40 micro-steps, cut at 2048; attention constant 7296; all-reduce 1.0 ms; fitted fixed time a = 11.5 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.89x | 2.76x (+46%) | 3.21x (+69%) | 3.03x (+60%) | 1.89x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 0.94x | 0.93x (-1%) | 0.96x (+2%) | 0.96x (+2%) | 0.97x (+3%) |
| random | Evencut, balanced padded size | 1.20x | 1.25x (+5%) | 1.29x (+8%) | 1.28x (+7%) | 1.18x (-1%) |
| length-grouped | equal counts | 1.00x | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) |
| length-grouped | Evencut, balanced tokens | 1.31x | 1.35x (+2%) | 1.58x (+20%) | 1.55x (+18%) | 1.34x (+2%) |
| length-grouped | Evencut, balanced padded size | 1.82x | 2.41x (+32%) | 2.54x (+40%) | 2.44x (+34%) | 1.71x (-6%) |
| any order | LPT on padded size (not contiguous) | 1.18x | 1.23x (+4%) | 1.26x (+7%) | 1.25x (+6%) | 1.17x (-1%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.92x | 3.00x (+57%) | 3.22x (+68%) | 3.03x (+58%) | 1.90x (-1%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.93x | 3.00x (+56%) | 3.22x (+67%) | 3.03x (+57%) | 1.90x (-2%) |

## Wikipedia-TR, 4x A100-SXM4-80GB, base model, accumulation 4 (`ddp_a100x4_wikipedia_base_accum4.json`)

4 GPUs, 24 sequences per micro-step, 40 micro-steps, cut at 2048; attention constant 9899; all-reduce 2.8 ms; fitted fixed time a = 29.9 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 2.35x | 2.50x (+6%) | 2.79x (+19%) | 2.74x (+16%) | 2.35x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.00x | 0.99x (-1%) | 1.01x (+1%) | 1.01x (+1%) | 1.01x (+1%) |
| random | Evencut, balanced padded size | 1.16x | 1.18x (+1%) | 1.19x (+2%) | 1.19x (+2%) | 1.17x (+0%) |
| length-grouped | equal counts | 0.92x | 0.91x (-2%) | 0.90x (-3%) | 0.90x (-2%) | 0.91x (-1%) |
| length-grouped | Evencut, balanced tokens | 1.28x | 1.22x (-5%) | 1.37x (+7%) | 1.36x (+7%) | 1.32x (+3%) |
| length-grouped | Evencut, balanced padded size | 2.05x | 2.33x (+13%) | 2.43x (+18%) | 2.40x (+17%) | 2.12x (+3%) |
| any order | LPT on padded size (not contiguous) | 1.13x | 1.14x (+1%) | 1.15x (+2%) | 1.15x (+2%) | 1.13x (+0%) |
| sorted dataset | token budget per GPU (fairseq-style) | 2.49x | 2.90x (+16%) | 3.09x (+24%) | 3.03x (+21%) | 2.54x (+2%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 2.49x | 2.90x (+16%) | 3.09x (+24%) | 3.03x (+21%) | 2.54x (+2%) |

## TinyStories-Turkish, 8x RTX 4090, small model, accumulation 1 (`ddp_batches.json`)

8 GPUs, 256 sequences per micro-step, 40 micro-steps, cut at 512; attention constant 7296; all-reduce 64.2 ms; fitted fixed time a = 79.5 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.01x | 1.24x (+23%) | 1.25x (+24%) | 1.05x (+4%) | 1.01x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.00x | 1.02x (+1%) | 1.02x (+1%) | 1.00x (-0%) | 1.00x (-0%) |
| random | Evencut, balanced padded size | 1.01x | 1.17x (+16%) | 1.18x (+17%) | 1.03x (+3%) | 1.01x (-0%) |
| length-grouped | equal counts | 1.00x | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) | 1.00x (-0%) |
| length-grouped | Evencut, balanced tokens | 1.01x | 1.37x (+36%) | 1.37x (+37%) | 1.06x (+6%) | 1.01x (+0%) |
| length-grouped | Evencut, balanced padded size | 1.01x | 1.58x (+57%) | 1.60x (+59%) | 1.09x (+8%) | 1.01x (+1%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.01x | 1.68x (+67%) | 1.71x (+70%) | 1.10x (+9%) | 1.01x (+1%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.02x | 1.68x (+66%) | 1.71x (+68%) | 1.10x (+8%) | 1.01x (-0%) |

## TinyStories-Turkish, 8x RTX 4090, small model, accumulation 8 (`ddp_batches_accum8.json`)

8 GPUs, 256 sequences per micro-step, 80 micro-steps, cut at 512; attention constant 7296; all-reduce 65.0 ms.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.00x | 1.12x (+13%) | 1.13x (+13%) | 1.10x (+11%) |  |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |  |
| random | Evencut, balanced tokens | 0.98x | 1.01x (+3%) | 1.01x (+3%) | 1.01x (+2%) |  |
| random | Evencut, balanced padded size | 0.98x | 1.08x (+10%) | 1.08x (+10%) | 1.07x (+9%) |  |
| length-grouped | equal counts | 1.00x | 0.91x (-9%) | 0.91x (-9%) | 0.92x (-8%) |  |
| length-grouped | Evencut, balanced tokens | 1.00x | 1.25x (+26%) | 1.25x (+25%) | 1.20x (+20%) |  |
| length-grouped | Evencut, balanced padded size | 0.99x | 1.45x (+47%) | 1.47x (+48%) | 1.36x (+37%) |  |
| sorted dataset | token budget per GPU (fairseq-style) | 1.01x | 1.54x (+54%) | 1.56x (+55%) | 1.43x (+42%) |  |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.01x | 1.54x (+53%) | 1.56x (+55%) | 1.43x (+42%) |  |

## TinyStories-Turkish, 8x RTX 4090, base model, accumulation 4 (`ddp_batches_base_accum4.json`)

8 GPUs, 384 sequences per micro-step, 40 micro-steps, cut at 512; attention constant 9899; all-reduce 195.9 ms; fitted fixed time a = 157.4 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.13x | 1.19x (+5%) | 1.19x (+6%) | 1.11x (-1%) | 1.13x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.01x | 1.01x (+0%) | 1.01x (+0%) | 1.01x (-0%) | 1.01x (-0%) |
| random | Evencut, balanced padded size | 1.07x | 1.10x (+3%) | 1.10x (+3%) | 1.06x (-1%) | 1.07x (-0%) |
| length-grouped | equal counts | 0.95x | 0.92x (-4%) | 0.92x (-4%) | 0.95x (-1%) | 0.94x (-1%) |
| length-grouped | Evencut, balanced tokens | 1.17x | 1.26x (+8%) | 1.26x (+8%) | 1.14x (-2%) | 1.17x (+0%) |
| length-grouped | Evencut, balanced padded size | 1.29x | 1.53x (+18%) | 1.54x (+19%) | 1.27x (-1%) | 1.32x (+2%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.32x | 1.63x (+24%) | 1.65x (+25%) | 1.32x (-0%) | 1.37x (+4%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.31x | 1.63x (+24%) | 1.65x (+26%) | 1.32x (+0%) | 1.37x (+5%) |

## Dolly-15k, 8x RTX 4090, small model, accumulation 1 (`ddp_batches_dolly.json`)

8 GPUs, 128 sequences per micro-step, 40 micro-steps, cut at 1024; attention constant 7296; all-reduce 64.9 ms; fitted fixed time a = 81.6 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.07x | 2.25x (+109%) | 2.41x (+125%) | 1.21x (+13%) | 1.07x (-0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.03x | 1.14x (+10%) | 1.15x (+11%) | 1.04x (+1%) | 1.02x (-2%) |
| random | Evencut, balanced padded size | 1.10x | 1.60x (+45%) | 1.64x (+49%) | 1.13x (+3%) | 1.05x (-5%) |
| length-grouped | equal counts | 1.01x | 1.00x (-1%) | 1.00x (-1%) | 1.00x (-1%) | 1.00x (-1%) |
| length-grouped | Evencut, balanced tokens | 1.12x | 3.17x (+183%) | 3.56x (+218%) | 1.27x (+14%) | 1.09x (-2%) |
| length-grouped | Evencut, balanced padded size | 1.12x | 4.65x (+316%) | 4.93x (+341%) | 1.31x (+17%) | 1.10x (-1%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.11x | 5.59x (+402%) | 6.08x (+447%) | 1.33x (+20%) | 1.11x (-0%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.12x | 5.59x (+399%) | 6.08x (+444%) | 1.33x (+19%) | 1.11x (-1%) |

## Dolly-15k, 8x RTX 4090, small model, accumulation 8 (`ddp_batches_dolly_accum8.json`)

8 GPUs, 128 sequences per micro-step, 80 micro-steps, cut at 1024; attention constant 7296; all-reduce 64.8 ms; fitted fixed time a = 375.3 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.02x | 1.62x (+59%) | 1.70x (+67%) | 1.52x (+50%) | 1.02x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.01x | 1.01x (+0%) | 1.02x (+1%) | 1.01x (+1%) | 1.00x (-1%) |
| random | Evencut, balanced padded size | 1.01x | 1.17x (+15%) | 1.18x (+17%) | 1.15x (+13%) | 1.01x (-1%) |
| length-grouped | equal counts | 0.91x | 0.73x (-20%) | 0.71x (-22%) | 0.75x (-18%) | 0.98x (+8%) |
| length-grouped | Evencut, balanced tokens | 1.04x | 2.26x (+117%) | 2.48x (+138%) | 1.99x (+91%) | 1.03x (-1%) |
| length-grouped | Evencut, balanced padded size | 1.04x | 3.40x (+228%) | 3.65x (+252%) | 2.54x (+145%) | 1.03x (-0%) |
| sorted dataset | token budget per GPU (fairseq-style) | 1.02x | 4.18x (+310%) | 4.46x (+338%) | 2.84x (+179%) | 1.03x (+2%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 1.04x | 4.18x (+301%) | 4.46x (+328%) | 2.84x (+173%) | 1.03x (-1%) |

## Dolly-15k, 8x RTX 4090, base model, accumulation 4 (`ddp_batches_dolly_base_accum4.json`)

8 GPUs, 192 sequences per micro-step, 40 micro-steps, cut at 1024; attention constant 9899; all-reduce 196.6 ms; fitted fixed time a = 173.6 ms per optimizer step.

| order | split | measured | padded | attention | +comm | 2 points |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1.65x | 1.96x (+18%) | 2.04x (+24%) | 1.61x (-2%) | 1.65x (+0%) |
| random | equal counts | 1.00x | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) | 1.00x (+0%) |
| random | Evencut, balanced tokens | 1.02x | 1.04x (+2%) | 1.05x (+2%) | 1.03x (+1%) | 1.03x (+1%) |
| random | Evencut, balanced padded size | 1.19x | 1.23x (+3%) | 1.24x (+4%) | 1.17x (-2%) | 1.17x (-1%) |
| length-grouped | equal counts | 0.89x | 0.86x (-4%) | 0.85x (-5%) | 0.89x (-1%) | 0.88x (-1%) |
| length-grouped | Evencut, balanced tokens | 1.98x | 2.77x (+39%) | 2.99x (+51%) | 1.98x (-0%) | 2.06x (+4%) |
| length-grouped | Evencut, balanced padded size | 2.04x | 4.15x (+104%) | 4.33x (+113%) | 2.33x (+14%) | 2.46x (+21%) |
| sorted dataset | token budget per GPU (fairseq-style) | 2.02x | 5.12x (+154%) | 5.40x (+168%) | 2.53x (+25%) | 2.70x (+34%) |
| sorted dataset | Evencut, padded_slices over the sorted data | 2.03x | 5.12x (+153%) | 5.40x (+167%) | 2.53x (+25%) | 2.70x (+33%) |

## Summary: absolute error of the predicted speed-up (median / worst)

| runs | rows | padded | attention | +comm | 2 points |
|---|--:|--:|--:|--:|--:|
| 4060 sim | 51 | 6% / 24% | 5% / 33% | 5% / 33% | 4% / 38% |
| 4x A100-SXM4-80GB | 74 | 5% / 145% | 7% / 168% | 7% / 140% | 2% / 37% |
| 8x RTX 4090 small | 14 | 51% / 402% | 54% / 447% | 7% / 20% | 1% / 5% |
| 8x RTX 4090 | 28 | 22% / 310% | 24% / 338% | 8% / 179% | 1% / 34% |

## Does the prediction pick the right split?

For every run, the split with the highest predicted speed-up is taken; the table gives its measured speed as a share of the best measured split (worst run / runs within 3% of the best).

| runs | runs | padded | attention | +comm | 2 points |
|---|--:|--:|--:|--:|--:|
| 4060 sim | 5 | 88% / 3 of 5 | 88% / 3 of 5 | 88% / 3 of 5 | 88% / 3 of 4 |
| 4x A100-SXM4-80GB | 10 | 99% / 10 of 10 | 99% / 10 of 10 | 99% / 10 of 10 | 99% / 9 of 9 |
| 8x RTX 4090 small | 2 | 99% / 2 of 2 | 99% / 2 of 2 | 99% / 2 of 2 | 99% / 2 of 2 |
| 8x RTX 4090 | 4 | 98% / 4 of 4 | 98% / 4 of 4 | 98% / 4 of 4 | 98% / 3 of 3 |

Among splits that keep every step's sequences (what `diagnose` advises first; it shows length grouping and whole-dataset sorting only as cautioned extras):

| runs | runs | padded | attention | +comm | 2 points |
|---|--:|--:|--:|--:|--:|
| 4060 sim | 5 | 96% / 4 of 5 | 100% / 5 of 5 | 100% / 5 of 5 | 100% / 4 of 4 |
| 4x A100-SXM4-80GB | 10 | 100% / 10 of 10 | 100% / 10 of 10 | 100% / 10 of 10 | 100% / 9 of 9 |
| 8x RTX 4090 small | 2 | 100% / 2 of 2 | 100% / 2 of 2 | 100% / 2 of 2 | 100% / 2 of 2 |
| 8x RTX 4090 | 4 | 100% / 4 of 4 | 100% / 4 of 4 | 100% / 4 of 4 | 100% / 3 of 3 |
