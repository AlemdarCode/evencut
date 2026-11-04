# Test 4: real data on a real GPU (Wikipedia-TR)

evencut 1.0.0, 2026-11-01, NVIDIA GeForce RTX 4060, PyTorch 2.14.1+cu130. Produced by `bench/gpu_batches.py`. 534,988 sequences (mean 510 tokens, median 275, 90th percentile 1474, cut at 2048), 29.9M-parameter decoder, bf16, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

## B. Eight data-parallel GPUs: one step waits for the slowest GPU

Every GPU's slice was run and timed on this GPU; step time = the slowest of the eight. The noise floor gives all eight GPUs the same slice, so its idle share is measurement noise alone.

| data order | split | step time | GPUs idle (average) | real tokens / s (8 GPUs) | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|--:|
| noise floor | identical slices | 63.2 ms | 4.3% | 266,538 | 51.9% |  |  |
| random | equal counts | 203.2 ms | 34.3% | 122,798 | 63.0% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 226.1 ms | 39.7% | 110,372 | 63.9% | 0.90x | 0.90x |
| random | Evencut, balanced padded size | 153.7 ms | 19.1% | 162,320 | 59.7% | 1.32x | 1.32x |
| random | Evencut, padded size + attention | 151.6 ms | 18.8% | 164,565 | 59.6% | 1.34x | 1.34x |
| length-grouped | equal counts | 200.0 ms | 64.6% | 124,762 | 17.3% | 1.00x | 1.02x |
| length-grouped | Evencut, balanced tokens | 100.4 ms | 40.0% | 248,552 | 20.8% | 1.99x | 2.02x |
| length-grouped | Evencut, balanced padded size | 68.4 ms | 18.5% | 364,747 | 15.1% | 2.92x | 2.97x |
| length-grouped | Evencut, padded size + attention | 65.7 ms | 15.5% | 379,654 | 16.4% | 3.04x | 3.09x |
| any order | LPT on padded size (not contiguous) | 181.8 ms | 14.7% | 137,240 | 65.7% | 1.00x | 1.12x |
| sorted dataset | token budget per GPU (fairseq-style) | 61.9 ms | 19.5% | 403,177 | 0.6% | 1.00x | 3.28x |
| sorted dataset | Evencut, padded_slices over the sorted data | 67.0 ms | 21.0% | 372,636 | 0.6% | 0.92x | 3.03x |
| sorted dataset | Evencut, padded + attention over the sorted data | 64.0 ms | 21.5% | 389,730 | 0.8% | 0.97x | 3.17x |
| random | Hugging Face batch_rebalance | 76.4 ms | 23.9% | 326,562 | 18.5% | 2.66x | 2.66x |
| pooled (like group_by_length) | Hugging Face group_by_length | 89.5 ms | 27.2% | 278,976 | 3.6% | 1.00x | 2.27x |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 70.6 ms | 8.8% | 353,677 | 3.7% | 1.27x | 2.88x |

"sorted dataset" sorts all sequences of the run, cuts them into one batch per GPU and step, and shuffles the batches; its first row is greedy max-token batching (fairseq-style), the strongest common baseline. Here its first two rows hand out exactly the same batches (checked by the script): with the whole dataset sorted, the best max-token budget already gives the optimal split, and the gap between the two rows is run-to-run timing variation.

### Adding gradient communication (modelled, not measured)

Each step also all-reduces the 29.9M bf16 gradients (~60 MB) across the eight GPUs, the same for every split. Rough ring all-reduce times are added below; real frameworks overlap part of it with the backward pass, so these are upper bounds.

| data order | best Evencut split vs the order's first row: none | NVLink-class, ~0.5 ms | PCIe 4.0, ~7 ms |
|---|--:|--:|--:|
| random (Hugging Face batch_rebalance vs equal counts) | 2.66x | 2.65x | 2.52x |
| length-grouped (Evencut, padded size + attention vs equal counts) | 3.04x | 3.03x | 2.85x |
| sorted dataset (Evencut, padded + attention over the sorted data vs token budget per GPU (fairseq-style)) | 0.97x | 0.97x | 0.97x |
| pooled (like group_by_length) (Evencut, length-grouped steps + balanced vs Hugging Face group_by_length) | 1.27x | 1.27x | 1.24x |
