# Test 4: real data on a real GPU (Dolly-15k)

evencut 1.0.0, 2026-11-02, NVIDIA GeForce RTX 4060, PyTorch 2.14.1+cu130. Produced by `bench/gpu_batches.py`. 15,011 sequences (mean 166 tokens, median 103, 90th percentile 374, cut at 1024), 29.9M-parameter decoder, bf16, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

## B. Eight data-parallel GPUs: one step waits for the slowest GPU

Every GPU's slice was run and timed on this GPU; step time = the slowest of the eight. The noise floor gives all eight GPUs the same slice, so its idle share is measurement noise alone. Gradient accumulation 8: GPUs wait for each other once per optimizer step (DDP with no_sync), so each GPU's time is summed over the 8 micro-steps and the step waits for the slowest sum; times are per optimizer step.

| data order | split | step time | GPUs idle (average) | real tokens / s (8 GPUs) | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|--:|
| noise floor | identical slices | 1909.5 ms | 0.0% | 92,171 | 83.2% |  |  |
| random | equal counts | 1365.0 ms | 18.5% | 123,944 | 73.5% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 1573.6 ms | 18.3% | 107,509 | 73.3% | 0.87x | 0.87x |
| random | Evencut, balanced padded size | 1308.3 ms | 8.7% | 129,310 | 71.4% | 1.04x | 1.04x |
| length-grouped | equal counts | 2040.6 ms | 75.1% | 82,905 | 31.0% | 1.00x | 0.67x |
| length-grouped | Evencut, balanced tokens | 567.4 ms | 31.2% | 298,163 | 20.2% | 3.60x | 2.41x |
| length-grouped | Evencut, balanced padded size | 407.7 ms | 6.8% | 414,923 | 18.1% | 5.00x | 3.35x |
| any order | LPT on padded size (not contiguous) | 1432.4 ms | 3.2% | 118,110 | 74.9% | 1.00x | 0.95x |
| sorted dataset | token budget per GPU (fairseq-style) | 330.9 ms | 3.7% | 511,239 | 0.3% | 1.00x | 4.12x |
| sorted dataset | Evencut, padded_slices over the sorted data | 331.7 ms | 3.6% | 510,080 | 0.3% | 1.00x | 4.12x |
| random | Evencut, whole optimizer step balanced (window) | 337.2 ms | 3.6% | 501,786 | 2.9% | 4.05x | 4.05x |
| random | Hugging Face batch_rebalance | 339.7 ms | 3.7% | 498,075 | 3.1% | 4.02x | 4.02x |
| pooled (like group_by_length) | Hugging Face group_by_length | 395.9 ms | 7.3% | 427,347 | 1.4% | 1.00x | 3.45x |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 369.8 ms | 2.6% | 457,503 | 0.8% | 1.07x | 3.69x |

"sorted dataset" sorts all sequences of the run, cuts them into one batch per GPU and step, and shuffles the batches; its first row is greedy max-token batching (fairseq-style), the strongest common baseline. Here its first two rows hand out exactly the same batches (checked by the script): with the whole dataset sorted, the best max-token budget already gives the optimal split, and the gap between the two rows is run-to-run timing variation.

### Adding gradient communication (modelled, not measured)

Each step also all-reduces the 29.9M bf16 gradients (~60 MB) across the eight GPUs, the same for every split. Rough ring all-reduce times are added below; real frameworks overlap part of it with the backward pass, so these are upper bounds.

| data order | best Evencut split vs the order's first row: none | NVLink-class, ~0.5 ms | PCIe 4.0, ~7 ms |
|---|--:|--:|--:|
| random (Evencut, whole optimizer step balanced (window) vs equal counts) | 4.05x | 4.04x | 3.99x |
| length-grouped (Evencut, balanced padded size vs equal counts) | 5.00x | 5.00x | 4.94x |
| sorted dataset (Evencut, padded_slices over the sorted data vs token budget per GPU (fairseq-style)) | 1.00x | 1.00x | 1.00x |
| pooled (like group_by_length) (Evencut, length-grouped steps + balanced vs Hugging Face group_by_length) | 1.07x | 1.07x | 1.07x |
