# Test 4: real data on a real GPU (TinyStories-Turkish)

evencut 1.0.0, 2026-11-01, NVIDIA GeForce RTX 4060, PyTorch 2.14.1+cu130. Produced by `bench/gpu_batches.py`. 359,954 sequences (mean 154 tokens, median 152, 90th percentile 191, cut at 512), 29.9M-parameter decoder, bf16, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

## A. One GPU: forming padded batches (full training steps)

| method | steps | padding (wasted share) | time | real tokens / s | vs random order |
|---|--:|--:|--:|--:|--:|
| random order, fixed size | 640 | 30.7% | 61.5 s | 51,217 | 1.00x |
| sorted, fixed size | 640 | 0.3% | 42.9 s | 73,508 | 1.44x |
| sorted, greedy budget | 195 | 0.5% | 44.1 s | 71,466 | 1.40x |
| sorted, Evencut token budget (ignores padding) | 346 | 0.3% | 42.8 s | 73,534 | 1.44x |
| sorted, Evencut padded budget | 195 | 0.5% | 44.7 s | 70,517 | 1.38x |

## B. Eight data-parallel GPUs: one step waits for the slowest GPU

Every GPU's slice was run and timed on this GPU; step time = the slowest of the eight. The noise floor gives all eight GPUs the same slice, so its idle share is measurement noise alone.

| data order | split | step time | GPUs idle (average) | real tokens / s (8 GPUs) | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|--:|
| noise floor | identical slices | 93.5 ms | 2.2% | 405,391 | 30.5% |  |  |
| random | equal counts | 114.5 ms | 17.4% | 343,825 | 30.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 113.8 ms | 16.3% | 345,880 | 30.6% | 1.01x | 1.01x |
| random | Evencut, balanced padded size | 97.2 ms | 5.2% | 405,071 | 30.1% | 1.18x | 1.18x |
| length-grouped | equal counts | 111.1 ms | 38.4% | 354,145 | 7.6% | 1.00x | 1.03x |
| length-grouped | Evencut, balanced tokens | 78.8 ms | 15.2% | 499,436 | 6.6% | 1.41x | 1.45x |
| length-grouped | Evencut, balanced padded size | 68.9 ms | 3.5% | 571,553 | 6.0% | 1.61x | 1.66x |
| sorted dataset | token budget per GPU (fairseq-style) | 65.0 ms | 3.9% | 605,349 | 0.3% | 1.00x | 1.76x |
| sorted dataset | Evencut, padded_slices over the sorted data | 64.8 ms | 3.3% | 606,992 | 0.3% | 1.00x | 1.77x |

"sorted dataset" sorts all sequences of the run, cuts them into one batch per GPU and step, and shuffles the batches; its first row is greedy max-token batching (fairseq-style), the strongest common baseline. Here its first two rows hand out exactly the same batches (checked by the script): with the whole dataset sorted, the best max-token budget already gives the optimal split, and the gap between the two rows is run-to-run timing variation.

### Adding gradient communication (modelled, not measured)

Each step also all-reduces the 29.9M bf16 gradients (~60 MB) across the eight GPUs, the same for every split. Rough ring all-reduce times are added below; real frameworks overlap part of it with the backward pass, so these are upper bounds.

| data order | best Evencut split vs the order's first row: none | NVLink-class, ~0.5 ms | PCIe 4.0, ~7 ms |
|---|--:|--:|--:|
| random (Evencut, balanced padded size vs equal counts) | 1.18x | 1.18x | 1.17x |
| length-grouped (Evencut, balanced padded size vs equal counts) | 1.61x | 1.61x | 1.56x |
| sorted dataset (Evencut, padded_slices over the sorted data vs token budget per GPU (fairseq-style)) | 1.00x | 1.00x | 1.00x |
