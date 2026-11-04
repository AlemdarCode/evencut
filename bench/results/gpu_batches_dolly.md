# Test 4: real data on a real GPU (Dolly-15k)

evencut 1.0.0, 2026-11-01, NVIDIA GeForce RTX 4060, PyTorch 2.14.1+cu130. Produced by `bench/gpu_batches.py`. 15,011 sequences (mean 166 tokens, median 103, 90th percentile 374, cut at 1024), 29.9M-parameter decoder, bf16, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

## B. Eight data-parallel GPUs: one step waits for the slowest GPU

Every GPU's slice was run and timed on this GPU; step time = the slowest of the eight. The noise floor gives all eight GPUs the same slice, so its idle share is measurement noise alone.

| data order | split | step time | GPUs idle (average) | real tokens / s (8 GPUs) | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|--:|
| noise floor | identical slices | 93.3 ms | 0.2% | 223,527 | 63.5% |  |  |
| random | equal counts | 236.7 ms | 41.0% | 89,708 | 73.2% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 207.9 ms | 34.2% | 102,123 | 72.5% | 1.14x | 1.14x |
| random | Evencut, balanced padded size | 145.7 ms | 11.1% | 145,793 | 71.0% | 1.63x | 1.63x |
| length-grouped | equal counts | 238.3 ms | 71.6% | 89,128 | 31.8% | 1.00x | 0.99x |
| length-grouped | Evencut, balanced tokens | 64.3 ms | 29.0% | 330,411 | 20.3% | 3.71x | 3.68x |
| length-grouped | Evencut, balanced padded size | 48.0 ms | 7.3% | 442,269 | 18.5% | 4.96x | 4.93x |
| sorted dataset | token budget per GPU (fairseq-style) | 41.6 ms | 6.3% | 510,208 | 0.6% | 1.00x | 5.69x |
| sorted dataset | Evencut, padded_slices over the sorted data | 44.1 ms | 7.7% | 481,446 | 0.6% | 0.94x | 5.37x |

"sorted dataset" sorts all sequences of the run, cuts them into one batch per GPU and step, and shuffles the batches; its first row is greedy max-token batching (fairseq-style), the strongest common baseline. Here its first two rows hand out exactly the same batches (checked by the script): with the whole dataset sorted, the best max-token budget already gives the optimal split, and the gap between the two rows is run-to-run timing variation.

### Adding gradient communication (modelled, not measured)

Each step also all-reduces the 29.9M bf16 gradients (~60 MB) across the eight GPUs, the same for every split. Rough ring all-reduce times are added below; real frameworks overlap part of it with the backward pass, so these are upper bounds.

| data order | best Evencut split vs the order's first row: none | NVLink-class, ~0.5 ms | PCIe 4.0, ~7 ms |
|---|--:|--:|--:|
| random (Evencut, balanced padded size vs equal counts) | 1.63x | 1.62x | 1.60x |
| length-grouped (Evencut, balanced padded size vs equal counts) | 4.96x | 4.92x | 4.46x |
| sorted dataset (Evencut, padded_slices over the sorted data vs token budget per GPU (fairseq-style)) | 0.94x | 0.94x | 0.95x |
