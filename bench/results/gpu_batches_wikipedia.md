# Test 4: real data on a real GPU (Wikipedia-TR)

evencut 1.0.0, 2026-11-01, NVIDIA GeForce RTX 4060, PyTorch 2.14.1+cu130. Produced by `bench/gpu_batches.py`. 534,988 sequences (mean 510 tokens, median 275, 90th percentile 1474, cut at 2048), 29.9M-parameter decoder, bf16, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

## B. Eight data-parallel GPUs: one step waits for the slowest GPU

Every GPU's slice was run and timed on this GPU; step time = the slowest of the eight. The noise floor gives all eight GPUs the same slice, so its idle share is measurement noise alone.

| data order | split | step time | GPUs idle (average) | real tokens / s (8 GPUs) | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|--:|
| noise floor | identical slices | 68.8 ms | 2.0% | 244,656 | 51.9% |  |  |
| random | equal counts | 209.3 ms | 34.6% | 119,208 | 63.0% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 231.0 ms | 39.9% | 108,039 | 63.9% | 0.91x | 0.91x |
| random | Evencut, balanced padded size | 157.7 ms | 18.6% | 158,283 | 59.7% | 1.33x | 1.33x |
| random | Evencut, padded size + attention | 148.4 ms | 16.8% | 168,214 | 59.6% | 1.41x | 1.41x |
| length-grouped | equal counts | 204.0 ms | 65.6% | 122,348 | 17.3% | 1.00x | 1.03x |
| length-grouped | Evencut, balanced tokens | 97.1 ms | 39.9% | 257,061 | 20.8% | 2.10x | 2.16x |
| length-grouped | Evencut, balanced padded size | 65.7 ms | 18.0% | 379,561 | 15.1% | 3.10x | 3.18x |
| length-grouped | Evencut, padded size + attention | 65.0 ms | 15.5% | 384,101 | 16.4% | 3.14x | 3.22x |
| any order | LPT on padded size (not contiguous) | 152.9 ms | 7.5% | 163,233 | 65.7% | 1.00x | 1.37x |
| sorted dataset | token budget per GPU (fairseq-style) | 59.0 ms | 19.2% | 423,044 | 0.6% | 1.00x | 3.55x |
| sorted dataset | Evencut, padded_slices over the sorted data | 59.2 ms | 19.4% | 421,584 | 0.6% | 1.00x | 3.54x |
| sorted dataset | Evencut, padded + attention over the sorted data | 60.6 ms | 20.3% | 411,863 | 0.8% | 0.97x | 3.46x |

"sorted dataset" sorts all sequences of the run, cuts them into one batch per GPU and step, and shuffles the batches; its first row is greedy max-token batching (fairseq-style), the strongest common baseline. Here its first two rows hand out exactly the same batches (checked by the script): with the whole dataset sorted, the best max-token budget already gives the optimal split, and the gap between the two rows is run-to-run timing variation.

### Adding gradient communication (modelled, not measured)

Each step also all-reduces the 29.9M bf16 gradients (~60 MB) across the eight GPUs, the same for every split. Rough ring all-reduce times are added below; real frameworks overlap part of it with the backward pass, so these are upper bounds.

| data order | best Evencut split vs the order's first row: none | NVLink-class, ~0.5 ms | PCIe 4.0, ~7 ms |
|---|--:|--:|--:|
| random (Evencut, padded size + attention vs equal counts) | 1.41x | 1.41x | 1.39x |
| length-grouped (Evencut, padded size + attention vs equal counts) | 3.14x | 3.12x | 2.93x |
| sorted dataset (Evencut, padded_slices over the sorted data vs token budget per GPU (fairseq-style)) | 1.00x | 1.00x | 1.00x |
