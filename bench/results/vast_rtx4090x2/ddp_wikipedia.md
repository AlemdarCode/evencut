# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-7.0.0-42-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 12 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 46.9 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 58.9 ms | 71,467 | 51.9% |  |  |
| random | equal counts | 76.1 ms | 84,517 | 62.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 74.5 ms | 86,304 | 62.5% | 1.02x | 1.02x |
| random | Evencut, balanced padded size | 70.5 ms | 91,127 | 60.5% | 1.08x | 1.08x |
| length-grouped | equal counts | 76.6 ms | 83,926 | 48.8% | 1.00x | 0.99x |
| length-grouped | Evencut, balanced tokens | 71.5 ms | 89,919 | 51.1% | 1.07x | 1.06x |
| length-grouped | Evencut, balanced padded size | 63.1 ms | 101,905 | 40.3% | 1.21x | 1.21x |
| any order | LPT on padded size (not contiguous) | 69.6 ms | 92,341 | 61.8% | 1.00x | 1.09x |
| sorted dataset | token budget per GPU (fairseq-style) | 58.6 ms | 109,780 | 2.2% | 1.00x | 1.30x |
| sorted dataset | Evencut, padded_slices over the sorted data | 58.5 ms | 109,853 | 2.2% | 1.00x | 1.30x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PHB	0-63	0		N/A
GPU1	PHB	 X 	0-63	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
