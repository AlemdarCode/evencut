# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-124-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 32 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 2.3 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 176.2 ms | 236,691 | 63.5% |  |  |
| random | equal counts | 236.1 ms | 175,610 | 73.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 234.6 ms | 176,668 | 73.3% | 1.01x | 1.01x |
| random | Evencut, balanced padded size | 221.0 ms | 187,612 | 72.1% | 1.07x | 1.07x |
| length-grouped | equal counts | 275.2 ms | 150,605 | 61.9% | 1.00x | 0.86x |
| length-grouped | Evencut, balanced tokens | 163.2 ms | 253,948 | 53.2% | 1.69x | 1.45x |
| length-grouped | Evencut, balanced padded size | 143.8 ms | 288,300 | 50.5% | 1.91x | 1.64x |
| any order | LPT on padded size (not contiguous) | 232.1 ms | 178,588 | 74.1% | 1.00x | 1.02x |
| sorted dataset | token budget per GPU (fairseq-style) | 139.7 ms | 296,795 | 1.3% | 1.00x | 1.69x |
| sorted dataset | Evencut, padded_slices over the sorted data | 139.0 ms | 298,318 | 1.3% | 1.01x | 1.70x |
| random | Evencut, whole optimizer step balanced (window) | 139.5 ms | 297,086 | 9.8% | 1.69x | 1.69x |
| pooled (like group_by_length) | Hugging Face group_by_length | 149.4 ms | 277,389 | 1.7% | 1.00x | 1.58x |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 139.3 ms | 297,547 | 1.8% | 1.07x | 1.69x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV4	48-63,112-127	3		N/A
GPU1	NV4	 X 	16-31,80-95	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
