# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1079-nvidia-x86_64-with-glibc2.39. Model small (29.9M parameters), 12 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 11.7 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 39.9 ms | 105,370 | 51.9% |  |  |
| random | equal counts | 49.1 ms | 130,938 | 62.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 47.6 ms | 134,997 | 62.5% | 1.03x | 1.03x |
| random | Evencut, balanced padded size | 45.9 ms | 139,925 | 60.5% | 1.07x | 1.07x |
| length-grouped | equal counts | 49.8 ms | 129,016 | 48.8% | 1.00x | 0.99x |
| length-grouped | Evencut, balanced tokens | 47.8 ms | 134,345 | 51.1% | 1.04x | 1.03x |
| length-grouped | Evencut, balanced padded size | 41.3 ms | 155,501 | 40.3% | 1.21x | 1.19x |
| any order | LPT on padded size (not contiguous) | 46.2 ms | 139,254 | 61.8% | 1.00x | 1.06x |
| sorted dataset | token budget per GPU (fairseq-style) | 39.2 ms | 163,780 | 2.2% | 1.00x | 1.25x |
| sorted dataset | Evencut, padded_slices over the sorted data | 39.0 ms | 164,982 | 2.2% | 1.01x | 1.26x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	SYS	0-63,128-191	0		N/A
GPU1	SYS	 X 	64-127,192-255	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
