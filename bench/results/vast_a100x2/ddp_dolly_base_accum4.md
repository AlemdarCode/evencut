# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1079-nvidia-x86_64-with-glibc2.39. Model base (91.2M parameters), 48 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 35.1 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 318.9 ms | 87,815 | 67.3% |  |  |
| random | equal counts | 503.0 ms | 59,996 | 76.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 473.1 ms | 63,784 | 76.1% | 1.06x | 1.06x |
| random | Evencut, balanced padded size | 452.5 ms | 66,695 | 75.5% | 1.11x | 1.11x |
| length-grouped | equal counts | 551.7 ms | 54,698 | 65.9% | 1.00x | 0.91x |
| length-grouped | Evencut, balanced tokens | 298.4 ms | 101,115 | 55.4% | 1.85x | 1.69x |
| length-grouped | Evencut, balanced padded size | 267.9 ms | 112,643 | 54.1% | 2.06x | 1.88x |
| any order | LPT on padded size (not contiguous) | 481.4 ms | 62,689 | 77.7% | 1.00x | 1.04x |
| sorted dataset | token budget per GPU (fairseq-style) | 221.4 ms | 136,285 | 2.5% | 1.00x | 2.27x |
| sorted dataset | Evencut, padded_slices over the sorted data | 221.2 ms | 136,398 | 2.5% | 1.00x | 2.27x |

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
