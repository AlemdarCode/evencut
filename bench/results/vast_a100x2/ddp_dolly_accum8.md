# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1079-nvidia-x86_64-with-glibc2.39. Model small (29.9M parameters), 32 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 11.7 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 281.2 ms | 148,258 | 63.5% |  |  |
| random | equal counts | 294.2 ms | 140,920 | 73.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 294.8 ms | 140,612 | 73.3% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 284.9 ms | 145,513 | 72.1% | 1.03x | 1.03x |
| length-grouped | equal counts | 309.9 ms | 133,765 | 61.9% | 1.00x | 0.95x |
| length-grouped | Evencut, balanced tokens | 270.2 ms | 153,433 | 53.2% | 1.15x | 1.09x |
| length-grouped | Evencut, balanced padded size | 267.5 ms | 154,972 | 50.5% | 1.16x | 1.10x |
| any order | LPT on padded size (not contiguous) | 285.2 ms | 145,346 | 74.1% | 1.00x | 1.03x |
| sorted dataset | token budget per GPU (fairseq-style) | 271.3 ms | 152,793 | 1.3% | 1.00x | 1.08x |
| sorted dataset | Evencut, padded_slices over the sorted data | 263.5 ms | 157,344 | 1.3% | 1.03x | 1.12x |
| random | Evencut, whole optimizer step balanced (window) | 265.5 ms | 156,127 | 9.8% | 1.11x | 1.11x |
| pooled (like group_by_length) | Hugging Face group_by_length | 269.6 ms | 153,784 | 1.7% | 1.00x | 1.09x |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 264.3 ms | 156,820 | 1.8% | 1.02x | 1.11x |

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
