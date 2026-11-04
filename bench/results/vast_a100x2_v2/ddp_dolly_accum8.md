# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-147-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 32 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 8.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 294.7 ms | 141,498 | 63.5% |  |  |
| random | equal counts | 310.7 ms | 133,433 | 73.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 307.8 ms | 134,691 | 73.3% | 1.01x | 1.01x |
| random | Evencut, balanced padded size | 308.3 ms | 134,441 | 72.1% | 1.01x | 1.01x |
| length-grouped | equal counts | 307.7 ms | 134,724 | 61.9% | 1.00x | 1.01x |
| length-grouped | Evencut, balanced tokens | 300.3 ms | 138,042 | 53.2% | 1.02x | 1.03x |
| length-grouped | Evencut, balanced padded size | 303.9 ms | 136,421 | 50.5% | 1.01x | 1.02x |
| any order | LPT on padded size (not contiguous) | 309.7 ms | 133,830 | 74.1% | 1.00x | 1.00x |
| sorted dataset | token budget per GPU (fairseq-style) | 295.0 ms | 140,498 | 1.3% | 1.00x | 1.05x |
| sorted dataset | Evencut, padded_slices over the sorted data | 296.2 ms | 139,973 | 1.3% | 1.00x | 1.05x |
| random | Evencut, whole optimizer step balanced (window) | 297.4 ms | 139,409 | 9.8% | 1.04x | 1.04x |
| pooled (like group_by_length) | Hugging Face group_by_length | 292.6 ms | 141,671 | 1.7% | 1.00x | 1.06x |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 291.0 ms | 142,455 | 1.8% | 1.01x | 1.07x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	SYS	0-31,64-95	0		N/A
GPU1	SYS	 X 	32-63,96-127	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
