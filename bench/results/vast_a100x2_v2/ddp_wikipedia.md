# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-147-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 12 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 9.1 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 40.6 ms | 103,625 | 51.9% |  |  |
| random | equal counts | 45.9 ms | 140,037 | 62.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 47.2 ms | 136,134 | 62.5% | 0.97x | 0.97x |
| random | Evencut, balanced padded size | 44.9 ms | 143,100 | 60.5% | 1.02x | 1.02x |
| length-grouped | equal counts | 46.7 ms | 137,764 | 48.8% | 1.00x | 0.98x |
| length-grouped | Evencut, balanced tokens | 45.4 ms | 141,586 | 51.1% | 1.03x | 1.01x |
| length-grouped | Evencut, balanced padded size | 42.3 ms | 152,016 | 40.3% | 1.10x | 1.09x |
| any order | LPT on padded size (not contiguous) | 44.5 ms | 144,328 | 61.8% | 1.00x | 1.03x |
| sorted dataset | token budget per GPU (fairseq-style) | 41.4 ms | 155,316 | 2.2% | 1.00x | 1.11x |
| sorted dataset | Evencut, padded_slices over the sorted data | 41.9 ms | 153,334 | 2.2% | 0.99x | 1.09x |

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
