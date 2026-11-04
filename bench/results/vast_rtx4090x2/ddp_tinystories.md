# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-7.0.0-42-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 64 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 47.2 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 62.1 ms | 152,609 | 30.5% |  |  |
| random | equal counts | 64.2 ms | 152,470 | 31.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 64.2 ms | 152,348 | 31.1% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 62.9 ms | 155,606 | 30.4% | 1.02x | 1.02x |
| length-grouped | equal counts | 64.9 ms | 150,736 | 20.8% | 1.00x | 0.99x |
| length-grouped | Evencut, balanced tokens | 62.5 ms | 156,643 | 19.5% | 1.04x | 1.03x |
| length-grouped | Evencut, balanced padded size | 61.1 ms | 160,226 | 18.6% | 1.06x | 1.05x |
| any order | LPT on padded size (not contiguous) | 63.0 ms | 155,445 | 31.4% | 1.00x | 1.02x |
| sorted dataset | token budget per GPU (fairseq-style) | 59.4 ms | 164,646 | 1.1% | 1.00x | 1.08x |
| sorted dataset | Evencut, padded_slices over the sorted data | 59.5 ms | 164,442 | 1.1% | 1.00x | 1.08x |

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
