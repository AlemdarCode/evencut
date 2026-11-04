# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-124-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 64 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 2.4 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 22.3 ms | 425,285 | 30.5% |  |  |
| random | equal counts | 24.2 ms | 404,926 | 31.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 24.4 ms | 401,736 | 31.1% | 0.99x | 0.99x |
| random | Evencut, balanced padded size | 23.1 ms | 423,917 | 30.4% | 1.05x | 1.05x |
| length-grouped | equal counts | 24.4 ms | 400,312 | 20.8% | 1.00x | 0.99x |
| length-grouped | Evencut, balanced tokens | 21.6 ms | 452,474 | 19.5% | 1.13x | 1.12x |
| length-grouped | Evencut, balanced padded size | 20.8 ms | 470,719 | 18.6% | 1.18x | 1.16x |
| any order | LPT on padded size (not contiguous) | 23.3 ms | 419,676 | 31.4% | 1.00x | 1.04x |
| sorted dataset | token budget per GPU (fairseq-style) | 19.2 ms | 510,158 | 1.1% | 1.00x | 1.26x |
| sorted dataset | Evencut, padded_slices over the sorted data | 19.1 ms | 513,504 | 1.1% | 1.01x | 1.27x |

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
