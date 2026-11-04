# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-194-generic-x86_64-with-glibc2.39. Model base (91.2M parameters), 48 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 99.2 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 438.4 ms | 63,892 | 67.3% |  |  |
| random | equal counts | 710.4 ms | 42,477 | 76.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 666.6 ms | 45,268 | 76.1% | 1.07x | 1.07x |
| random | Evencut, balanced padded size | 634.7 ms | 47,546 | 75.5% | 1.12x | 1.12x |
| length-grouped | equal counts | 780.0 ms | 38,691 | 65.9% | 1.00x | 0.91x |
| length-grouped | Evencut, balanced tokens | 404.6 ms | 74,582 | 55.4% | 1.93x | 1.76x |
| length-grouped | Evencut, balanced padded size | 376.1 ms | 80,235 | 54.1% | 2.07x | 1.89x |
| any order | LPT on padded size (not contiguous) | 672.6 ms | 44,864 | 77.7% | 1.00x | 1.06x |
| sorted dataset | token budget per GPU (fairseq-style) | 243.0 ms | 124,200 | 2.5% | 1.00x | 2.92x |
| sorted dataset | Evencut, padded_slices over the sorted data | 241.7 ms | 124,869 | 2.5% | 1.01x | 2.94x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NODE	0-127	0		N/A
GPU1	NODE	 X 	0-127	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
