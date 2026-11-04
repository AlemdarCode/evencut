# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-147-generic-x86_64-with-glibc2.39. Model base (91.2M parameters), 48 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 26.1 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 293.4 ms | 95,475 | 67.3% |  |  |
| random | equal counts | 467.0 ms | 64,617 | 76.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 438.0 ms | 68,898 | 76.1% | 1.07x | 1.07x |
| random | Evencut, balanced padded size | 417.0 ms | 72,365 | 75.5% | 1.12x | 1.12x |
| length-grouped | equal counts | 507.1 ms | 59,515 | 65.9% | 1.00x | 0.92x |
| length-grouped | Evencut, balanced tokens | 274.0 ms | 110,155 | 55.4% | 1.85x | 1.70x |
| length-grouped | Evencut, balanced padded size | 241.5 ms | 124,956 | 54.1% | 2.10x | 1.93x |
| any order | LPT on padded size (not contiguous) | 441.1 ms | 68,418 | 77.7% | 1.00x | 1.06x |
| sorted dataset | token budget per GPU (fairseq-style) | 200.0 ms | 150,881 | 2.5% | 1.00x | 2.34x |
| sorted dataset | Evencut, padded_slices over the sorted data | 203.5 ms | 148,317 | 2.5% | 0.98x | 2.30x |

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
