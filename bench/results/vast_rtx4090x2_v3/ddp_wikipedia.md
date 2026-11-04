# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-194-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 12 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 36.5 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 54.1 ms | 77,736 | 51.9% |  |  |
| random | equal counts | 71.8 ms | 89,516 | 62.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 70.7 ms | 90,974 | 62.5% | 1.02x | 1.02x |
| random | Evencut, balanced padded size | 66.2 ms | 97,058 | 60.5% | 1.08x | 1.08x |
| length-grouped | equal counts | 73.9 ms | 87,010 | 48.8% | 1.00x | 0.97x |
| length-grouped | Evencut, balanced tokens | 64.4 ms | 99,804 | 51.1% | 1.15x | 1.11x |
| length-grouped | Evencut, balanced padded size | 58.1 ms | 110,732 | 40.3% | 1.27x | 1.24x |
| any order | LPT on padded size (not contiguous) | 65.5 ms | 98,142 | 61.8% | 1.00x | 1.10x |
| sorted dataset | token budget per GPU (fairseq-style) | 53.9 ms | 119,202 | 2.2% | 1.00x | 1.33x |
| sorted dataset | Evencut, padded_slices over the sorted data | 53.8 ms | 119,507 | 2.2% | 1.00x | 1.34x |

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
