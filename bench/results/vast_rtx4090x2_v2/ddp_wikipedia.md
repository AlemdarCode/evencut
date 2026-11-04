# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-94-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 12 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 16.0 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 52.2 ms | 80,564 | 51.9% |  |  |
| random | equal counts | 52.3 ms | 122,817 | 62.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 53.7 ms | 119,717 | 62.5% | 0.97x | 0.97x |
| random | Evencut, balanced padded size | 52.3 ms | 122,809 | 60.5% | 1.00x | 1.00x |
| length-grouped | equal counts | 51.6 ms | 124,567 | 48.8% | 1.00x | 1.01x |
| length-grouped | Evencut, balanced tokens | 53.5 ms | 120,054 | 51.1% | 0.96x | 0.98x |
| length-grouped | Evencut, balanced padded size | 51.5 ms | 124,728 | 40.3% | 1.00x | 1.02x |
| any order | LPT on padded size (not contiguous) | 51.9 ms | 123,799 | 61.8% | 1.00x | 1.01x |
| sorted dataset | token budget per GPU (fairseq-style) | 51.6 ms | 124,552 | 2.2% | 1.00x | 1.01x |
| sorted dataset | Evencut, padded_slices over the sorted data | 51.5 ms | 124,805 | 2.2% | 1.00x | 1.02x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	NIC0	NIC1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PXB	NODE	NODE	0-31,64-95	0		N/A
GPU1	PXB	 X 	NODE	NODE	0-31,64-95	0		N/A
NIC0	NODE	NODE	 X 	PIX				
NIC1	NODE	NODE	PIX	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
```
