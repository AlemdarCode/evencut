# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-94-generic-x86_64-with-glibc2.39. Model base (91.2M parameters), 48 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 47.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 318.1 ms | 88,043 | 67.3% |  |  |
| random | equal counts | 558.4 ms | 54,041 | 76.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 521.5 ms | 57,864 | 76.1% | 1.07x | 1.07x |
| random | Evencut, balanced padded size | 492.3 ms | 61,300 | 75.5% | 1.13x | 1.13x |
| length-grouped | equal counts | 617.8 ms | 48,843 | 65.9% | 1.00x | 0.90x |
| length-grouped | Evencut, balanced tokens | 302.9 ms | 99,639 | 55.4% | 2.04x | 1.84x |
| length-grouped | Evencut, balanced padded size | 265.7 ms | 113,583 | 54.1% | 2.33x | 2.10x |
| any order | LPT on padded size (not contiguous) | 527.8 ms | 57,172 | 77.7% | 1.00x | 1.06x |
| sorted dataset | token budget per GPU (fairseq-style) | 240.3 ms | 125,561 | 2.5% | 1.00x | 2.32x |
| sorted dataset | Evencut, padded_slices over the sorted data | 244.7 ms | 123,330 | 2.5% | 0.98x | 2.28x |

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
