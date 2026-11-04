# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-94-generic-x86_64-with-glibc2.39. Model large (312.0M parameters), 12 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 24 micro-steps per plan. One all-reduce of the fp32 gradients (1248 MB) takes 200.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 241.8 ms | 17,404 | 51.9% |  |  |
| random | equal counts | 345.6 ms | 19,396 | 63.2% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 302.9 ms | 22,126 | 61.9% | 1.14x | 1.14x |

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
