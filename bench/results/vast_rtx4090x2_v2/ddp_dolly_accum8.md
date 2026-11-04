# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-94-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 32 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 15.9 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 346.9 ms | 120,191 | 63.5% |  |  |
| random | equal counts | 351.3 ms | 118,010 | 73.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 353.0 ms | 117,438 | 73.3% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 352.2 ms | 117,710 | 72.1% | 1.00x | 1.00x |
| length-grouped | equal counts | 349.4 ms | 118,658 | 61.9% | 1.00x | 1.01x |
| length-grouped | Evencut, balanced tokens | 349.5 ms | 118,608 | 53.2% | 1.00x | 1.01x |
| length-grouped | Evencut, balanced padded size | 351.9 ms | 117,810 | 50.5% | 0.99x | 1.00x |
| any order | LPT on padded size (not contiguous) | 351.8 ms | 117,847 | 74.1% | 1.00x | 1.00x |
| sorted dataset | token budget per GPU (fairseq-style) | 346.6 ms | 119,590 | 1.3% | 1.00x | 1.01x |
| sorted dataset | Evencut, padded_slices over the sorted data | 344.1 ms | 120,485 | 1.3% | 1.01x | 1.02x |
| random | Evencut, whole optimizer step balanced (window) | 345.3 ms | 120,059 | 9.8% | 1.02x | 1.02x |
| pooled (like group_by_length) | Hugging Face group_by_length | 347.0 ms | 119,447 | 1.7% | 1.00x | 1.01x |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 347.6 ms | 119,254 | 1.8% | 1.00x | 1.01x |

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
