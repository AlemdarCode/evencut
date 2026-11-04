# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-01, 8 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-134-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 256 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 65.0 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 382.0 ms | 793,379 | 30.5% |  |  |
| random | equal counts | 380.4 ms | 826,977 | 30.5% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 386.3 ms | 814,362 | 30.5% | 0.98x | 0.98x |
| random | Evencut, balanced padded size | 388.1 ms | 810,677 | 30.0% | 0.98x | 0.98x |
| length-grouped | equal counts | 379.6 ms | 828,729 | 7.6% | 1.00x | 1.00x |
| length-grouped | Evencut, balanced tokens | 381.6 ms | 824,400 | 6.6% | 0.99x | 1.00x |
| length-grouped | Evencut, balanced padded size | 383.5 ms | 820,374 | 6.0% | 0.99x | 0.99x |
| sorted dataset | token budget per GPU (fairseq-style) | 378.2 ms | 831,749 | 0.1% | 1.00x | 1.01x |
| sorted dataset | Evencut, padded_slices over the sorted data | 378.0 ms | 832,216 | 0.1% | 1.00x | 1.01x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PIX	PXB	PXB	NODE	NODE	NODE	NODE	0-27,56-83	0		N/A
GPU1	PIX	 X 	PXB	PXB	NODE	NODE	NODE	NODE	0-27,56-83	0		N/A
GPU2	PXB	PXB	 X 	PXB	NODE	NODE	NODE	NODE	0-27,56-83	0		N/A
GPU3	PXB	PXB	PXB	 X 	NODE	NODE	NODE	NODE	0-27,56-83	0		N/A
GPU4	NODE	NODE	NODE	NODE	 X 	PIX	PXB	PXB	0-27,56-83	0		N/A
GPU5	NODE	NODE	NODE	NODE	PIX	 X 	PXB	PXB	0-27,56-83	0		N/A
GPU6	NODE	NODE	NODE	NODE	PXB	PXB	 X 	PXB	0-27,56-83	0		N/A
GPU7	NODE	NODE	NODE	NODE	PXB	PXB	PXB	 X 	0-27,56-83	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
