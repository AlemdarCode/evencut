# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-01, 8 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-134-generic-x86_64-with-glibc2.39. Model base (91.2M parameters), 384 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 195.9 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 447.9 ms | 516,816 | 29.3% |  |  |
| random | equal counts | 504.4 ms | 468,118 | 33.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 498.3 ms | 473,803 | 33.0% | 1.01x | 1.01x |
| random | Evencut, balanced padded size | 470.6 ms | 501,725 | 32.5% | 1.07x | 1.07x |
| length-grouped | equal counts | 528.3 ms | 446,913 | 8.4% | 1.00x | 0.95x |
| length-grouped | Evencut, balanced tokens | 432.8 ms | 545,483 | 7.2% | 1.22x | 1.17x |
| length-grouped | Evencut, balanced padded size | 389.9 ms | 605,496 | 6.3% | 1.35x | 1.29x |
| sorted dataset | token budget per GPU (fairseq-style) | 381.8 ms | 618,468 | 0.3% | 1.00x | 1.32x |
| sorted dataset | Evencut, padded_slices over the sorted data | 384.0 ms | 614,941 | 0.3% | 0.99x | 1.31x |

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
