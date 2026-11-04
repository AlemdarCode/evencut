# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-01, 8 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-134-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 256 sequences per step, 40 steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 64.2 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 81.7 ms | 463,888 | 30.5% |  |  |
| random | equal counts | 82.2 ms | 478,712 | 30.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 81.9 ms | 480,504 | 30.6% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 81.7 ms | 481,835 | 30.1% | 1.01x | 1.01x |
| length-grouped | equal counts | 81.9 ms | 480,483 | 7.6% | 1.00x | 1.00x |
| length-grouped | Evencut, balanced tokens | 81.7 ms | 481,788 | 6.6% | 1.00x | 1.01x |
| length-grouped | Evencut, balanced padded size | 81.7 ms | 481,757 | 6.0% | 1.00x | 1.01x |
| sorted dataset | token budget per GPU (fairseq-style) | 81.7 ms | 481,842 | 0.3% | 1.00x | 1.01x |
| sorted dataset | Evencut, padded_slices over the sorted data | 80.9 ms | 486,296 | 0.3% | 1.01x | 1.02x |

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
