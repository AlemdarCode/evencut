# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-01, 8 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-134-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 128 sequences per step, 40 steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 64.9 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 86.1 ms | 242,183 | 63.5% |  |  |
| random | equal counts | 92.4 ms | 229,858 | 73.2% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 89.4 ms | 237,574 | 72.5% | 1.03x | 1.03x |
| random | Evencut, balanced padded size | 84.1 ms | 252,636 | 71.0% | 1.10x | 1.10x |
| length-grouped | equal counts | 91.8 ms | 231,321 | 31.8% | 1.00x | 1.01x |
| length-grouped | Evencut, balanced tokens | 82.6 ms | 257,202 | 20.3% | 1.11x | 1.12x |
| length-grouped | Evencut, balanced padded size | 82.6 ms | 256,932 | 18.5% | 1.11x | 1.12x |
| sorted dataset | token budget per GPU (fairseq-style) | 83.0 ms | 255,706 | 0.6% | 1.00x | 1.11x |
| sorted dataset | Evencut, padded_slices over the sorted data | 82.6 ms | 257,172 | 0.6% | 1.01x | 1.12x |

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
