# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-01, 8 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-134-generic-x86_64-with-glibc2.39. Model base (91.2M parameters), 192 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 196.6 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 462.1 ms | 242,464 | 67.3% |  |  |
| random | equal counts | 762.7 ms | 166,435 | 76.6% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 746.1 ms | 170,138 | 76.5% | 1.02x | 1.02x |
| random | Evencut, balanced padded size | 642.3 ms | 197,634 | 75.4% | 1.19x | 1.19x |
| length-grouped | equal counts | 852.5 ms | 148,910 | 32.3% | 1.00x | 0.89x |
| length-grouped | Evencut, balanced tokens | 384.3 ms | 330,330 | 20.5% | 2.22x | 1.98x |
| length-grouped | Evencut, balanced padded size | 374.7 ms | 338,772 | 19.1% | 2.28x | 2.04x |
| sorted dataset | token budget per GPU (fairseq-style) | 378.0 ms | 335,852 | 0.6% | 1.00x | 2.02x |
| sorted dataset | Evencut, padded_slices over the sorted data | 376.7 ms | 337,035 | 0.6% | 1.00x | 2.03x |

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
