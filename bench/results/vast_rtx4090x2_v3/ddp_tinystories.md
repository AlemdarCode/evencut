# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-194-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 64 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 36.7 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 57.2 ms | 165,582 | 30.5% |  |  |
| random | equal counts | 58.6 ms | 166,885 | 31.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 58.3 ms | 167,723 | 31.1% | 1.01x | 1.01x |
| random | Evencut, balanced padded size | 57.2 ms | 171,011 | 30.4% | 1.02x | 1.02x |
| length-grouped | equal counts | 59.1 ms | 165,655 | 20.8% | 1.00x | 0.99x |
| length-grouped | Evencut, balanced tokens | 56.4 ms | 173,534 | 19.5% | 1.05x | 1.04x |
| length-grouped | Evencut, balanced padded size | 55.2 ms | 177,234 | 18.6% | 1.07x | 1.06x |
| any order | LPT on padded size (not contiguous) | 57.0 ms | 171,562 | 31.4% | 1.00x | 1.03x |
| sorted dataset | token budget per GPU (fairseq-style) | 54.9 ms | 178,130 | 1.1% | 1.00x | 1.07x |
| sorted dataset | Evencut, padded_slices over the sorted data | 54.5 ms | 179,664 | 1.1% | 1.01x | 1.08x |

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
