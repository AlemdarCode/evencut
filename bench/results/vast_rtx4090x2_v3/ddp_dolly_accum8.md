# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-194-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 32 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 33.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 304.7 ms | 136,855 | 63.5% |  |  |
| random | equal counts | 323.9 ms | 128,000 | 73.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 326.0 ms | 127,142 | 73.3% | 0.99x | 0.99x |
| random | Evencut, balanced padded size | 313.7 ms | 132,145 | 72.1% | 1.03x | 1.03x |
| length-grouped | equal counts | 363.1 ms | 114,173 | 61.9% | 1.00x | 0.89x |
| length-grouped | Evencut, balanced tokens | 309.6 ms | 133,898 | 53.2% | 1.17x | 1.05x |
| length-grouped | Evencut, balanced padded size | 316.0 ms | 131,173 | 50.5% | 1.15x | 1.02x |
| any order | LPT on padded size (not contiguous) | 324.7 ms | 127,679 | 74.1% | 1.00x | 1.00x |
| sorted dataset | token budget per GPU (fairseq-style) | 299.0 ms | 138,639 | 1.3% | 1.00x | 1.08x |
| sorted dataset | Evencut, padded_slices over the sorted data | 300.5 ms | 137,968 | 1.3% | 1.00x | 1.08x |
| random | Evencut, whole optimizer step balanced (window) | 308.5 ms | 134,359 | 9.8% | 1.05x | 1.05x |
| pooled (like group_by_length) | Hugging Face group_by_length | 308.5 ms | 134,358 | 1.7% | 1.00x | 1.05x |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 306.3 ms | 135,357 | 1.8% | 1.01x | 1.06x |

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
