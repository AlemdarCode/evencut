# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-7.0.0-42-generic-x86_64-with-glibc2.39. Model base (91.2M parameters), 48 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 146.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 430.0 ms | 65,131 | 67.3% |  |  |
| random | equal counts | 678.1 ms | 44,504 | 76.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 637.2 ms | 47,360 | 76.1% | 1.06x | 1.06x |
| random | Evencut, balanced padded size | 605.8 ms | 49,810 | 75.5% | 1.12x | 1.12x |
| length-grouped | equal counts | 738.8 ms | 40,848 | 65.9% | 1.00x | 0.92x |
| length-grouped | Evencut, balanced tokens | 412.0 ms | 73,240 | 55.4% | 1.79x | 1.65x |
| length-grouped | Evencut, balanced padded size | 372.4 ms | 81,029 | 54.1% | 1.98x | 1.82x |
| any order | LPT on padded size (not contiguous) | 636.6 ms | 47,401 | 77.7% | 1.00x | 1.07x |
| sorted dataset | token budget per GPU (fairseq-style) | 247.9 ms | 121,722 | 2.5% | 1.00x | 2.74x |
| sorted dataset | Evencut, padded_slices over the sorted data | 248.0 ms | 121,669 | 2.5% | 1.00x | 2.73x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PHB	0-63	0		N/A
GPU1	PHB	 X 	0-63	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
