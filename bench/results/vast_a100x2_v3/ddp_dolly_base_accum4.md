# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-124-generic-x86_64-with-glibc2.39. Model base (91.2M parameters), 48 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 6.6 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 279.0 ms | 100,385 | 67.3% |  |  |
| random | equal counts | 457.5 ms | 65,965 | 76.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 428.5 ms | 70,433 | 76.1% | 1.07x | 1.07x |
| random | Evencut, balanced padded size | 405.9 ms | 74,338 | 75.5% | 1.13x | 1.13x |
| length-grouped | equal counts | 500.6 ms | 60,286 | 65.9% | 1.00x | 0.91x |
| length-grouped | Evencut, balanced tokens | 257.8 ms | 117,056 | 55.4% | 1.94x | 1.77x |
| length-grouped | Evencut, balanced padded size | 225.8 ms | 133,622 | 54.1% | 2.22x | 2.03x |
| any order | LPT on padded size (not contiguous) | 429.8 ms | 70,212 | 77.7% | 1.00x | 1.06x |
| sorted dataset | token budget per GPU (fairseq-style) | 117.6 ms | 256,656 | 2.5% | 1.00x | 3.89x |
| sorted dataset | Evencut, padded_slices over the sorted data | 117.7 ms | 256,425 | 2.5% | 1.00x | 3.89x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV4	48-63,112-127	3		N/A
GPU1	NV4	 X 	16-31,80-95	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
