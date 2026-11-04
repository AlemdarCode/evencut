# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-124-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 12 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 2.3 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 19.3 ms | 218,407 | 51.9% |  |  |
| random | equal counts | 34.8 ms | 184,557 | 62.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 33.7 ms | 190,486 | 62.5% | 1.03x | 1.03x |
| random | Evencut, balanced padded size | 30.2 ms | 213,093 | 60.5% | 1.15x | 1.15x |
| length-grouped | equal counts | 35.5 ms | 181,283 | 48.8% | 1.00x | 0.98x |
| length-grouped | Evencut, balanced tokens | 30.7 ms | 209,691 | 51.1% | 1.16x | 1.14x |
| length-grouped | Evencut, balanced padded size | 22.9 ms | 280,711 | 40.3% | 1.55x | 1.52x |
| any order | LPT on padded size (not contiguous) | 29.4 ms | 218,679 | 61.8% | 1.00x | 1.18x |
| sorted dataset | token budget per GPU (fairseq-style) | 19.3 ms | 333,648 | 2.2% | 1.00x | 1.81x |
| sorted dataset | Evencut, padded_slices over the sorted data | 19.1 ms | 337,076 | 2.2% | 1.01x | 1.83x |

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
