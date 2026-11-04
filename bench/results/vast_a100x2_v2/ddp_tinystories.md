# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-147-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 64 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 9.0 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 40.3 ms | 234,831 | 30.5% |  |  |
| random | equal counts | 43.2 ms | 226,636 | 31.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 42.8 ms | 228,809 | 31.1% | 1.01x | 1.01x |
| random | Evencut, balanced padded size | 42.2 ms | 231,669 | 30.4% | 1.02x | 1.02x |
| length-grouped | equal counts | 42.1 ms | 232,692 | 20.8% | 1.00x | 1.03x |
| length-grouped | Evencut, balanced tokens | 41.6 ms | 235,191 | 19.5% | 1.01x | 1.04x |
| length-grouped | Evencut, balanced padded size | 42.0 ms | 233,152 | 18.6% | 1.00x | 1.03x |
| any order | LPT on padded size (not contiguous) | 42.0 ms | 232,737 | 31.4% | 1.00x | 1.03x |
| sorted dataset | token budget per GPU (fairseq-style) | 41.7 ms | 234,746 | 1.1% | 1.00x | 1.04x |
| sorted dataset | Evencut, padded_slices over the sorted data | 41.3 ms | 236,865 | 1.1% | 1.01x | 1.05x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	SYS	0-31,64-95	0		N/A
GPU1	SYS	 X 	32-63,96-127	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
