# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-02, 1 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-6.8.0-147-generic-x86_64-with-glibc2.39. Model large (312.0M parameters), 16 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 24 micro-steps per plan. One all-reduce of the fp32 gradients (1248 MB) takes 0.0 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 277.2 ms | 35,353 | 28.1% |  |  |
| random | equal counts | 288.4 ms | 34,237 | 27.8% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 288.5 ms | 34,232 | 27.8% | 1.00x | 1.00x |
| random | Evencut, whole optimizer step balanced (window) | 245.6 ms | 40,203 | 9.4% | 1.17x | 1.17x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	0-11	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
