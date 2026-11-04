# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA A100-SXM4-40GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1079-nvidia-x86_64-with-glibc2.39. Model small (29.9M parameters), 64 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 11.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 40.5 ms | 233,637 | 30.5% |  |  |
| random | equal counts | 42.9 ms | 228,129 | 31.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 42.8 ms | 228,584 | 31.1% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 41.6 ms | 235,160 | 30.4% | 1.03x | 1.03x |
| length-grouped | equal counts | 42.5 ms | 230,345 | 20.8% | 1.00x | 1.01x |
| length-grouped | Evencut, balanced tokens | 41.3 ms | 236,689 | 19.5% | 1.03x | 1.04x |
| length-grouped | Evencut, balanced padded size | 40.7 ms | 240,497 | 18.6% | 1.04x | 1.05x |
| any order | LPT on padded size (not contiguous) | 41.9 ms | 233,560 | 31.4% | 1.00x | 1.02x |
| sorted dataset | token budget per GPU (fairseq-style) | 40.0 ms | 244,820 | 1.1% | 1.00x | 1.07x |
| sorted dataset | Evencut, padded_slices over the sorted data | 40.0 ms | 244,855 | 1.1% | 1.00x | 1.07x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	SYS	0-63,128-191	0		N/A
GPU1	SYS	 X 	64-127,192-255	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
