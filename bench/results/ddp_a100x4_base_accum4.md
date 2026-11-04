# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-01, 4 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1090-nvidia-x86_64-with-glibc2.39. Model base (91.2M parameters), 192 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 2.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 249.1 ms | 464,636 | 29.3% |  |  |
| random | equal counts | 276.8 ms | 426,336 | 32.8% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 277.1 ms | 425,787 | 32.7% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 262.3 ms | 449,920 | 32.3% | 1.06x | 1.06x |
| length-grouped | equal counts | 286.5 ms | 411,813 | 13.4% | 1.00x | 0.97x |
| length-grouped | Evencut, balanced tokens | 232.5 ms | 507,455 | 12.0% | 1.23x | 1.19x |
| length-grouped | Evencut, balanced padded size | 208.4 ms | 566,169 | 11.1% | 1.37x | 1.33x |
| sorted dataset | token budget per GPU (fairseq-style) | 190.3 ms | 620,180 | 0.6% | 1.00x | 1.45x |
| sorted dataset | Evencut, padded_slices over the sorted data | 190.2 ms | 620,342 | 0.6% | 1.00x | 1.46x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	GPU2	GPU3	NIC0	NIC1	NIC2	NIC3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV12	NV12	NV12	SYS	SYS	SYS	SYS	48-63,176-191	3		N/A
GPU1	NV12	 X 	NV12	NV12	SYS	SYS	SYS	SYS	48-63,176-191	3		N/A
GPU2	NV12	NV12	 X 	NV12	SYS	SYS	SYS	SYS	16-31,144-159	1		N/A
GPU3	NV12	NV12	NV12	 X 	SYS	SYS	SYS	SYS	16-31,144-159	1		N/A
NIC0	SYS	SYS	SYS	SYS	 X 	PIX	SYS	SYS				
NIC1	SYS	SYS	SYS	SYS	PIX	 X 	SYS	SYS				
NIC2	SYS	SYS	SYS	SYS	SYS	SYS	 X 	PIX				
NIC3	SYS	SYS	SYS	SYS	SYS	SYS	PIX	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_4
  NIC1: mlx5_5
  NIC2: mlx5_10
  NIC3: mlx5_11
```
