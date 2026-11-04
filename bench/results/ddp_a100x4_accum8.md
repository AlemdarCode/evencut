# Test 4, real multi-GPU run (TinyStories-Turkish)

evencut 1.0.0, 2026-11-01, 4 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1090-nvidia-x86_64-with-glibc2.39. Model small (29.9M parameters), 128 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 1.0 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 153.6 ms | 986,384 | 30.5% |  |  |
| random | equal counts | 171.1 ms | 920,316 | 30.7% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 170.7 ms | 922,026 | 30.7% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 165.0 ms | 954,098 | 30.2% | 1.04x | 1.04x |
| length-grouped | equal counts | 181.7 ms | 866,301 | 12.5% | 1.00x | 0.94x |
| length-grouped | Evencut, balanced tokens | 143.5 ms | 1,096,946 | 11.2% | 1.27x | 1.19x |
| length-grouped | Evencut, balanced padded size | 135.9 ms | 1,158,434 | 10.5% | 1.34x | 1.26x |
| sorted dataset | token budget per GPU (fairseq-style) | 132.6 ms | 1,187,292 | 0.3% | 1.00x | 1.29x |
| sorted dataset | Evencut, padded_slices over the sorted data | 131.9 ms | 1,193,727 | 0.3% | 1.01x | 1.30x |

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
