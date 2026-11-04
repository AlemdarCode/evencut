# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-01, 4 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1090-nvidia-x86_64-with-glibc2.39. Model base (91.2M parameters), 96 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 2.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 263.1 ms | 212,872 | 67.3% |  |  |
| random | equal counts | 461.7 ms | 137,537 | 76.4% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 439.4 ms | 144,506 | 76.2% | 1.05x | 1.05x |
| random | Evencut, balanced padded size | 401.3 ms | 158,225 | 75.0% | 1.15x | 1.15x |
| length-grouped | equal counts | 534.1 ms | 118,892 | 49.8% | 1.00x | 0.86x |
| length-grouped | Evencut, balanced tokens | 211.1 ms | 300,727 | 34.9% | 2.53x | 2.19x |
| length-grouped | Evencut, balanced padded size | 163.5 ms | 388,278 | 32.8% | 3.27x | 2.82x |
| sorted dataset | token budget per GPU (fairseq-style) | 115.8 ms | 548,224 | 1.2% | 1.00x | 3.99x |
| sorted dataset | Evencut, padded_slices over the sorted data | 115.8 ms | 548,411 | 1.2% | 1.00x | 3.99x |

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
