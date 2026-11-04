# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-01, 4 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1090-nvidia-x86_64-with-glibc2.39. Model small (29.9M parameters), 64 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 1.0 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 173.5 ms | 480,575 | 63.5% |  |  |
| random | equal counts | 245.5 ms | 345,940 | 73.2% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 238.4 ms | 356,327 | 72.7% | 1.03x | 1.03x |
| random | Evencut, balanced padded size | 218.6 ms | 388,498 | 71.4% | 1.12x | 1.12x |
| length-grouped | equal counts | 312.7 ms | 271,679 | 47.7% | 1.00x | 0.79x |
| length-grouped | Evencut, balanced tokens | 142.8 ms | 594,671 | 34.7% | 2.19x | 1.72x |
| length-grouped | Evencut, balanced padded size | 139.5 ms | 609,021 | 31.1% | 2.24x | 1.76x |
| sorted dataset | token budget per GPU (fairseq-style) | 138.8 ms | 612,105 | 0.6% | 1.00x | 1.77x |
| sorted dataset | Evencut, padded_slices over the sorted data | 138.0 ms | 615,302 | 0.6% | 1.01x | 1.78x |

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
