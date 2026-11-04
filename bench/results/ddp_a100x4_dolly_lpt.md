# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-01, 4 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1090-nvidia-x86_64-with-glibc2.39. Model small (29.9M parameters), 64 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 1.0 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 23.9 ms | 435,901 | 63.5% |  |  |
| random | equal counts | 40.1 ms | 258,684 | 73.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 35.8 ms | 289,703 | 72.7% | 1.12x | 1.12x |
| random | Evencut, balanced padded size | 29.0 ms | 356,794 | 71.3% | 1.38x | 1.38x |
| length-grouped | equal counts | 40.0 ms | 259,217 | 48.0% | 1.00x | 1.00x |
| length-grouped | Evencut, balanced tokens | 20.1 ms | 515,953 | 34.3% | 1.99x | 1.99x |
| length-grouped | Evencut, balanced padded size | 18.9 ms | 549,457 | 31.2% | 2.12x | 2.12x |
| any order | LPT on padded size (not contiguous) | 30.6 ms | 338,128 | 74.6% | 1.00x | 1.31x |
| sorted dataset | token budget per GPU (fairseq-style) | 18.7 ms | 553,544 | 1.3% | 1.00x | 2.14x |
| sorted dataset | Evencut, padded_slices over the sorted data | 18.7 ms | 555,345 | 1.3% | 1.00x | 2.15x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	GPU2	GPU3	NIC0	NIC1	NIC2	NIC3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV12	NV12	NV12	SYS	SYS	SYS	SYS	112-127,240-255	7		N/A
GPU1	NV12	 X 	NV12	NV12	SYS	SYS	SYS	SYS	112-127,240-255	7		N/A
GPU2	NV12	NV12	 X 	NV12	SYS	SYS	SYS	SYS	80-95,208-223	5		N/A
GPU3	NV12	NV12	NV12	 X 	SYS	SYS	SYS	SYS	80-95,208-223	5		N/A
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
