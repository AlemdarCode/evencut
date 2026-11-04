# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-01, 4 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1090-nvidia-x86_64-with-glibc2.39. Model base (91.2M parameters), 24 sequences per micro-step, 4 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (365 MB) takes 2.8 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 122.8 ms | 274,167 | 51.9% |  |  |
| random | equal counts | 288.8 ms | 182,673 | 62.9% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 288.2 ms | 183,040 | 63.1% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 248.1 ms | 212,590 | 59.8% | 1.16x | 1.16x |
| length-grouped | equal counts | 313.2 ms | 168,415 | 29.5% | 1.00x | 0.92x |
| length-grouped | Evencut, balanced tokens | 225.9 ms | 233,547 | 34.7% | 1.39x | 1.28x |
| length-grouped | Evencut, balanced padded size | 140.5 ms | 375,349 | 25.0% | 2.23x | 2.05x |
| any order | LPT on padded size (not contiguous) | 255.8 ms | 206,175 | 64.7% | 1.00x | 1.13x |
| sorted dataset | token budget per GPU (fairseq-style) | 115.8 ms | 455,701 | 1.2% | 1.00x | 2.49x |
| sorted dataset | Evencut, padded_slices over the sorted data | 115.8 ms | 455,439 | 1.2% | 1.00x | 2.49x |

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
