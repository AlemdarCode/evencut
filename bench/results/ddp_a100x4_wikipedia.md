# Test 4, real multi-GPU run (Wikipedia-TR)

evencut 1.0.0, 2026-11-01, 4 x NVIDIA A100-SXM4-80GB, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-5.15.0-1090-nvidia-x86_64-with-glibc2.39. Model small (29.9M parameters), 24 sequences per micro-step, 1 micro-steps per optimizer step (gradient accumulation), 40 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 1.0 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 19.3 ms | 435,736 | 51.9% |  |  |
| random | equal counts | 36.6 ms | 360,504 | 62.9% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 38.9 ms | 339,352 | 63.1% | 0.94x | 0.94x |
| random | Evencut, balanced padded size | 30.6 ms | 431,191 | 59.8% | 1.20x | 1.20x |
| length-grouped | equal counts | 36.5 ms | 360,989 | 29.5% | 1.00x | 1.00x |
| length-grouped | Evencut, balanced tokens | 27.8 ms | 473,622 | 34.7% | 1.31x | 1.31x |
| length-grouped | Evencut, balanced padded size | 20.1 ms | 655,572 | 25.0% | 1.82x | 1.82x |
| any order | LPT on padded size (not contiguous) | 31.0 ms | 425,021 | 64.7% | 1.00x | 1.18x |
| sorted dataset | token budget per GPU (fairseq-style) | 19.1 ms | 691,288 | 1.2% | 1.00x | 1.92x |
| sorted dataset | Evencut, padded_slices over the sorted data | 18.9 ms | 696,081 | 1.2% | 1.01x | 1.93x |

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
