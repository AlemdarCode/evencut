# Test 4, real multi-GPU run (Dolly-15k)

evencut 1.0.0, 2026-11-02, 2 x NVIDIA GeForce RTX 4090, nccl 2.29.3, PyTorch 2.14.1+cu126, Linux-7.0.0-42-generic-x86_64-with-glibc2.39. Model small (29.9M parameters), 32 sequences per micro-step, 8 micro-steps per optimizer step (gradient accumulation), 80 micro-steps per plan. One all-reduce of the fp32 gradients (120 MB) takes 47.2 ms. Produced by `bench/ddp_batches.py`; wall time per step, communication included.

| data order | split | step time | real tokens / s | padding | vs the order's first row | vs default (random, equal counts) |
|---|---|--:|--:|--:|--:|--:|
| noise floor | identical slices | 210.8 ms | 197,803 | 63.5% |  |  |
| random | equal counts | 288.6 ms | 143,634 | 73.1% | 1.00x | 1.00x |
| random | Evencut, balanced tokens | 287.6 ms | 144,139 | 73.3% | 1.00x | 1.00x |
| random | Evencut, balanced padded size | 269.8 ms | 153,668 | 72.1% | 1.07x | 1.07x |
| length-grouped | equal counts | 337.0 ms | 123,017 | 61.9% | 1.00x | 0.86x |
| length-grouped | Evencut, balanced tokens | 204.0 ms | 203,167 | 53.2% | 1.65x | 1.41x |
| length-grouped | Evencut, balanced padded size | 177.1 ms | 234,118 | 50.5% | 1.90x | 1.63x |
| any order | LPT on padded size (not contiguous) | 283.3 ms | 146,317 | 74.1% | 1.00x | 1.02x |
| sorted dataset | token budget per GPU (fairseq-style) | 163.1 ms | 254,116 | 1.3% | 1.00x | 1.77x |
| sorted dataset | Evencut, padded_slices over the sorted data | 155.3 ms | 267,003 | 1.3% | 1.05x | 1.86x |
| random | Evencut, whole optimizer step balanced (window) | 169.0 ms | 245,322 | 9.8% | 1.71x | 1.71x |
| pooled (like group_by_length) | Hugging Face group_by_length | 176.1 ms | 235,407 | 1.7% | 1.00x | 1.64x |
| pooled (like group_by_length) | Evencut, length-grouped steps + balanced | 170.4 ms | 243,259 | 1.8% | 1.03x | 1.69x |

GPU topology (`nvidia-smi topo -m`):

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PHB	0-63	0		N/A
GPU1	PHB	 X 	0-63	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
