# Test 5: when the order does not matter

evencut 1.0.0, 2026-11-01. Produced by `bench/order_free.py`.

## Part 1: plain work items, largest part / average part (1.0000 = perfect)

Evencut keeps the order (contiguous parts); LPT may put any item anywhere. The bound is max(total / k, largest item): no assignment of any kind beats it.

| dataset | items | k | bound | Evencut (contiguous) | LPT (any order) |
|---|--:|--:|--:|--:|--:|
| psmigr_1 | 3,140 | 2 | 1.0000 | 1.0000 | 1.0000 |
| psmigr_1 | 3,140 | 8 | 1.0000 | 1.0031 | 1.0001 |
| psmigr_1 | 3,140 | 64 | 1.0000 | 1.0243 | 1.0012 |
| psmigr_1 | 3,140 | 1024 | 4.3248 | 4.3248 | 4.3248 |
| bcsstk17 | 10,974 | 2 | 1.0000 | 1.0001 | 1.0000 |
| bcsstk17 | 10,974 | 8 | 1.0000 | 1.0004 | 1.0000 |
| bcsstk17 | 10,974 | 64 | 1.0001 | 1.0035 | 1.0001 |
| bcsstk17 | 10,974 | 1024 | 1.0009 | 1.0631 | 1.0081 |
| bcsstk17 | 10,974 | 4096 | 1.4333 | 1.4333 | 1.4333 |
| memplus | 17,758 | 2 | 1.0000 | 1.0000 | 1.0000 |
| memplus | 17,758 | 8 | 1.0000 | 1.0013 | 1.0000 |
| memplus | 17,758 | 64 | 1.0005 | 1.0177 | 1.0005 |
| memplus | 17,758 | 1024 | 4.6593 | 4.6593 | 4.6593 |
| memplus | 17,758 | 4096 | 18.6374 | 18.6374 | 18.6374 |
| af23560 | 23,560 | 2 | 1.0000 | 1.0000 | 1.0000 |
| af23560 | 23,560 | 8 | 1.0000 | 1.0001 | 1.0000 |
| af23560 | 23,560 | 64 | 1.0001 | 1.0010 | 1.0010 |
| af23560 | 23,560 | 1024 | 1.0002 | 1.0213 | 1.0171 |
| af23560 | 23,560 | 4096 | 1.0065 | 1.0658 | 1.0573 |
| bcsstk30 | 28,924 | 2 | 1.0000 | 1.0000 | 1.0000 |
| bcsstk30 | 28,924 | 8 | 1.0000 | 1.0001 | 1.0000 |
| bcsstk30 | 28,924 | 64 | 1.0000 | 1.0013 | 1.0000 |
| bcsstk30 | 28,924 | 1024 | 1.0002 | 1.0212 | 1.0042 |
| bcsstk30 | 28,924 | 4096 | 1.0002 | 1.0804 | 1.0182 |
| amazon0601 | 403,394 | 2 | 1.0000 | 1.0000 | 1.0000 |
| amazon0601 | 403,394 | 8 | 1.0000 | 1.0000 | 1.0000 |
| amazon0601 | 403,394 | 64 | 1.0000 | 1.0001 | 1.0000 |
| amazon0601 | 403,394 | 1024 | 1.0000 | 1.0015 | 1.0000 |
| amazon0601 | 403,394 | 4096 | 1.0000 | 1.0060 | 1.0000 |
| web-Google | 875,713 | 2 | 1.0000 | 1.0000 | 1.0000 |
| web-Google | 875,713 | 8 | 1.0000 | 1.0000 | 1.0000 |
| web-Google | 875,713 | 64 | 1.0000 | 1.0001 | 1.0000 |
| web-Google | 875,713 | 1024 | 1.0001 | 1.0013 | 1.0001 |
| web-Google | 875,713 | 4096 | 1.0005 | 1.0053 | 1.0005 |
| roadNet-CA | 1,965,206 | 2 | 1.0000 | 1.0000 | 1.0000 |
| roadNet-CA | 1,965,206 | 8 | 1.0000 | 1.0000 | 1.0000 |
| roadNet-CA | 1,965,206 | 64 | 1.0000 | 1.0000 | 1.0000 |
| roadNet-CA | 1,965,206 | 1024 | 1.0001 | 1.0003 | 1.0001 |
| roadNet-CA | 1,965,206 | 4096 | 1.0001 | 1.0008 | 1.0001 |
| wiki-Talk | 2,394,385 | 2 | 1.0000 | 1.0000 | 1.0000 |
| wiki-Talk | 2,394,385 | 8 | 1.0000 | 1.0007 | 1.0000 |
| wiki-Talk | 2,394,385 | 64 | 1.2748 | 1.2748 | 1.2748 |
| wiki-Talk | 2,394,385 | 1024 | 20.3972 | 20.3972 | 20.3972 |
| wiki-Talk | 2,394,385 | 4096 | 81.5887 | 81.5887 | 81.5887 |
| cit-Patents | 3,774,768 | 2 | 1.0000 | 1.0000 | 1.0000 |
| cit-Patents | 3,774,768 | 8 | 1.0000 | 1.0000 | 1.0000 |
| cit-Patents | 3,774,768 | 64 | 1.0000 | 1.0000 | 1.0000 |
| cit-Patents | 3,774,768 | 1024 | 1.0000 | 1.0005 | 1.0000 |
| cit-Patents | 3,774,768 | 4096 | 1.0000 | 1.0022 | 1.0000 |

## Part 2: data-parallel steps with padding, predicted share of time GPUs wait

| dataset | GPUs | sequences per step | Evencut, contiguous in length order: idle / largest padded slice | LPT on padded size, any order: idle / largest padded slice |
|---|--:|--:|--:|--:|
| TinyStories-Turkish | 8 | 256 | 1.5% / 5,313 | 1.2% / 7,337 |
| Dolly-15k | 8 | 128 | 5.6% / 3,450 | 3.0% / 10,830 |
| Wikipedia-TR | 8 | 48 | 14.8% / 4,309 | 6.6% / 9,734 |

Step time follows the largest padded slice, not the idle share: LPT mixes lengths on every GPU, so each GPU pads more and the step gets slower even though the GPUs wait less.
