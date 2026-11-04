# Test 3: speed

evencut 1.0.0, 2026-11-01, AMD Ryzen 5 5600 6-Core Processor, Windows 11, Python 3.13.6. Produced by `bench/speed.py`. Median of 5 runs (3 for 10M items; Python and alternatives: 3, or 1 for 10M; pure Python: 1).

## Evencut

| dataset | items | C++ min_max k=64 | k=4096 | C++ sum_squares k=64 | k=4096 | from Python min_max k=64 | sum_squares k=64 | pure Python min_max k=64 | sum_squares k=64 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| psmigr_1 | 3,140 | 0.03 ms | k > items | 0.59 ms | k > items | 0.34 ms | 1 ms | 2 ms | 24 ms |
| bcsstk17 | 10,974 | 0.09 ms | 2 ms | 1 ms | 2 ms | 0.66 ms | 2 ms | 3 ms | 81 ms |
| memplus | 17,758 | 0.15 ms | 3 ms | 3 ms | 3 ms | 0.83 ms | 4 ms | 5 ms | 143 ms |
| af23560 | 23,560 | 0.16 ms | 1 ms | 2 ms | 4 ms | 0.97 ms | 4 ms | 6 ms | 136 ms |
| bcsstk30 | 28,924 | 0.10 ms | 2 ms | 5 ms | 7 ms | 1 ms | 7 ms | 7 ms | 251 ms |
| amazon0601 | 403,394 | 2 ms | 10 ms | 62 ms | 97 ms | 15 ms | 89 ms | 102 ms | 3.6 s |
| web-Google | 875,713 | 4 ms | 22 ms | 153 ms | 270 ms | 33 ms | 229 ms | 219 ms | 6.7 s |
| roadNet-CA | 1,965,206 | 9 ms | 24 ms | 310 ms | 509 ms | 76 ms | 447 ms | 479 ms | 17.7 s |
| wiki-Talk | 2,394,385 | 11 ms | 30 ms | 201 ms | 284 ms | 94 ms | 317 ms | 679 ms | 12.1 s |
| cit-Patents | 3,774,768 | 18 ms | 34 ms | 506 ms | 809 ms | 149 ms | 723 ms | 890 ms | 23.2 s |
| uniform-10M | 10,000,000 | 48 ms | 72 ms | 1.3 s | 2.5 s | 567 ms | 2.3 s | - | - |
| heavy-tail-10M | 10,000,000 | 49 ms | 77 ms | 1.3 s | 3.1 s | 604 ms | 2.3 s | - | - |

## Other ways to get the same exact answer

| dataset | items | min_max k=64: Evencut from Python | binary search + greedy (Python) | min_max k=4096: Evencut from Python | binary search + greedy (Python) | sum_squares k=64: Evencut C++ | textbook DP (C) | plain DP (C) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| psmigr_1 | 3,140 | 0.34 ms | 0.26 ms | 2 ms | 0.67 ms | 0.59 ms | 14 ms | 311 ms |
| bcsstk17 | 10,974 | 0.66 ms | 0.77 ms | 3 ms | 7 ms | 1 ms | 21 ms | 3.8 s |
| memplus | 17,758 | 0.83 ms | 0.62 ms | 4 ms | 0.99 ms | 3 ms | 30 ms | - |
| af23560 | 23,560 | 0.97 ms | 0.83 ms | 3 ms | 7 ms | 2 ms | 35 ms | - |
| bcsstk30 | 28,924 | 1 ms | 1 ms | 4 ms | 9 ms | 5 ms | 45 ms | - |
| amazon0601 | 403,394 | 15 ms | 16 ms | 26 ms | 29 ms | 62 ms | 608 ms | - |
| web-Google | 875,713 | 33 ms | 39 ms | 60 ms | 58 ms | 153 ms | 1.5 s | - |
| roadNet-CA | 1,965,206 | 76 ms | 85 ms | 92 ms | 108 ms | 310 ms | 3.5 s | - |
| wiki-Talk | 2,394,385 | 94 ms | 103 ms | 120 ms | 107 ms | 201 ms | 5.1 s | - |
| cit-Patents | 3,774,768 | 149 ms | 134 ms | 172 ms | 158 ms | 506 ms | 6.8 s | - |
| uniform-10M | 10,000,000 | 567 ms | 579 ms | 614 ms | 635 ms | 1.3 s | - | - |
| heavy-tail-10M | 10,000,000 | 604 ms | 598 ms | 613 ms | 679 ms | 1.3 s | - | - |

All alternatives were checked to return the same optimum as Evencut. "-" = not run (too slow for that size). The textbook DP computes every k up to the requested one, which is how it works.
