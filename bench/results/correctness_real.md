# Test 1: optimality on real data

evencut 1.0.0 (native backend), 2026-11-01, AMD Ryzen 5 5600 6-Core Processor, Windows 11, Python 3.13.6.
Produced by `bench/correctness_real.py`. Matrices: one item per row, cost = non-zeros. Networks: one item per node ID, cost = out-degree.

| dataset | field | items | k | min_max proven optimal | sum_squares proven optimal | sum_squares = textbook DP | pure-Python backend: same split | pure-Python: slowest call |
|---|---|--:|---|--:|--:|--:|--:|--:|
| psmigr_1 | economics (inter-county migration) | 3,140 | 2..1024 | 10/10 | 10/10 | 10/10 (k <= 1024) | 20/20 | 0.0 s |
| bcsstk17 | structural engineering | 10,974 | 2..4096 | 11/11 | 11/11 | 11/11 (k <= 4096) | 22/22 | 0.1 s |
| memplus | circuit simulation (memory chip) | 17,758 | 2..4096 | 11/11 | 11/11 | 11/11 (k <= 4096) | 22/22 | 0.2 s |
| af23560 | fluid dynamics (airfoil) | 23,560 | 2..4096 | 11/11 | 11/11 | 11/11 (k <= 4096) | 22/22 | 0.3 s |
| bcsstk30 | structural engineering | 28,924 | 2..4096 | 11/11 | 11/11 | 11/11 (k <= 4096) | 22/22 | 0.4 s |
| amazon0601 | co-purchase network | 403,394 | 2..4096 | 11/11 | 11/11 | 9/9 (k <= 256) | 22/22 | 5.9 s |
| web-Google | web graph | 916,428 | 2..4096 | 11/11 | 11/11 | 9/9 (k <= 256) | 22/22 | 11.0 s |
| roadNet-CA | road network | 1,971,281 | 2..4096 | 11/11 | 11/11 | 8/8 (k <= 128) | 22/22 | 29.1 s |
| wiki-Talk | communication network | 2,394,385 | 2..4096 | 11/11 | 11/11 | 7/7 (k <= 64) | 22/22 | 18.7 s |
| cit-Patents | citation network | 6,009,555 | 2..4096 | 11/11 | 11/11 | 6/6 (k <= 32) | 22/22 | 47.7 s |

Proofs: min_max by the greedy certificate (bound M - 1 needs more than k parts); sum_squares by a Lagrangian-duality lower bound computed exactly by `bench/verify/ssq_check.c`, which shares no code with Evencut. The textbook DP column is a second, independent opinion where it is affordable.

Sources: NIST Matrix Market (math.nist.gov/MatrixMarket) and the Stanford Large Network Dataset Collection (snap.stanford.edu/data). Checksums are in the JSON file.
