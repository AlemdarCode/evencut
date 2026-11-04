# Test 2: Evencut against simple methods

evencut 1.0.0, 2026-11-01. Produced by `bench/compare_simple.py`. Score: how many times longer the slowest worker runs than with Evencut (1.00x = as good as Evencut). Ten real datasets; SNAP node IDs compacted.

## Summary over the ten datasets

| k | equal count: median / worst | nearest-target cut: median / worst | greedy fill: median / worst | recursive bisection: median / worst |
|--:|--:|--:|--:|--:|
| 2 | 1.07x / 1.99x | 1.00x / 1.00x | 1.00x / 1.00x | 1.00x / 1.00x |
| 4 | 1.12x / 3.94x | 1.00x / 1.00x | 1.00x / 1.00x | 1.00x / 1.00x |
| 8 | 1.17x / 7.82x | 1.00x / 1.00x | 1.00x / 1.01x | 1.00x / 1.00x |
| 16 | 1.19x / 15x | 1.00x / 1.12x | 1.00x / 1.15x | 1.00x / 1.12x |
| 32 | 1.20x / 30x | 1.00x / 1.25x | 1.00x / 1.23x | 1.00x / 1.12x |
| 64 | 1.25x / 44x | 1.00x / 1.21x | 1.00x / 1.68x | 1.00x / 1.03x |
| 128 | 1.34x / 39x | 1.00x / 1.16x | 1.01x / 1.24x | 1.00x / 1.16x |
| 256 | 1.46x / 30x | 1.01x / 1.33x | 1.01x / 1.88x | 1.00x / 1.28x |
| 1024 | 1.74x / 12x | 1.01x / 1.08x | 1.02x / 1.22x | 1.00x / 1.02x |
| 4096 | 1.95x / 3.28x | 1.05x / 1.19x | 1.09x / 1.27x | 1.00x / 1.13x |

## k = 8: heaviest part / average part (1.00 = perfect)

| dataset | field | items | best possible | Evencut | equal count | nearest-target cut | greedy fill | recursive bisection |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| psmigr_1 | economics (inter-county migration) | 3,140 | 1.000 | **1.003** | 1.310 | 1.004 | 1.008 | 1.004 |
| bcsstk17 | structural engineering | 10,974 | 1.000 | **1.000** | 1.196 | 1.000 | 1.001 | 1.000 |
| memplus | circuit simulation (memory chip) | 17,758 | 1.000 | **1.001** | 3.301 | 1.004 | 1.010 | 1.004 |
| af23560 | fluid dynamics (airfoil) | 23,560 | 1.000 | **1.000** | 1.004 | 1.000 | 1.000 | 1.000 |
| bcsstk30 | structural engineering | 28,924 | 1.000 | **1.000** | 1.115 | 1.000 | 1.000 | 1.000 |
| amazon0601 | co-purchase network | 403,394 | 1.000 | **1.000** | 1.137 | 1.000 | 1.000 | 1.000 |
| web-Google | web graph | 875,713 | 1.000 | **1.000** | 1.011 | 1.000 | 1.000 | 1.000 |
| roadNet-CA | road network | 1,965,206 | 1.000 | **1.000** | 1.046 | 1.000 | 1.000 | 1.000 |
| wiki-Talk | communication network | 2,394,385 | 1.000 | **1.001** | 7.821 | 1.001 | 1.002 | 1.001 |
| cit-Patents | citation network | 3,774,768 | 1.000 | **1.000** | 2.440 | 1.000 | 1.000 | 1.000 |

## k = 64: heaviest part / average part (1.00 = perfect)

| dataset | field | items | best possible | Evencut | equal count | nearest-target cut | greedy fill | recursive bisection |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| psmigr_1 | economics (inter-county migration) | 3,140 | 1.000 | **1.024** | 2.623 | 1.051 | 1.146 | 1.045 |
| bcsstk17 | structural engineering | 10,974 | 1.000 | **1.003** | 1.318 | 1.008 | 1.008 | 1.006 |
| memplus | circuit simulation (memory chip) | 17,758 | 1.000 | **1.018** | 14.084 | 1.052 | 1.111 | 1.048 |
| af23560 | fluid dynamics (airfoil) | 23,560 | 1.000 | **1.001** | 1.009 | 1.002 | 1.003 | 1.002 |
| bcsstk30 | structural engineering | 28,924 | 1.000 | **1.001** | 1.191 | 1.002 | 1.005 | 1.002 |
| amazon0601 | co-purchase network | 403,394 | 1.000 | **1.000** | 1.167 | 1.000 | 1.000 | 1.000 |
| web-Google | web graph | 875,713 | 1.000 | **1.000** | 1.031 | 1.000 | 1.000 | 1.000 |
| roadNet-CA | road network | 1,965,206 | 1.000 | **1.000** | 1.130 | 1.000 | 1.000 | 1.000 |
| wiki-Talk | communication network | 2,394,385 | 1.275 | **1.275** | 56.560 | 1.544 | 2.142 | 1.275 |
| cit-Patents | citation network | 3,774,768 | 1.000 | **1.000** | 2.570 | 1.000 | 1.000 | 1.000 |

## k = 1024: heaviest part / average part (1.00 = perfect)

| dataset | field | items | best possible | Evencut | equal count | nearest-target cut | greedy fill | recursive bisection |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| psmigr_1 | economics (inter-county migration) | 3,140 | 4.325 | **4.325** | 6.936 | 4.325 | 5.286 | 4.325 |
| bcsstk17 | structural engineering | 10,974 | 1.001 | **1.063** | 2.637 | 1.147 | 1.218 | 1.089 |
| memplus | circuit simulation (memory chip) | 17,758 | 4.659 | **4.659** | 32.818 | 4.659 | 4.781 | 4.659 |
| af23560 | fluid dynamics (airfoil) | 23,560 | 1.000 | **1.021** | 1.066 | 1.038 | 1.040 | 1.021 |
| bcsstk30 | structural engineering | 28,924 | 1.000 | **1.021** | 1.913 | 1.062 | 1.079 | 1.046 |
| amazon0601 | co-purchase network | 403,394 | 1.000 | **1.002** | 1.184 | 1.003 | 1.003 | 1.002 |
| web-Google | web graph | 875,713 | 1.000 | **1.001** | 1.122 | 1.013 | 1.032 | 1.011 |
| roadNet-CA | road network | 1,965,206 | 1.000 | **1.000** | 1.235 | 1.001 | 1.001 | 1.000 |
| wiki-Talk | communication network | 2,394,385 | 20.397 | **20.397** | 249.008 | 20.397 | 20.538 | 20.397 |
| cit-Patents | citation network | 3,774,768 | 1.000 | **1.001** | 2.718 | 1.011 | 1.010 | 1.011 |

"Best possible" is max(total / k, largest item): no split of any kind can beat it. When it is above 1.00, a single item is heavier than a fair share and no method can be perfect.
