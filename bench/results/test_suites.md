# Test 1: test suites (random inputs against brute force)

Run on 2026-11-01, AMD Ryzen 5 5600, Windows 11. Every optimum is compared with brute force and
with an independent dynamic program, on random inputs that include zeros, equal weights, totals
near 2^64, forbidden cuts, secondary limits and trees.

| build | command | checks | failures |
|---|---|--:|--:|
| GCC 16, 128-bit fast path | `evencut_tests 200000 1101` | 13,003,193 | 0 |
| GCC 16, `-DEVENCUT_NO_INT128` (the path MSVC uses) | `evencut_tests 200000 1102` | 7,313,303 | 0 |
| MSVC 14.42 | `evencut_tests 200000 1103` | 7,314,835 | 0 |
| GCC 16, UndefinedBehaviorSanitizer (trap mode) | `evencut_tests 20000 1104` | 1,301,125 | 0 |
| **total** | | **28,932,456** | **0** |

The number of checks is not a score: it grows with the number of random rounds you ask for.
`evencut_tests` with no argument runs 2,000 rounds (130,467 checks with the 128-bit path, 73,565
without it); 4,000 rounds give about 260,000. The runs above use 200,000 rounds. Any build can
be pushed further: `evencut_tests <rounds> <seed>`.

The default run (`ctest`, 500 iterations) and the C ABI test pass on GCC and MSVC (8/8 tests).
Python: 56 tests pass on the native backend; 55 pass on the pure-Python backend, plus one skipped
because it needs the native library. Both include Hypothesis property tests against brute force
and the HTTP service.
