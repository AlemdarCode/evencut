// Times evencut::partition (C++ only, no Python) on a file of little-endian uint64 weights.
// Usage: speed_core <weights.u64> <reps> <k1> [k2 ...]
// Prints one line per objective and k: "<objective> <k> <median ms> <max_load>".

#include "evencut/evencut.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: speed_core <weights.u64> <reps> <k1> [k2 ...]\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    const auto bytes = static_cast<std::size_t>(in.tellg());
    std::vector<std::uint64_t> w(bytes / 8);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(w.data()), static_cast<std::streamsize>(w.size() * 8));
    const int reps = std::max(1, std::atoi(argv[2]));
    for (const auto goal : {evencut::objective::min_max, evencut::objective::sum_squares}) {
        for (int a = 3; a < argc; ++a) {
            const auto k = static_cast<std::size_t>(std::atoll(argv[a]));
            evencut::options opt;
            opt.goal = goal;
            std::vector<double> ms;
            std::uint64_t max_load = 0;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                const auto res = evencut::partition(w, k, opt);
                const auto t1 = std::chrono::steady_clock::now();
                max_load = res.max_load;
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            std::printf("%s %zu %.3f %llu\n", goal == evencut::objective::min_max ? "min_max" : "sum_squares", k,
                        ms[ms.size() / 2], static_cast<unsigned long long>(max_load));
        }
    }
    return 0;
}
