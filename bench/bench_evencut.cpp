// Timing benchmark for evencut. Prints a Markdown table (median of `reps` runs).
// Usage: bench_evencut [max_n] [reps]   (defaults: 1000000, 3)

#include "evencut/evencut.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<std::uint64_t> make_weights(const std::string& dist, std::size_t n, std::mt19937_64& rng) {
    std::vector<std::uint64_t> w(n);
    std::lognormal_distribution<double> heavy(0.0, 1.5);
    for (std::size_t i = 0; i < n; ++i) {
        if (dist == "uniform") {
            w[i] = 1 + rng() % 1000;
        } else if (dist == "heavy-tail") {
            w[i] = static_cast<std::uint64_t>(std::llround(heavy(rng) * 1000.0)) + 1;
        } else if (dist == "equal") {
            w[i] = 7;
        } else if (dist == "mostly-zero") {
            w[i] = (rng() % 10 == 0) ? 1 + rng() % 100 : 0;
        } else {  // "huge": totals far above 2^40 (exercises the 256-bit path)
            w[i] = (rng() >> 1) / n;
        }
    }
    return w;
}

double median_ms(const std::vector<std::uint64_t>& w, std::size_t k, evencut::objective goal, int reps) {
    std::vector<double> times;
    evencut::options opt;
    opt.goal = goal;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto res = evencut::partition(w, k, opt);
        const auto t1 = std::chrono::steady_clock::now();
        if (res.parts() == 0) std::abort();  // keep the result alive
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t max_n = argc > 1 ? static_cast<std::size_t>(std::atoll(argv[1])) : 1000000;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    std::mt19937_64 rng(12345);
    std::printf("evencut %s benchmark (median of %d runs, milliseconds, native_int128=%d)\n\n", evencut::version(), reps,
                EVENCUT_NATIVE_INT128);
    std::printf("| distribution | n | k | min_max ms | sum_squares ms |\n|---|--:|--:|--:|--:|\n");
    for (const std::string dist : {"uniform", "heavy-tail", "equal", "mostly-zero", "huge"}) {
        for (std::size_t n = 1000; n <= max_n; n *= 10) {
            const auto w = make_weights(dist, n, rng);
            for (const std::size_t k : {2, 8, 64, 1024}) {
                if (k > n) continue;
                const double mm = median_ms(w, k, evencut::objective::min_max, reps);
                const double sq = median_ms(w, k, evencut::objective::sum_squares, reps);
                std::printf("| %s | %zu | %zu | %.2f | %.2f |\n", dist.c_str(), n, k, mm, sq);
                std::fflush(stdout);
            }
        }
    }
    return 0;
}
