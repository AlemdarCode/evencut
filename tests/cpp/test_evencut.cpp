// Tests for evencut.hpp. Every optimum is checked against references written independently of
// the library: an O(k*n^2) dynamic program, brute-force enumeration for small inputs, an
// edge-subset enumeration for trees, and a separate 32-bit-limb 128-bit arithmetic.
//
// Usage: evencut_tests [iterations] [seed]   (defaults: 2000, 20261031)

#include "evencut/evencut.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace harness {
inline int failures = 0;
inline long long checks = 0;
inline std::vector<std::pair<const char*, std::function<void()>>>& registry() {
    static std::vector<std::pair<const char*, std::function<void()>>> r;
    return r;
}
struct registrar {
    registrar(const char* name, std::function<void()> fn) { registry().emplace_back(name, std::move(fn)); }
};
}  // namespace harness

#define TEST_CASE(name)                                          \
    static void name();                                          \
    static const harness::registrar name##_registrar(#name, name); \
    static void name()

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++harness::checks;                                                                \
        if (!(cond)) {                                                                    \
            ++harness::failures;                                                          \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

#define CHECK_THROWS(expr)                                                                       \
    do {                                                                                         \
        ++harness::checks;                                                                       \
        bool thrown_ = false;                                                                    \
        try {                                                                                    \
            (void)(expr);                                                                        \
        } catch (const evencut::error&) {                                                        \
            thrown_ = true;                                                                      \
        }                                                                                        \
        if (!thrown_) {                                                                          \
            ++harness::failures;                                                                 \
            std::fprintf(stderr, "%s:%d: expected evencut::error: %s\n", __FILE__, __LINE__, #expr); \
        }                                                                                        \
    } while (0)

static long long g_iterations = 2000;
static std::uint64_t g_seed = 20261031;

// ---------------------------------------------------------------------------------------
// Independent reference arithmetic: unsigned 128-bit via 32-bit limbs
// ---------------------------------------------------------------------------------------
struct ref_u128 {
    std::uint64_t hi = 0, lo = 0;
    friend bool operator<(const ref_u128& a, const ref_u128& b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; }
    friend bool operator==(const ref_u128& a, const ref_u128& b) { return a.hi == b.hi && a.lo == b.lo; }
    friend ref_u128 operator+(ref_u128 a, const ref_u128& b) {
        const std::uint64_t low = a.lo + b.lo;
        a.hi = a.hi + b.hi + (low < a.lo ? 1U : 0U);
        a.lo = low;
        return a;
    }
};
static ref_u128 ref_square(std::uint64_t x) {
    // (x1*2^32 + x0)^2 with 32-bit pieces
    const std::uint64_t x0 = x & 0xffffffffULL, x1 = x >> 32;
    const std::uint64_t p00 = x0 * x0, p01 = x0 * x1, p11 = x1 * x1;
    ref_u128 r;
    r.lo = p00;
    r.hi = p11;
    // add 2 * p01 * 2^32
    const ref_u128 mid1{p01 >> 32, p01 << 32};
    r = r + mid1;
    r = r + mid1;
    return r;
}
static std::string ref_to_string(ref_u128 v) {
    std::uint32_t d[4] = {static_cast<std::uint32_t>(v.lo), static_cast<std::uint32_t>(v.lo >> 32),
                          static_cast<std::uint32_t>(v.hi), static_cast<std::uint32_t>(v.hi >> 32)};
    std::string s;
    for (;;) {
        bool zero = true;
        for (const std::uint32_t x : d) zero = zero && x == 0;
        if (zero) break;
        std::uint64_t rem = 0;
        for (int i = 3; i >= 0; --i) {
            const std::uint64_t cur = (rem << 32) | d[i];
            d[i] = static_cast<std::uint32_t>(cur / 10);
            rem = cur % 10;
        }
        s.insert(s.begin(), static_cast<char>('0' + rem));
    }
    return s.empty() ? "0" : s;
}

// ---------------------------------------------------------------------------------------
// Reference optimisers (exactly `parts` parts, cuts only where allowed)
// ---------------------------------------------------------------------------------------
struct problem {
    std::vector<std::uint64_t> w;
    std::vector<std::size_t> forbidden;
    std::vector<unsigned char> joined;  // joined[i]: no cut between i and i+1
    std::vector<std::uint64_t> sec;
    std::uint64_t sec_limit = UINT64_MAX;
    std::size_t k = 1;
    std::size_t blocks() const {
        std::size_t b = 1;
        for (std::size_t i = 0; i + 1 < w.size(); ++i) b += joined[i] ? std::size_t{0} : std::size_t{1};
        return b;
    }
    bool cut_ok(std::size_t pos) const { return pos == 0 || pos == w.size() || joined[pos - 1] == 0; }
};

static const std::uint64_t INF64 = UINT64_MAX;

static std::uint64_t ref_minmax(const problem& p, std::size_t parts) {
    const std::size_t n = p.w.size();
    std::vector<std::uint64_t> pre(n + 1, 0), spre(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
        pre[i + 1] = pre[i] + p.w[i];
        spre[i + 1] = spre[i] + (p.sec.empty() ? 0 : p.sec[i]);
    }
    std::vector<std::vector<std::uint64_t>> best(parts + 1, std::vector<std::uint64_t>(n + 1, INF64));
    best[0][0] = 0;
    for (std::size_t j = 1; j <= parts; ++j) {
        for (std::size_t i = 1; i <= n; ++i) {
            if (!p.cut_ok(i)) continue;
            for (std::size_t q = 0; q < i; ++q) {
                if (!p.cut_ok(q) || best[j - 1][q] == INF64) continue;
                if (!p.sec.empty() && spre[i] - spre[q] > p.sec_limit) continue;
                const std::uint64_t cand = std::max(best[j - 1][q], pre[i] - pre[q]);
                best[j][i] = std::min(best[j][i], cand);
            }
        }
    }
    return best[parts][n];
}

static std::string ref_sumsq(const problem& p, std::size_t parts) {
    const std::size_t n = p.w.size();
    std::vector<std::uint64_t> pre(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) pre[i + 1] = pre[i] + p.w[i];
    const ref_u128 inf{UINT64_MAX, UINT64_MAX};
    std::vector<std::vector<ref_u128>> best(parts + 1, std::vector<ref_u128>(n + 1, inf));
    best[0][0] = ref_u128{};
    for (std::size_t j = 1; j <= parts; ++j) {
        for (std::size_t i = 1; i <= n; ++i) {
            if (!p.cut_ok(i)) continue;
            for (std::size_t q = 0; q < i; ++q) {
                if (!p.cut_ok(q) || best[j - 1][q] == inf) continue;
                const ref_u128 cand = best[j - 1][q] + ref_square(pre[i] - pre[q]);
                if (cand < best[j][i]) best[j][i] = cand;
            }
        }
    }
    return best[parts][n] == inf ? "inf" : ref_to_string(best[parts][n]);
}

// Brute force over every set of parts-1 allowed cut positions (small n only).
static void brute_force(const problem& p, std::size_t parts, std::uint64_t& best_max, std::string& best_sq) {
    const std::size_t n = p.w.size();
    std::vector<std::size_t> allowed;
    for (std::size_t pos = 1; pos < n; ++pos) {
        if (p.cut_ok(pos)) allowed.push_back(pos);
    }
    best_max = INF64;
    ref_u128 best{UINT64_MAX, UINT64_MAX};
    const std::size_t a = allowed.size();
    for (std::uint64_t mask = 0; mask < (std::uint64_t{1} << a); ++mask) {
        if (static_cast<std::size_t>(std::popcount(mask)) + 1 != parts) continue;
        std::size_t prev = 0;
        std::uint64_t mx = 0;
        ref_u128 sq{};
        bool sec_ok = true;
        for (std::size_t b = 0; b <= a; ++b) {
            if (b < a && !((mask >> b) & 1U)) continue;
            const std::size_t end = b < a ? allowed[b] : n;
            std::uint64_t load = 0, sl = 0;
            for (std::size_t i = prev; i < end; ++i) {
                load += p.w[i];
                sl += p.sec.empty() ? 0 : p.sec[i];
            }
            if (!p.sec.empty() && sl > p.sec_limit) sec_ok = false;
            mx = std::max(mx, load);
            sq = sq + ref_square(load);
            prev = end;
        }
        if (sec_ok) best_max = std::min(best_max, mx);
        if (sq < best) best = sq;
    }
    best_sq = ref_to_string(best);
}

// ---------------------------------------------------------------------------------------
// Result validation (invariants that must hold for every answer)
// ---------------------------------------------------------------------------------------
static bool valid_partition(const problem& p, const evencut::partition_result& r, std::size_t parts) {
    const std::size_t n = p.w.size();
    if (r.parts() != parts || r.loads.size() != parts) return false;
    if (r.ends.empty() || r.ends.back() != n) return false;
    std::size_t prev = 0;
    std::uint64_t total = 0, mx = 0;
    ref_u128 sq{};
    for (std::size_t s = 0; s < parts; ++s) {
        const std::size_t e = r.ends[s];
        if (e <= prev || !p.cut_ok(e)) return false;
        std::uint64_t load = 0, sl = 0;
        for (std::size_t i = prev; i < e; ++i) {
            load += p.w[i];
            sl += p.sec.empty() ? 0 : p.sec[i];
        }
        if (load != r.loads[s]) return false;
        if (!p.sec.empty() && sl > p.sec_limit) return false;
        total += load;
        mx = std::max(mx, load);
        sq = sq + ref_square(load);
        if (r.part_begin(s) != prev || r.part_end(s) != e) return false;
        prev = e;
    }
    return total == r.total && mx == r.max_load && ref_to_string(sq) == r.sum_of_squares;
}

// ---------------------------------------------------------------------------------------
// Random problem generation
// ---------------------------------------------------------------------------------------
static problem random_problem(std::mt19937_64& rng, std::size_t max_n, bool allow_secondary) {
    problem p;
    const std::size_t n = 1 + rng() % max_n;
    const int dist = static_cast<int>(rng() % 9);
    std::uint64_t equal_value = rng() % 4;
    // totals between 2^41 and 2^62: just above the 128-bit fast-path limit, where an overflow
    // would show up if the limit were set too high
    const std::uint64_t mid_cap = (std::uint64_t{1} << (41 + rng() % 22)) / n + 1;
    p.w.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        switch (dist) {
            case 7: p.w[i] = mid_cap - 1 - rng() % (mid_cap / 8 + 1); break;     // large and nearly equal
            case 8: p.w[i] = rng() % mid_cap; break;                              // large and uneven
            case 0: p.w[i] = rng() % 10; break;                                   // small
            case 1: p.w[i] = (rng() % 10 < 7) ? 0 : 1 + rng() % 5; break;         // mostly zeros
            case 2: p.w[i] = equal_value; break;                                  // all equal
            case 3: p.w[i] = rng() % ((std::uint64_t{1} << 40) / n + 1); break;   // around the fast-path limit
            case 4: p.w[i] = rng() % (UINT64_MAX / n); break;                     // huge totals (wide path)
            case 5: p.w[i] = (i == n / 2) ? 1000 + rng() % 1000 : rng() % 5; break;  // one heavy item
            default: p.w[i] = (std::uint64_t{1} << (rng() % 20)) + rng() % 3; break;  // heavy tail
        }
    }
    p.joined.assign(n > 0 ? n - 1 : 0, 0);
    const unsigned density = static_cast<unsigned>(rng() % 3);  // 0%, ~20%, ~50%
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const bool forbid = density == 1 ? rng() % 5 == 0 : (density == 2 ? rng() % 2 == 0 : false);
        if (forbid) {
            p.joined[i] = 1;
            p.forbidden.push_back(i + 1);  // numbered like ends: cut between item i and i+1
        }
    }
    p.k = 1 + rng() % (n + 2);
    if (allow_secondary && rng() % 2 == 0) {
        p.sec.resize(n);
        for (auto& s : p.sec) s = rng() % 10;
        // pick a limit around the largest block, sometimes infeasible
        std::uint64_t max_block = 0, cur = 0;
        for (std::size_t i = 0; i < n; ++i) {
            cur += p.sec[i];
            if (i + 1 == n || !p.joined[i]) {
                max_block = std::max(max_block, cur);
                cur = 0;
            }
        }
        const std::uint64_t total = [&] { std::uint64_t t = 0; for (auto s : p.sec) t += s; return t; }();
        p.sec_limit = max_block + rng() % (total - max_block + 2);
        if (rng() % 10 == 0 && max_block > 0) p.sec_limit = max_block - 1;  // infeasible
    }
    return p;
}

static evencut::options make_options(const problem& p, evencut::objective goal) {
    evencut::options o;
    o.goal = goal;
    o.forbidden_cuts = p.forbidden;
    if (!p.sec.empty()) {
        o.secondary = p.sec;
        o.secondary_limit = p.sec_limit;
    }
    return o;
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------
TEST_CASE(wide_matches_known_values) {
    using W2 = evencut::detail::wide<2>;
    using W4 = evencut::detail::wide<4>;
    const W2 two64 = W2::from_u64(UINT64_MAX) + W2(1);
    CHECK(two64.to_string() == "18446744073709551616");
    W4 max128 = W4::from_u64(UINT64_MAX);
    max128.limb[1] = UINT64_MAX;  // 2^128 - 1
    CHECK(max128.to_string() == "340282366920938463463374607431768211455");
    CHECK((W4(0) - max128).to_string() == "-340282366920938463463374607431768211455");
    CHECK(W2(-7).half() == W2(-4));
    CHECK(W2(7).half() == W2(3));
    CHECK(W2(-1) < W2(0) && W2(0) < W2(1) && !(W2(1) < W2(1)));
    CHECK(evencut::detail::sum_of_squares(std::vector<std::uint64_t>{UINT64_MAX}) == "340282366920938463426481119284349108225");
}

TEST_CASE(wide_matches_reference_arithmetic) {
    std::mt19937_64 rng(g_seed ^ 0x51ULL);
    using W4 = evencut::detail::wide<4>;
    for (long long it = 0; it < g_iterations * 5; ++it) {
        const std::uint64_t a = rng() >> (rng() % 64);
        const std::uint64_t b = rng() >> (rng() % 64);
        // a*a over the full 64-bit range
        const W4 sq = W4::from_u64(a) * W4::from_u64(a);
        CHECK(sq.to_string() == ref_to_string(ref_square(a)));
        // a*a + b*b with operands below 2^63, so the sum fits the 128-bit reference
        // (with full-range operands the sum can reach 2^129 and the reference would wrap)
        const std::uint64_t ha = a >> 1, hb = b >> 1;
        const W4 lib = W4::from_u64(ha) * W4::from_u64(ha) + W4::from_u64(hb) * W4::from_u64(hb);
        CHECK(lib.to_string() == ref_to_string(ref_square(ha) + ref_square(hb)));
        // (a - b) * (b - a) == -(a - b)^2 checks signed products
        const W4 d = W4::from_u64(a) - W4::from_u64(b);
        const W4 neg = d * (W4(0) - d);
        const std::uint64_t ad = a > b ? a - b : b - a;
        CHECK(neg.to_string() == (ad == 0 ? std::string("0") : "-" + ref_to_string(ref_square(ad))));
        CHECK((W4::from_u64(a) < W4::from_u64(b)) == (a < b));
#if EVENCUT_NATIVE_INT128
        using N = evencut::detail::native_i128;
        using W2 = evencut::detail::wide<2>;
        const std::int64_t x = static_cast<std::int64_t>(rng());
        const std::int64_t y = static_cast<std::int64_t>(rng());
        const W2 wp = W2(x) * W2(y) + W2(x) - W2(y);
        const N np = static_cast<N>(x) * y + x - y;
        CHECK(wp.limb[0] == static_cast<std::uint64_t>(np) && wp.limb[1] == static_cast<std::uint64_t>(static_cast<evencut::detail::native_u128>(np) >> 64));
        CHECK((W2(x) < W2(y)) == (x < y));
        using P = evencut::detail::portable_i128;
        const P pp = P(x) * P(y) + P(x) - P(y);
        CHECK(pp.lo == static_cast<std::uint64_t>(np) && pp.hi == static_cast<std::uint64_t>(static_cast<evencut::detail::native_u128>(np) >> 64));
        CHECK((P(x) < P(y)) == (x < y) && (P(x) == P(y)) == (x == y));
        const N big = static_cast<N>(x) * y;
        const P pbig = P(x) * P(y);
        CHECK(pbig.half().lo == static_cast<std::uint64_t>(big >> 1));  // arithmetic shift
        if (big >= 0) {
            const std::uint64_t divisor = (rng() >> (rng() % 63)) | 1U;
            const N q = big / static_cast<N>(divisor);
            const P pq = pbig.div_u64(divisor);
            CHECK(pq.lo == static_cast<std::uint64_t>(q) && pq.hi == static_cast<std::uint64_t>(static_cast<evencut::detail::native_u128>(q) >> 64));
        }
#endif
    }
}

TEST_CASE(min_max_matches_reference) {
    std::mt19937_64 rng(g_seed ^ 0x6d6dULL);
    for (long long it = 0; it < g_iterations; ++it) {
        const problem p = random_problem(rng, 40, true);
        const std::size_t parts = std::min(p.k, p.blocks());
        const std::uint64_t expected = ref_minmax(p, parts);
        for (const bool polish : {true, false}) {
            evencut::options o = make_options(p, evencut::objective::min_max);
            o.polish = polish;
            if (expected == INF64) {
                CHECK_THROWS(evencut::partition(p.w, p.k, o));
                continue;
            }
            const auto r = evencut::partition(p.w, p.k, o);
            CHECK(valid_partition(p, r, parts));
            CHECK(r.max_load == expected);
        }
    }
}

TEST_CASE(sum_squares_matches_reference) {
    std::mt19937_64 rng(g_seed ^ 0x5353ULL);
    for (long long it = 0; it < g_iterations; ++it) {
        const problem p = random_problem(rng, 40, false);
        const std::size_t parts = std::min(p.k, p.blocks());
        const auto r = evencut::partition(p.w, p.k, make_options(p, evencut::objective::sum_squares));
        CHECK(valid_partition(p, r, parts));
        CHECK(r.sum_of_squares == ref_sumsq(p, parts));
    }
}

TEST_CASE(brute_force_agrees_on_small_inputs) {
    std::mt19937_64 rng(g_seed ^ 0xb0b0ULL);
    for (long long it = 0; it < g_iterations; ++it) {
        const problem p = random_problem(rng, 11, true);
        const std::size_t parts = std::min(p.k, p.blocks());
        std::uint64_t bf_max = 0;
        std::string bf_sq;
        brute_force(p, parts, bf_max, bf_sq);
        CHECK(bf_max == ref_minmax(p, parts));  // the references agree with each other
        if (p.sec.empty()) {
            CHECK(bf_sq == ref_sumsq(p, parts));
            const auto r = evencut::partition(p.w, p.k, make_options(p, evencut::objective::sum_squares));
            CHECK(r.sum_of_squares == bf_sq);
        }
        if (bf_max != INF64) {
            const auto r = evencut::partition(p.w, p.k, make_options(p, evencut::objective::min_max));
            CHECK(r.max_load == bf_max);
        }
    }
}

TEST_CASE(sum_squares_paths_agree) {
    // The 128-bit fast path and the 256-bit path must find partitions of equal cost.
    std::mt19937_64 rng(g_seed ^ 0x7a7aULL);
    using namespace evencut::detail;
    for (long long it = 0; it < g_iterations; ++it) {
        problem p = random_problem(rng, 60, false);
        for (auto& x : p.w) x %= (std::uint64_t{1} << 30);  // keep within the fast-path limit
        const block_view bv = make_blocks(p.w, make_options(p, evencut::objective::sum_squares));
        const std::size_t m = bv.blocks();
        if (m < 3) continue;
        const std::size_t k = 2 + rng() % (m - 2);
        const auto wide4 = sum_squares_blocks<wide<4>>(bv.prefix, k);
        const auto wide2 = sum_squares_blocks<wide<2>>(bv.prefix, k);
        const auto cost = [&](const std::vector<std::size_t>& b) {
            std::vector<std::uint64_t> loads;
            for (std::size_t t = 1; t < b.size(); ++t) loads.push_back(bv.prefix[b[t]] - bv.prefix[b[t - 1]]);
            return std::make_pair(b.size() - 1, sum_of_squares(loads));
        };
        CHECK(cost(wide4) == cost(wide2));
        CHECK(cost(wide4) == cost(sum_squares_blocks<portable_i128>(bv.prefix, k)));
        CHECK(cost(wide4).first == k);
#if EVENCUT_NATIVE_INT128
        CHECK(cost(sum_squares_blocks<native_i128>(bv.prefix, k)) == cost(wide4));
#endif
    }
}

TEST_CASE(evaluate_and_exact_k) {
    const std::vector<std::uint64_t> w{5, 3, 8, 2, 7, 4, 6, 1, 9, 5};
    const auto r = evencut::partition(w, 3);
    CHECK(r.parts() == 3);
    CHECK(r.max_load == 18);  // 17 is infeasible: the greedy needs 4 parts ([5,3,8] [2,7,4] [6,1,9] [5])
    const std::vector<std::size_t> uniform{4, 7, 10};
    const auto e = evencut::evaluate(w, uniform);
    CHECK(e.loads == (std::vector<std::uint64_t>{18, 17, 15}));
    CHECK(e.max_load == 18);
    CHECK(r.max_load <= e.max_load);
    // more parts than items: every item becomes its own part
    const auto many = evencut::partition(w, 50);
    CHECK(many.parts() == w.size());
    // all zeros
    const std::vector<std::uint64_t> zeros(9, 0);
    const auto z = evencut::partition(zeros, 4, evencut::options{evencut::objective::sum_squares});
    CHECK(z.parts() == 4 && z.sum_of_squares == "0" && z.imbalance == 1.0);
}

TEST_CASE(invalid_input_is_rejected) {
    const std::vector<std::uint64_t> w{1, 2, 3};
    const std::vector<std::uint64_t> empty;
    CHECK_THROWS(evencut::partition(empty, 1));
    CHECK_THROWS(evencut::partition(w, 0));
    const std::vector<std::uint64_t> overflow{UINT64_MAX, 1};
    CHECK_THROWS(evencut::partition(overflow, 1));
    for (const std::size_t bad : {std::size_t{0}, std::size_t{3}}) {  // valid cut positions are 1 .. n-1
        const std::vector<std::size_t> bad_cut{bad};
        evencut::options o;
        o.forbidden_cuts = bad_cut;
        CHECK_THROWS(evencut::partition(w, 2, o));
    }
    const std::vector<std::size_t> ok_cut{2};
    evencut::options o_ok;
    o_ok.forbidden_cuts = ok_cut;
    CHECK(evencut::partition(w, 2, o_ok).ends == (std::vector<std::size_t>{1, 3}));  // 2 is never an end
    const std::vector<std::uint64_t> sec{1, 1};
    evencut::options o2;
    o2.secondary = sec;
    o2.secondary_limit = 5;
    CHECK_THROWS(evencut::partition(w, 2, o2));  // wrong size
    const std::vector<std::uint64_t> sec_ok{1, 1, 1};
    evencut::options no_limit;
    no_limit.secondary = sec_ok;
    CHECK_THROWS(evencut::partition(w, 2, no_limit));  // secondary without a limit is not silently unlimited
    evencut::options only_limit;
    only_limit.secondary_limit = 5;
    CHECK_THROWS(evencut::partition(w, 2, only_limit));  // a limit without values
    const std::vector<std::uint64_t> sec3{1, 5, 1};
    evencut::options o3;
    o3.secondary = sec3;
    o3.secondary_limit = 4;
    CHECK_THROWS(evencut::partition(w, 3, o3));  // item 1 alone exceeds the limit
    o3.goal = evencut::objective::sum_squares;
    o3.secondary_limit = 10;
    CHECK_THROWS(evencut::partition(w, 3, o3));  // secondary not supported with sum_squares
    const std::vector<std::size_t> bad_ends{1, 1, 3};
    CHECK_THROWS(evencut::evaluate(w, bad_ends));
    const std::vector<double> nan_values{1.0, std::nan("")};
    CHECK_THROWS(evencut::quantize(nan_values, 10.0));
    const std::vector<double> neg{-1.0};
    CHECK_THROWS(evencut::quantize(neg, 10.0));
    const std::vector<double> ok{0.25, 1.5};
    CHECK((evencut::quantize(ok, 100.0) == std::vector<std::uint64_t>{25, 150}));
    // trees
    const std::vector<std::size_t> two_roots{evencut::no_parent, evencut::no_parent};
    const std::vector<std::uint64_t> tw{1, 1};
    CHECK_THROWS(evencut::partition_tree(two_roots, tw, 1));
    const std::vector<std::size_t> cycle{evencut::no_parent, 2, 1};
    const std::vector<std::uint64_t> cw{1, 1, 1};
    CHECK_THROWS(evencut::partition_tree(cycle, cw, 2));
    const std::vector<std::size_t> self{evencut::no_parent, 1};
    CHECK_THROWS(evencut::partition_tree(self, tw, 2));
}

TEST_CASE(results_are_deterministic) {
    std::mt19937_64 rng(g_seed ^ 0xdeULL);
    for (int it = 0; it < 200; ++it) {
        const problem p = random_problem(rng, 200, false);
        for (const auto goal : {evencut::objective::min_max, evencut::objective::sum_squares}) {
            const auto a = evencut::partition(p.w, p.k, make_options(p, goal));
            const auto b = evencut::partition(p.w, p.k, make_options(p, goal));
            CHECK(a.ends == b.ends);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------------------
static std::vector<std::size_t> random_tree(std::mt19937_64& rng, std::size_t n) {
    std::vector<std::size_t> label(n);
    for (std::size_t i = 0; i < n; ++i) label[i] = i;
    std::shuffle(label.begin(), label.end(), rng);
    std::vector<std::size_t> parent(n, evencut::no_parent);
    const int shape = static_cast<int>(rng() % 3);
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t p;
        if (shape == 0) p = rng() % i;            // random recursive tree
        else if (shape == 1) p = i - 1;           // path
        else p = (rng() % 4 == 0) ? rng() % i : 0;  // star-like
        parent[label[i]] = label[p];
    }
    return parent;
}

static std::uint64_t tree_brute_force(const std::vector<std::size_t>& parent, const std::vector<std::uint64_t>& w, std::size_t parts) {
    const std::size_t n = parent.size();
    std::vector<std::size_t> nonroot;
    for (std::size_t v = 0; v < n; ++v) {
        if (parent[v] != evencut::no_parent) nonroot.push_back(v);
    }
    std::uint64_t best = INF64;
    for (std::uint64_t mask = 0; mask < (std::uint64_t{1} << nonroot.size()); ++mask) {
        if (static_cast<std::size_t>(std::popcount(mask)) + 1 != parts) continue;
        std::vector<unsigned char> cut(n, 0);
        for (std::size_t b = 0; b < nonroot.size(); ++b) {
            if ((mask >> b) & 1U) cut[nonroot[b]] = 1;
        }
        // component id: walk up until a root or a cut node
        std::vector<std::uint64_t> load(n, 0);
        for (std::size_t v = 0; v < n; ++v) {
            std::size_t u = v;
            while (parent[u] != evencut::no_parent && !cut[u]) u = parent[u];
            load[u] += w[v];
        }
        best = std::min(best, *std::max_element(load.begin(), load.end()));
    }
    return best;
}

TEST_CASE(tree_matches_brute_force) {
    std::mt19937_64 rng(g_seed ^ 0x7ee7ULL);
    for (long long it = 0; it < g_iterations; ++it) {
        const std::size_t n = 1 + rng() % 11;
        const auto parent = random_tree(rng, n);
        std::vector<std::uint64_t> w(n);
        const int dist = static_cast<int>(rng() % 3);
        for (auto& x : w) x = dist == 0 ? rng() % 10 : (dist == 1 ? rng() % 2 : rng() % 1000);
        const std::size_t k = 1 + rng() % (n + 2);
        const std::size_t parts = std::min(k, n);
        const auto r = evencut::partition_tree(parent, w, k);
        CHECK(r.parts() == parts);
        CHECK(r.max_load == tree_brute_force(parent, w, parts));
        // invariants: part ids, connectivity, loads, cut list
        std::vector<std::uint64_t> load(parts, 0);
        bool ok = true;
        std::size_t cuts = 0;
        for (std::size_t v = 0; v < n; ++v) {
            if (r.part_of[v] >= parts) ok = false;
            else load[r.part_of[v]] += w[v];
            const bool is_cut = std::find(r.cut_nodes.begin(), r.cut_nodes.end(), v) != r.cut_nodes.end();
            cuts += is_cut ? 1 : 0;
            if (parent[v] != evencut::no_parent && !is_cut && r.part_of[v] != r.part_of[parent[v]]) ok = false;
            if (parent[v] != evencut::no_parent && is_cut && r.part_of[v] == r.part_of[parent[v]]) ok = false;
        }
        CHECK(ok);
        CHECK(cuts + 1 == parts);
        CHECK(load == r.loads);
    }
}

TEST_CASE(deep_tree_does_not_recurse) {
    const std::size_t n = 200000;  // a path this deep would overflow a recursive implementation
    std::vector<std::size_t> parent(n);
    std::vector<std::uint64_t> w(n, 1);
    parent[0] = evencut::no_parent;
    for (std::size_t v = 1; v < n; ++v) parent[v] = v - 1;
    const auto r = evencut::partition_tree(parent, w, 8);
    CHECK(r.parts() == 8);
    CHECK(r.max_load == n / 8);
}

int main(int argc, char** argv) {
    if (argc > 1) g_iterations = std::atoll(argv[1]);
    if (argc > 2) g_seed = std::strtoull(argv[2], nullptr, 10);
    std::printf("evencut %s tests: iterations=%lld seed=%llu native_int128=%d\n", evencut::version(), g_iterations,
                static_cast<unsigned long long>(g_seed), EVENCUT_NATIVE_INT128);
    for (const auto& [name, fn] : harness::registry()) {
        const int before = harness::failures;
        fn();
        std::printf("  %-40s %s\n", name, harness::failures == before ? "ok" : "FAILED");
    }
    std::printf("%lld checks, %d failures\n", harness::checks, harness::failures);
    return harness::failures == 0 ? 0 : 1;
}
