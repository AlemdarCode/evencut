/**
 * @file evencut.hpp
 * @brief Evencut: exact, balanced partitioning of ordered workloads.
 *
 * Evencut splits an ordered list of non-negative integer weights into k contiguous parts,
 * or a tree into k connected parts, so that the parts are as even as possible:
 *
 *   - objective::min_max      minimise the largest part (the bottleneck / makespan)
 *   - objective::sum_squares  minimise the sum of squared part loads (overall evenness)
 *
 * Every result is an exact optimum and comes with its cut positions. Typical uses are
 * pipeline-parallel stage assignment, token-balanced sharding across GPUs, splitting
 * ordered jobs, logs or files across workers, and range sharding.
 *
 * Properties: C++20, header-only, no dependencies, no global state (thread-safe),
 * deterministic, exact for any input whose total weight fits in 64 bits.
 *
 * Complexity (n items, k parts, S total weight):
 *   - min_max      O(n + k log n log S)
 *   - sum_squares  O(n log S)  (Lagrangian relaxation + convex hull trick)
 *   - tree         O(n log n log S)  (Kundu-Misra greedy + binary search)
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#define EVENCUT_VERSION_MAJOR 1
#define EVENCUT_VERSION_MINOR 0
#define EVENCUT_VERSION_PATCH 0
#define EVENCUT_VERSION_STRING "1.0.0"

// Native 128-bit integers are used when the compiler has them (GCC, Clang). MSVC does not,
// so a portable fixed-width integer is used there. Define EVENCUT_NO_INT128 to force the
// portable path on any compiler (this is how the MSVC path is tested with GCC).
#if !defined(EVENCUT_NO_INT128) && defined(__SIZEOF_INT128__)
#define EVENCUT_NATIVE_INT128 1
#else
#define EVENCUT_NATIVE_INT128 0
#endif

namespace evencut {

/// Library version, e.g. "1.0.0".
inline constexpr const char* version() noexcept { return EVENCUT_VERSION_STRING; }

/// Thrown for invalid input: empty input, k == 0, mismatched sizes, total weight above
/// 2^64-1, out-of-range forbidden cuts, infeasible secondary limits, malformed trees.
class error : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/// What "balanced" means.
enum class objective {
    min_max,     ///< minimise the largest part load
    sum_squares  ///< minimise the sum of squared part loads
};

/// Parent value that marks the root of a tree.
inline constexpr std::size_t no_parent = std::numeric_limits<std::size_t>::max();

/// Options for partition().
struct options {
    objective goal = objective::min_max;
    /// Cut positions that are not allowed, numbered like `ends`: b forbids the cut between
    /// item b-1 and item b, so a result's `ends` never contains b. Valid entries are 1 .. n-1;
    /// duplicates are allowed.
    std::span<const std::size_t> forbidden_cuts{};
    /// Optional second resource per item (for example memory per layer). When given, every
    /// part must satisfy sum(secondary) <= *secondary_limit, and the limit is required.
    /// Only supported with min_max.
    std::span<const std::uint64_t> secondary{};
    std::optional<std::uint64_t> secondary_limit{};
    /// Whether to even out the non-maximal parts after the optimal maximum is found
    /// (min_max only). The maximum itself is optimal either way.
    bool polish = true;
};

/// Result of partition() and evaluate().
struct partition_result {
    std::vector<std::size_t> ends;     ///< exclusive end index of each part; ends.back() == n
    std::vector<std::uint64_t> loads;  ///< weight of each part
    std::uint64_t max_load = 0;        ///< largest part weight
    std::uint64_t total = 0;           ///< sum of all weights
    std::string sum_of_squares;        ///< exact decimal value of sum(load^2)
    double imbalance = 1.0;            ///< max_load / (total / parts); 1.0 means perfectly even

    /// Number of parts. Equals min(k, number of allowed blocks).
    [[nodiscard]] std::size_t parts() const noexcept { return ends.size(); }
    /// First item index of a part.
    [[nodiscard]] std::size_t part_begin(std::size_t part) const noexcept { return part == 0 ? 0 : ends[part - 1]; }
    /// One past the last item index of a part.
    [[nodiscard]] std::size_t part_end(std::size_t part) const noexcept { return ends[part]; }
};

/// Result of partition_tree().
struct tree_result {
    std::vector<std::size_t> part_of;    ///< part id of every node (0 .. parts()-1); the root is in part 0
    std::vector<std::uint64_t> loads;    ///< weight of each part
    std::vector<std::size_t> cut_nodes;  ///< nodes whose edge to their parent is cut, ascending
    std::uint64_t max_load = 0;
    std::uint64_t total = 0;
    std::string sum_of_squares;
    double imbalance = 1.0;

    [[nodiscard]] std::size_t parts() const noexcept { return loads.size(); }
};

namespace detail {

#if EVENCUT_NATIVE_INT128
__extension__ typedef __int128 native_i128;
__extension__ typedef unsigned __int128 native_u128;
#endif

/// Full 64x64 -> 128-bit unsigned product.
inline void mul64(std::uint64_t a, std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo) noexcept {
#if EVENCUT_NATIVE_INT128
    const native_u128 p = static_cast<native_u128>(a) * b;
    lo = static_cast<std::uint64_t>(p);
    hi = static_cast<std::uint64_t>(p >> 64);
#elif defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
    lo = _umul128(a, b, &hi);
#elif defined(_MSC_VER) && !defined(__clang__) && defined(_M_ARM64)
    lo = a * b;
    hi = __umulh(a, b);
#else
    const std::uint64_t mask = 0xffffffffULL;
    const std::uint64_t a0 = a & mask, a1 = a >> 32, b0 = b & mask, b1 = b >> 32;
    const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const std::uint64_t mid = (p00 >> 32) + (p01 & mask) + (p10 & mask);
    lo = (p00 & mask) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
}

/// Signed fixed-width integer with N 64-bit limbs (two's complement, little-endian).
/// Supports exactly what the solvers need: +, -, *, comparisons, halving and decimal output.
template <std::size_t N>
struct wide {
    static_assert(N >= 2, "wide<N> needs at least two limbs");
    std::array<std::uint64_t, N> limb{};

    constexpr wide() noexcept = default;
    // Implicit on purpose so small constants (wide<N>(0), wide<N>(1)) read naturally.
    constexpr wide(std::int64_t v) noexcept {  // NOLINT(google-explicit-constructor)
        limb[0] = static_cast<std::uint64_t>(v);
        const std::uint64_t ext = v < 0 ? ~std::uint64_t{0} : std::uint64_t{0};
        for (std::size_t i = 1; i < N; ++i) limb[i] = ext;
    }
    static constexpr wide from_u64(std::uint64_t v) noexcept {
        wide r;
        r.limb[0] = v;
        return r;
    }

    [[nodiscard]] constexpr bool negative() const noexcept { return (limb[N - 1] >> 63) != 0; }
    [[nodiscard]] constexpr bool is_zero() const noexcept {
        for (const std::uint64_t x : limb) {
            if (x != 0) return false;
        }
        return true;
    }

    friend constexpr wide operator+(wide a, const wide& b) noexcept {
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const std::uint64_t s = a.limb[i] + carry;
            const std::uint64_t c1 = s < carry ? 1U : 0U;
            const std::uint64_t t = s + b.limb[i];
            const std::uint64_t c2 = t < s ? 1U : 0U;
            a.limb[i] = t;
            carry = c1 + c2;
        }
        return a;
    }
    friend constexpr wide operator-(wide a, const wide& b) noexcept {
        std::uint64_t borrow = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const std::uint64_t bi = b.limb[i];
            const std::uint64_t d = a.limb[i] - bi;
            const std::uint64_t b1 = a.limb[i] < bi ? 1U : 0U;
            const std::uint64_t d2 = d - borrow;
            const std::uint64_t b2 = d < borrow ? 1U : 0U;
            a.limb[i] = d2;
            borrow = b1 + b2;
        }
        return a;
    }
    friend constexpr wide operator-(const wide& a) noexcept { return wide{} - a; }
    // Product, exact whenever the true product fits in N limbs. Multiplies magnitudes and
    // skips their zero high limbs, which is most of them for the values the solvers use.
    friend wide operator*(const wide& a, const wide& b) noexcept {
        const bool negate = a.negative() != b.negative();
        const wide ua = a.negative() ? -a : a;
        const wide ub = b.negative() ? -b : b;
        std::size_t na = N;
        while (na > 0 && ua.limb[na - 1] == 0) --na;
        std::size_t nb = N;
        while (nb > 0 && ub.limb[nb - 1] == 0) --nb;
        wide r;
        for (std::size_t i = 0; i < na; ++i) {
            if (ua.limb[i] == 0) continue;
            std::uint64_t carry = 0;
            std::size_t j = 0;
            for (; j < nb && i + j < N; ++j) {
                std::uint64_t hi = 0;
                std::uint64_t lo = 0;
                mul64(ua.limb[i], ub.limb[j], hi, lo);
                std::uint64_t t = r.limb[i + j] + carry;
                std::uint64_t c = t < carry ? 1U : 0U;
                t += lo;
                c += t < lo ? 1U : 0U;
                r.limb[i + j] = t;
                carry = hi + c;  // cannot overflow: r + carry + a*b < 2^128
            }
            for (; carry != 0 && i + j < N; ++j) {
                const std::uint64_t t = r.limb[i + j] + carry;
                carry = t < carry ? 1U : 0U;
                r.limb[i + j] = t;
            }
        }
        return negate ? -r : r;
    }

    friend constexpr bool operator==(const wide& a, const wide& b) noexcept { return a.limb == b.limb; }
    friend constexpr bool operator<(const wide& a, const wide& b) noexcept {
        const bool an = a.negative();
        const bool bn = b.negative();
        if (an != bn) return an;
        for (std::size_t i = N; i-- > 0;) {
            if (a.limb[i] != b.limb[i]) return a.limb[i] < b.limb[i];
        }
        return false;
    }
    friend constexpr bool operator>(const wide& a, const wide& b) noexcept { return b < a; }
    friend constexpr bool operator<=(const wide& a, const wide& b) noexcept { return !(b < a); }
    friend constexpr bool operator>=(const wide& a, const wide& b) noexcept { return !(a < b); }

    /// Floor division of a non-negative value by d > 0 (binary long division; rarely called).
    [[nodiscard]] constexpr wide div_u64(std::uint64_t d) const noexcept {
        wide q;
        std::uint64_t r = 0;
        for (std::size_t bit = 64 * N; bit-- > 0;) {
            const std::uint64_t top = r >> 63;
            r = (r << 1) | ((limb[bit / 64] >> (bit % 64)) & 1U);
            if (top != 0 || r >= d) {
                r -= d;
                q.limb[bit / 64] |= std::uint64_t{1} << (bit % 64);
            }
        }
        return q;
    }

    /// Arithmetic shift right by one (floor division by two).
    [[nodiscard]] constexpr wide half() const noexcept {
        wide r;
        const std::uint64_t fill = negative() ? ~std::uint64_t{0} : std::uint64_t{0};
        for (std::size_t i = 0; i < N; ++i) {
            const std::uint64_t next = (i + 1 < N) ? limb[i + 1] : fill;
            r.limb[i] = (limb[i] >> 1) | (next << 63);
        }
        return r;
    }

    /// Exact decimal representation.
    [[nodiscard]] std::string to_string() const {
        if (negative()) return "-" + (-*this).to_string();
        if (is_zero()) return "0";
        std::array<std::uint32_t, 2 * N> digits32{};
        for (std::size_t i = 0; i < N; ++i) {
            digits32[2 * i] = static_cast<std::uint32_t>(limb[i] & 0xffffffffULL);
            digits32[2 * i + 1] = static_cast<std::uint32_t>(limb[i] >> 32);
        }
        std::vector<std::uint32_t> chunks;  // base 1e9, least significant first
        const auto all_zero = [&digits32] {
            return std::all_of(digits32.begin(), digits32.end(), [](std::uint32_t d) { return d == 0; });
        };
        while (!all_zero()) {
            std::uint64_t rem = 0;
            for (std::size_t i = 2 * N; i-- > 0;) {
                const std::uint64_t cur = (rem << 32) | digits32[i];
                digits32[i] = static_cast<std::uint32_t>(cur / 1000000000ULL);
                rem = cur % 1000000000ULL;
            }
            chunks.push_back(static_cast<std::uint32_t>(rem));
        }
        std::string s = std::to_string(chunks.back());
        for (std::size_t i = chunks.size() - 1; i-- > 0;) {
            const std::string c = std::to_string(chunks[i]);
            s.append(9 - c.size(), '0');
            s += c;
        }
        return s;
    }
};

/// Signed 128-bit integer for compilers without __int128 (MSVC). Two's complement in two limbs;
/// the truncated product needs no sign handling, so * costs one full and two low products.
struct portable_i128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    constexpr portable_i128() noexcept = default;
    // Implicit on purpose so small constants read naturally, like the native type.
    constexpr portable_i128(std::int64_t v) noexcept  // NOLINT(google-explicit-constructor)
        : lo(static_cast<std::uint64_t>(v)), hi(v < 0 ? ~std::uint64_t{0} : std::uint64_t{0}) {}
    static constexpr portable_i128 from_u64(std::uint64_t v) noexcept {
        portable_i128 r;
        r.lo = v;
        return r;
    }

    friend constexpr portable_i128 operator+(portable_i128 a, const portable_i128& b) noexcept {
        const std::uint64_t low = a.lo + b.lo;
        a.hi = a.hi + b.hi + (low < a.lo ? 1U : 0U);
        a.lo = low;
        return a;
    }
    friend constexpr portable_i128 operator-(portable_i128 a, const portable_i128& b) noexcept {
        const std::uint64_t borrow = a.lo < b.lo ? 1U : 0U;
        a.lo -= b.lo;
        a.hi = a.hi - b.hi - borrow;
        return a;
    }
    friend constexpr portable_i128 operator-(const portable_i128& a) noexcept { return portable_i128{} - a; }
    friend portable_i128 operator*(const portable_i128& a, const portable_i128& b) noexcept {
        portable_i128 r;
        mul64(a.lo, b.lo, r.hi, r.lo);
        r.hi += a.lo * b.hi + a.hi * b.lo;  // mod 2^128
        return r;
    }
    friend constexpr bool operator==(const portable_i128& a, const portable_i128& b) noexcept {
        return a.lo == b.lo && a.hi == b.hi;
    }
    friend constexpr bool operator<(const portable_i128& a, const portable_i128& b) noexcept {
        if (a.hi != b.hi) return static_cast<std::int64_t>(a.hi) < static_cast<std::int64_t>(b.hi);
        return a.lo < b.lo;
    }
    friend constexpr bool operator>(const portable_i128& a, const portable_i128& b) noexcept { return b < a; }
    friend constexpr bool operator<=(const portable_i128& a, const portable_i128& b) noexcept { return !(b < a); }
    friend constexpr bool operator>=(const portable_i128& a, const portable_i128& b) noexcept { return !(a < b); }

    [[nodiscard]] constexpr portable_i128 half() const noexcept {
        portable_i128 r;
        r.lo = (lo >> 1) | (hi << 63);
        r.hi = static_cast<std::uint64_t>(static_cast<std::int64_t>(hi) >> 1);  // arithmetic shift (C++20)
        return r;
    }
    /// Floor division of a non-negative value by d > 0 (binary long division; rarely called).
    [[nodiscard]] constexpr portable_i128 div_u64(std::uint64_t d) const noexcept {
        portable_i128 q;
        std::uint64_t r = 0;
        for (int bit = 127; bit >= 0; --bit) {
            const std::uint64_t word = bit >= 64 ? hi : lo;
            const std::uint64_t top = r >> 63;
            r = (r << 1) | ((word >> (bit % 64)) & 1U);
            if (top != 0 || r >= d) {
                r -= d;
                (bit >= 64 ? q.hi : q.lo) |= std::uint64_t{1} << (bit % 64);
            }
        }
        return q;
    }
};

/// Uniform construction/halving over native and portable integer types.
template <class I>
struct int_ops;

template <>
struct int_ops<portable_i128> {
    static portable_i128 from_u64(std::uint64_t v) noexcept { return portable_i128::from_u64(v); }
    static portable_i128 half(const portable_i128& v) noexcept { return v.half(); }
    static portable_i128 div_u64(const portable_i128& v, std::uint64_t d) noexcept { return v.div_u64(d); }
};

#if EVENCUT_NATIVE_INT128
template <>
struct int_ops<native_i128> {
    static native_i128 from_u64(std::uint64_t v) noexcept { return static_cast<native_i128>(v); }
    static native_i128 half(native_i128 v) noexcept { return v >> 1; }  // arithmetic shift (C++20)
    static native_i128 div_u64(native_i128 v, std::uint64_t d) noexcept { return v / static_cast<native_i128>(d); }
};
#endif

template <std::size_t N>
struct int_ops<wide<N>> {
    static wide<N> from_u64(std::uint64_t v) noexcept { return wide<N>::from_u64(v); }
    static wide<N> half(const wide<N>& v) noexcept { return v.half(); }
    static wide<N> div_u64(const wide<N>& v, std::uint64_t d) noexcept { return v.div_u64(d); }
};

/// Exact sum of squares of 64-bit loads (fits: sum <= (2^64-1)^2 < 2^128 < 2^191).
inline std::string sum_of_squares(std::span<const std::uint64_t> loads) {
    wide<3> acc;
    for (const std::uint64_t x : loads) {
        std::uint64_t hi = 0;
        std::uint64_t lo = 0;
        mul64(x, x, hi, lo);
        wide<3> sq;
        sq.limb[0] = lo;
        sq.limb[1] = hi;
        acc = acc + sq;
    }
    return acc.to_string();
}

inline std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw error("evencut: total weight exceeds 2^64-1; scale the weights down");
    }
    return a + b;
}

inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return b > std::numeric_limits<std::uint64_t>::max() - a ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

inline double imbalance_of(std::uint64_t max_load, std::uint64_t total, std::size_t parts) noexcept {
    if (total == 0 || parts == 0) return 1.0;
    return static_cast<double>(max_load) / (static_cast<double>(total) / static_cast<double>(parts));
}

/// Items grouped into blocks that may not be split (because of forbidden cuts).
struct block_view {
    std::vector<std::uint64_t> prefix;            // prefix sums of block weights, size m+1
    std::vector<std::uint64_t> sec_prefix;        // prefix sums of secondary weights (empty if unused)
    std::vector<std::size_t> item_end;            // exclusive end item index of each block
    std::uint64_t max_block = 0;
    std::uint64_t max_sec_block = 0;
    [[nodiscard]] std::size_t blocks() const noexcept { return item_end.size(); }
    [[nodiscard]] std::uint64_t total() const noexcept { return prefix.back(); }
};

inline block_view make_blocks(std::span<const std::uint64_t> weights, const options& opt) {
    const std::size_t n = weights.size();
    std::vector<unsigned char> joined(n, 0);  // joined[i] != 0: no cut between item i and i+1
    for (const std::size_t b : opt.forbidden_cuts) {
        if (b == 0 || b >= n) throw error("evencut: forbidden_cuts entries must be in [1, n-1] (numbered like ends)");
        joined[b - 1] = 1;
    }
    const bool has_sec = !opt.secondary.empty();
    block_view bv;
    bv.prefix.reserve(n + 1);
    bv.prefix.push_back(0);
    if (has_sec) {
        bv.sec_prefix.reserve(n + 1);
        bv.sec_prefix.push_back(0);
    }
    bv.item_end.reserve(n);
    std::uint64_t w = 0;
    std::uint64_t s = 0;
    std::uint64_t total = 0;
    std::uint64_t sec_total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        w = checked_add(w, weights[i]);
        total = checked_add(total, weights[i]);
        if (has_sec) {
            if (opt.secondary[i] > std::numeric_limits<std::uint64_t>::max() - sec_total) {
                throw error("evencut: total secondary value exceeds 2^64-1; scale it down");
            }
            s += opt.secondary[i];
            sec_total += opt.secondary[i];
        }
        if (i + 1 == n || joined[i] == 0) {
            bv.prefix.push_back(total);
            if (has_sec) bv.sec_prefix.push_back(sec_total);
            bv.item_end.push_back(i + 1);
            bv.max_block = std::max(bv.max_block, w);
            bv.max_sec_block = std::max(bv.max_sec_block, s);
            w = 0;
            s = 0;
        }
    }
    return bv;
}

/// Converts block boundaries [0, b1, ..., m] into item end indices.
inline std::vector<std::size_t> to_item_ends(const block_view& bv, const std::vector<std::size_t>& bounds) {
    std::vector<std::size_t> ends;
    ends.reserve(bounds.size() - 1);
    for (std::size_t t = 1; t < bounds.size(); ++t) ends.push_back(bv.item_end[bounds[t] - 1]);
    return ends;
}

inline partition_result summarize(std::span<const std::uint64_t> weights, std::vector<std::size_t> ends) {
    partition_result r;
    r.ends = std::move(ends);
    r.loads.reserve(r.ends.size());
    std::size_t begin = 0;
    for (const std::size_t e : r.ends) {
        std::uint64_t load = 0;
        for (std::size_t i = begin; i < e; ++i) load = checked_add(load, weights[i]);
        r.loads.push_back(load);
        r.total = checked_add(r.total, load);
        r.max_load = std::max(r.max_load, load);
        begin = e;
    }
    r.sum_of_squares = sum_of_squares(r.loads);
    r.imbalance = imbalance_of(r.max_load, r.total, r.ends.size());
    return r;
}

// ----------------------------------------------------------------------------------------
// min_max: binary search on the bound + greedy feasibility (optimal for hereditary limits)
// ----------------------------------------------------------------------------------------

/// Largest block index e > start such that the part [start, e) stays within both limits.
/// Returns start when even a single block does not fit.
inline std::size_t furthest_end(const block_view& bv, std::size_t start, std::uint64_t bound, std::uint64_t sec_limit) {
    const auto& P = bv.prefix;
    const std::uint64_t pt = saturating_add(P[start], bound);
    std::size_t e = static_cast<std::size_t>(std::upper_bound(P.begin() + static_cast<std::ptrdiff_t>(start) + 1, P.end(), pt) - P.begin()) - 1;
    if (!bv.sec_prefix.empty()) {
        const auto& Q = bv.sec_prefix;
        const std::uint64_t qt = saturating_add(Q[start], sec_limit);
        const std::size_t e2 = static_cast<std::size_t>(std::upper_bound(Q.begin() + static_cast<std::ptrdiff_t>(start) + 1, Q.end(), qt) - Q.begin()) - 1;
        e = std::min(e, e2);
    }
    return e;
}

/// Parts the greedy needs for the given bound; stops counting once it exceeds stop_after.
inline std::size_t greedy_parts(const block_view& bv, std::uint64_t bound, std::uint64_t sec_limit, std::size_t stop_after) {
    const std::size_t m = bv.blocks();
    std::size_t start = 0;
    std::size_t parts = 0;
    while (start < m) {
        const std::size_t e = furthest_end(bv, start, bound, sec_limit);
        if (e == start) return std::numeric_limits<std::size_t>::max();
        ++parts;
        if (parts > stop_after) return parts;
        start = e;
    }
    return parts;
}

/// Most balanced split point c of the block range [s, e) (requires e - s >= 2).
inline std::size_t balanced_split(const std::vector<std::uint64_t>& P, std::size_t s, std::size_t e) {
    const std::uint64_t half_point = P[s] + (P[e] - P[s]) / 2;
    std::size_t c = static_cast<std::size_t>(std::lower_bound(P.begin() + static_cast<std::ptrdiff_t>(s) + 1,
                                                              P.begin() + static_cast<std::ptrdiff_t>(e), half_point) - P.begin());
    c = std::clamp(c, s + 1, e - 1);
    const auto worst = [&](std::size_t x) { return std::max(P[x] - P[s], P[e] - P[x]); };
    if (c > s + 1 && worst(c - 1) <= worst(c)) c = c - 1;
    return c;
}

/// Splits the heaviest splittable parts until there are exactly `target` parts.
inline void expand_parts(const block_view& bv, std::vector<std::size_t>& bounds, std::size_t target) {
    const auto& P = bv.prefix;
    if (bounds.size() - 1 >= target) return;
    using item = std::tuple<std::uint64_t, std::size_t, std::size_t>;  // load, start, end
    const auto cmp = [](const item& a, const item& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
        return std::get<1>(a) > std::get<1>(b);  // prefer the earlier part on ties
    };
    std::priority_queue<item, std::vector<item>, decltype(cmp)> heap(cmp);
    std::vector<std::size_t> out;
    for (std::size_t t = 1; t < bounds.size(); ++t) heap.emplace(P[bounds[t]] - P[bounds[t - 1]], bounds[t - 1], bounds[t]);
    std::size_t parts = bounds.size() - 1;
    std::vector<item> done;
    while (parts < target && !heap.empty()) {
        const item top = heap.top();
        heap.pop();
        const std::size_t s = std::get<1>(top);
        const std::size_t e = std::get<2>(top);
        if (e - s < 2) {
            done.push_back(top);
            continue;
        }
        const std::size_t c = balanced_split(P, s, e);
        heap.emplace(P[c] - P[s], s, c);
        heap.emplace(P[e] - P[c], c, e);
        ++parts;
    }
    std::vector<std::size_t> starts;
    while (!heap.empty()) {
        starts.push_back(std::get<1>(heap.top()));
        heap.pop();
    }
    for (const item& d : done) starts.push_back(std::get<1>(d));
    std::sort(starts.begin(), starts.end());
    bounds = std::move(starts);
    bounds.push_back(bv.blocks());
}

/// Moves each inner cut to the most even position between its neighbours while keeping every
/// part within `bound` and the secondary limit. Strictly lowers the sum of squares on every
/// move, so it terminates; the largest part never grows.
inline void polish_parts(const block_view& bv, std::vector<std::size_t>& bounds, std::uint64_t bound, std::uint64_t sec_limit) {
    const auto& P = bv.prefix;
    const auto& Q = bv.sec_prefix;
    const bool has_sec = !Q.empty();
    const std::size_t k = bounds.size() - 1;
    if (k < 2) return;
    for (int pass = 0; pass < 64; ++pass) {
        bool moved = false;
        for (std::size_t s = 1; s < k; ++s) {
            const std::size_t lo_b = bounds[s - 1];
            const std::size_t hi_b = bounds[s + 1];
            // feasible cut positions c in [cmin, cmax], lo_b < c < hi_b
            std::size_t cmax = furthest_end(bv, lo_b, bound, sec_limit);  // left part [lo_b, c) fits
            cmax = std::min(cmax, hi_b - 1);
            // right part [c, hi_b) fits: smallest c with P[hi_b] - P[c] <= bound (and secondary)
            const std::uint64_t need = P[hi_b] >= bound ? P[hi_b] - bound : 0;
            std::size_t cmin = static_cast<std::size_t>(std::lower_bound(P.begin(), P.end(), need) - P.begin());
            if (has_sec) {
                const std::uint64_t qneed = Q[hi_b] >= sec_limit ? Q[hi_b] - sec_limit : 0;
                cmin = std::max(cmin, static_cast<std::size_t>(std::lower_bound(Q.begin(), Q.end(), qneed) - Q.begin()));
            }
            cmin = std::max(cmin, lo_b + 1);
            if (cmin > cmax) continue;  // cannot happen for a feasible partition, but stay safe
            const std::size_t cur = bounds[s];
            const auto gap = [&](std::size_t c) {
                const std::uint64_t left = P[c] - P[lo_b];
                const std::uint64_t right = P[hi_b] - P[c];
                return left > right ? left - right : right - left;
            };
            const std::uint64_t half_point = P[lo_b] + (P[hi_b] - P[lo_b]) / 2;
            std::size_t best = static_cast<std::size_t>(std::lower_bound(P.begin() + static_cast<std::ptrdiff_t>(cmin),
                                                                         P.begin() + static_cast<std::ptrdiff_t>(cmax) + 1, half_point) - P.begin());
            best = std::clamp(best, cmin, cmax);
            if (best > cmin && gap(best - 1) <= gap(best)) best = best - 1;
            if (gap(best) < gap(cur)) {
                bounds[s] = best;
                moved = true;
            }
        }
        if (!moved) break;
    }
}

inline std::vector<std::size_t> min_max_blocks(const block_view& bv, std::size_t target, const options& opt) {
    const std::size_t m = bv.blocks();
    const std::uint64_t sec_limit = bv.sec_prefix.empty() ? std::numeric_limits<std::uint64_t>::max() : *opt.secondary_limit;
    if (!bv.sec_prefix.empty() && bv.max_sec_block > sec_limit) {
        throw error("evencut: an item (or a group of items joined by forbidden_cuts) exceeds secondary_limit on its own");
    }
    const std::uint64_t total = bv.total();
    std::uint64_t lo = std::max(bv.max_block, total / target + (total % target != 0 ? 1U : 0U));
    std::uint64_t hi = total;
    if (lo > hi) lo = hi;
    if (greedy_parts(bv, hi, sec_limit, target) > target) {
        throw error("evencut: no partition into the requested number of parts satisfies secondary_limit");
    }
    while (lo < hi) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        if (greedy_parts(bv, mid, sec_limit, target) <= target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    const std::uint64_t best = lo;
    std::vector<std::size_t> bounds{0};
    for (std::size_t start = 0; start < m;) {
        start = furthest_end(bv, start, best, sec_limit);
        bounds.push_back(start);
    }
    expand_parts(bv, bounds, target);
    if (opt.polish) polish_parts(bv, bounds, best, sec_limit);
    return bounds;
}

// ----------------------------------------------------------------------------------------
// sum_squares: Lagrangian relaxation with odd penalties + convex hull trick + Monge splice
// ----------------------------------------------------------------------------------------
//
// For a penalty mu we solve  min over partitions of  2 * sum(load^2) + mu * parts.
// With odd mu the optimal number of parts is unique (2*cost is even, mu odd, cost convex
// in the number of parts), so the count is a well-defined non-increasing function of mu
// and a binary search over odd mu is exact. If the requested k sits on a straight piece of
// the cost curve, no odd mu yields it; then the optimal partitions for the two neighbouring
// odd penalties are spliced (quadrangle inequality) into an optimal partition with exactly k
// parts.

template <class I>
class lagrange_kernel {
public:
    explicit lagrange_kernel(std::span<const std::uint64_t> prefix)
        : prefix_(prefix), m_(prefix.size() - 1), slope_(m_ + 1), icpt_(m_ + 1), line_(m_ + 1), count_(m_ + 1), from_(m_ + 1) {}

    /// Returns the optimal number of parts for penalty mu and remembers the partition.
    std::size_t run(const I& mu) {
        using ops = int_ops<I>;
        std::size_t head = 0;
        std::size_t tail = 1;
        slope_[0] = I(0);
        icpt_[0] = I(0);
        line_[0] = 0;
        count_[0] = 0;
        for (std::size_t i = 1; i <= m_; ++i) {
            const I x = ops::from_u64(prefix_[i]);
            // next line is at least as good at x:  m1*x + c1 <= m0*x + c0
            while (tail - head >= 2 && (slope_[head + 1] - slope_[head]) * x <= icpt_[head] - icpt_[head + 1]) ++head;
            const std::size_t j = line_[head];
            const I x2 = x * x;
            const I two_x2 = x2 + x2;
            const I dp = slope_[head] * x + icpt_[head] + two_x2 + mu;
            count_[i] = count_[j] + 1;
            from_[i] = j;
            if (i == m_) break;
            const I two_x = x + x;
            const I slope = I(0) - (two_x + two_x);
            const I icpt = dp + two_x2;
            if (tail > head && slope_[tail - 1] == slope) {
                if (icpt_[tail - 1] <= icpt) continue;  // existing line is never worse
                --tail;
            }
            while (tail - head >= 2 &&
                   (icpt - icpt_[tail - 2]) * (slope_[tail - 2] - slope_[tail - 1]) <=
                       (icpt_[tail - 1] - icpt_[tail - 2]) * (slope_[tail - 2] - slope)) {
                --tail;
            }
            slope_[tail] = slope;
            icpt_[tail] = icpt;
            line_[tail] = i;
            ++tail;
        }
        return count_[m_];
    }

    /// Block boundaries [0, ..., m] of the partition found by the last run().
    [[nodiscard]] std::vector<std::size_t> boundaries() const {
        std::vector<std::size_t> b;
        for (std::size_t i = m_;; i = from_[i]) {
            b.push_back(i);
            if (i == 0) break;
        }
        std::reverse(b.begin(), b.end());
        return b;
    }

private:
    std::span<const std::uint64_t> prefix_;
    std::size_t m_;
    std::vector<I> slope_;
    std::vector<I> icpt_;
    std::vector<std::size_t> line_;
    std::vector<std::size_t> count_;
    std::vector<std::size_t> from_;
};

/// Combines an optimal partition `a` with more than k parts and an optimal partition `b` with
/// fewer than k parts (both optimal for the same penalty) into one with exactly k parts.
inline std::vector<std::size_t> splice(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b, std::size_t k) {
    const std::size_t q = b.size() - 1;
    for (std::size_t t = 1; t <= q; ++t) {
        const std::size_t L = static_cast<std::size_t>(std::lower_bound(a.begin(), a.end(), b[t - 1]) - a.begin());
        const std::size_t R = static_cast<std::size_t>(std::upper_bound(a.begin(), a.end(), b[t]) - a.begin()) - 1;
        if (R <= L) continue;
        // a-parts (a[s], a[s+1]] with L <= s < R lie inside b-part (b[t-1], b[t]];
        // a[0..s] + (a[s], b[t]] + b[t..q] has s + 1 + q - t parts.
        const std::size_t low = L + 1 + q - t;
        const std::size_t high = R + q - t;
        if (k >= low && k <= high) {
            const std::size_t s = (k + t) - (1 + q);
            std::vector<std::size_t> out(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(s) + 1);
            out.insert(out.end(), b.begin() + static_cast<std::ptrdiff_t>(t), b.end());
            return out;
        }
    }
    throw std::logic_error("evencut: internal error while assembling an exact-k partition");
}

/// Exact sum-of-squares partition of blocks into k parts (2 <= k < m) using integer type I.
///
/// Searches the odd penalty mu with a safeguarded chord (Newton-like) method: the two known
/// optima on either side of k define a chord of the convex cost curve, and its slope s points
/// at the next penalty to try. A chord step that finds no new part count is followed by a
/// bisection step, so the range always shrinks geometrically. When s is an exact integer, k
/// may lie on a straight piece of the curve; the odd penalties 2s-1 and 2s+1 then pin it down.
template <class I>
std::vector<std::size_t> sum_squares_blocks(std::span<const std::uint64_t> prefix, std::size_t k) {
    using ops = int_ops<I>;
    const std::size_t m = prefix.size() - 1;
    lagrange_kernel<I> kernel(prefix);
    const auto cost_of = [&prefix](const std::vector<std::size_t>& b) {
        I f(0);
        for (std::size_t t = 1; t < b.size(); ++t) {
            const I d = ops::from_u64(prefix[b[t]] - prefix[b[t - 1]]);
            f = f + d * d;
        }
        return f;
    };
    struct side {
        I mu;
        std::size_t count;
        I cost;
        std::vector<std::size_t> bounds;
    };
    // Known optima at the ends of the range: mu = -1 keeps every block separate (m > k parts),
    // mu = 2*S^2 + 1 keeps everything together (1 < k part).
    std::vector<std::size_t> all(m + 1);
    std::iota(all.begin(), all.end(), std::size_t{0});
    const I S = ops::from_u64(prefix[m]);
    const I S2 = S * S;
    side lo{I(-1), m, cost_of(all), std::move(all)};
    side hi{S2 + S2 + I(1), 1, S2, std::vector<std::size_t>{0, m}};
    bool bisect = false;
    while (I(2) < hi.mu - lo.mu) {
        I mu(0);
        bool chord = false;
        bool probe = false;
        if (!bisect) {
            const std::uint64_t dc = static_cast<std::uint64_t>(lo.count - hi.count);
            const I diff = hi.cost - lo.cost;  // >= 0: fewer parts never cost less
            const I s = ops::div_u64(diff, dc);
            const bool exact = s * ops::from_u64(dc) == diff;
            const I below = s + s - I(1);
            const I above = s + s + I(1);
            if (exact && lo.mu < below && below < hi.mu) {
                mu = below;
                chord = probe = true;
            } else if (lo.mu < above && above < hi.mu) {
                mu = above;
                chord = true;
                probe = exact;
            }
        }
        if (!chord) {
            const I t = ops::half(ops::half(lo.mu - I(1)) + ops::half(hi.mu - I(1)));
            mu = t + t + I(1);
        }
        const std::size_t c = kernel.run(mu);
        if (c == k) return kernel.boundaries();
        std::vector<std::size_t> b = kernel.boundaries();
        const I f = cost_of(b);
        const bool new_count = c != lo.count && c != hi.count;
        if (c > k) {
            lo = side{mu, c, f, std::move(b)};
        } else {
            hi = side{mu, c, f, std::move(b)};
        }
        bisect = chord && !probe && !new_count;
    }
    // Adjacent odd penalties with counts on both sides of k: both partitions are optimal for
    // the even penalty between them, and splicing yields an optimal partition with k parts.
    return splice(lo.bounds, hi.bounds, k);
}

/// Largest total weight for which the 128-bit path cannot overflow (24 * S^3 < 2^127).
inline constexpr std::uint64_t fast_path_limit = std::uint64_t{1} << 40;

inline std::vector<std::size_t> sum_squares_dispatch(std::span<const std::uint64_t> prefix, std::size_t target) {
    const std::size_t m = prefix.size() - 1;
    if (target >= m) {
        std::vector<std::size_t> all(m + 1);
        std::iota(all.begin(), all.end(), std::size_t{0});
        return all;
    }
    if (target == 1) return {0, m};
    if (prefix[m] <= fast_path_limit) {
#if EVENCUT_NATIVE_INT128
        return sum_squares_blocks<native_i128>(prefix, target);
#else
        return sum_squares_blocks<portable_i128>(prefix, target);
#endif
    }
    return sum_squares_blocks<wide<4>>(prefix, target);
}

}  // namespace detail

/**
 * Splits `weights` into min(k, b) contiguous parts, where b is the number of blocks left
 * after applying `forbidden_cuts` (b == n without them).
 *
 * @throws evencut::error on invalid input or infeasible secondary limits.
 */
[[nodiscard]] inline partition_result partition(std::span<const std::uint64_t> weights, std::size_t k, const options& opt = {}) {
    const std::size_t n = weights.size();
    if (n == 0) throw error("evencut: weights must not be empty");
    if (k == 0) throw error("evencut: k must be at least 1");
    if (!opt.secondary.empty() && opt.secondary.size() != n) {
        throw error("evencut: secondary must be empty or have exactly one value per weight");
    }
    if (opt.goal == objective::sum_squares && (!opt.secondary.empty() || opt.secondary_limit)) {
        throw error("evencut: secondary limits are only supported with objective::min_max");
    }
    if (!opt.secondary.empty() && !opt.secondary_limit) {
        throw error("evencut: secondary_limit is required when secondary is given");
    }
    if (opt.secondary.empty() && opt.secondary_limit) {
        throw error("evencut: secondary_limit is set but secondary is empty");
    }
    const detail::block_view bv = detail::make_blocks(weights, opt);
    const std::size_t target = std::min(k, bv.blocks());
    std::vector<std::size_t> bounds;
    if (opt.goal == objective::min_max) {
        bounds = detail::min_max_blocks(bv, target, opt);
    } else {
        bounds = detail::sum_squares_dispatch(bv.prefix, target);
    }
    return detail::summarize(weights, detail::to_item_ends(bv, bounds));
}

/// Convenience overload for initializer lists and vectors.
[[nodiscard]] inline partition_result partition(const std::vector<std::uint64_t>& weights, std::size_t k, const options& opt = {}) {
    return partition(std::span<const std::uint64_t>(weights), k, opt);
}

/**
 * Measures an existing split: `ends` are exclusive end indices, strictly increasing, last == n.
 * Useful for comparing a hand-made or framework-default split with the optimum.
 */
[[nodiscard]] inline partition_result evaluate(std::span<const std::uint64_t> weights, std::span<const std::size_t> ends) {
    const std::size_t n = weights.size();
    if (n == 0) throw error("evencut: weights must not be empty");
    if (ends.empty() || ends.back() != n) throw error("evencut: ends must be non-empty and its last entry must equal n");
    std::size_t prev = 0;
    for (const std::size_t e : ends) {
        if (e <= prev) throw error("evencut: ends must be strictly increasing and start above 0");
        prev = e;
    }
    return detail::summarize(weights, std::vector<std::size_t>(ends.begin(), ends.end()));
}

/**
 * Splits a tree into min(k, n) connected parts, minimising the largest part weight.
 *
 * @param parent  parent[v] is the parent of node v; exactly one node has parent no_parent.
 * @param weights weight of each node.
 * @throws evencut::error on malformed trees or invalid input.
 */
[[nodiscard]] inline tree_result partition_tree(std::span<const std::size_t> parent, std::span<const std::uint64_t> weights, std::size_t k) {
    const std::size_t n = parent.size();
    if (n == 0) throw error("evencut: the tree must have at least one node");
    if (weights.size() != n) throw error("evencut: weights must have exactly one value per node");
    if (k == 0) throw error("evencut: k must be at least 1");

    std::size_t root = no_parent;
    std::vector<std::size_t> child_start(n + 1, 0);
    for (std::size_t v = 0; v < n; ++v) {
        const std::size_t p = parent[v];
        if (p == no_parent) {
            if (root != no_parent) throw error("evencut: the tree has more than one root");
            root = v;
        } else {
            if (p >= n || p == v) throw error("evencut: parent indices must be in [0, n) and differ from the node");
            ++child_start[p + 1];
        }
    }
    if (root == no_parent) throw error("evencut: the tree has no root (exactly one parent must be no_parent)");
    for (std::size_t v = 0; v < n; ++v) child_start[v + 1] += child_start[v];
    std::vector<std::size_t> children(n > 0 ? n - 1 : 0);
    {
        std::vector<std::size_t> fill(child_start.begin(), child_start.end() - 1);
        for (std::size_t v = 0; v < n; ++v) {
            if (parent[v] != no_parent) children[fill[parent[v]]++] = v;
        }
    }
    std::vector<std::size_t> order;
    order.reserve(n);
    order.push_back(root);
    for (std::size_t idx = 0; idx < order.size(); ++idx) {
        const std::size_t v = order[idx];
        for (std::size_t c = child_start[v]; c < child_start[v + 1]; ++c) order.push_back(children[c]);
    }
    if (order.size() != n) throw error("evencut: parent array does not form a single tree (cycle or detached nodes)");

    std::uint64_t total = 0;
    std::uint64_t max_w = 0;
    for (const std::uint64_t w : weights) {
        total = detail::checked_add(total, w);
        max_w = std::max(max_w, w);
    }
    const std::size_t target = std::min(k, n);

    std::vector<std::uint64_t> resid(n, 0);
    std::vector<std::pair<std::uint64_t, std::size_t>> heap;
    std::vector<unsigned char> cut(n, 0);
    // Kundu-Misra: fewest cuts so that every component weighs at most `bound`.
    const auto cuts_needed = [&](std::uint64_t bound, bool record) -> std::size_t {
        std::size_t cuts = 0;
        if (record) std::fill(cut.begin(), cut.end(), static_cast<unsigned char>(0));
        for (std::size_t idx = n; idx-- > 0;) {
            const std::size_t v = order[idx];
            std::uint64_t sum = weights[v];
            const std::size_t cb = child_start[v];
            const std::size_t ce = child_start[v + 1];
            for (std::size_t c = cb; c < ce; ++c) sum += resid[children[c]];  // <= total, no overflow
            if (sum > bound) {
                heap.clear();
                for (std::size_t c = cb; c < ce; ++c) heap.emplace_back(resid[children[c]], children[c]);
                // largest residual first; on ties the smaller node index first (deterministic)
                const auto less = [](const std::pair<std::uint64_t, std::size_t>& x, const std::pair<std::uint64_t, std::size_t>& y) {
                    return x.first != y.first ? x.first < y.first : x.second > y.second;
                };
                std::make_heap(heap.begin(), heap.end(), less);
                while (sum > bound) {
                    std::pop_heap(heap.begin(), heap.end(), less);
                    const auto [r, c] = heap.back();
                    heap.pop_back();
                    sum -= r;
                    ++cuts;
                    if (record) cut[c] = 1;
                }
            }
            resid[v] = sum;
        }
        return cuts;
    };

    std::uint64_t lo = std::max(max_w, total / target + (total % target != 0 ? 1U : 0U));
    std::uint64_t hi = total;
    if (lo > hi) lo = hi;
    while (lo < hi) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        if (cuts_needed(mid, false) + 1 <= target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    std::size_t parts = cuts_needed(lo, true) + 1;
    // Reach exactly `target` parts: extra cuts never raise the maximum. Prefer the heaviest
    // remaining subtrees so the new parts are as large (and the others as light) as possible.
    if (parts < target) {
        std::vector<std::size_t> cand;
        cand.reserve(n);
        for (std::size_t v = 0; v < n; ++v) {
            if (v != root && cut[v] == 0) cand.push_back(v);
        }
        const std::size_t need = target - parts;
        std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(need), cand.end(),
                          [&](std::size_t x, std::size_t y) { return resid[x] != resid[y] ? resid[x] > resid[y] : x < y; });
        for (std::size_t i = 0; i < need; ++i) cut[cand[i]] = 1;
        parts = target;
    }

    tree_result r;
    r.part_of.assign(n, 0);
    r.loads.assign(parts, 0);
    std::size_t next_id = 1;
    for (std::size_t idx = 0; idx < n; ++idx) {
        const std::size_t v = order[idx];
        if (v == root) {
            r.part_of[v] = 0;
        } else if (cut[v] != 0) {
            r.part_of[v] = next_id++;
        } else {
            r.part_of[v] = r.part_of[parent[v]];
        }
        r.loads[r.part_of[v]] += weights[v];
    }
    for (std::size_t v = 0; v < n; ++v) {
        if (cut[v] != 0) r.cut_nodes.push_back(v);
    }
    r.total = total;
    r.max_load = *std::max_element(r.loads.begin(), r.loads.end());
    r.sum_of_squares = detail::sum_of_squares(r.loads);
    r.imbalance = detail::imbalance_of(r.max_load, r.total, parts);
    return r;
}

/**
 * Converts non-negative real values (seconds, megabytes, FLOPs...) into integer weights:
 * round(value * scale). Pick `scale` so the integers keep the precision you need, e.g.
 * 1e6 turns seconds into microseconds.
 *
 * @throws evencut::error for NaN, infinity, negative values, a non-positive scale or overflow.
 */
[[nodiscard]] inline std::vector<std::uint64_t> quantize(std::span<const double> values, double scale) {
    if (!(scale > 0.0) || !std::isfinite(scale)) throw error("evencut: scale must be a positive finite number");
    std::vector<std::uint64_t> out;
    out.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double v = values[i];
        if (!std::isfinite(v) || v < 0.0) {
            throw error("evencut: value at index " + std::to_string(i) + " must be finite and non-negative");
        }
        const double q = std::round(v * scale);
        if (!(q < 18446744073709551616.0)) {
            throw error("evencut: value at index " + std::to_string(i) + " does not fit in 64 bits after scaling");
        }
        out.push_back(static_cast<std::uint64_t>(q));
    }
    return out;
}

}  // namespace evencut
