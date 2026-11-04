// C ABI implementation. No exception ever crosses this boundary.

#ifndef EVENCUT_BUILDING_DLL
#define EVENCUT_BUILDING_DLL
#endif
#include "evencut/evencut.h"

#include "evencut/evencut.hpp"

#include <cstddef>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

namespace {

// Smallest evencut_options accepted: the version-1 layout. Later versions may append fields;
// callers built against an older header pass a smaller struct_size and keep working.
constexpr std::size_t options_v1_size = offsetof(evencut_options, polish) + sizeof(std::int32_t);

thread_local std::string last_error;

evencut_status fail(evencut_status status, const std::string& message) {
    last_error = message;
    return status;
}

template <class Fn>
evencut_status guarded(Fn&& fn) noexcept {
    try {
        last_error.clear();
        return fn();
    } catch (const evencut::error& e) {
        return fail(EVENCUT_INVALID_ARGUMENT, e.what());
    } catch (const std::bad_alloc&) {
        return fail(EVENCUT_OUT_OF_MEMORY, "evencut: out of memory");
    } catch (const std::exception& e) {
        return fail(EVENCUT_INTERNAL_ERROR, e.what());
    } catch (...) {
        return fail(EVENCUT_INTERNAL_ERROR, "evencut: unknown internal error");
    }
}

void fill_summary(evencut_summary* s, std::size_t parts, std::uint64_t max_load, std::uint64_t total, double imbalance,
                  const std::string& sum_of_squares) {
    if (s == nullptr) return;
    s->parts = parts;
    s->max_load = max_load;
    s->total = total;
    s->imbalance = imbalance;
    const std::size_t len = std::min(sum_of_squares.size(), sizeof(s->sum_of_squares) - 1);  // <= 39 digits in practice
    std::memcpy(s->sum_of_squares, sum_of_squares.data(), len);
    s->sum_of_squares[len] = '\0';
}

std::span<const std::uint64_t> weight_span(const std::uint64_t* weights, std::size_t n) {
    if (n > 0 && weights == nullptr) throw evencut::error("evencut: weights is NULL");
    return {weights, n};
}

}  // namespace

extern "C" {

EVENCUT_API void evencut_options_init(evencut_options* options) {
    if (options == nullptr) return;
    std::memset(options, 0, sizeof(*options));
    options->struct_size = static_cast<std::uint32_t>(sizeof(evencut_options));
    options->objective = EVENCUT_MIN_MAX;
    options->polish = 1;  // secondary_limit stays 0: a forgotten limit fails loudly
}

EVENCUT_API evencut_status evencut_partition(const std::uint64_t* weights, std::size_t n, std::size_t k,
                                             const evencut_options* options, std::size_t* ends_out,
                                             std::size_t ends_capacity, evencut_summary* summary) {
    return guarded([&]() -> evencut_status {
        evencut::options opt;
        if (options != nullptr) {
            if (options->struct_size < options_v1_size) {
                return fail(EVENCUT_INVALID_ARGUMENT, "evencut: options not initialised; call evencut_options_init() first");
            }
            if (options->objective == EVENCUT_MIN_MAX) {
                opt.goal = evencut::objective::min_max;
            } else if (options->objective == EVENCUT_SUM_SQUARES) {
                opt.goal = evencut::objective::sum_squares;
            } else {
                return fail(EVENCUT_INVALID_ARGUMENT, "evencut: unknown objective");
            }
            if (options->forbidden_count > 0) {
                if (options->forbidden_cuts == nullptr) return fail(EVENCUT_INVALID_ARGUMENT, "evencut: forbidden_cuts is NULL");
                opt.forbidden_cuts = {options->forbidden_cuts, options->forbidden_count};
            }
            if (options->secondary != nullptr) {
                opt.secondary = {options->secondary, n};
                opt.secondary_limit = options->secondary_limit;  // any value, including UINT64_MAX
            } else if (options->secondary_limit != 0) {
                return fail(EVENCUT_INVALID_ARGUMENT, "evencut: secondary_limit is set but secondary is NULL");
            }
            opt.polish = options->polish != 0;
        }
        if (ends_out == nullptr) return fail(EVENCUT_INVALID_ARGUMENT, "evencut: ends_out is NULL");
        const evencut::partition_result r = evencut::partition(weight_span(weights, n), k, opt);
        if (ends_capacity < r.parts()) {
            return fail(EVENCUT_BUFFER_TOO_SMALL,
                        "evencut: ends_out needs room for " + std::to_string(r.parts()) + " entries");
        }
        std::copy(r.ends.begin(), r.ends.end(), ends_out);
        fill_summary(summary, r.parts(), r.max_load, r.total, r.imbalance, r.sum_of_squares);
        return EVENCUT_OK;
    });
}

EVENCUT_API evencut_status evencut_evaluate(const std::uint64_t* weights, std::size_t n, const std::size_t* ends,
                                            std::size_t parts, evencut_summary* summary) {
    return guarded([&]() -> evencut_status {
        if (parts > 0 && ends == nullptr) return fail(EVENCUT_INVALID_ARGUMENT, "evencut: ends is NULL");
        const evencut::partition_result r = evencut::evaluate(weight_span(weights, n), {ends, parts});
        fill_summary(summary, r.parts(), r.max_load, r.total, r.imbalance, r.sum_of_squares);
        return EVENCUT_OK;
    });
}

EVENCUT_API evencut_status evencut_partition_tree(const std::size_t* parent, const std::uint64_t* weights, std::size_t n,
                                                  std::size_t k, std::size_t* part_out, std::size_t part_capacity,
                                                  evencut_summary* summary) {
    return guarded([&]() -> evencut_status {
        if (n > 0 && parent == nullptr) return fail(EVENCUT_INVALID_ARGUMENT, "evencut: parent is NULL");
        if (part_out == nullptr) return fail(EVENCUT_INVALID_ARGUMENT, "evencut: part_out is NULL");
        if (part_capacity < n) {
            return fail(EVENCUT_BUFFER_TOO_SMALL, "evencut: part_out needs room for " + std::to_string(n) + " entries");
        }
        const evencut::tree_result r = evencut::partition_tree({parent, n}, weight_span(weights, n), k);
        std::copy(r.part_of.begin(), r.part_of.end(), part_out);
        fill_summary(summary, r.parts(), r.max_load, r.total, r.imbalance, r.sum_of_squares);
        return EVENCUT_OK;
    });
}

EVENCUT_API const char* evencut_last_error(void) { return last_error.c_str(); }

EVENCUT_API const char* evencut_version(void) { return EVENCUT_VERSION_STRING; }

EVENCUT_API int evencut_abi_version(void) { return EVENCUT_ABI_VERSION; }

}  // extern "C"
