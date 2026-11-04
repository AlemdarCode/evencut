/*
 * evencut.h: C interface to the Evencut partitioning engine.
 *
 * A stable, exception-free ABI for C and for foreign-function interfaces (Python ctypes,
 * C#, Go cgo, Rust, Java, Node). Every function returns an evencut_status; on failure
 * evencut_last_error() describes the problem (per thread).
 *
 * Indices are 0-based. "ends" are exclusive end indices of the parts: for n items split into
 * three parts, ends = {e0, e1, n} means items [0, e0), [e0, e1), [e1, n).
 */
#ifndef EVENCUT_H
#define EVENCUT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(EVENCUT_BUILDING_DLL)
#    define EVENCUT_API __declspec(dllexport)
#  elif defined(EVENCUT_USING_DLL)
#    define EVENCUT_API __declspec(dllimport)
#  else
#    define EVENCUT_API
#  endif
#elif defined(__GNUC__)
#  define EVENCUT_API __attribute__((visibility("default")))
#else
#  define EVENCUT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Incremented when the binary interface changes incompatibly. */
#define EVENCUT_ABI_VERSION 1

/** Parent value that marks the root node in evencut_partition_tree(). */
#define EVENCUT_NO_PARENT ((size_t)-1)

typedef enum evencut_status {
    EVENCUT_OK = 0,
    EVENCUT_INVALID_ARGUMENT = 1, /* bad input or infeasible limits; see evencut_last_error() */
    EVENCUT_BUFFER_TOO_SMALL = 2, /* an output buffer is smaller than the result */
    EVENCUT_OUT_OF_MEMORY = 3,
    EVENCUT_INTERNAL_ERROR = 4
} evencut_status;

typedef enum evencut_objective {
    EVENCUT_MIN_MAX = 0,     /* minimise the largest part */
    EVENCUT_SUM_SQUARES = 1  /* minimise the sum of squared part loads */
} evencut_objective;

typedef struct evencut_options {
    uint32_t struct_size;          /* set by evencut_options_init(); the library accepts any size from the
                                      version-1 layout up, so programs built with older headers keep working */
    int32_t objective;             /* an evencut_objective value */
    const size_t* forbidden_cuts;  /* NULL, or forbidden_count entries numbered like ends: b forbids the
                                      cut between item b-1 and item b (valid: 1 .. n-1) */
    size_t forbidden_count;
    const uint64_t* secondary;     /* NULL, or one value per item (min_max only), e.g. memory */
    uint64_t secondary_limit;      /* per-part limit for secondary, used only when secondary is set. The
                                      default 0 makes a forgotten limit fail loudly instead of being
                                      skipped; setting it without secondary is an error */
    int32_t polish;                /* min_max: 1 (default) evens out the non-maximal parts */
} evencut_options;

typedef struct evencut_summary {
    size_t parts;            /* number of parts written */
    uint64_t max_load;       /* largest part weight */
    uint64_t total;          /* sum of all weights */
    double imbalance;        /* max_load / (total / parts); 1.0 is perfectly even */
    char sum_of_squares[48]; /* exact decimal value of sum(load^2), NUL-terminated */
} evencut_summary;

/** Fills *options with defaults (min_max, no constraints, polish on). Always call it first. */
EVENCUT_API void evencut_options_init(evencut_options* options);

/**
 * Splits weights[0..n) into min(k, blocks) contiguous parts (blocks == n unless cuts are
 * forbidden). ends_out must have room for ends_capacity entries; min(k, n) is always enough.
 * options and summary may be NULL.
 */
EVENCUT_API evencut_status evencut_partition(const uint64_t* weights, size_t n, size_t k,
                                             const evencut_options* options,
                                             size_t* ends_out, size_t ends_capacity,
                                             evencut_summary* summary);

/** Measures an existing split given by `parts` strictly increasing ends (last == n). */
EVENCUT_API evencut_status evencut_evaluate(const uint64_t* weights, size_t n,
                                            const size_t* ends, size_t parts,
                                            evencut_summary* summary);

/**
 * Splits a tree into min(k, n) connected parts, minimising the largest part. parent[v] is the
 * parent of node v and exactly one node has EVENCUT_NO_PARENT. part_out (part_capacity >= n
 * entries) receives the part id of every node; the root is in part 0. summary may be NULL.
 */
EVENCUT_API evencut_status evencut_partition_tree(const size_t* parent, const uint64_t* weights,
                                                  size_t n, size_t k, size_t* part_out,
                                                  size_t part_capacity, evencut_summary* summary);

/** Message for the last failed call on this thread (empty string if none). */
EVENCUT_API const char* evencut_last_error(void);

/** Library version string, e.g. "1.0.0". */
EVENCUT_API const char* evencut_version(void);

/** EVENCUT_ABI_VERSION of the loaded library. */
EVENCUT_API int evencut_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif /* EVENCUT_H */
