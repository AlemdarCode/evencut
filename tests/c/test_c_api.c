/* Exercises the C ABI from plain C (C11), linked against the shared library. */
#include "evencut/evencut.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++checks;                                                            \
        if (!(cond)) {                                                       \
            ++failures;                                                      \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

int main(void) {
    const uint64_t w[] = {5, 3, 8, 2, 7, 4, 6, 1, 9, 5};
    const size_t n = sizeof(w) / sizeof(w[0]);
    size_t ends[16];
    evencut_summary s;
    evencut_options opt;

    CHECK(evencut_abi_version() == EVENCUT_ABI_VERSION);
    CHECK(strlen(evencut_version()) > 0);

    /* defaults (opt == NULL): min-max */
    CHECK(evencut_partition(w, n, 3, NULL, ends, 16, &s) == EVENCUT_OK);
    CHECK(s.parts == 3 && ends[2] == n && s.max_load == 18 && s.total == 50);

    /* sum of squares */
    evencut_options_init(&opt);
    opt.objective = EVENCUT_SUM_SQUARES;
    CHECK(evencut_partition(w, n, 3, &opt, ends, 16, &s) == EVENCUT_OK);
    CHECK(s.parts == 3 && strcmp(s.sum_of_squares, "838") == 0);

    /* forbidden cut, numbered like ends: 3 keeps items 2 and 3 together */
    evencut_options_init(&opt);
    size_t forbid[] = {3};
    opt.forbidden_cuts = forbid;
    opt.forbidden_count = 1;
    CHECK(evencut_partition(w, n, 3, &opt, ends, 16, &s) == EVENCUT_OK);
    CHECK(ends[0] != 3 && ends[1] != 3);

    /* secondary limit */
    evencut_options_init(&opt);
    const uint64_t mem[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    opt.secondary = mem;
    opt.secondary_limit = 3;
    CHECK(evencut_partition(w, n, 4, &opt, ends, 16, &s) == EVENCUT_OK);
    CHECK(s.parts == 4 && ends[0] <= 3 && ends[1] - ends[0] <= 3 && ends[2] - ends[1] <= 3 && n - ends[2] <= 3);
    opt.secondary_limit = 2; /* 10 items, 4 parts of at most 2 items: infeasible */
    CHECK(evencut_partition(w, n, 4, &opt, ends, 16, &s) == EVENCUT_INVALID_ARGUMENT);
    CHECK(strlen(evencut_last_error()) > 0);
    evencut_options_init(&opt); /* forgotten limit: the default 0 fails loudly, never "unlimited" */
    opt.secondary = mem;
    CHECK(evencut_partition(w, n, 4, &opt, ends, 16, &s) == EVENCUT_INVALID_ARGUMENT);
    CHECK(strstr(evencut_last_error(), "secondary_limit") != NULL);
    opt.secondary_limit = UINT64_MAX; /* an explicit maximum is a valid limit */
    CHECK(evencut_partition(w, n, 4, &opt, ends, 16, &s) == EVENCUT_OK && s.parts == 4);
    evencut_options_init(&opt); /* a limit without secondary values is a mistake too */
    opt.secondary_limit = 3;
    CHECK(evencut_partition(w, n, 4, &opt, ends, 16, &s) == EVENCUT_INVALID_ARGUMENT);
    evencut_options_init(&opt); /* forbidden cut outside [1, n-1] */
    size_t bad_forbid[] = {0};
    opt.forbidden_cuts = bad_forbid;
    opt.forbidden_count = 1;
    CHECK(evencut_partition(w, n, 3, &opt, ends, 16, &s) == EVENCUT_INVALID_ARGUMENT);

    /* errors */
    CHECK(evencut_partition(w, n, 0, NULL, ends, 16, &s) == EVENCUT_INVALID_ARGUMENT);
    CHECK(strstr(evencut_last_error(), "k must be") != NULL);
    CHECK(evencut_partition(w, n, 5, NULL, ends, 2, &s) == EVENCUT_BUFFER_TOO_SMALL);
    CHECK(evencut_partition(NULL, n, 2, NULL, ends, 16, &s) == EVENCUT_INVALID_ARGUMENT);
    evencut_options bad;
    memset(&bad, 0, sizeof(bad)); /* not initialised: struct_size == 0 */
    CHECK(evencut_partition(w, n, 2, &bad, ends, 16, &s) == EVENCUT_INVALID_ARGUMENT);
    evencut_options_init(&opt);
    opt.struct_size = (uint32_t)(sizeof(opt) + 16); /* a caller built against a future, larger header */
    CHECK(evencut_partition(w, n, 2, &opt, ends, 16, &s) == EVENCUT_OK);
    opt.struct_size = 8; /* smaller than the version-1 layout */
    CHECK(evencut_partition(w, n, 2, &opt, ends, 16, &s) == EVENCUT_INVALID_ARGUMENT);
    CHECK(evencut_partition(w, n, 2, NULL, ends, 16, NULL) == EVENCUT_OK); /* summary optional */
    CHECK(evencut_last_error()[0] == '\0');                              /* cleared on success */

    /* evaluate */
    const size_t uniform[] = {4, 7, 10};
    CHECK(evencut_evaluate(w, n, uniform, 3, &s) == EVENCUT_OK);
    CHECK(s.max_load == 18 && s.parts == 3);
    const size_t broken[] = {4, 4, 10};
    CHECK(evencut_evaluate(w, n, broken, 3, &s) == EVENCUT_INVALID_ARGUMENT);

    /* tree: root 0 with children 1, 2; node 3 under 1 */
    const size_t parent[] = {EVENCUT_NO_PARENT, 0, 0, 1};
    const uint64_t tw[] = {4, 3, 5, 2};
    size_t part[4];
    CHECK(evencut_partition_tree(parent, tw, 4, 2, part, 4, &s) == EVENCUT_OK);
    CHECK(s.parts == 2 && s.max_load == 9 && part[0] == 0);
    CHECK(evencut_partition_tree(parent, tw, 4, 2, part, 3, &s) == EVENCUT_BUFFER_TOO_SMALL);
    const size_t cyclic[] = {EVENCUT_NO_PARENT, 2, 1, 0};
    CHECK(evencut_partition_tree(cyclic, tw, 4, 2, part, 4, &s) == EVENCUT_INVALID_ARGUMENT);

    printf("C API: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
