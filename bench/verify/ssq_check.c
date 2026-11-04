/* Independent exact checker for the sum_squares objective. It shares no code with Evencut.
 *
 * Reads n little-endian uint64 weights from a file. Every mode prints one line per k,
 * "k value", where value is the minimum of sum(load^2) over all splits into k non-empty
 * contiguous parts.
 *
 *   ssq_check <weights.u64> dp   <K>          k = 1..K, divide-and-conquer DP, O(K n log n)
 *   ssq_check <weights.u64> full <K>          k = 1..K, plain DP, O(K n^2), assumes nothing
 *   ssq_check <weights.u64> dual <k1,k2,...>  the listed k, Lagrangian dual, O(n log S) per k
 *
 * dp:   relies on the textbook fact that cost(i, j) = (P[j] - P[i])^2 satisfies the
 *       quadrangle inequality, so the leftmost optimal split point never moves left.
 * full: only practical for small n; used to check dp itself.
 * dual: for any lambda, g(lambda) = min over splits into ANY number m of parts of
 *       (sum of squares + lambda * m) gives a lower bound: every k-part split costs at least
 *       g(lambda) - lambda * k (weak duality, no assumption). The program maximises this bound
 *       over integer lambda. The printed value is therefore a proven lower bound, and a split
 *       that achieves it is proven optimal. (The bound is tight because the optimal cost is
 *       convex in k; if it were not, the check would fail, never pass wrongly.) g(lambda) is
 *       computed exactly in O(n) with a convex hull trick. A third column gives lambda.
 *
 * Timing goes to stderr.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 u128;
typedef __int128 i128;
#define INF (~(u128)0)

static long n;
static uint64_t *P;        /* prefix sums: P[0] = 0, P[i] = w[0] + ... + w[i-1] */

static void print_u128(u128 v) {
    char buf[64];
    int pos = 63;
    buf[pos] = '\0';
    if (v == 0) buf[--pos] = '0';
    while (v > 0) { buf[--pos] = (char)('0' + (int)(v % 10)); v /= 10; }
    fputs(buf + pos, stdout);
}

static u128 sq(uint64_t x) { return (u128)x * x; }

/* ---- dp and full ---------------------------------------------------------------------- */

static u128 *prev_row, *cur_row;

/* No overflow: prev_row[t] <= P[t]^2, so prev_row[t] + (P[i] - P[t])^2 <= P[i]^2 < 2^128. */
static void solve(long lo, long hi, long optlo, long opthi) {
    if (lo > hi) return;
    long mid = lo + (hi - lo) / 2;
    long last = opthi < mid - 1 ? opthi : mid - 1;
    long best_t = optlo;
    u128 best = INF;
    for (long t = optlo; t <= last; ++t) {
        if (prev_row[t] == INF) continue;
        u128 v = prev_row[t] + sq(P[mid] - P[t]);
        if (v < best) { best = v; best_t = t; }   /* strict: keeps the leftmost optimum */
    }
    cur_row[mid] = best;
    solve(lo, mid - 1, optlo, best_t);
    solve(mid + 1, hi, best_t, opthi);
}

static void full_row(long j) {
    for (long i = j; i <= n; ++i) {
        u128 best = INF;
        for (long t = j - 1; t <= i - 1; ++t) {
            if (prev_row[t] == INF) continue;
            u128 v = prev_row[t] + sq(P[i] - P[t]);
            if (v < best) best = v;
        }
        cur_row[i] = best;
    }
}

static int run_dp(long K, int full) {
    prev_row = malloc((size_t)(n + 1) * sizeof *prev_row);
    cur_row = malloc((size_t)(n + 1) * sizeof *cur_row);
    if (!prev_row || !cur_row) { fprintf(stderr, "out of memory\n"); return 2; }
    if (K > n) K = n;
    for (long i = 0; i <= n; ++i) prev_row[i] = i == 0 ? INF : sq(P[i]);   /* k = 1 */
    printf("1 "); print_u128(prev_row[n]); printf("\n");
    for (long j = 2; j <= K; ++j) {
        for (long i = 0; i < j; ++i) cur_row[i] = INF;
        if (full) full_row(j);
        else solve(j, n, j - 1, n - 1);
        printf("%ld ", j); print_u128(cur_row[n]); printf("\n");
        u128 *tmp = prev_row; prev_row = cur_row; cur_row = tmp;
    }
    return 0;
}

/* ---- dual ----------------------------------------------------------------------------- */

static i128 *G;   /* G[i] = g restricted to the first i items */
static long *dq;  /* deque of split points t, lines y = A(t) x + B(t) */

static i128 A(long t) { return -2 * (i128)P[t]; }
static i128 B(long t) { return G[t] + (i128)P[t] * (i128)P[t]; }
static i128 val(long t, i128 x) { return A(t) * x + B(t); }

/* g(lam) = min over splits into any m >= 1 non-empty parts of sum(load^2) + lam * m.
 * G[i] = min_{t<i} G[t] + (P[i] - P[t])^2 + lam = P[i]^2 + lam + min_t (A(t) P[i] + B(t)).
 * Queries P[i] never decrease and slopes A(t) never increase, so a monotone deque suffices.
 * Magnitudes (total S < 2^40, lam <= S^2): B < 2^82, slope gaps < 2^42, products < 2^124. */
static i128 g_of(i128 lam) {
    long head = 0, tail = 0;
    G[0] = 0;
    dq[tail++] = 0;
    for (long i = 1; i <= n; ++i) {
        i128 x = (i128)P[i];
        while (tail - head >= 2 && val(dq[head + 1], x) <= val(dq[head], x)) ++head;
        G[i] = x * x + lam + val(dq[head], x);
        /* add the line of split point i */
        i128 a3 = A(i), b3 = B(i);
        if (tail > head && A(dq[tail - 1]) == a3) {          /* equal slope: keep the lower line */
            if (B(dq[tail - 1]) <= b3) continue;
            --tail;
        }
        while (tail - head >= 2) {
            long t1 = dq[tail - 2], t2 = dq[tail - 1];
            i128 a1 = A(t1), b1 = B(t1), a2 = A(t2), b2 = B(t2);
            if ((b3 - b1) * (a1 - a2) <= (b2 - b1) * (a1 - a3)) --tail;   /* t2 never the minimum */
            else break;
        }
        dq[tail++] = i;
    }
    return G[n];
}

static i128 h_of(i128 lam, long k) { return g_of(lam) - lam * k; }

static int run_dual(const char *list) {
    if (P[n] >= ((uint64_t)1 << 40)) { fprintf(stderr, "dual mode needs a total below 2^40\n"); return 2; }
    G = malloc((size_t)(n + 1) * sizeof *G);
    dq = malloc((size_t)(n + 1) * sizeof *dq);
    if (!G || !dq) { fprintf(stderr, "out of memory\n"); return 2; }
    const i128 lam_max = (i128)P[n] * (i128)P[n];
    char *copy = strdup(list);
    for (char *tok = strtok(copy, ","); tok; tok = strtok(NULL, ",")) {
        long k = atol(tok);
        if (k < 1 || k > n) { fprintf(stderr, "k=%ld out of range\n", k); return 2; }
        /* h is concave in lam: find the first lam where it stops increasing */
        i128 lo = 0, hi = lam_max;
        while (lo < hi) {
            i128 mid = lo + (hi - lo) / 2;
            if (h_of(mid + 1, k) <= h_of(mid, k)) hi = mid;
            else lo = mid + 1;
        }
        i128 best = h_of(lo, k);
        printf("%ld ", k); print_u128((u128)best); printf(" "); print_u128((u128)lo); printf("\n");
        fflush(stdout);
    }
    free(copy);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || (strcmp(argv[2], "dp") && strcmp(argv[2], "full") && strcmp(argv[2], "dual"))) {
        fprintf(stderr, "usage: ssq_check <weights.u64> dp|full <K>\n       ssq_check <weights.u64> dual <k1,k2,...>\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % 8 != 0) { fprintf(stderr, "bad file size\n"); return 2; }
    n = bytes / 8;
    uint64_t *w = malloc((size_t)n * sizeof *w);
    P = malloc((size_t)(n + 1) * sizeof *P);
    if (!w || !P) { fprintf(stderr, "out of memory\n"); return 2; }
    if (fread(w, 8, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);
    P[0] = 0;
    for (long i = 0; i < n; ++i) {
        if (P[i] > UINT64_MAX - w[i]) { fprintf(stderr, "total does not fit in 64 bits\n"); return 2; }
        P[i + 1] = P[i] + w[i];
    }
    free(w);
    clock_t t0 = clock();
    int rc = strcmp(argv[2], "dual") == 0 ? run_dual(argv[3]) : run_dp(atol(argv[3]), strcmp(argv[2], "full") == 0);
    fprintf(stderr, "n=%ld mode=%s seconds=%.2f\n", n, argv[2], (double)(clock() - t0) / CLOCKS_PER_SEC);
    return rc;
}
