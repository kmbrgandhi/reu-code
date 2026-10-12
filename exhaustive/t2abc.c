/*
 * t2abc.c -- exhaustive search for binary type-2 alternating code sets of 8 codes,
 * using the forced type composition of Sulzer (1993).
 *
 * Definitions. For a +-1 code c of length N and 1 <= l <= k, l + k <= N - 1,
 *     v(c)[l,k] = sum_{j=0}^{N-1-l-k} c_j c_{j+l} c_{j+k} c_{j+l+k}.
 * A multiset of 8 codes is a type-2 set iff the 8 vectors sum to zero.
 * Codes are normalized to c_0 = c_1 = +1 (negation and alternation leave v
 * unchanged) and grouped by vector; solutions are multisets of 8 distinct
 * vectors, repetition allowed (Sulzer's counting).
 *
 * Types. The lag-(N-3) products t_j = c_j c_{j+N-3}, j = 0, 1, 2, give
 *     L1 = t0 t1 + t1 t2 = v[1, N-3]   and   L2 = t0 t2 = v[2, N-3].
 * Every code has one of
 *     A: (L1, L2) = (2, 1),   B: (0, -1),   C: (-2, 1).
 * In a type-2 set both coordinates sum to zero: 2#A - 2#C = 0 and
 * #A - #B + #C = 0, so a set of 8 codes has exactly 2 A, 4 B and 2 C codes.
 *
 * Search. Every solution is a unique triple
 *     (A-pair a1 <= a2, C-pair c1 <= c2, B-multiset b1 <= b2 <= b3 <= b4),
 * and it needs  v(a1)+v(a2)+v(c1)+v(c2) = -(v(b1)+v(b2)+v(b3)+v(b4)).
 * The left side is an (A-pair, C-pair) combination. The right side is the
 * B-pair (b1, b2) plus the B-pair (b3, b4) with b2 <= b3: the sorted split of
 * the B-multiset, so each B-multiset is covered exactly once.
 *   - W(v) = sum_d w_d v_d (mod 2^64), w_d random: exactly additive, so a
 *     true solution always has W(left) + W(right) = 0 (no false negatives).
 *   - Tasks: r(v) = sum_d u_d v_d (mod P), P prime, u_d random. A solution has
 *     r(left) + r(right) = 0 (mod P), so task t joins the left entries with
 *     r = t against the right entries with r = -t. (A prime modulus keeps
 *     the residues uniform. The low bits of W are not uniform, because every
 *     coordinate of v has a fixed parity.)
 *   - Pairs are stored sorted by residue, so a task reads its buckets in
 *     order. The left entries of a task go into a table of 31-bit
 *     fingerprints of W, in groups of 8 slots (half a cache line) that are
 *     filled in order and compared 8 at a time with NEON. The table is
 *     sized for a load of 0.5; the group comes from the top 32 bits of W
 *     (multiply-shift range reduction) and the fingerprint from bits 5-35. Right
 *     entries are probed in prefetched batches. A fingerprint match makes a
 *     candidate. At the end of the task the left side is regenerated,
 *     candidates are matched on the full 64-bit W, and every match is
 *     confirmed by an exact integer vector sum.
 *   - Threads take tasks from a shared counter. Progress, with an estimate
 *     of the remaining time, goes to stderr about once a minute.
 *
 * Build: clang -O3 -mcpu=native -pthread -o t2abc t2abc.c
 * Run:   ./t2abc N [-t threads] [-m MB] [-p prime] [-s seed] [-r first:last] [-o solutions.txt] [-n]
 *   -m sets the table size per thread (default 32 MB). The number of tasks P
 *      follows from it: more memory means fewer tasks and less per-task
 *      overhead, but slower table lookups.
 *   -r runs only tasks first..last-1, so a long search can be split across
 *      runs. Use the same N, prime and seed for every part, then merge the
 *      solution files (runs/t2abc_chunked.sh does this).
 *   -n prints the setup line, including the number of tasks, and exits.
 */
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

typedef struct { uint64_t w; uint16_t i, j; uint32_t r; } Pair;          /* 16 bytes */
typedef struct { uint64_t key; uint32_t b1, b2; } Cand;

static int N, D, M;
static int8_t *V;              /* M x D distinct vectors */
static uint32_t *rep;          /* smallest code word per vector (c_0 = most significant bit, 1 = -1) */
static uint64_t *Wv;           /* W of each vector */
static uint32_t *Rv;           /* r of each vector */
static uint32_t P;             /* prime modulus = number of tasks */
static Pair *PA, *PC, *PB;     /* pairs within each type, sorted by residue, then by (i, j) */
static size_t nPA, nPC, nPB;
static uint32_t *sA, *sC, *sB; /* bucket starts by residue, size P + 1 */
static uint32_t taskFirst, taskLast;
static atomic_uint_fast64_t nextTask, doneTasks, nCand, workDone;
static double t0;
static pthread_mutex_t outMu = PTHREAD_MUTEX_INITIALIZER;
static uint16_t (*sols)[8];
static size_t nSols, capSols;
static _Atomic double lastPrint;

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "out of memory\n"); exit(1); } return p; }
static void *xrealloc(void *p, size_t n) { p = realloc(p, n ? n : 1); if (!p) { fprintf(stderr, "out of memory\n"); exit(1); } return p; }

static uint64_t rng_state;
static uint64_t rnd(void) {                       /* splitmix64 */
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint32_t sub_mod(uint32_t a, uint32_t b) { return a >= b ? a - b : a + P - b; }

static int is_prime(uint64_t n) { if (n < 2) return 0; for (uint64_t q = 2; q * q <= n; q++) if (n % q == 0) return 0; return 1; }
static int cmp_rec(const void *x, const void *y) { return memcmp(x, y, (size_t)D + 4); }

/* Build vectors for all codes with c0 = c1 = +1 and keep the distinct ones. */
static void build_vectors(void) {
    D = 0;
    for (int l = 1; l < N; l++) for (int k = l; k < N - l; k++) D++;
    uint32_t nc = 1u << (N - 2);
    size_t rec = (size_t)D + 4;                    /* vector bytes + 4-byte big-endian code word */
    uint8_t *buf = xmalloc(rec * nc);
    int c[64];
    for (uint32_t m = 0; m < nc; m++) {            /* the code word is m: bit N-1-j is c_j */
        c[0] = c[1] = 1;
        for (int j = 2; j < N; j++) c[j] = (m >> (N - 1 - j)) & 1 ? -1 : 1;
        uint8_t *p = buf + rec * m;
        int d = 0;
        for (int l = 1; l < N; l++) for (int k = l; k < N - l; k++) {
            int s = 0;
            for (int j = 0; j + l + k < N; j++) s += c[j] * c[j + l] * c[j + k] * c[j + l + k];
            ((int8_t *)p)[d++] = (int8_t)s;
        }
        uint8_t be[4] = { (uint8_t)(m >> 24), (uint8_t)(m >> 16), (uint8_t)(m >> 8), (uint8_t)m };
        memcpy(p + D, be, 4);
    }
    qsort(buf, nc, rec, cmp_rec);                  /* by vector, then by code word */
    V = xmalloc((size_t)nc * D); rep = xmalloc(4 * (size_t)nc);
    M = 0;
    for (uint32_t m = 0; m < nc; m++) {
        uint8_t *p = buf + rec * m;
        if (M > 0 && memcmp(V + (size_t)(M - 1) * D, p, (size_t)D) == 0) continue;
        memcpy(V + (size_t)M * D, p, (size_t)D);
        rep[M] = (uint32_t)p[D] << 24 | (uint32_t)p[D + 1] << 16 | (uint32_t)p[D + 2] << 8 | p[D + 3];
        M++;
    }
    free(buf);
}

static int coord_index(int l, int k) {             /* position of (l, k), l <= k, in the vector */
    int d = 0;
    for (int a = 1; a < N; a++) for (int b = a; b < N - a; b++) { if (a == l && b == k) return d; d++; }
    return -1;
}

/* All pairs x <= y of the given vectors, sorted by residue (stable, so (i, j) order within a bucket). */
static Pair *make_pairs(const int *ids, int n, size_t *np, uint32_t **start) {
    size_t cnt = (size_t)n * (n + 1) / 2;
    Pair *tmp = xmalloc(sizeof(Pair) * cnt), *out = xmalloc(sizeof(Pair) * cnt);
    uint32_t *s = calloc((size_t)P + 1, 4);
    if (!s) { fprintf(stderr, "out of memory\n"); exit(1); }
    size_t q = 0;
    for (int x = 0; x < n; x++) for (int y = x; y < n; y++) {
        int i = ids[x], j = ids[y];
        tmp[q].w = Wv[i] + Wv[j];
        tmp[q].r = (uint32_t)(((uint64_t)Rv[i] + Rv[j]) % P);
        tmp[q].i = (uint16_t)i; tmp[q].j = (uint16_t)j;
        s[tmp[q].r + 1]++;
        q++;
    }
    for (uint32_t t = 0; t < P; t++) s[t + 1] += s[t];
    uint32_t *pos = xmalloc(4 * (size_t)P); memcpy(pos, s, 4 * (size_t)P);
    for (q = 0; q < cnt; q++) out[pos[tmp[q].r]++] = tmp[q];
    free(pos); free(tmp);
    *np = cnt; *start = s;
    return out;
}

static int exact_zero(const uint16_t idx[8]) {
    int16_t acc[512];
    for (int d = 0; d < D; d++) acc[d] = 0;
    for (int q = 0; q < 8; q++) { const int8_t *v = V + (size_t)idx[q] * D; for (int d = 0; d < D; d++) acc[d] += v[d]; }
    for (int d = 0; d < D; d++) if (acc[d]) return 0;
    return 1;
}

static int cmp_u16(const void *x, const void *y) { return (int)*(const uint16_t *)x - (int)*(const uint16_t *)y; }
static int cmp_sol(const void *x, const void *y) { return memcmp(x, y, 16); }
static int cmp_cand(const void *x, const void *y) { uint64_t p = ((const Cand *)x)->key, q = ((const Cand *)y)->key; return (p > q) - (p < q); }
static int cmp_u32(const void *x, const void *y) { uint32_t p = *(const uint32_t *)x, q = *(const uint32_t *)y; return (p > q) - (p < q); }
static int cmp_line(const void *x, const void *y) {
    for (int i = 0; i < 8; i++) { int c = cmp_u32((const uint32_t *)x + i, (const uint32_t *)y + i); if (c) return c; }
    return 0;
}

/* Group table: 8 slots per group, filled in order; 0 marks an empty slot. */
static inline uint32_t fingerprint(uint64_t key) { return (uint32_t)(key >> 4) | 1u; }
static inline uint64_t group_of(uint64_t key, uint64_t G) { return ((key >> 32) * G) >> 32; }

static inline void tab_insert(uint32_t *tab, uint64_t G, uint64_t key) {
    uint64_t g = group_of(key, G);
    uint32_t fp = fingerprint(key);
    for (;;) {
        uint32_t *gp = tab + 8 * g;
        uint16x8_t z = vcombine_u16(vmovn_u32(vceqzq_u32(vld1q_u32(gp))), vmovn_u32(vceqzq_u32(vld1q_u32(gp + 4))));
        uint64_t e = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(z)), 0);   /* one 0xFF byte per empty slot */
        if (e) { gp[__builtin_ctzll(e) >> 3] = fp; return; }
        g = g + 1 == G ? 0 : g + 1;
    }
}

static inline int tab_find(const uint32_t *tab, uint64_t G, uint64_t key) {
    uint64_t g = group_of(key, G);
    uint32x4_t f = vdupq_n_u32(fingerprint(key));
    for (;;) {
        const uint32_t *gp = tab + 8 * g;
        uint32x4_t x0 = vld1q_u32(gp), x1 = vld1q_u32(gp + 4);
        if (vmaxvq_u32(vorrq_u32(vceqq_u32(x0, f), vceqq_u32(x1, f)))) return 1;
        if (vgetq_lane_u32(x1, 3) == 0) return 0;     /* group not full: the key is not further on */
        g = g + 1 == G ? 0 : g + 1;
    }
}

/* Number of left entries of task t: sum over A-buckets of |A bucket| * |C bucket|. */
static uint64_t left_count(uint32_t t) {
    uint64_t n = 0;
    for (uint32_t ra = 0; ra < P; ra++) {
        uint32_t na = sA[ra + 1] - sA[ra];
        if (na) { uint32_t rc = sub_mod(t, ra); n += (uint64_t)na * (sC[rc + 1] - sC[rc]); }
    }
    return n;
}

/* Regenerate the left side of task t and confirm the candidates (sorted by key). */
static void confirm(uint32_t t, Cand *cand, size_t nc) {
    for (uint32_t ra = 0; ra < P; ra++) {
        uint32_t a0 = sA[ra], a1 = sA[ra + 1];
        if (a0 == a1) continue;
        uint32_t rc = sub_mod(t, ra), c0 = sC[rc], c1 = sC[rc + 1];
        for (uint32_t a = a0; a < a1; a++) for (uint32_t c = c0; c < c1; c++) {
            uint64_t key = PA[a].w + PC[c].w;
            size_t lo = 0, hi = nc;                  /* first candidate with cand.key >= key */
            while (lo < hi) { size_t mid = (lo + hi) / 2; if (cand[mid].key < key) lo = mid + 1; else hi = mid; }
            for (size_t q = lo; q < nc && cand[q].key == key; q++) {
                uint16_t idx[8] = { PA[a].i, PA[a].j, PC[c].i, PC[c].j,
                                    PB[cand[q].b1].i, PB[cand[q].b1].j, PB[cand[q].b2].i, PB[cand[q].b2].j };
                if (!exact_zero(idx)) continue;
                qsort(idx, 8, 2, cmp_u16);
                pthread_mutex_lock(&outMu);
                if (nSols == capSols) { capSols = capSols ? 2 * capSols : 64; sols = xrealloc(sols, 16 * capSols); }
                memcpy(sols[nSols++], idx, 16);
                pthread_mutex_unlock(&outMu);
            }
        }
    }
}

#ifndef BATCH
#define BATCH 128
#endif
#ifndef LOAD
#define LOAD 0.5
#endif

static void *worker(void *arg) {
    (void)arg;
    size_t capT = 0, capC = 64, nc;
    uint32_t *tab = NULL;
    Cand *cand = xmalloc(sizeof(Cand) * capC);
    uint64_t bk[BATCH], bi[BATCH];
    for (;;) {
        uint64_t tt = atomic_fetch_add(&nextTask, 1);
        if (tt >= taskLast) break;
        uint32_t t = (uint32_t)tt;
        uint64_t nL = left_count(t), work = nL;
        uint64_t G = nL / (8 * LOAD) + 2;                 /* groups of 8 slots, load about LOAD */
        size_t slots = (size_t)8 * G;
        if (slots > capT) {
            free(tab); capT = slots;
            if (posix_memalign((void **)&tab, 64, 4 * capT)) { fprintf(stderr, "out of memory\n"); exit(1); }
        }
        memset(tab, 0, 4 * slots);
        /* left side: A-bucket ra with C-bucket t - ra, inserted in prefetched batches */
        int nb = 0;
        for (uint32_t ra = 0; ra < P; ra++) {
            uint32_t a0 = sA[ra], a1 = sA[ra + 1];
            if (a0 == a1) continue;
            uint32_t rc = sub_mod(t, ra), c0 = sC[rc], c1 = sC[rc + 1];
            for (uint32_t a = a0; a < a1; a++) {
                uint64_t wa = PA[a].w;
                for (uint32_t c = c0; c < c1; c++) {
                    uint64_t key = wa + PC[c].w;
                    __builtin_prefetch(tab + 8 * group_of(key, G), 1);
                    bk[nb] = key;
                    if (++nb == BATCH) { for (int x = 0; x < nb; x++) tab_insert(tab, G, bk[x]); nb = 0; }
                }
            }
        }
        for (int x = 0; x < nb; x++) tab_insert(tab, G, bk[x]);
        /* right side: B-bucket rb with B-bucket -t - rb, b2 <= b3 */
        nc = 0;
        nb = 0;
        uint32_t u = t ? P - t : 0;
#define FLUSH() do { \
            for (int x = 0; x < nb; x++) if (tab_find(tab, G, bk[x])) { \
                if (nc == capC) { capC *= 2; cand = xrealloc(cand, sizeof(Cand) * capC); } \
                cand[nc].key = bk[x]; cand[nc].b1 = (uint32_t)(bi[x] >> 32); cand[nc].b2 = (uint32_t)bi[x]; nc++; \
            } \
            work += nb; nb = 0; \
        } while (0)
        for (uint32_t rb = 0; rb < P; rb++) {
            uint32_t p0 = sB[rb], p1 = sB[rb + 1];
            if (p0 == p1) continue;
            uint32_t s = sub_mod(u, rb), q0 = sB[s], q1 = sB[s + 1];
            if (q0 == q1) continue;
            for (uint32_t b1 = p0; b1 < p1; b1++) {
                uint16_t jmin = PB[b1].j;
                uint64_t nw1 = 0 - PB[b1].w;
                /* a bucket is in (i, j) order: scan down and stop below jmin */
                for (uint32_t q = q1; q-- > q0; ) {
                    if (PB[q].i < jmin) break;
                    uint64_t key = nw1 - PB[q].w;
                    __builtin_prefetch(tab + 8 * group_of(key, G));
                    bk[nb] = key; bi[nb] = (uint64_t)b1 << 32 | q;
                    if (++nb == BATCH) FLUSH();
                }
            }
        }
        FLUSH();
#undef FLUSH
        if (nc) {
            atomic_fetch_add(&nCand, nc);
            qsort(cand, nc, sizeof(Cand), cmp_cand);
            confirm(t, cand, nc);
        }
        atomic_fetch_add(&workDone, work);
        uint64_t dn = atomic_fetch_add(&doneTasks, 1) + 1;
        double tn = now(), lp = atomic_load(&lastPrint);
        if ((tn - lp >= 60.0 && atomic_compare_exchange_strong(&lastPrint, &lp, tn)) || dn == taskLast - taskFirst) {
            double el = tn - t0, frac = (double)dn / (taskLast - taskFirst);
            pthread_mutex_lock(&outMu);
            size_t ns = nSols;
            pthread_mutex_unlock(&outMu);
            fprintf(stderr, "  [N=%d] %llu/%u tasks (%.1f%%), %.0f s elapsed, about %.0f s left, %zu solutions, %llu candidates\n",
                    N, (unsigned long long)dn, taskLast - taskFirst, 100 * frac, el, el * (1 - frac) / frac,
                    ns, (unsigned long long)atomic_load(&nCand));
        }
    }
    free(tab); free(cand);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s N [-t threads] [-m MB] [-p prime] [-s seed] [-r first:last] [-o file] [-n]\n", argv[0]); return 1; }
    N = atoi(argv[1]);
    int T = 8, dry = 0; uint64_t pOpt = 0, seed = 0x7432ABCULL; double mb = 32; const char *out = NULL, *range = NULL;
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], "-t") && a + 1 < argc) T = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-p") && a + 1 < argc) pOpt = strtoull(argv[++a], 0, 10);
        else if (!strcmp(argv[a], "-m") && a + 1 < argc) mb = atof(argv[++a]);
        else if (!strcmp(argv[a], "-s") && a + 1 < argc) seed = strtoull(argv[++a], 0, 0);
        else if (!strcmp(argv[a], "-r") && a + 1 < argc) range = argv[++a];
        else if (!strcmp(argv[a], "-o") && a + 1 < argc) out = argv[++a];
        else if (!strcmp(argv[a], "-n")) dry = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 1; }
    }
    if (N < 8 || N > 18) { fprintf(stderr, "N must be between 8 and 18\n"); return 1; }
    double ts = now();
    build_vectors();
    if (M > 65535) { fprintf(stderr, "too many vectors for 16-bit indices\n"); return 1; }
    int dL1 = coord_index(1, N - 3), dL2 = coord_index(2, N - 3);
    int *ia = xmalloc(4 * (size_t)M), *ib = xmalloc(4 * (size_t)M), *ic = xmalloc(4 * (size_t)M), na = 0, nb = 0, ncc = 0;
    for (int m = 0; m < M; m++) {
        int l1 = V[(size_t)m * D + dL1], l2 = V[(size_t)m * D + dL2];
        if (l1 == 2 && l2 == 1) ia[na++] = m;
        else if (l1 == 0 && l2 == -1) ib[nb++] = m;
        else if (l1 == -2 && l2 == 1) ic[ncc++] = m;
        else { fprintf(stderr, "unexpected type (%d, %d)\n", l1, l2); return 1; }
    }
    /* work estimate: left = A-pairs * C-pairs, right = B-multisets of size 4 */
    double pa = (double)na * (na + 1) / 2, pc = (double)ncc * (ncc + 1) / 2;
    double wl = pa * pc, wr = (double)nb * (nb + 1) * (nb + 2) * (nb + 3) / 24;
    /* tasks: a task's table holds about wl / P entries at 4 / LOAD bytes each */
    uint64_t p = pOpt ? pOpt : (uint64_t)(wl * (4 / LOAD) / (mb * 1048576.0));
    if (p < 7) p = 7;
    while (!is_prime(p)) p++;
    if (p > 0xFFFFFFF0ULL) { fprintf(stderr, "prime too large\n"); return 1; }
    P = (uint32_t)p;
    taskFirst = 0; taskLast = P;
    if (range) {
        unsigned long long f, l;
        if (sscanf(range, "%llu:%llu", &f, &l) != 2 || f >= l || l > P) { fprintf(stderr, "bad range %s (prime is %u)\n", range, P); return 1; }
        taskFirst = (uint32_t)f; taskLast = (uint32_t)l;
    }
    rng_state = seed;
    Wv = xmalloc(8 * (size_t)M); Rv = xmalloc(4 * (size_t)M);
    uint64_t *w = xmalloc(8 * (size_t)D); uint32_t *uu = xmalloc(4 * (size_t)D);
    for (int d = 0; d < D; d++) { w[d] = rnd() | 1; uu[d] = (uint32_t)(rnd() % P); }
    for (int m = 0; m < M; m++) {
        uint64_t h = 0; int64_t r = 0;
        for (int d = 0; d < D; d++) { int v = V[(size_t)m * D + d]; h += w[d] * (uint64_t)(int64_t)v; r += (int64_t)uu[d] * v; }
        Wv[m] = h; r %= (int64_t)P; if (r < 0) r += P; Rv[m] = (uint32_t)r;
    }
    PA = make_pairs(ia, na, &nPA, &sA); PC = make_pairs(ic, ncc, &nPC, &sC); PB = make_pairs(ib, nb, &nPB, &sB);
    fprintf(stderr, "N=%d D=%d distinct_vectors=%d A=%d B=%d C=%d left=%.3g right=%.3g prime=%u tasks=%u..%u "
            "table=%.1fMB/thread scan_overhead=%.2f threads=%d seed=0x%llx setup=%.1fs\n",
            N, D, M, na, nb, ncc, wl, wr, P, taskFirst, taskLast, wl / P * (4 / LOAD) / 1048576.0,
            (double)P * (pa + (double)nb * (nb + 1) / 2) / (wl + wr), T, (unsigned long long)seed, now() - ts);
    if (dry) return 0;
    t0 = now(); atomic_store(&lastPrint, t0); atomic_store(&nextTask, taskFirst);
    pthread_t *th = xmalloc(sizeof(pthread_t) * T);
    for (int i = 0; i < T; i++) pthread_create(&th[i], NULL, worker, NULL);
    for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
    qsort(sols, nSols, 16, cmp_sol);
    size_t u = 0;
    for (size_t q = 0; q < nSols; q++) if (!u || memcmp(sols[u - 1], sols[q], 16)) memcpy(sols[u++], sols[q], 16);
    if (u != nSols) fprintf(stderr, "warning: %zu duplicate solutions removed (expected none)\n", nSols - u);
    printf("RESULT N=%d count=%zu tasks=%u..%u of %u search_time=%.1f s candidates=%llu work=%llu\n",
           N, u, taskFirst, taskLast, P, now() - t0, (unsigned long long)atomic_load(&nCand),
           (unsigned long long)atomic_load(&workDone));
    if (out) {                                     /* same format as cleanroom/t2search: words sorted per line, lines sorted */
        uint32_t (*wlines)[8] = xmalloc(32 * (u ? u : 1));
        for (size_t q = 0; q < u; q++) {
            for (int x = 0; x < 8; x++) wlines[q][x] = rep[sols[q][x]];
            qsort(wlines[q], 8, 4, cmp_u32);
        }
        qsort(wlines, u, 32, cmp_line);
        FILE *f = fopen(out, "w");
        if (!f) { perror(out); return 1; }
        int hexd = (N + 3) / 4;
        for (size_t q = 0; q < u; q++)
            for (int x = 0; x < 8; x++) fprintf(f, "%0*x%c", hexd, wlines[q][x], x == 7 ? '\n' : ' ');
        fclose(f);
    }
    return 0;
}
