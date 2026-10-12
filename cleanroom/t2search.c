/*
 * t2search.c -- clean-room exhaustive search for binary type-2 alternating
 * code sets (Sulzer 1993 type-2 condition), K = 8 codes.
 *
 * For a +-1 code c of length N and 1 <= l <= k, l + k <= N-1:
 *   v(c)[l,k] = sum_{j=0}^{N-1-l-k} c_j c_{j+l} c_{j+k} c_{j+l+k}.
 * A multiset of 8 codes is a type-2 set iff the 8 vectors sum to zero.
 * Codes are normalized to c_0 = c_1 = +1, grouped by vector, and solutions
 * are counted as multisets of 8 distinct vectors (repetition allowed).
 *
 * Algorithm (meet in the middle, 4 + 4):
 *   - W(v) = sum_d w_d v_d mod 2^64 with random w_d. W is exactly additive,
 *     so v1 + v2 = 0 implies W(v1) + W(v2) = 0 (no false negatives). Every
 *     candidate is confirmed by exact integer vector comparison.
 *   - Every sorted 4-multiset (i1<=i2<=i3<=i4) is generated exactly once as
 *     pair (i1,i2) + pair (i3,i4) with i2 <= i3. Pairs are bucketed by
 *     W mod R, so the 4-multisets with W = t (mod R) are generated directly.
 *   - Task {t, -t mod R}: generate L_t (keys W) and L_{-t} (keys -W), split
 *     both by further key bits into cache-sized sub-buckets, hash-join on
 *     the full 64-bit key.
 *   - Every split (Y, X) of every solution into two 4-multisets is found.
 *     The canonical split (first four / last four of the sorted 8-tuple) is
 *     counted once; all splits are also collected and deduplicated by sorted
 *     8-tuple, and the two counts are compared.
 *   - Optional brute force (-b): DFS over sorted 8-tuples with bound pruning,
 *     compared tuple-by-tuple with the MITM result.
 *
 * Build: clang -O3 -mcpu=native -pthread -o t2search t2search.c
 * Run:   ./t2search N [-t threads] [-b] [-B] [-s seed] [-L log2_list] [-o file]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#define MAXD 64

static int N, D, NT = 8;
static int M;                 /* number of distinct vectors */
static int8_t *V;             /* M x D, lexicographically sorted */
static uint32_t *repCode;     /* smallest normalized code with this vector */
static uint32_t *clsSize;     /* number of normalized codes per vector */
static uint64_t W[MAXD];
static uint64_t *VW;          /* W(V[i]) */
static int lp[MAXD], kp[MAXD];

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static uint64_t sm_state;
static uint64_t splitmix(void) {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void die(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(1); }
static volatile double lastPrint = 0.0;  /* time of the last progress line */
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory"); return p; }
static void *xrealloc(void *q, size_t n) { void *p = realloc(q, n ? n : 1); if (!p) die("out of memory"); return p; }

/* code x: bit (N-1-j) is c_j; bit 0 -> +1, bit 1 -> -1 */
static void code_vec(uint32_t x, int8_t *out) {
    int c[64];
    for (int j = 0; j < N; j++) c[j] = ((x >> (N - 1 - j)) & 1) ? -1 : 1;
    for (int d = 0; d < D; d++) {
        int l = lp[d], k = kp[d], s = 0;
        for (int j = 0; j + l + k <= N - 1; j++) s += c[j] * c[j + l] * c[j + k] * c[j + l + k];
        out[d] = (int8_t)s;
    }
}

static int vcmp(const int8_t *a, const int8_t *b) {
    for (int d = 0; d < D; d++) if (a[d] != b[d]) return a[d] < b[d] ? -1 : 1;
    return 0;
}

typedef struct { int8_t v[MAXD]; uint32_t code; } CodeRec;
static int cmp_coderec(const void *x, const void *y) {
    const CodeRec *a = x, *b = y;
    int c = vcmp(a->v, b->v);
    if (c) return c;
    return a->code < b->code ? -1 : a->code > b->code;
}

/* ---------------- setup: vectors ---------------- */
static void verify_symmetries(void) {
    uint32_t full = 1u << N, mask = full - 1, alt = 0;
    for (int j = 1; j < N; j += 2) alt |= 1u << (N - 1 - j);
    int8_t a[MAXD], b[MAXD];
    for (uint32_t x = 0; x < full; x++) {
        code_vec(x, a);
        code_vec(x ^ mask, b); if (vcmp(a, b)) die("negation symmetry fails");
        code_vec(x ^ alt, b);  if (vcmp(a, b)) die("alternation symmetry fails");
        uint32_t r = 0;
        for (int j = 0; j < N; j++) if ((x >> j) & 1) r |= 1u << (N - 1 - j);
        code_vec(r, b);        if (vcmp(a, b)) die("reversal symmetry fails");
    }
}

static void setup_vectors(void) {
    D = 0;
    for (int l = 1; 2 * l <= N - 1; l++)
        for (int k = l; l + k <= N - 1; k++) { lp[D] = l; kp[D] = k; D++; }
    if (D > MAXD) die("D too large");
    if (N <= 16) verify_symmetries();
    uint32_t nc = 1u << (N - 2);
    CodeRec *r = xmalloc(sizeof(CodeRec) * nc);
    for (uint32_t x = 0; x < nc; x++) { memset(r[x].v, 0, MAXD); code_vec(x, r[x].v); r[x].code = x; }
    qsort(r, nc, sizeof(CodeRec), cmp_coderec);
    V = xmalloc((size_t)nc * D); repCode = xmalloc(4 * nc); clsSize = xmalloc(4 * nc);
    M = 0;
    for (uint32_t x = 0; x < nc; x++) {
        if (x == 0 || vcmp(r[x].v, r[x - 1].v)) {
            memcpy(V + (size_t)M * D, r[x].v, D); repCode[M] = r[x].code; clsSize[M] = 1; M++;
        } else clsSize[M - 1]++;
    }
    free(r);
    if (M > 65535) die("M too large for uint16 indices");
    VW = xmalloc(8 * (size_t)M);
    for (int d = 0; d < D; d++) W[d] = splitmix();
    for (int i = 0; i < M; i++) {
        uint64_t h = 0;
        for (int d = 0; d < D; d++) h += W[d] * (uint64_t)(int64_t)V[(size_t)i * D + d];
        VW[i] = h;
    }
}

/* ---------------- pairs ---------------- */
static int logR, SB;
static uint32_t R, SBN;
static uint64_t P;
static uint32_t *bstart;                 /* R+1 */
static uint64_t *k1, *k2;
static uint16_t *a1, *b1, *a2, *b2;      /* role 1: sorted b asc; role 2: sorted a desc */

typedef struct { uint64_t key; uint16_t a, b; } PairRec;
static int cmp_p1(const void *x, const void *y) {      /* b ascending, then a */
    const PairRec *p = x, *q = y;
    if (p->b != q->b) return p->b < q->b ? -1 : 1;
    return (p->a > q->a) - (p->a < q->a);
}
static int cmp_p2(const void *x, const void *y) {      /* a descending, then b */
    const PairRec *p = x, *q = y;
    if (p->a != q->a) return p->a > q->a ? -1 : 1;
    return (p->b > q->b) - (p->b < q->b);
}

static void setup_pairs(void) {
    P = (uint64_t)M * (M + 1) / 2;
    bstart = calloc(R + 1, 4);
    PairRec *pr = xmalloc(sizeof(PairRec) * P);
    for (int a = 0; a < M; a++) for (int b = a; b < M; b++) bstart[((VW[a] + VW[b]) & (R - 1)) + 1]++;
    for (uint32_t r = 0; r < R; r++) bstart[r + 1] += bstart[r];
    uint32_t *pos = xmalloc(4 * (size_t)R); memcpy(pos, bstart, 4 * (size_t)R);
    for (int a = 0; a < M; a++) for (int b = a; b < M; b++) {
        uint64_t key = VW[a] + VW[b];
        PairRec *e = &pr[pos[key & (R - 1)]++];
        e->key = key; e->a = (uint16_t)a; e->b = (uint16_t)b;
    }
    free(pos);
    k1 = xmalloc(8 * P); k2 = xmalloc(8 * P);
    a1 = xmalloc(2 * P); b1 = xmalloc(2 * P); a2 = xmalloc(2 * P); b2 = xmalloc(2 * P);
    for (uint32_t r = 0; r < R; r++) {
        uint32_t s = bstart[r], e = bstart[r + 1];
        qsort(pr + s, e - s, sizeof(PairRec), cmp_p1);
        for (uint32_t i = s; i < e; i++) { k1[i] = pr[i].key; a1[i] = pr[i].a; b1[i] = pr[i].b; }
        qsort(pr + s, e - s, sizeof(PairRec), cmp_p2);
        for (uint32_t i = s; i < e; i++) { k2[i] = pr[i].key; a2[i] = pr[i].a; b2[i] = pr[i].b; }
    }
    free(pr);
}

/* ---------------- MITM worker ---------------- */
typedef struct { uint64_t key; uint16_t i[4]; } Entry;

typedef struct {
    uint32_t *cntA, *cntB, *offA, *offB, *posA, *posB;
    Entry *A, *B; size_t capA, capB;
    int32_t *head; size_t capH;
    uint64_t *nodeKey; int32_t *nodeNext, *nodeFirst, *entNext; size_t capNode;
    uint16_t *sols; size_t nsols, capSols;
    uint64_t quads, keymatch, falsepos, exact, canon, twiceSplits;
} Thr;

static atomic_uint nextTask;
static atomic_uint doneTasks;

static void gen_count(uint32_t t, int neg, uint32_t *cnt) {
    const int sh = logR; const uint64_t sm = SBN - 1; const uint32_t rm = R - 1;
    for (uint32_t u = 0; u < R; u++) {
        uint32_t v = (t - u) & rm;
        uint32_t s2 = bstart[v], e2 = bstart[v + 1];
        if (s2 == e2) continue;
        uint16_t maxa = a2[s2];
        for (uint32_t p = bstart[u], pe = bstart[u + 1]; p < pe; p++) {
            uint16_t bb = b1[p];
            if (bb > maxa) break;
            uint64_t kk = k1[p];
            for (uint32_t q = s2; q < e2 && a2[q] >= bb; q++) {
                uint64_t key = kk + k2[q];
                if (neg) key = (uint64_t)0 - key;
                cnt[(key >> sh) & sm]++;
            }
        }
    }
}

static void gen_fill(uint32_t t, int neg, uint32_t *pos, Entry *out) {
    const int sh = logR; const uint64_t sm = SBN - 1; const uint32_t rm = R - 1;
    for (uint32_t u = 0; u < R; u++) {
        uint32_t v = (t - u) & rm;
        uint32_t s2 = bstart[v], e2 = bstart[v + 1];
        if (s2 == e2) continue;
        uint16_t maxa = a2[s2];
        for (uint32_t p = bstart[u], pe = bstart[u + 1]; p < pe; p++) {
            uint16_t bb = b1[p];
            if (bb > maxa) break;
            uint64_t kk = k1[p]; uint16_t aa = a1[p];
            for (uint32_t q = s2; q < e2 && a2[q] >= bb; q++) {
                uint64_t key = kk + k2[q];
                if (neg) key = (uint64_t)0 - key;
                Entry *e = &out[pos[(key >> sh) & sm]++];
                e->key = key; e->i[0] = aa; e->i[1] = bb; e->i[2] = a2[q]; e->i[3] = b2[q];
            }
        }
    }
}

static void push_sol(uint16_t **arr, size_t *n, size_t *cap, const uint16_t *t8) {
    if (*n == *cap) { *cap = *cap ? 2 * *cap : 1024; *arr = xrealloc(*arr, 16 * *cap); }
    memcpy(*arr + 8 * *n, t8, 16); (*n)++;
}

static void handle(Thr *T, const Entry *y, const Entry *x, int self) {
    T->keymatch++;
    int s[MAXD];
    for (int d = 0; d < D; d++) s[d] = 0;
    for (int m = 0; m < 4; m++) {
        const int8_t *vy = V + (size_t)y->i[m] * D, *vx = V + (size_t)x->i[m] * D;
        for (int d = 0; d < D; d++) s[d] += vy[d] + vx[d];
    }
    for (int d = 0; d < D; d++) if (s[d]) { T->falsepos++; return; }
    T->exact++;
    /* twice the number of unordered splits {Y,X}: a non-self task sees each
       split once; a self task sees (Y,X) and (X,Y), or once if Y == X */
    if (!self) T->twiceSplits += 2;
    else T->twiceSplits += memcmp(y->i, x->i, sizeof y->i) ? 1 : 2;
    int canon = self ? (y->i[3] <= x->i[0]) : (y->i[3] <= x->i[0] || x->i[3] <= y->i[0]);
    if (canon) T->canon++;
    uint16_t t8[8]; int a = 0, b = 0, o = 0;      /* merge two sorted 4-lists */
    while (a < 4 && b < 4) t8[o++] = (y->i[a] <= x->i[b]) ? y->i[a++] : x->i[b++];
    while (a < 4) t8[o++] = y->i[a++];
    while (b < 4) t8[o++] = x->i[b++];
    push_sol(&T->sols, &T->nsols, &T->capSols, t8);
}

static void join(Thr *T, int self) {
    const int hs = logR + SB;
    for (uint32_t j = 0; j < SBN; j++) {
        uint32_t sa = T->offA[j], ea = T->offA[j + 1], sb = T->offB[j], eb = T->offB[j + 1];
        if (sa == ea || sb == eb) continue;
        uint32_t nA = ea - sa, H = 8;
        while (H < 2 * nA) H <<= 1;
        if (H > T->capH) { T->capH = H; T->head = xrealloc(T->head, 4 * (size_t)H); }
        if (nA > T->capNode) {
            T->capNode = nA;
            T->nodeKey = xrealloc(T->nodeKey, 8 * (size_t)nA);
            T->nodeNext = xrealloc(T->nodeNext, 4 * (size_t)nA);
            T->nodeFirst = xrealloc(T->nodeFirst, 4 * (size_t)nA);
            T->entNext = xrealloc(T->entNext, 4 * (size_t)nA);
        }
        int32_t *head = T->head, *nodeNext = T->nodeNext, *nodeFirst = T->nodeFirst, *entNext = T->entNext;
        uint64_t *nodeKey = T->nodeKey;
        for (uint32_t h = 0; h < H; h++) head[h] = -1;
        const Entry *A = T->A + sa;
        int32_t nn = 0;
        for (uint32_t i = 0; i < nA; i++) {
            uint64_t key = A[i].key; uint32_t h = (uint32_t)(key >> hs) & (H - 1);
            int32_t n = head[h];
            while (n >= 0 && nodeKey[n] != key) n = nodeNext[n];
            if (n >= 0) { entNext[i] = nodeFirst[n]; nodeFirst[n] = (int32_t)i; }
            else { n = nn++; nodeKey[n] = key; nodeFirst[n] = (int32_t)i; entNext[i] = -1; nodeNext[n] = head[h]; head[h] = n; }
        }
        const Entry *B = T->B + sb;
        for (uint32_t b = 0; b < eb - sb; b++) {
            uint64_t key = B[b].key; uint32_t h = (uint32_t)(key >> hs) & (H - 1);
            int32_t n = head[h];
            while (n >= 0 && nodeKey[n] != key) n = nodeNext[n];
            if (n >= 0) for (int32_t i = nodeFirst[n]; i >= 0; i = entNext[i]) handle(T, &A[i], &B[b], self);
        }
    }
}

static int progress = 1;
static double tStart;

static void *worker(void *arg) {
    Thr *T = arg;
    uint32_t last = R / 2;
    for (;;) {
        uint32_t t = atomic_fetch_add(&nextTask, 1);
        if (t > last) break;
        uint32_t s = (R - t) & (R - 1);
        int self = (s == t);
        memset(T->cntA, 0, 4 * (size_t)SBN); memset(T->cntB, 0, 4 * (size_t)SBN);
        gen_count(t, 0, T->cntA);
        gen_count(s, 1, T->cntB);
        T->offA[0] = T->offB[0] = 0;
        for (uint32_t j = 0; j < SBN; j++) { T->offA[j + 1] = T->offA[j] + T->cntA[j]; T->offB[j + 1] = T->offB[j] + T->cntB[j]; }
        size_t nA = T->offA[SBN], nB = T->offB[SBN];
        if (nA > T->capA) { T->capA = nA + nA / 8; free(T->A); T->A = xmalloc(sizeof(Entry) * T->capA); }
        if (nB > T->capB) { T->capB = nB + nB / 8; free(T->B); T->B = xmalloc(sizeof(Entry) * T->capB); }
        memcpy(T->posA, T->offA, 4 * (size_t)SBN); memcpy(T->posB, T->offB, 4 * (size_t)SBN);
        gen_fill(t, 0, T->posA, T->A);
        gen_fill(s, 1, T->posB, T->B);
        for (uint32_t j = 0; j < SBN; j++) if (T->posA[j] != T->offA[j + 1] || T->posB[j] != T->offB[j + 1]) die("count/fill mismatch");
        T->quads += nA + (self ? 0 : nB);
        join(T, self);
        uint32_t dn = atomic_fetch_add(&doneTasks, 1) + 1;
        /* progress: at least once a minute (edited for long runs; races only duplicate a line) */
        if (progress) {
            double tn = now();
            if (tn - lastPrint >= 60.0 || dn == last + 1) {
                lastPrint = tn;
                fprintf(stderr, "  [N=%d] %u/%u tasks, %.1f s\n", N, dn, last + 1, tn - tStart);
            }
        }
    }
    return NULL;
}

static int cmp_t8_lex(const void *x, const void *y) {
    const uint16_t *a = x, *b = y;
    for (int i = 0; i < 8; i++) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
static size_t dedupe(uint16_t *arr, size_t n) {
    if (!n) return 0;
    qsort(arr, n, 16, cmp_t8_lex);
    size_t m = 1;
    for (size_t i = 1; i < n; i++) if (cmp_t8_lex(arr + 8 * i, arr + 8 * (m - 1))) memcpy(arr + 8 * m++, arr + 8 * i, 16);
    return m;
}

/* ---------------- brute force (independent check) ---------------- */
static int8_t *sufmin, *sufmax;
static atomic_int bfNext;
typedef struct { uint16_t *sols; size_t nsols, capSols; uint64_t leaves; } BThr;

static int find_vec(const int8_t *v) {
    int lo = 0, hi = M - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, c = vcmp(V + (size_t)mid * D, v);
        if (!c) return mid;
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static void bf_rec(BThr *T, int d, int start, int (*sum)[MAXD], uint16_t *idx) {
    /* d vectors chosen; sum[d] holds their sum */
    if (d == 7) {
        int8_t need[MAXD];
        for (int c = 0; c < D; c++) need[c] = (int8_t)(-sum[7][c]);
        T->leaves++;
        int j = find_vec(need);
        if (j >= start) { idx[7] = (uint16_t)j; push_sol(&T->sols, &T->nsols, &T->capSols, idx); }
        return;
    }
    int r = 7 - d;                /* vectors still to choose after this one */
    for (int i = start; i < M; i++) {
        const int8_t *v = V + (size_t)i * D, *mn = sufmin + (size_t)i * D, *mx = sufmax + (size_t)i * D;
        int ok = 1;
        for (int c = 0; c < D; c++) {
            int s = sum[d][c] + v[c];
            sum[d + 1][c] = s;
            if (s + r * mn[c] > 0 || s + r * mx[c] < 0) { ok = 0; break; }
        }
        if (!ok) continue;
        idx[d] = (uint16_t)i;
        bf_rec(T, d + 1, i, sum, idx);
    }
}

static void *bf_worker(void *arg) {
    BThr *T = arg;
    int sum[8][MAXD]; uint16_t idx[8];
    for (int c = 0; c < MAXD; c++) sum[0][c] = 0;
    for (;;) {
        int i = atomic_fetch_add(&bfNext, 1);
        if (i >= M) break;
        const int8_t *v = V + (size_t)i * D, *mn = sufmin + (size_t)i * D, *mx = sufmax + (size_t)i * D;
        int ok = 1;
        for (int c = 0; c < D; c++) { sum[1][c] = v[c]; if (v[c] + 7 * mn[c] > 0 || v[c] + 7 * mx[c] < 0) ok = 0; }
        if (!ok) continue;
        idx[0] = (uint16_t)i;
        bf_rec(T, 1, i, sum, idx);
    }
    return NULL;
}

/* ---------------- main ---------------- */
static uint64_t binom4(uint64_t n) { /* C(n,4) */
    if (n < 4) return 0;
    return n * (n - 1) / 2 * (n - 2) / 3 * (n - 3) / 4;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s N [-t threads] [-b] [-B] [-s seed] [-L log2_list] [-o file] [-q]\n", argv[0]); return 1; }
    N = atoi(argv[1]);
    int brute = 0, bruteOnly = 0, Ltarget = 21; uint64_t seed = 0x5EEDC0DE12345ULL; const char *outFile = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) NT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b")) brute = 1;
        else if (!strcmp(argv[i], "-B")) brute = bruteOnly = 1;
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "-L") && i + 1 < argc) Ltarget = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outFile = argv[++i];
        else if (!strcmp(argv[i], "-q")) progress = 0;
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    if (N < 4 || N > 18) die("N out of range (4..18)");
    sm_state = seed;
    double t0 = now();
    setup_vectors();
    uint64_t T4 = binom4((uint64_t)M + 3);
    printf("N=%d D=%d codes(c0=c1=+1)=%u distinct_vectors M=%d seed=0x%llx threads=%d\n",
           N, D, 1u << (N - 2), M, (unsigned long long)seed, NT);

    uint16_t *mitmSols = NULL; size_t mitmN = 0;
    if (!bruteOnly) {
        logR = 6;
        while (logR < 20 && (T4 >> logR) > (1ULL << Ltarget)) logR++;
        R = 1u << logR;
        uint64_t avg = T4 >> logR;
        SB = 0;
        while (SB < 14 && (avg >> SB) > 1024) SB++;
        SBN = 1u << SB;
        setup_pairs();
        printf("pairs=%llu 4-multisets=%llu R=2^%d subbuckets=2^%d tasks=%u\n",
               (unsigned long long)P, (unsigned long long)T4, logR, SB, R / 2 + 1);
        fflush(stdout);
        tStart = now(); lastPrint = tStart;
        Thr *th = calloc(NT, sizeof(Thr));
        pthread_t *pt = xmalloc(sizeof(pthread_t) * NT);
        for (int i = 0; i < NT; i++) {
            th[i].cntA = xmalloc(4 * (size_t)SBN); th[i].cntB = xmalloc(4 * (size_t)SBN);
            th[i].posA = xmalloc(4 * (size_t)SBN); th[i].posB = xmalloc(4 * (size_t)SBN);
            th[i].offA = xmalloc(4 * ((size_t)SBN + 1)); th[i].offB = xmalloc(4 * ((size_t)SBN + 1));
            pthread_create(&pt[i], NULL, worker, &th[i]);
        }
        uint64_t quads = 0, km = 0, fp = 0, ex = 0, canon = 0, tw = 0;
        for (int i = 0; i < NT; i++) {
            pthread_join(pt[i], NULL);
            quads += th[i].quads; km += th[i].keymatch; fp += th[i].falsepos; ex += th[i].exact; canon += th[i].canon; tw += th[i].twiceSplits;
            mitmSols = xrealloc(mitmSols, 16 * (mitmN + th[i].nsols + 1));
            memcpy(mitmSols + 8 * mitmN, th[i].sols, 16 * th[i].nsols); mitmN += th[i].nsols;
        }
        size_t uniq = dedupe(mitmSols, mitmN);
        mitmN = uniq;
        double t1 = now();
        printf("4-multisets generated=%llu (expected %llu) %s\n", (unsigned long long)quads,
               (unsigned long long)T4, quads == T4 ? "OK" : "MISMATCH");
        printf("key matches=%llu false positives=%llu exact split matches=%llu canonical=%llu unique solutions=%zu %s\n",
               (unsigned long long)km, (unsigned long long)fp, (unsigned long long)ex, (unsigned long long)canon,
               uniq, (uint64_t)uniq == canon ? "(canonical==unique OK)" : "(CANONICAL/UNIQUE MISMATCH)");
        printf("RESULT N=%d count=%zu search_time=%.2f s total_time=%.2f s\n", N, uniq, t1 - tStart, t1 - t0);
        fflush(stdout);
        /* independent count of unordered 4+4 splits of every solution */
        uint64_t tw2 = 0;
        for (size_t q = 0; q < uniq; q++) {
            const uint16_t *S = mitmSols + 8 * q;
            uint16_t Ys[70][4]; int ny = 0;
            for (int m = 0; m < 256; m++) {
                if (__builtin_popcount(m) != 4) continue;
                uint16_t y[4]; int o = 0;
                for (int b = 0; b < 8; b++) if (m >> b & 1) y[o++] = S[b];
                int dup = 0;
                for (int z = 0; z < ny && !dup; z++) dup = !memcmp(Ys[z], y, sizeof y);
                if (!dup) memcpy(Ys[ny++], y, sizeof y);
            }
            for (int z = 0; z < ny; z++) {   /* Y == complement(Y) adds one more */
                uint16_t x[4]; int o = 0, used[8] = {0};
                for (int b = 0; b < 4; b++) for (int c = 0; c < 8; c++) if (!used[c] && S[c] == Ys[z][b]) { used[c] = 1; break; }
                for (int c = 0; c < 8; c++) if (!used[c]) x[o++] = S[c];
                tw2 += 1 + (memcmp(x, Ys[z], sizeof x) == 0);
            }
        }
        printf("all 4+4 splits found: %llu, expected %llu %s\n", (unsigned long long)tw / 2,
               (unsigned long long)tw2 / 2, tw == tw2 ? "OK" : "MISMATCH");
        if (quads != T4 || (uint64_t)uniq != canon || tw != tw2) die("internal consistency check failed");
    }

    if (brute) {
        sufmin = xmalloc((size_t)M * D); sufmax = xmalloc((size_t)M * D);
        for (int d = 0; d < D; d++) {
            int mn = 127, mx = -128;
            for (int i = M - 1; i >= 0; i--) {
                int v = V[(size_t)i * D + d];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sufmin[(size_t)i * D + d] = (int8_t)mn; sufmax[(size_t)i * D + d] = (int8_t)mx;
            }
        }
        double b0 = now();
        BThr *bt = calloc(NT, sizeof(BThr)); pthread_t *pt = xmalloc(sizeof(pthread_t) * NT);
        for (int i = 0; i < NT; i++) pthread_create(&pt[i], NULL, bf_worker, &bt[i]);
        uint16_t *bs = NULL; size_t bn = 0; uint64_t leaves = 0;
        for (int i = 0; i < NT; i++) {
            pthread_join(pt[i], NULL); leaves += bt[i].leaves;
            bs = xrealloc(bs, 16 * (bn + bt[i].nsols + 1));
            memcpy(bs + 8 * bn, bt[i].sols, 16 * bt[i].nsols); bn += bt[i].nsols;
        }
        size_t bu = dedupe(bs, bn);
        printf("BRUTE N=%d count=%zu (raw %zu, leaves %llu) time=%.2f s\n", N, bu, bn, (unsigned long long)leaves, now() - b0);
        if (!bruteOnly) {
            int same = (bu == mitmN) && (bu == 0 || memcmp(bs, mitmSols, 16 * bu) == 0);
            printf("BRUTE vs MITM solution sets: %s\n", same ? "IDENTICAL" : "DIFFERENT");
        }
        if (bruteOnly) { mitmSols = bs; mitmN = bu; }
    }

    if (outFile) {
        int width = (N + 3) / 4;
        char (*lines)[8 * 6] = xmalloc(sizeof(*lines) * (mitmN + 1));
        for (size_t s = 0; s < mitmN; s++) {
            uint32_t c[8];
            for (int m = 0; m < 8; m++) c[m] = repCode[mitmSols[8 * s + m]];
            for (int a = 1; a < 8; a++) for (int b = a; b > 0 && c[b - 1] > c[b]; b--) { uint32_t tmp = c[b]; c[b] = c[b - 1]; c[b - 1] = tmp; }
            char *p = lines[s];
            for (int m = 0; m < 8; m++) p += sprintf(p, m ? " %0*x" : "%0*x", width, c[m]);
        }
        qsort(lines, mitmN, sizeof(*lines), (int (*)(const void *, const void *))strcmp);
        FILE *f = fopen(outFile, "w");
        if (!f) die("cannot open output file");
        for (size_t s = 0; s < mitmN; s++) fprintf(f, "%s\n", lines[s]);
        fclose(f);
        printf("wrote %zu solutions to %s\n", mitmN, outFile);
    }
    return 0;
}
