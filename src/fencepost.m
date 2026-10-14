/* fencepost - a happens-before race detector for Metal GPU synchronisation.
 *
 *   DYLD_INSERT_LIBRARIES=/path/to/libfencepost.dylib  your-metal-program
 *
 * Metal orders GPU work for you only on resources it tracks. Resources created with
 * MTLHazardTrackingModeUntracked (what every translation layer and most engines use, for
 * speed) are ordered only by the MTLFences and MTLEvents the program encodes. Forget one
 * and two passes overlap on the GPU: a read sees half-written data, and the bug comes and
 * goes with timing. Metal's API validation does not report it.
 *
 * fencepost watches untracked resources: which encoder reads or writes which texture or
 * buffer (render attachments, useResource, bindings, blits), and which fences and events
 * order those encoders. Each encoder is a node; a fence update -> wait pair, an event
 * signal -> wait pair, or a command buffer that completed before another was committed is
 * an edge. Two conflicting accesses (at least one a write) with no path between their
 * encoders are a race. Like ThreadSanitizer, it judges by the synchronisation the program
 * encoded, not by when the GPU happened to run things, so it reports a race even on runs
 * where the race did not show.
 *
 * Render encoders have two stages here, vertex (0) and fragment (1): updateFence:afterStages:
 * and waitForFence:beforeStages: order only the stages they name. Compute and blit encoders
 * are one stage. See README.md for the model, the knobs and the limits. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define FP_W            4096            /* nodes an ordering query can look back */
#define FP_WORDS        (2 * FP_W / 64) /* bit 2d+s = (node d analyses back, stage s) */
#define FP_POOL         16384           /* live nodes (encoders + event signals) */
#define FP_CBPOOL       4096
#define FP_OPENCB       1024
#define FP_OPENENC      256
#define FP_QRECENT      1024
#define FP_MAXQ         32
#define FP_OBJMAP       (1 << 16)
#define FP_FENCEMAP     8192
#define FP_EVMAP        2048
#define FP_EVHIST       64
#define FP_SIGMAP       (1 << 14)
#define FP_ENTCAP       512
#define FP_PEND_TIMEOUT 256             /* commits a node may wait for an event signal */
#define FP_HOOKCLS      8

enum { FP_RD = 1, FP_WR = 2, FP_MAYBE = 4 };    /* FP_MAYBE: declared read+write, may not write */
enum { FPK_RENDER = 1, FPK_COMPUTE, FPK_BLIT, FPK_SIGNAL };
static const char *const kind_name[] = { "?", "render", "compute", "blit", "event signal" };
enum { VIA_ATT_LOAD, VIA_ATT, VIA_RESOLVE, VIA_USE, VIA_BIND, VIA_BLIT_SRC, VIA_BLIT_DST,
       VIA_FILL, VIA_MIPGEN, VIA_INDEX, VIA_INDIRECT, VIA_VISIBILITY, VIA_SYNC };
static const char *const via_name[] = { "attachment load", "attachment", "MSAA resolve", "useResource",
    "binding", "blit source", "blit destination", "fillBuffer", "generateMipmaps", "index buffer",
    "indirect buffer", "visibility result buffer", "synchronizeResource" };

enum { ST_RACE_UNIQ, ST_POSS_UNIQ, ST_RACE_HITS, ST_POSS_HITS, ST_NODES, ST_ACCESSES, ST_CBS,
       ST_HEAP_USES, ST_PARALLEL, ST_TRACKED, ST_TIMEOUTS, ST_UNCERTAIN, ST_WINDOW, ST_LOST,
       ST_ORDER_UNIQ, ST_ORDER_HITS, ST_PARTIAL_UNIQ, ST_PARTIAL_HITS, ST_BARRIERS, ST_COUNT };

typedef struct { uint64_t w[FP_WORDS]; } fp_bits;

typedef struct {
    void *store;            /* root storage: texture (views resolved) or buffer */
    uint64_t a0, a1;        /* texture: mip levels [a0,a1); buffer: bytes [a0,a1) */
    uint32_t s0, s1;        /* texture: slices [s0,s1) */
    uint8_t is_tex, rw, sf, sl, via;   /* sf/sl: first and last stage the access happens in */
} fp_acc;

typedef struct { uint64_t src; uint8_t after, before; } fp_edge;
typedef struct { void *ev; uint64_t val; uint8_t before; } fp_evwait;
typedef struct { void *f; uint64_t src; uint8_t st; } fp_fwait;
typedef struct { void *f; uint8_t st; } fp_fupd;

struct fp_queue;
typedef struct fp_node {
    uint64_t uid, aseq;             /* aseq: analysis order, 0 = not analysed yet */
    uint64_t cb_uid, commit_tick, done_tick, pend_since;
    int kind, state, idx;           /* state: 1 encoding, 2 ended, 3 committed, 4 analysed */
    void *enc;
    uint8_t uncertain, exempt;
    int att[10];                    /* acc index of colour 0-7, depth, stencil attachments */
    char label[64];
    fp_acc *acc;     int nacc, cacc;
    fp_fwait *fw;    int nfw, cfw;
    fp_fupd *fu;     int nfu, cfu;
    fp_edge *ed;     int ned, ced;
    fp_evwait *ew;   int nev, cev;
    struct fp_queue *req_q;         /* declared barrier: must follow req_q's nodes before req_mark */
    uint64_t req_mark, bar_cb;
    int bar_idx;
    fp_bits anc[2];                 /* what is ordered before this node's vertex / fragment work */
} fp_node;

typedef struct { uint8_t type; uint64_t uid; void *ev; uint64_t val; } fp_item;   /* 0 encoder, 1 signal, 2 wait, 3 barrier */
typedef struct {
    uint64_t uid, enq, commit_tick;
    void *cb;
    struct fp_queue *q;
    int state, nenc;
    fp_item *it; int nit, cit;
    char label[48];
} fp_cb;

typedef struct fp_queue {
    void *q;
    int idx;
    uint64_t enq_assign, enq_next;
    uint64_t recent[FP_QRECENT]; uint64_t nrecent;
    uint64_t bar_mark, bar_cb; int bar_idx;   /* the last declared barrier */
    uint64_t *held; int nheld, cheld;
} fp_queue;

typedef struct { fp_acc a; uint64_t uid, aseq; } fp_ent;
typedef struct { void *key; fp_ent *w; int nw, cw; fp_ent *r; int nr, cr; } fp_obj;   /* writes (incl. may-write) and pure reads */
typedef struct { void *ev; int known, gpu; uint64_t cpu_max; struct { uint64_t val, uid; } h[FP_EVHIST]; unsigned nh; } fp_ev;

#define PUSH(arr, n, cap, val) do { if ((n) == (cap)) { (cap) = (cap) ? (cap) * 2 : 8; \
    (arr) = realloc((arr), (size_t)(cap) * sizeof *(arr)); } (arr)[(n)++] = (val); } while (0)

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static fp_node g_pool[FP_POOL];
static fp_cb g_cbs[FP_CBPOOL];
static fp_queue g_q[FP_MAXQ]; static int g_nq;
static struct { void *cb; uint64_t uid; } g_opencb[FP_OPENCB]; static unsigned g_opencb_rr;
static struct { void *enc; uint64_t uid; } g_openenc[FP_OPENENC]; static unsigned g_openenc_rr;
static struct { void *f; uint64_t uid; uint8_t after; } g_fence[FP_FENCEMAP];
static fp_ev g_ev[FP_EVMAP];
static fp_obj g_obj[FP_OBJMAP];
static uint64_t g_sig[FP_SIGMAP];
static uint64_t *g_pend; static int g_npend, g_cpend;
static uint64_t g_uid, g_cbuid, g_aseq, g_tick;
static long g_st[ST_COUNT];
static FILE *g_out;
static long g_max_reports = 200, g_printed;
static int g_show_possible = 1, g_summary_secs = 10, g_started, g_bind_maywrite;
static time_t g_last_summary;
static long g_last_summary_hits = -1;

static __thread int tl_depth;   /* >0 inside a hooked Metal call: nested hooked calls pass through */
static __thread struct { void *enc; uint64_t uid; } tl_cache;

/* ---------------------------------------------------------------- utilities */

static inline unsigned ptr_hash(const void *p, unsigned bits)
{
    return (unsigned)((((uintptr_t)p >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - bits));
}
static inline void bit_set(fp_bits *b, uint64_t i) { if (i < 2 * FP_W) b->w[i >> 6] |= 1ull << (i & 63); }
static inline int bit_test(const fp_bits *b, uint64_t i) { return i < 2 * FP_W && ((b->w[i >> 6] >> (i & 63)) & 1); }
static void bits_or_shift(fp_bits *dst, const fp_bits *src, uint64_t sh)
{
    unsigned ws = (unsigned)(sh >> 6), bs = (unsigned)(sh & 63);
    int i;
    if (ws >= FP_WORDS) return;
    for (i = FP_WORDS - 1; i >= (int)ws; i--) {
        uint64_t v = src->w[i - ws] << bs;
        if (bs && i - (int)ws - 1 >= 0) v |= src->w[i - ws - 1] >> (64 - bs);
        dst->w[i] |= v;
    }
}

static void fp_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void fp_log(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_out ? g_out : stderr, fmt, ap);
    va_end(ap);
    fflush(g_out ? g_out : stderr);
}

static fp_node *node_get(uint64_t uid)
{
    fp_node *n = &g_pool[uid % FP_POOL];
    return uid && n->uid == uid ? n : NULL;
}

static fp_node *node_new(int kind)   /* lock held */
{
    uint64_t uid = ++g_uid;
    fp_node *n = &g_pool[uid % FP_POOL];
    if (n->uid && n->state && n->state < 4) g_st[ST_LOST]++;
    n->uid = uid; n->aseq = 0; n->cb_uid = 0; n->commit_tick = 0; n->done_tick = 0; n->pend_since = 0;
    n->kind = kind; n->state = 1; n->idx = 0; n->enc = NULL; n->uncertain = 0; n->exempt = 0; n->label[0] = 0;
    n->req_q = NULL; n->req_mark = 0; n->bar_cb = 0; n->bar_idx = 0;
    n->nacc = n->nfw = n->nfu = n->ned = n->nev = 0;
    for (int i = 0; i < 10; i++) n->att[i] = -1;
    return n;
}

static fp_cb *cb_get(uint64_t uid)
{
    fp_cb *c = &g_cbs[uid % FP_CBPOOL];
    return uid && c->uid == uid ? c : NULL;
}

static fp_cb *cb_find_open(void *cb)   /* lock held */
{
    for (unsigned i = 0; i < FP_OPENCB; i++)
        if (g_opencb[i].cb == cb) return cb_get(g_opencb[i].uid);
    return NULL;
}

static void cb_close(void *cb)
{
    for (unsigned i = 0; i < FP_OPENCB; i++)
        if (g_opencb[i].cb == cb) { g_opencb[i].cb = NULL; g_opencb[i].uid = 0; }
}

static fp_queue *queue_get(void *q)   /* lock held */
{
    for (int i = 0; i < g_nq; i++) if (g_q[i].q == q) return &g_q[i];
    if (g_nq == FP_MAXQ) return NULL;
    fp_queue *fq = &g_q[g_nq];
    memset(fq, 0, sizeof *fq);
    fq->q = q; fq->idx = g_nq++; fq->enq_next = 1;
    return fq;
}

static fp_ev *ev_get(void *ev, int create)   /* lock held */
{
    unsigned h = ptr_hash(ev, 11);
    for (unsigned i = 0; i < FP_EVMAP; i++) {
        fp_ev *e = &g_ev[(h + i) & (FP_EVMAP - 1)];
        if (e->ev == ev) return e;
        if (!e->ev) {
            if (!create) return NULL;
            memset(e, 0, sizeof *e); e->ev = ev;
            return e;
        }
    }
    return NULL;
}

static int fence_slot(void *f, int create)   /* lock held */
{
    unsigned h = ptr_hash(f, 13);
    for (unsigned i = 0; i < FP_FENCEMAP; i++) {
        unsigned s = (h + i) & (FP_FENCEMAP - 1);
        if (g_fence[s].f == f) return (int)s;
        if (!g_fence[s].f) {
            if (!create) return -1;
            g_fence[s].f = f; g_fence[s].uid = 0;
            return (int)s;
        }
    }
    return -1;
}

static fp_obj *obj_get(void *key)   /* lock held */
{
    unsigned h = ptr_hash(key, 16);
    for (unsigned i = 0; i < FP_OBJMAP; i++) {
        fp_obj *o = &g_obj[(h + i) & (FP_OBJMAP - 1)];
        if (o->key == key) return o;
        if (!o->key) { o->key = key; o->nw = o->nr = 0; return o; }
    }
    return NULL;
}

/* ---------------------------------------------------------------- reporting */

static uint64_t fnv(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = p;
    while (n--) { h ^= *b++; h *= 0x100000001b3ull; }
    return h;
}

static int sig_seen(uint64_t s)   /* lock held; returns 1 if already reported */
{
    if (!s) s = 1;
    unsigned h = (unsigned)(s >> 50);
    for (unsigned i = 0; i < FP_SIGMAP; i++) {
        uint64_t *slot = &g_sig[(h + i) & (FP_SIGMAP - 1)];
        if (*slot == s) return 1;
        if (!*slot) { *slot = s; return 0; }
    }
    return 1;
}

static const char *stage_name(const fp_node *n, int s)
{
    if (n->kind != FPK_RENDER) return "";
    return s ? ", fragment stage" : ", vertex stage";
}

static void describe_node(char *buf, size_t len, const fp_node *n)
{
    const fp_cb *c = cb_get(n->cb_uid);
    snprintf(buf, len, "%s encoder #%llu%s%s%s (command buffer #%llu%s%s%s, encoder %d, queue %d)",
             kind_name[n->kind], (unsigned long long)n->uid,
             n->label[0] ? " \"" : "", n->label, n->label[0] ? "\"" : "",
             (unsigned long long)n->cb_uid, c && c->label[0] ? " \"" : "", c ? c->label : "",
             c && c->label[0] ? "\"" : "", n->idx, c && c->q ? c->q->idx : -1);
}

static void describe_store(char *buf, size_t len, const fp_acc *a, int safe)
{
    char what[160] = "";
    if (a->is_tex) {
        if (safe) {
            id<MTLTexture> t = (id<MTLTexture>)a->store;
            const char *l = t.label ? t.label.UTF8String : NULL;
            snprintf(what, sizeof what, "texture %s%s%s%lux%lu, pixel format %lu, %lu levels",
                     l ? "\"" : "", l ? l : "", l ? "\" " : "", (unsigned long)t.width, (unsigned long)t.height,
                     (unsigned long)t.pixelFormat, (unsigned long)t.mipmapLevelCount);
        } else snprintf(what, sizeof what, "texture");
        char lv[48], sl[48];
        if (a->a1 - a->a0 > 1) snprintf(lv, sizeof lv, "levels %llu-%llu", (unsigned long long)a->a0, (unsigned long long)(a->a1 - 1));
        else snprintf(lv, sizeof lv, "level %llu", (unsigned long long)a->a0);
        if (a->s1 - a->s0 > 1) snprintf(sl, sizeof sl, "slices %u-%u", a->s0, a->s1 - 1);
        else snprintf(sl, sizeof sl, "slice %u", a->s0);
        snprintf(buf, len, "%s %p, %s, %s", what, a->store, lv, sl);
    } else {
        if (safe) {
            id<MTLBuffer> b = (id<MTLBuffer>)a->store;
            const char *l = b.label ? b.label.UTF8String : NULL;
            snprintf(what, sizeof what, "buffer %s%s%s%lu bytes", l ? "\"" : "", l ? l : "", l ? "\" " : "",
                     (unsigned long)b.length);
        } else snprintf(what, sizeof what, "buffer");
        snprintf(buf, len, "%s %p, bytes [%llu, %llu)", what, a->store,
                 (unsigned long long)a->a0, (unsigned long long)a->a1);
    }
}

static const char *rw_verb(int rw)
{
    if (rw & FP_WR) return (rw & FP_RD) ? "reads and writes" : "writes";
    if (rw & FP_MAYBE) return "reads and may write";
    return "reads";
}

/* x's access e and y's access a conflict and nothing orders them. "first" is the one
 * committed first; the race is symmetric, the naming is for the reader. */
static void report(fp_node *x, const fp_acc *xa, fp_node *y, const fp_acc *ya)
{
    fp_node *f = x, *s = y;
    const fp_acc *fa = xa, *sa = ya;
    if (x->commit_tick > y->commit_tick) { f = y; s = x; fa = ya; sa = xa; }
    int fw = fa->rw & FP_WR, sw = sa->rw & FP_WR;
    int certain = (fw || sw) && !x->uncertain && !y->uncertain;
    const char *what = fw && sw ? "write-after-write" : fw ? "read-after-write" : sw ? "write-after-read"
                     : "write conflict (both declared read+write)";

    uint64_t h = 0xcbf29ce484222325ull;
    h = fnv(h, &certain, sizeof certain);
    h = fnv(h, what, strlen(what));
    h = fnv(h, &ya->store, sizeof ya->store);
    h = fnv(h, &f->kind, sizeof f->kind); h = fnv(h, &fa->via, 1); h = fnv(h, f->label, strlen(f->label));
    h = fnv(h, &s->kind, sizeof s->kind); h = fnv(h, &sa->via, 1); h = fnv(h, s->label, strlen(s->label));
    g_st[certain ? ST_RACE_HITS : ST_POSS_HITS]++;
    if (sig_seen(h)) return;
    g_st[certain ? ST_RACE_UNIQ : ST_POSS_UNIQ]++;
    if (!certain && !g_show_possible) return;
    if (g_printed >= g_max_reports) {
        if (g_printed++ == g_max_reports) fp_log("fencepost: report limit reached (FENCEPOST_MAX_REPORTS); counting only\n");
        return;
    }
    g_printed++;

    char d1[256], d2[256], st[320];
    describe_node(d1, sizeof d1, f);
    describe_node(d2, sizeof d2, s);
    describe_store(st, sizeof st, ya, y->done_tick == 0);
    fp_log("fencepost: %s %s on %s\n", certain ? "RACE" : "possible race", what, st);
    fp_log("  first : %s %s it: %s%s\n", d1, rw_verb(fa->rw), via_name[fa->via], stage_name(f, fa->sl));
    fp_log("  second: %s %s it: %s%s\n", d2, rw_verb(sa->rw), via_name[sa->via], stage_name(s, sa->sf));

    /* why: look for the fence that was meant to order them */
    int said = 0;
    for (int i = 0; i < s->nfw && !said; i++)
        for (int j = 0; j < f->nfu && !said; j++) {
            if (s->fw[i].f != f->fu[j].f) continue;
            said = 1;
            if (s->fw[i].src && s->fw[i].src != f->uid)
                fp_log("  why   : the second waits on fence %p, but encoder #%llu updated that fence after the first "
                       "did, so the wait pairs with the later update (fence reused too early)\n",
                       s->fw[i].f, (unsigned long long)s->fw[i].src);
            else if (!s->fw[i].src)
                fp_log("  why   : the second waits on fence %p, which the first updates, but the wait was submitted "
                       "first (the update must be submitted before the wait)\n", s->fw[i].f);
            else
                fp_log("  why   : fence %p is updated %s and waited for %s, which does not cover these accesses\n",
                       s->fw[i].f, f->kind != FPK_RENDER ? "at the end of the first" :
                       f->fu[j].st ? "after the first's fragment stage" : "after the first's vertex stage only",
                       s->kind != FPK_RENDER ? "before the second starts" :
                       s->fw[i].st ? "before the second's fragment stage only" : "before the second's vertex stage");
        }
    if (!said) {
        if (f->cb_uid == s->cb_uid)
            fp_log("  why   : both are in command buffer #%llu and no fence orders them; untracked resources need "
                   "updateFence in the first and waitForFence in the second\n", (unsigned long long)f->cb_uid);
        else if (!f->done_tick || f->done_tick >= s->commit_tick)
            fp_log("  why   : the first's command buffer had not completed when the second's was committed, and "
                   "no fence or event orders them\n");
        else
            fp_log("  why   : no fence or event orders them\n");
        if (s->nfw) fp_log("          (the second waits on %d fence(s), none updated by the first or after it)\n", s->nfw);
    }
    if (x->uncertain || y->uncertain)
        fp_log("  note  : one side waited on an event fencepost never saw signalled; the ordering may exist "
               "outside its view\n");
}

static void summary(int final)
{
    pthread_mutex_lock(&g_lock);
    long hits = g_st[ST_RACE_HITS] + g_st[ST_POSS_HITS];
    if (!final && hits == g_last_summary_hits) { pthread_mutex_unlock(&g_lock); return; }
    g_last_summary_hits = hits;
    fp_log("fencepost: %s: %ld race(s) (%ld hits), %ld possible (%ld hits); %ld encoders analysed, "
           "%ld command buffers, %ld accesses\n", final ? "final" : "so far",
           g_st[ST_RACE_UNIQ], g_st[ST_RACE_HITS], g_st[ST_POSS_UNIQ], g_st[ST_POSS_HITS],
           g_st[ST_NODES], g_st[ST_CBS], g_st[ST_ACCESSES]);
    if (g_st[ST_BARRIERS])
        fp_log("fencepost:   %ld declared barriers: %ld encoder(s) not ordered after them (%ld hits), %ld ordered for the "
               "fragment stage only (%ld hits)\n", g_st[ST_BARRIERS], g_st[ST_ORDER_UNIQ], g_st[ST_ORDER_HITS],
               g_st[ST_PARTIAL_UNIQ], g_st[ST_PARTIAL_HITS]);
    if (g_st[ST_TRACKED])
        fp_log("fencepost:   %ld uses of tracked resources skipped (Metal orders those itself)\n", g_st[ST_TRACKED]);
    if (g_st[ST_HEAP_USES] || g_st[ST_PARALLEL] || g_st[ST_TIMEOUTS] || g_st[ST_UNCERTAIN] || g_st[ST_WINDOW] || g_st[ST_LOST])
        fp_log("fencepost:   blind spots: %ld useHeap calls (resources inside heaps not seen), %ld parallel render "
               "encoders (not followed), %ld event waits timed out, %ld nodes with unresolved event waits, "
               "%ld checks past the %d-encoder window, %ld nodes dropped\n",
               g_st[ST_HEAP_USES], g_st[ST_PARALLEL], g_st[ST_TIMEOUTS], g_st[ST_UNCERTAIN], g_st[ST_WINDOW],
               FP_W, g_st[ST_LOST]);
    pthread_mutex_unlock(&g_lock);
}

/* ---------------------------------------------------------------- analysis */

static int overlap(const fp_acc *a, const fp_acc *b)
{
    if (a->a0 >= b->a1 || b->a0 >= a->a1) return 0;
    if (a->is_tex && (a->s0 >= b->s1 || b->s0 >= a->s1)) return 0;
    return 1;
}

static int covers(const fp_acc *a, const fp_acc *b)   /* a's range contains b's */
{
    return a->a0 <= b->a0 && a->a1 >= b->a1 && (!a->is_tex || (a->s0 <= b->s0 && a->s1 >= b->s1));
}

static int ordered(const fp_ent *e, fp_node *x, const fp_node *y, int sf)
{
    if (!x) return 1;
    if (x->done_tick && x->done_tick < y->commit_tick) return 1;   /* completed before y was committed */
    uint64_t d = y->aseq - e->aseq;
    if (d >= FP_W) { g_st[ST_WINDOW]++; return 1; }
    return bit_test(&y->anc[sf], 2 * d + e->a.sl);
}

static int same_range(const fp_acc *a, const fp_acc *b)
{
    return a->a0 == b->a0 && a->a1 == b->a1 && a->s0 == b->s0 && a->s1 == b->s1;
}

/* Check a against one shadow list, dropping entries that no longer matter (gone, retired,
 * out of the window, or superseded by an ordered write that covers them). */
static void scan(fp_ent *list, int *n, fp_node *y, const fp_acc *a, int *merged)
{
    int i, j;
    for (i = 0, j = 0; i < *n; i++) {
        fp_ent *e = &list[i];
        fp_node *x = node_get(e->uid);
        int keep = 1;
        if (!x || (x->done_tick && x->done_tick < y->commit_tick) || y->aseq - e->aseq >= FP_W) keep = 0;
        else if (overlap(&e->a, a)) {
            if (e->uid == y->uid) {
                if (merged && same_range(&e->a, a)) {
                    e->a.rw |= a->rw;
                    if (a->sl > e->a.sl) e->a.sl = a->sl;
                    *merged = 1;
                }
            } else if ((e->a.rw | a->rw) & (FP_WR | FP_MAYBE)) {
                if (!ordered(e, x, y, a->sf)) report(x, &e->a, y, a);
                else if ((a->rw & FP_WR) && covers(a, &e->a)) keep = 0;   /* superseded by an ordered write */
            }
        }
        if (keep) list[j++] = *e;
    }
    *n = j;
}

/* A read is checked against earlier writes only; a write against reads and writes. Most
 * accesses are reads, so they never walk the (long) list of earlier reads. */
static void check_access(fp_node *y, const fp_acc *a)
{
    fp_obj *o = obj_get(a->store);
    int merged = 0;
    if (!o) return;
    int writes = a->rw & (FP_WR | FP_MAYBE);
    scan(o->w, &o->nw, y, a, writes ? &merged : NULL);
    if (writes) scan(o->r, &o->nr, y, a, NULL);
    else if (o->nr && o->r[o->nr - 1].uid == y->uid && same_range(&o->r[o->nr - 1].a, a)) merged = 1;
    if (merged) return;
    fp_ent **list = writes ? &o->w : &o->r;
    int *n = writes ? &o->nw : &o->nr, *cap = writes ? &o->cw : &o->cr;
    if (*n >= FP_ENTCAP) {
        memmove(*list, *list + FP_ENTCAP / 2, (size_t)(*n - FP_ENTCAP / 2) * sizeof **list);
        *n -= FP_ENTCAP / 2;
    }
    fp_ent ne = { *a, y->uid, y->aseq };
    PUSH(*list, *n, *cap, ne);
}

/* A barrier the program declared (fencepost_barrier): every node of the queue submitted
 * before it must be ordered before y. Checks the synchronisation a runtime encoded against
 * the ordering it meant, whether or not fencepost can see the resources involved. */
static void check_barrier(fp_node *y)
{
    fp_queue *q = y->req_q;
    uint64_t hi = y->req_mark, lo = hi > FP_QRECENT ? hi - FP_QRECENT : 0;
    if (q->nrecent > FP_QRECENT && lo < q->nrecent - FP_QRECENT) lo = q->nrecent - FP_QRECENT;
    fp_node *bad = NULL, *part = NULL;
    long nbad = 0;
    for (uint64_t k = hi; k-- > lo;) {
        fp_node *x = node_get(q->recent[k % FP_QRECENT]);
        if (!x || x == y || x->kind == FPK_SIGNAL || x->exempt || !x->aseq || x->aseq >= y->aseq) continue;
        if (x->done_tick && x->done_tick < y->commit_tick) continue;
        uint64_t d = y->aseq - x->aseq;
        if (d >= FP_W) continue;
        if (bit_test(&y->anc[0], 2 * d + 1)) continue;
        if (bit_test(&y->anc[1], 2 * d + 1)) { if (!part) part = x; continue; }
        if (!bad) bad = x;
        nbad++;
    }
    fp_node *x = bad ? bad : part;
    if (!x) return;
    uint64_t h = 0x84222325cbf29ce4ull;
    h = fnv(h, &y->kind, sizeof y->kind); h = fnv(h, y->label, strlen(y->label));
    h = fnv(h, &x->kind, sizeof x->kind); h = fnv(h, x->label, strlen(x->label));
    h ^= bad ? 0x5bd1e995ull : 0;
    g_st[bad ? ST_ORDER_HITS : ST_PARTIAL_HITS]++;
    if (sig_seen(h)) return;
    g_st[bad ? ST_ORDER_UNIQ : ST_PARTIAL_UNIQ]++;
    if (g_printed >= g_max_reports) return;
    g_printed++;
    char d1[256], d2[256];
    describe_node(d1, sizeof d1, x);
    describe_node(d2, sizeof d2, y);
    if (bad) fp_log("fencepost: ORDER a declared barrier is not honoured\n");
    else fp_log("fencepost: ORDER a declared barrier is honoured for the fragment stage only\n");
    fp_log("  barrier: command buffer #%llu, after its encoder %d\n", (unsigned long long)y->bar_cb, y->bar_idx);
    fp_log("  before : %s\n", d1);
    if (bad) fp_log("  after  : %s is not ordered after it%s\n", d2, nbad > 1 ? " (nor after other encoders from before the barrier)" : "");
    else fp_log("  after  : %s waits for it only before its fragment stage; its vertex stage can overlap\n", d2);
    for (int i = 0; i < y->nfw; i++) {
        fp_node *u = node_get(y->fw[i].src);
        fp_log("  waits  : fence %p before its %s, last updated by %s #%llu\n", y->fw[i].f,
               y->kind != FPK_RENDER ? "start" : y->fw[i].st ? "fragment stage" : "vertex stage",
               u ? kind_name[u->kind] : "an old or unknown encoder", (unsigned long long)y->fw[i].src);
    }
    if (!y->nfw && !y->ned) fp_log("  waits  : nothing\n");
}

static void analyse(fp_node *y)   /* lock held; every edge source is analysed or gone */
{
    y->aseq = ++g_aseq;
    memset(y->anc, 0, sizeof y->anc);
    for (int i = 0; i < y->ned; i++) {
        const fp_edge *e = &y->ed[i];
        fp_node *x = node_get(e->src);
        if (!x || !x->aseq || x->aseq >= y->aseq) continue;
        uint64_t d = y->aseq - x->aseq;
        if (d >= FP_W) continue;
        fp_bits *t = &y->anc[e->before];
        bit_set(t, 2 * d);
        if (e->after) bit_set(t, 2 * d + 1);
        bits_or_shift(t, &x->anc[e->after], 2 * d);
    }
    for (int i = 0; i < FP_WORDS; i++) y->anc[1].w[i] |= y->anc[0].w[i];   /* fragment work starts after vertex work */
    y->state = 4;
    if (y->kind != FPK_SIGNAL) g_st[ST_NODES]++;
    if (y->kind != FPK_SIGNAL && y->req_q && y->req_mark) check_barrier(y);
    for (int i = 0; i < y->nacc; i++) check_access(y, &y->acc[i]);
}

/* force 0: wait for event signals and unanalysed sources; 1: resolve what is left */
static int ready(fp_node *n, int force)
{
    int i, j;
    for (i = 0, j = 0; i < n->nev; i++) {
        fp_evwait w = n->ew[i];
        fp_ev *ev = ev_get(w.ev, 0);
        int done = 0;
        if (!ev || !ev->known) done = 1;   /* event made outside our view (e.g. from a mach port): CPU-side */
        else {
            unsigned first = ev->nh > FP_EVHIST ? ev->nh - FP_EVHIST : 0;
            for (unsigned k = first; k < ev->nh; k++)
                if (ev->h[k % FP_EVHIST].val >= w.val) {
                    fp_edge e = { ev->h[k % FP_EVHIST].uid, 1, w.before };
                    PUSH(n->ed, n->ned, n->ced, e);
                    done = 1;
                    break;
                }
            if (!done && ev->cpu_max >= w.val) done = 1;
            if (!done && (force || g_tick - n->pend_since > FP_PEND_TIMEOUT)) {
                if (ev->gpu) { n->uncertain = 1; g_st[ST_UNCERTAIN]++; }
                if (!force) g_st[ST_TIMEOUTS]++;
                done = 1;
            }
        }
        if (!done) n->ew[j++] = w;
    }
    n->nev = j;
    if (n->nev) return 0;
    if (force || g_tick - n->pend_since > FP_PEND_TIMEOUT) return 1;
    for (i = 0; i < n->ned; i++) {
        fp_node *x = node_get(n->ed[i].src);
        if (x && !x->aseq) return 0;
    }
    if (n->req_q && n->req_mark) {   /* a declared barrier is judged once what it follows is analysed */
        fp_queue *q = n->req_q;
        uint64_t hi = n->req_mark, lo = hi > FP_QRECENT ? hi - FP_QRECENT : 0;
        if (q->nrecent > FP_QRECENT && lo < q->nrecent - FP_QRECENT) lo = q->nrecent - FP_QRECENT;
        for (uint64_t k = hi; k-- > lo;) {
            fp_node *x = node_get(q->recent[k % FP_QRECENT]);
            if (x && x != n && x->state == 3) return 0;
        }
    }
    return 1;
}

static void process_pending(int force)   /* lock held */
{
    for (int pass = 0; pass < (force ? 2 : 1); pass++) {
        int progress = 1;
        while (progress) {
            progress = 0;
            for (int i = 0; i < g_npend;) {
                fp_node *n = node_get(g_pend[i]);
                if (!n || n->state != 3 || ready(n, 0)) {
                    if (n && n->state == 3) { analyse(n); progress = 1; }
                    memmove(g_pend + i, g_pend + i + 1, (size_t)(g_npend - i - 1) * sizeof *g_pend);
                    g_npend--;
                    continue;
                }
                i++;
            }
        }
        if (force && pass == 0)   /* settle event waits, then retry in order */
            for (int i = 0; i < g_npend; i++) { fp_node *n = node_get(g_pend[i]); if (n) (void)ready(n, 1); }
    }
    if (force) {   /* whatever still waits on a node that never got analysed */
        for (int i = 0; i < g_npend; i++) { fp_node *n = node_get(g_pend[i]); if (n && n->state == 3) analyse(n); }
        g_npend = 0;
    }
}

static void queue_recent(fp_queue *q, uint64_t uid)
{
    q->recent[q->nrecent++ % FP_QRECENT] = uid;
}

/* Phase 1, in submission order: resolve fence waits against the latest update, record
 * fence updates and event signals, queue the nodes for analysis. */
static void phase1(fp_cb *c)
{
    fp_evwait active[32]; int nactive = 0;
    fp_queue *q = c->q;
    g_st[ST_CBS]++;
    for (int i = 0; i < c->nit; i++) {
        fp_item *it = &c->it[i];
        if (it->type == 3) {
            q->bar_mark = q->nrecent; q->bar_cb = c->uid; q->bar_idx = it->val > 0 ? (int)it->val : 0;
            g_st[ST_BARRIERS]++;
            continue;
        }
        if (it->type == 2) {
            if (nactive < 32) { active[nactive].ev = it->ev; active[nactive].val = it->val; active[nactive].before = 0; nactive++; }
            continue;
        }
        fp_node *n;
        if (it->type == 1) {
            n = node_new(FPK_SIGNAL);
            it->uid = n->uid;
            n->cb_uid = c->uid; n->idx = 0; n->state = 2;
            unsigned first = q->nrecent > FP_QRECENT ? q->nrecent - FP_QRECENT : 0;
            for (unsigned k = first; k < q->nrecent; k++) {   /* the signal follows earlier work on this queue */
                fp_node *p = node_get(q->recent[k % FP_QRECENT]);
                if (!p || (p->done_tick && p->done_tick < c->commit_tick)) continue;
                fp_edge e = { p->uid, 1, 0 };
                PUSH(n->ed, n->ned, n->ced, e);
            }
            fp_ev *ev = ev_get(it->ev, 1);
            if (ev) { ev->gpu = 1; ev->h[ev->nh % FP_EVHIST].val = it->val; ev->h[ev->nh % FP_EVHIST].uid = n->uid; ev->nh++; }
        } else {
            n = node_get(it->uid);
            if (!n) continue;
            for (int k = 0; k < n->nfw; k++) {
                int s = fence_slot(n->fw[k].f, 0);
                n->fw[k].src = s >= 0 ? g_fence[s].uid : 0;
                if (s >= 0 && g_fence[s].uid) {
                    fp_edge e = { g_fence[s].uid, g_fence[s].after, n->fw[k].st };
                    PUSH(n->ed, n->ned, n->ced, e);
                }
            }
            for (int k = 0; k < n->nfu; k++) {
                int s = fence_slot(n->fu[k].f, 1);
                if (s >= 0) { g_fence[s].uid = n->uid; g_fence[s].after = n->fu[k].st; }
            }
        }
        for (int k = 0; k < nactive; k++) PUSH(n->ew, n->nev, n->cev, active[k]);
        n->commit_tick = c->commit_tick;
        n->pend_since = g_tick;
        n->req_q = q; n->req_mark = q->bar_mark; n->bar_cb = q->bar_cb; n->bar_idx = q->bar_idx;
        n->state = 3;
        queue_recent(q, n->uid);
        PUSH(g_pend, g_npend, g_cpend, n->uid);
    }
    c->state = 3;
}

static void queue_submit(fp_queue *q, fp_cb *c)   /* in enqueue order */
{
    PUSH(q->held, q->nheld, q->cheld, c->uid);
    for (;;) {
        int best = -1;
        for (int i = 0; i < q->nheld; i++) {
            fp_cb *h = cb_get(q->held[i]);
            if (!h) { q->held[i--] = q->held[--q->nheld]; continue; }
            if (h->enq == q->enq_next || (q->nheld > 256 && (best < 0 || h->enq < cb_get(q->held[best])->enq))) best = i;
            if (h->enq == q->enq_next) break;
        }
        if (best < 0) break;
        fp_cb *h = cb_get(q->held[best]);
        q->held[best] = q->held[--q->nheld];
        q->enq_next = h->enq + 1;
        phase1(h);
    }
}

static void cb_done(uint64_t uid)
{
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_get(uid);
    if (c)
        for (int i = 0; i < c->nit; i++) {
            fp_node *n = c->it[i].type != 2 ? node_get(c->it[i].uid) : NULL;
            if (n && !n->done_tick) n->done_tick = g_tick;
        }
    pthread_mutex_unlock(&g_lock);
}

/* ---------------------------------------------------------------- recording */

static fp_node *cur_node(id enc)
{
    if (tl_cache.enc == (void *)enc) {
        fp_node *n = node_get(tl_cache.uid);
        if (n && n->enc == (void *)enc && n->state == 1) return n;
    }
    uint64_t uid = 0;
    pthread_mutex_lock(&g_lock);
    for (unsigned i = 0; i < FP_OPENENC; i++) if (g_openenc[i].enc == (void *)enc) { uid = g_openenc[i].uid; break; }
    pthread_mutex_unlock(&g_lock);
    fp_node *n = node_get(uid);
    if (!n || n->state != 1 || n->enc != (void *)enc) return NULL;
    tl_cache.enc = (void *)enc; tl_cache.uid = uid;
    return n;
}

static void push_acc(fp_node *n, const fp_acc *a)
{
    for (int i = n->nacc - 1; i >= 0 && i >= n->nacc - 32; i--) {
        fp_acc *p = &n->acc[i];
        if (p->store == a->store && p->a0 == a->a0 && p->a1 == a->a1 && p->s0 == a->s0 && p->s1 == a->s1 && p->via == a->via) {
            p->rw |= a->rw;
            if (a->sf < p->sf) p->sf = a->sf;
            if (a->sl > p->sl) p->sl = a->sl;
            return;
        }
    }
    PUSH(n->acc, n->nacc, n->cacc, *a);
    __atomic_fetch_add(&g_st[ST_ACCESSES], 1, __ATOMIC_RELAXED);
}

static struct { Class c; int k; } g_ck[64]; static atomic_int g_nck;
static int res_kind(id r)   /* 1 texture, 2 buffer, 0 other */
{
    Class c = object_getClass(r);
    int n = atomic_load_explicit(&g_nck, memory_order_acquire);
    for (int i = 0; i < n; i++) if (g_ck[i].c == c) return g_ck[i].k;
    int k = [r respondsToSelector:@selector(pixelFormat)] ? 1 : [r respondsToSelector:@selector(contents)] ? 2 : 0;
    pthread_mutex_lock(&g_lock);
    n = atomic_load(&g_nck);
    if (n < 64) { g_ck[n].c = c; g_ck[n].k = k; atomic_store_explicit(&g_nck, n + 1, memory_order_release); }
    pthread_mutex_unlock(&g_lock);
    return k;
}

static uint32_t tex_slices(id<MTLTexture> t)
{
    NSUInteger n = t.arrayLength;
    if (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray) n *= 6;
    return n ? (uint32_t)n : 1;
}

static int acc_buf(fp_node *n, id<MTLBuffer> b, uint64_t off, uint64_t end, int rw, int sf, int sl, int via)
{
    if (!b || !n) return -1;
    if (b.hazardTrackingMode != MTLHazardTrackingModeUntracked) { __atomic_fetch_add(&g_st[ST_TRACKED], 1, __ATOMIC_RELAXED); return -1; }
    uint64_t len = b.length;
    if (end > len) end = len;
    if (off >= end) return -1;
    fp_acc a = { (void *)b, off, end, 0, 0, 0, (uint8_t)rw, (uint8_t)sf, (uint8_t)sl, (uint8_t)via };
    push_acc(n, &a);
    return n->nacc - 1;
}

/* level/slice ranges are in t's own terms; views are mapped onto their storage */
static int acc_tex(fp_node *n, id<MTLTexture> t, uint64_t level, uint64_t nlev, uint32_t slice, uint32_t nsl,
                   int rw, int sf, int sl, int via)
{
    if (!t || !n) return -1;
    while (t.parentTexture) {
        level += t.parentRelativeLevel;
        slice += (uint32_t)t.parentRelativeSlice;
        t = t.parentTexture;
    }
    if (t.buffer) return acc_buf(n, t.buffer, t.bufferOffset, UINT64_MAX, rw, sf, sl, via);
    if (t.hazardTrackingMode != MTLHazardTrackingModeUntracked) { __atomic_fetch_add(&g_st[ST_TRACKED], 1, __ATOMIC_RELAXED); return -1; }
    fp_acc a = { (void *)t, level, level + (nlev ? nlev : 1), slice, slice + (nsl ? nsl : 1), 1,
                 (uint8_t)rw, (uint8_t)sf, (uint8_t)sl, (uint8_t)via };
    push_acc(n, &a);
    return n->nacc - 1;
}

static void acc_res(fp_node *n, id r, int rw, int sf, int sl, int via)   /* whole resource */
{
    if (!r || !n) return;
    int k = res_kind(r);
    if (k == 1) { id<MTLTexture> t = r; acc_tex(n, t, 0, t.mipmapLevelCount, 0, tex_slices(t), rw, sf, sl, via); }
    else if (k == 2) acc_buf(n, r, 0, UINT64_MAX, rw, sf, sl, via);
}

static int usage_rw(NSUInteger u)
{
    if (u & MTLResourceUsageWrite) return (u & ~(NSUInteger)MTLResourceUsageWrite) ? (FP_RD | FP_MAYBE) : FP_WR;
    return FP_RD;
}
static int stages_first(NSUInteger st)
{
    if (!st) return 0;
    return (st & (MTLRenderStageVertex | MTLRenderStageObject | MTLRenderStageMesh)) ? 0 : 1;
}
static int stages_last(NSUInteger st)
{
    if (!st) return 1;
    return (st & (MTLRenderStageFragment | MTLRenderStageTile)) ? 1 : 0;
}

static void att(fp_node *n, int slot, MTLRenderPassAttachmentDescriptor *d, NSUInteger layers)
{
    id<MTLTexture> t = d.texture;
    if (!t) return;
    MTLLoadAction la = d.loadAction;
    MTLStoreAction sa = d.storeAction;
    int rw = (la == MTLLoadActionLoad ? FP_RD : 0) | ((la == MTLLoadActionLoad && sa == MTLStoreActionDontCare) ? 0 : FP_WR);
    uint32_t slice = t.textureType == MTLTextureType3D ? 0 : (uint32_t)d.slice;
    n->att[slot] = acc_tex(n, t, d.level, 1, slice, layers > 1 ? (uint32_t)layers : 1, rw, 1, 1,
                           la == MTLLoadActionLoad ? VIA_ATT_LOAD : VIA_ATT);
    if (d.resolveTexture && (sa == MTLStoreActionMultisampleResolve || sa == MTLStoreActionStoreAndMultisampleResolve))
        acc_tex(n, d.resolveTexture, d.resolveLevel, 1, (uint32_t)d.resolveSlice, 1, FP_WR, 1, 1, VIA_RESOLVE);
}

static void store_action(id enc, int slot, MTLStoreAction sa)   /* a deferred store action arrives */
{
    fp_node *n = cur_node(enc);
    if (!n || n->att[slot] < 0 || n->att[slot] >= n->nacc) return;
    fp_acc *a = &n->acc[n->att[slot]];
    if (a->via == VIA_ATT_LOAD) a->rw = sa == MTLStoreActionDontCare ? FP_RD : (FP_RD | FP_WR);
}

/* ---------------------------------------------------------------- hooking */

enum {
    H_DEV_NEWQ, H_DEV_NEWQ_MAX, H_DEV_NEWQ_DESC, H_DEV_NEWSEV, H_DEV_NEWEV,
    H_Q_CB, H_Q_CBU, H_Q_CBD,
    H_CB_RENC, H_CB_PRENC, H_CB_CENC, H_CB_CENC_DT, H_CB_CENC_D, H_CB_BENC, H_CB_BENC_D,
    H_CB_SIG, H_CB_WAIT, H_CB_COMMIT, H_CB_ENQ,
    H_EV_SET,
    H_END, H_USE, H_USES, H_HEAP, H_HEAPS, H_UPD, H_WAIT,
    H_R_USE_ST, H_R_USES_ST, H_R_HEAP_ST, H_R_HEAPS_ST, H_R_UPD, H_R_WAIT,
    H_R_VTEX, H_R_FTEX, H_R_VTEXS, H_R_FTEXS, H_R_VBUF, H_R_FBUF, H_R_OBUF, H_R_MBUF, H_R_VBUFS, H_R_FBUFS,
    H_R_DIDX, H_R_DIDX_I, H_R_DIDX_IBB, H_R_DIND, H_R_DIDXIND, H_R_MESHIND,
    H_R_CSA, H_R_DSA, H_R_SSA,
    H_C_TEX, H_C_TEXS, H_C_BUF, H_C_BUFS, H_C_DISPIND,
    H_B_BB, H_B_BT, H_B_BT_O, H_B_TB, H_B_TB_O, H_B_TT, H_B_TT_N, H_B_TT_ALL, H_B_FILL, H_B_MIP, H_B_SYNCR, H_B_SYNCT,
    H_COUNT
};

static struct { Class c; IMP imp; } g_orig[H_COUNT][FP_HOOKCLS];
static int g_norig[H_COUNT];

static IMP orig_imp(int h, id self)
{
    for (Class c = object_getClass(self); c; c = class_getSuperclass(c))
        for (int i = 0; i < g_norig[h]; i++)
            if (g_orig[h][i].c == c) return g_orig[h][i].imp;
    fp_log("fencepost: internal error: no original for hook %d on %s\n", h, class_getName(object_getClass(self)));
    abort();
}

static void hook(Class cls, int h, const char *selname, IMP repl)   /* g_lock held */
{
    SEL sel = sel_registerName(selname);
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) return;
    IMP cur = method_getImplementation(m);
    if (cur == repl) return;   /* inherited from a class already hooked */
    if (g_norig[h] == FP_HOOKCLS) { fp_log("fencepost: too many classes for %s\n", selname); return; }
    g_orig[h][g_norig[h]].c = cls;
    g_orig[h][g_norig[h]].imp = cur;
    g_norig[h]++;
    if (!class_addMethod(cls, sel, repl, method_getTypeEncoding(m))) method_setImplementation(m, repl);
}

#define ORIG(h, T) ((T)orig_imp((h), self))
#define ENTER int outer_ = !tl_depth; tl_depth++
#define LEAVE tl_depth--
#define OUTER outer_

static void hook_device(Class cls);
static void hook_queue(Class cls);
static void hook_cb(Class cls);
static void hook_encoder(Class cls, int kind);
static void hook_event(Class cls);

/* device */

static void on_queue(id q)
{
    if (!q) return;
    pthread_mutex_lock(&g_lock);
    queue_get((void *)q);
    pthread_mutex_unlock(&g_lock);
    hook_queue(object_getClass(q));
}
static void on_event(id ev)
{
    if (!ev) return;
    pthread_mutex_lock(&g_lock);
    fp_ev *e = ev_get((void *)ev, 1);
    if (e) e->known = 1;
    pthread_mutex_unlock(&g_lock);
    hook_event(object_getClass(ev));
}

static id h_dev_newq(id self, SEL _cmd)
{
    ENTER; id q = ORIG(H_DEV_NEWQ, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER) on_queue(q);
    return q;
}
static id h_dev_newq_max(id self, SEL _cmd, NSUInteger n)
{
    ENTER; id q = ORIG(H_DEV_NEWQ_MAX, id (*)(id, SEL, NSUInteger))(self, _cmd, n); LEAVE;
    if (OUTER) on_queue(q);
    return q;
}
static id h_dev_newq_desc(id self, SEL _cmd, id d)
{
    ENTER; id q = ORIG(H_DEV_NEWQ_DESC, id (*)(id, SEL, id))(self, _cmd, d); LEAVE;
    if (OUTER) on_queue(q);
    return q;
}
static id h_dev_newsev(id self, SEL _cmd)
{
    ENTER; id e = ORIG(H_DEV_NEWSEV, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER) on_event(e);
    return e;
}
static id h_dev_newev(id self, SEL _cmd)
{
    ENTER; id e = ORIG(H_DEV_NEWEV, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER) on_event(e);
    return e;
}

/* queue */

static void on_cb(id q, id cb)
{
    if (!cb) return;
    pthread_mutex_lock(&g_lock);
    fp_queue *fq = queue_get((void *)q);
    if (fq) {
        uint64_t uid = ++g_cbuid;
        fp_cb *c = &g_cbs[uid % FP_CBPOOL];
        c->uid = uid; c->cb = (void *)cb; c->q = fq; c->state = 1; c->nenc = 0; c->nit = 0;
        c->enq = 0; c->commit_tick = 0; c->label[0] = 0;
        unsigned slot = FP_OPENCB;
        for (unsigned i = 0; i < FP_OPENCB; i++) {
            if (g_opencb[i].cb == (void *)cb) { slot = i; break; }   /* stale entry for a reused address */
            if (slot == FP_OPENCB && !g_opencb[i].cb) slot = i;
        }
        if (slot == FP_OPENCB) slot = g_opencb_rr++ % FP_OPENCB;
        g_opencb[slot].cb = (void *)cb; g_opencb[slot].uid = uid;
    }
    pthread_mutex_unlock(&g_lock);
    hook_cb(object_getClass(cb));
}
static id h_q_cb(id self, SEL _cmd)
{
    ENTER; id cb = ORIG(H_Q_CB, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}
static id h_q_cbu(id self, SEL _cmd)
{
    ENTER; id cb = ORIG(H_Q_CBU, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}
static id h_q_cbd(id self, SEL _cmd, id d)
{
    ENTER; id cb = ORIG(H_Q_CBD, id (*)(id, SEL, id))(self, _cmd, d); LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}

/* command buffer */

static fp_node *on_encoder(id cb, id enc, int kind)
{
    if (!enc) return NULL;
    hook_encoder(object_getClass(enc), kind);
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    fp_node *n = NULL;
    if (c) {
        n = node_new(kind);
        n->enc = (void *)enc; n->cb_uid = c->uid; n->idx = ++c->nenc;
        fp_item it = { 0, n->uid, NULL, 0 };
        PUSH(c->it, c->nit, c->cit, it);
        unsigned slot = FP_OPENENC;
        for (unsigned i = 0; i < FP_OPENENC; i++) {
            if (g_openenc[i].enc == (void *)enc) { slot = i; break; }
            if (slot == FP_OPENENC && !g_openenc[i].enc) slot = i;
        }
        if (slot == FP_OPENENC) slot = g_openenc_rr++ % FP_OPENENC;
        g_openenc[slot].enc = (void *)enc; g_openenc[slot].uid = n->uid;
        tl_cache.enc = (void *)enc; tl_cache.uid = n->uid;
    }
    pthread_mutex_unlock(&g_lock);
    return n;
}

static id h_cb_renc(id self, SEL _cmd, MTLRenderPassDescriptor *d)
{
    ENTER; id enc = ORIG(H_CB_RENC, id (*)(id, SEL, id))(self, _cmd, d); LEAVE;
    if (!OUTER) return enc;
    fp_node *n = on_encoder(self, enc, FPK_RENDER);
    if (n && d) {
        NSUInteger layers = d.renderTargetArrayLength;
        for (int i = 0; i < 8; i++) att(n, i, d.colorAttachments[i], layers);
        att(n, 8, d.depthAttachment, layers);
        att(n, 9, d.stencilAttachment, layers);
        if (d.visibilityResultBuffer) acc_buf(n, d.visibilityResultBuffer, 0, UINT64_MAX, FP_WR, 1, 1, VIA_VISIBILITY);
    }
    return enc;
}
static id h_cb_prenc(id self, SEL _cmd, id d)
{
    ENTER; id enc = ORIG(H_CB_PRENC, id (*)(id, SEL, id))(self, _cmd, d); LEAVE;
    if (OUTER) __atomic_fetch_add(&g_st[ST_PARALLEL], 1, __ATOMIC_RELAXED);
    return enc;
}
static id h_cb_cenc(id self, SEL _cmd)
{
    ENTER; id enc = ORIG(H_CB_CENC, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_COMPUTE);
    return enc;
}
static id h_cb_cenc_dt(id self, SEL _cmd, NSUInteger dt)
{
    ENTER; id enc = ORIG(H_CB_CENC_DT, id (*)(id, SEL, NSUInteger))(self, _cmd, dt); LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_COMPUTE);
    return enc;
}
static id h_cb_cenc_d(id self, SEL _cmd, id d)
{
    ENTER; id enc = ORIG(H_CB_CENC_D, id (*)(id, SEL, id))(self, _cmd, d); LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_COMPUTE);
    return enc;
}
static id h_cb_benc(id self, SEL _cmd)
{
    ENTER; id enc = ORIG(H_CB_BENC, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_BLIT);
    return enc;
}
static id h_cb_benc_d(id self, SEL _cmd, id d)
{
    ENTER; id enc = ORIG(H_CB_BENC_D, id (*)(id, SEL, id))(self, _cmd, d); LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_BLIT);
    return enc;
}

static void cb_item(id cb, int type, id ev, uint64_t val)
{
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    if (c) { fp_item it = { (uint8_t)type, 0, (void *)ev, val }; PUSH(c->it, c->nit, c->cit, it); }
    pthread_mutex_unlock(&g_lock);
}
static void h_cb_sig(id self, SEL _cmd, id ev, uint64_t v)
{
    ENTER; ORIG(H_CB_SIG, void (*)(id, SEL, id, uint64_t))(self, _cmd, ev, v); LEAVE;
    if (OUTER) cb_item(self, 1, ev, v);
}
static void h_cb_wait(id self, SEL _cmd, id ev, uint64_t v)
{
    ENTER; ORIG(H_CB_WAIT, void (*)(id, SEL, id, uint64_t))(self, _cmd, ev, v); LEAVE;
    if (OUTER) cb_item(self, 2, ev, v);
}
static void h_cb_enq(id self, SEL _cmd)
{
    ENTER;
    if (OUTER) {
        pthread_mutex_lock(&g_lock);
        fp_cb *c = cb_find_open((void *)self);
        if (c && !c->enq) c->enq = ++c->q->enq_assign;
        pthread_mutex_unlock(&g_lock);
    }
    ORIG(H_CB_ENQ, void (*)(id, SEL))(self, _cmd);
    LEAVE;
}

static void maybe_summary(void)
{
    time_t now = time(NULL);
    if (g_summary_secs > 0 && now - g_last_summary >= g_summary_secs) {
        g_last_summary = now;
        summary(0);
    }
}

static void h_cb_commit(id self, SEL _cmd)
{
    ENTER;
    if (OUTER) {
        uint64_t uid = 0;
        NSString *l = [self label];
        pthread_mutex_lock(&g_lock);
        fp_cb *c = cb_find_open((void *)self);
        if (c && c->state == 1) {
            uid = c->uid;
            if (l) snprintf(c->label, sizeof c->label, "%s", l.UTF8String);
            cb_close((void *)self);
            c->state = 2;
            c->commit_tick = ++g_tick;
            if (!c->enq) c->enq = ++c->q->enq_assign;
            queue_submit(c->q, c);
            process_pending(0);
        }
        pthread_mutex_unlock(&g_lock);
        if (uid) {
            [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> b) { (void)b; cb_done(uid); }];
            maybe_summary();
        }
    }
    ORIG(H_CB_COMMIT, void (*)(id, SEL))(self, _cmd);
    LEAVE;
}

/* shared event, CPU side */

static void h_ev_set(id self, SEL _cmd, uint64_t v)
{
    ENTER; ORIG(H_EV_SET, void (*)(id, SEL, uint64_t))(self, _cmd, v); LEAVE;
    if (!OUTER) return;
    pthread_mutex_lock(&g_lock);
    fp_ev *e = ev_get((void *)self, 0);
    if (e && v > e->cpu_max) { e->cpu_max = v; process_pending(0); }
    pthread_mutex_unlock(&g_lock);
}

/* every encoder */

static void h_end(id self, SEL _cmd)
{
    ENTER; ORIG(H_END, void (*)(id, SEL))(self, _cmd); LEAVE;
    if (!OUTER) return;
    fp_node *n = cur_node(self);
    if (!n) return;
    NSString *l = [self label];
    pthread_mutex_lock(&g_lock);
    if (l) snprintf(n->label, sizeof n->label, "%s", l.UTF8String);
    n->state = 2;
    for (unsigned i = 0; i < FP_OPENENC; i++) if (g_openenc[i].enc == (void *)self) { g_openenc[i].enc = NULL; g_openenc[i].uid = 0; }
    pthread_mutex_unlock(&g_lock);
    tl_cache.enc = NULL;
}

static void fence_upd(id self, id f, int after)
{
    fp_node *n = cur_node(self);
    if (n && f) { fp_fupd u = { (void *)f, (uint8_t)after }; PUSH(n->fu, n->nfu, n->cfu, u); }
}
static void fence_wait(id self, id f, int before)
{
    fp_node *n = cur_node(self);
    if (n && f) { fp_fwait w = { (void *)f, 0, (uint8_t)before }; PUSH(n->fw, n->nfw, n->cfw, w); }
}

static void h_upd(id self, SEL _cmd, id f)
{
    ENTER; ORIG(H_UPD, void (*)(id, SEL, id))(self, _cmd, f); LEAVE;
    if (OUTER) fence_upd(self, f, 1);
}
static void h_wait(id self, SEL _cmd, id f)
{
    ENTER; ORIG(H_WAIT, void (*)(id, SEL, id))(self, _cmd, f); LEAVE;
    if (OUTER) fence_wait(self, f, 0);
}
static void h_r_upd(id self, SEL _cmd, id f, NSUInteger st)
{
    ENTER; ORIG(H_R_UPD, void (*)(id, SEL, id, NSUInteger))(self, _cmd, f, st); LEAVE;
    if (OUTER) fence_upd(self, f, stages_last(st));
}
static void h_r_wait(id self, SEL _cmd, id f, NSUInteger st)
{
    ENTER; ORIG(H_R_WAIT, void (*)(id, SEL, id, NSUInteger))(self, _cmd, f, st); LEAVE;
    if (OUTER) fence_wait(self, f, stages_first(st));
}

static void use(id self, id r, NSUInteger usage, NSUInteger st)
{
    fp_node *n = cur_node(self);
    if (n) acc_res(n, r, usage_rw(usage), stages_first(st), stages_last(st), VIA_USE);
}
static void h_use(id self, SEL _cmd, id r, NSUInteger u)
{
    ENTER; ORIG(H_USE, void (*)(id, SEL, id, NSUInteger))(self, _cmd, r, u); LEAVE;
    if (OUTER) use(self, r, u, 0);
}
static void h_uses(id self, SEL _cmd, const id *r, NSUInteger cnt, NSUInteger u)
{
    ENTER; ORIG(H_USES, void (*)(id, SEL, const id *, NSUInteger, NSUInteger))(self, _cmd, r, cnt, u); LEAVE;
    if (OUTER) for (NSUInteger i = 0; i < cnt; i++) use(self, r[i], u, 0);
}
static void h_r_use_st(id self, SEL _cmd, id r, NSUInteger u, NSUInteger st)
{
    ENTER; ORIG(H_R_USE_ST, void (*)(id, SEL, id, NSUInteger, NSUInteger))(self, _cmd, r, u, st); LEAVE;
    if (OUTER) use(self, r, u, st);
}
static void h_r_uses_st(id self, SEL _cmd, const id *r, NSUInteger cnt, NSUInteger u, NSUInteger st)
{
    ENTER; ORIG(H_R_USES_ST, void (*)(id, SEL, const id *, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, r, cnt, u, st); LEAVE;
    if (OUTER) for (NSUInteger i = 0; i < cnt; i++) use(self, r[i], u, st);
}
static void h_heap(id self, SEL _cmd, id hp)
{
    ENTER; ORIG(H_HEAP, void (*)(id, SEL, id))(self, _cmd, hp); LEAVE;
    if (OUTER) __atomic_fetch_add(&g_st[ST_HEAP_USES], 1, __ATOMIC_RELAXED);
}
static void h_heaps(id self, SEL _cmd, const id *hp, NSUInteger cnt)
{
    ENTER; ORIG(H_HEAPS, void (*)(id, SEL, const id *, NSUInteger))(self, _cmd, hp, cnt); LEAVE;
    if (OUTER) __atomic_fetch_add(&g_st[ST_HEAP_USES], (long)cnt, __ATOMIC_RELAXED);
}
static void h_r_heap_st(id self, SEL _cmd, id hp, NSUInteger st)
{
    ENTER; ORIG(H_R_HEAP_ST, void (*)(id, SEL, id, NSUInteger))(self, _cmd, hp, st); LEAVE;
    if (OUTER) __atomic_fetch_add(&g_st[ST_HEAP_USES], 1, __ATOMIC_RELAXED);
}
static void h_r_heaps_st(id self, SEL _cmd, const id *hp, NSUInteger cnt, NSUInteger st)
{
    ENTER; ORIG(H_R_HEAPS_ST, void (*)(id, SEL, const id *, NSUInteger, NSUInteger))(self, _cmd, hp, cnt, st); LEAVE;
    if (OUTER) __atomic_fetch_add(&g_st[ST_HEAP_USES], (long)cnt, __ATOMIC_RELAXED);
}

/* render bindings: reads in the stage they are bound to */

static void bind_tex(id self, id t, int stage)
{
    fp_node *n = cur_node(self);
    if (n && t) acc_tex(n, t, 0, ((id<MTLTexture>)t).mipmapLevelCount, 0, tex_slices(t), FP_RD, stage, stage, VIA_BIND);
}
static void bind_buf(id self, id b, NSUInteger off, int stage)
{
    fp_node *n = cur_node(self);
    if (n && b) acc_buf(n, b, off, UINT64_MAX, FP_RD, stage, stage, VIA_BIND);
}
static void h_r_vtex(id self, SEL _cmd, id t, NSUInteger i)
{
    ENTER; ORIG(H_R_VTEX, void (*)(id, SEL, id, NSUInteger))(self, _cmd, t, i); LEAVE;
    if (OUTER) bind_tex(self, t, 0);
}
static void h_r_ftex(id self, SEL _cmd, id t, NSUInteger i)
{
    ENTER; ORIG(H_R_FTEX, void (*)(id, SEL, id, NSUInteger))(self, _cmd, t, i); LEAVE;
    if (OUTER) bind_tex(self, t, 1);
}
static void h_r_vtexs(id self, SEL _cmd, const id *t, NSRange r)
{
    ENTER; ORIG(H_R_VTEXS, void (*)(id, SEL, const id *, NSRange))(self, _cmd, t, r); LEAVE;
    if (OUTER) for (NSUInteger i = 0; i < r.length; i++) bind_tex(self, t[i], 0);
}
static void h_r_ftexs(id self, SEL _cmd, const id *t, NSRange r)
{
    ENTER; ORIG(H_R_FTEXS, void (*)(id, SEL, const id *, NSRange))(self, _cmd, t, r); LEAVE;
    if (OUTER) for (NSUInteger i = 0; i < r.length; i++) bind_tex(self, t[i], 1);
}
static void h_r_vbuf(id self, SEL _cmd, id b, NSUInteger off, NSUInteger i)
{
    ENTER; ORIG(H_R_VBUF, void (*)(id, SEL, id, NSUInteger, NSUInteger))(self, _cmd, b, off, i); LEAVE;
    if (OUTER) bind_buf(self, b, off, 0);
}
static void h_r_fbuf(id self, SEL _cmd, id b, NSUInteger off, NSUInteger i)
{
    ENTER; ORIG(H_R_FBUF, void (*)(id, SEL, id, NSUInteger, NSUInteger))(self, _cmd, b, off, i); LEAVE;
    if (OUTER) bind_buf(self, b, off, 1);
}
static void h_r_obuf(id self, SEL _cmd, id b, NSUInteger off, NSUInteger i)
{
    ENTER; ORIG(H_R_OBUF, void (*)(id, SEL, id, NSUInteger, NSUInteger))(self, _cmd, b, off, i); LEAVE;
    if (OUTER) bind_buf(self, b, off, 0);
}
static void h_r_mbuf(id self, SEL _cmd, id b, NSUInteger off, NSUInteger i)
{
    ENTER; ORIG(H_R_MBUF, void (*)(id, SEL, id, NSUInteger, NSUInteger))(self, _cmd, b, off, i); LEAVE;
    if (OUTER) bind_buf(self, b, off, 0);
}
static void h_r_vbufs(id self, SEL _cmd, const id *b, const NSUInteger *off, NSRange r)
{
    ENTER; ORIG(H_R_VBUFS, void (*)(id, SEL, const id *, const NSUInteger *, NSRange))(self, _cmd, b, off, r); LEAVE;
    if (OUTER) for (NSUInteger i = 0; i < r.length; i++) bind_buf(self, b[i], off ? off[i] : 0, 0);
}
static void h_r_fbufs(id self, SEL _cmd, const id *b, const NSUInteger *off, NSRange r)
{
    ENTER; ORIG(H_R_FBUFS, void (*)(id, SEL, const id *, const NSUInteger *, NSRange))(self, _cmd, b, off, r); LEAVE;
    if (OUTER) for (NSUInteger i = 0; i < r.length; i++) bind_buf(self, b[i], off ? off[i] : 0, 1);
}

/* index and indirect buffers are read by the vertex stage */
static void h_r_didx(id self, SEL _cmd, NSUInteger p, NSUInteger cnt, NSUInteger it, id ib, NSUInteger off)
{
    ENTER; ORIG(H_R_DIDX, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, id, NSUInteger))(self, _cmd, p, cnt, it, ib, off); LEAVE;
    if (OUTER) acc_buf(cur_node(self), ib, off, off + cnt * (it == MTLIndexTypeUInt16 ? 2 : 4), FP_RD, 0, 0, VIA_INDEX);
}
static void h_r_didx_i(id self, SEL _cmd, NSUInteger p, NSUInteger cnt, NSUInteger it, id ib, NSUInteger off, NSUInteger inst)
{
    ENTER; ORIG(H_R_DIDX_I, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, id, NSUInteger, NSUInteger))(self, _cmd, p, cnt, it, ib, off, inst); LEAVE;
    if (OUTER) acc_buf(cur_node(self), ib, off, off + cnt * (it == MTLIndexTypeUInt16 ? 2 : 4), FP_RD, 0, 0, VIA_INDEX);
}
static void h_r_didx_ibb(id self, SEL _cmd, NSUInteger p, NSUInteger cnt, NSUInteger it, id ib, NSUInteger off, NSUInteger inst, NSInteger bv, NSUInteger bi)
{
    ENTER; ORIG(H_R_DIDX_IBB, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, id, NSUInteger, NSUInteger, NSInteger, NSUInteger))(self, _cmd, p, cnt, it, ib, off, inst, bv, bi); LEAVE;
    if (OUTER) acc_buf(cur_node(self), ib, off, off + cnt * (it == MTLIndexTypeUInt16 ? 2 : 4), FP_RD, 0, 0, VIA_INDEX);
}
static void h_r_dind(id self, SEL _cmd, NSUInteger p, id ind, NSUInteger off)
{
    ENTER; ORIG(H_R_DIND, void (*)(id, SEL, NSUInteger, id, NSUInteger))(self, _cmd, p, ind, off); LEAVE;
    if (OUTER) acc_buf(cur_node(self), ind, off, off + 16, FP_RD, 0, 0, VIA_INDIRECT);
}
static void h_r_didxind(id self, SEL _cmd, NSUInteger p, NSUInteger it, id ib, NSUInteger ioff, id ind, NSUInteger off)
{
    ENTER; ORIG(H_R_DIDXIND, void (*)(id, SEL, NSUInteger, NSUInteger, id, NSUInteger, id, NSUInteger))(self, _cmd, p, it, ib, ioff, ind, off); LEAVE;
    if (!OUTER) return;
    fp_node *n = cur_node(self);
    acc_buf(n, ib, ioff, UINT64_MAX, FP_RD, 0, 0, VIA_INDEX);
    acc_buf(n, ind, off, off + 20, FP_RD, 0, 0, VIA_INDIRECT);
}
static void h_r_meshind(id self, SEL _cmd, id ind, NSUInteger off, MTLSize a, MTLSize b)
{
    ENTER; ORIG(H_R_MESHIND, void (*)(id, SEL, id, NSUInteger, MTLSize, MTLSize))(self, _cmd, ind, off, a, b); LEAVE;
    if (OUTER) acc_buf(cur_node(self), ind, off, off + 12, FP_RD, 0, 0, VIA_INDIRECT);
}

static void h_r_csa(id self, SEL _cmd, MTLStoreAction sa, NSUInteger i)
{
    ENTER; ORIG(H_R_CSA, void (*)(id, SEL, MTLStoreAction, NSUInteger))(self, _cmd, sa, i); LEAVE;
    if (OUTER && i < 8) store_action(self, (int)i, sa);
}
static void h_r_dsa(id self, SEL _cmd, MTLStoreAction sa)
{
    ENTER; ORIG(H_R_DSA, void (*)(id, SEL, MTLStoreAction))(self, _cmd, sa); LEAVE;
    if (OUTER) store_action(self, 8, sa);
}
static void h_r_ssa(id self, SEL _cmd, MTLStoreAction sa)
{
    ENTER; ORIG(H_R_SSA, void (*)(id, SEL, MTLStoreAction))(self, _cmd, sa); LEAVE;
    if (OUTER) store_action(self, 9, sa);
}

/* compute bindings: reads. A kernel may write a texture or buffer bound directly, which we
 * cannot tell without the pipeline's reflection; translation layers bind argument buffers
 * this way and declare writes with useResource. FENCEPOST_BIND_MAYWRITE=1 makes writable
 * bindings "may write" (for programs that write through direct bindings). */
static void cbind_tex(id self, id t)
{
    fp_node *n = cur_node(self);
    if (!n || !t) return;
    id<MTLTexture> tt = t;
    int rw = (g_bind_maywrite && (tt.usage & MTLTextureUsageShaderWrite)) ? (FP_RD | FP_MAYBE) : FP_RD;
    acc_tex(n, tt, 0, tt.mipmapLevelCount, 0, tex_slices(tt), rw, 0, 1, VIA_BIND);
}
static void cbind_buf(id self, id b, NSUInteger off)
{
    fp_node *n = cur_node(self);
    if (n && b) acc_buf(n, b, off, UINT64_MAX, g_bind_maywrite ? (FP_RD | FP_MAYBE) : FP_RD, 0, 1, VIA_BIND);
}
static void h_c_tex(id self, SEL _cmd, id t, NSUInteger i)
{
    ENTER; ORIG(H_C_TEX, void (*)(id, SEL, id, NSUInteger))(self, _cmd, t, i); LEAVE;
    if (OUTER) cbind_tex(self, t);
}
static void h_c_texs(id self, SEL _cmd, const id *t, NSRange r)
{
    ENTER; ORIG(H_C_TEXS, void (*)(id, SEL, const id *, NSRange))(self, _cmd, t, r); LEAVE;
    if (OUTER) for (NSUInteger i = 0; i < r.length; i++) cbind_tex(self, t[i]);
}
static void h_c_buf(id self, SEL _cmd, id b, NSUInteger off, NSUInteger i)
{
    ENTER; ORIG(H_C_BUF, void (*)(id, SEL, id, NSUInteger, NSUInteger))(self, _cmd, b, off, i); LEAVE;
    if (OUTER) cbind_buf(self, b, off);
}
static void h_c_bufs(id self, SEL _cmd, const id *b, const NSUInteger *off, NSRange r)
{
    ENTER; ORIG(H_C_BUFS, void (*)(id, SEL, const id *, const NSUInteger *, NSRange))(self, _cmd, b, off, r); LEAVE;
    if (OUTER) for (NSUInteger i = 0; i < r.length; i++) cbind_buf(self, b[i], off ? off[i] : 0);
}
static void h_c_dispind(id self, SEL _cmd, id ind, NSUInteger off, MTLSize tg)
{
    ENTER; ORIG(H_C_DISPIND, void (*)(id, SEL, id, NSUInteger, MTLSize))(self, _cmd, ind, off, tg); LEAVE;
    if (OUTER) acc_buf(cur_node(self), ind, off, off + 12, FP_RD, 0, 0, VIA_INDIRECT);
}

/* blits: exact ranges */
static uint64_t img_bytes(NSUInteger bpr, NSUInteger bpi, MTLSize sz)
{
    return bpi ? (uint64_t)bpi * (sz.depth ? sz.depth : 1) : (uint64_t)bpr * (sz.height ? sz.height : 1);
}
static void h_b_bb(id self, SEL _cmd, id s, NSUInteger so, id d, NSUInteger dof, NSUInteger size)
{
    ENTER; ORIG(H_B_BB, void (*)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger))(self, _cmd, s, so, d, dof, size); LEAVE;
    if (!OUTER) return;
    fp_node *n = cur_node(self);
    acc_buf(n, s, so, so + size, FP_RD, 0, 1, VIA_BLIT_SRC);
    acc_buf(n, d, dof, dof + size, FP_WR, 0, 1, VIA_BLIT_DST);
}
static void blit_bt(id self, id s, NSUInteger so, NSUInteger bpr, NSUInteger bpi, MTLSize sz, id d, NSUInteger dsl, NSUInteger dl)
{
    fp_node *n = cur_node(self);
    acc_buf(n, s, so, so + img_bytes(bpr, bpi, sz), FP_RD, 0, 1, VIA_BLIT_SRC);
    acc_tex(n, d, dl, 1, (uint32_t)dsl, 1, FP_WR, 0, 1, VIA_BLIT_DST);
}
static void h_b_bt(id self, SEL _cmd, id s, NSUInteger so, NSUInteger bpr, NSUInteger bpi, MTLSize sz, id d, NSUInteger dsl, NSUInteger dl, MTLOrigin o)
{
    ENTER; ORIG(H_B_BT, void (*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger, MTLSize, id, NSUInteger, NSUInteger, MTLOrigin))(self, _cmd, s, so, bpr, bpi, sz, d, dsl, dl, o); LEAVE;
    if (OUTER) blit_bt(self, s, so, bpr, bpi, sz, d, dsl, dl);
}
static void h_b_bt_o(id self, SEL _cmd, id s, NSUInteger so, NSUInteger bpr, NSUInteger bpi, MTLSize sz, id d, NSUInteger dsl, NSUInteger dl, MTLOrigin o, NSUInteger opt)
{
    ENTER; ORIG(H_B_BT_O, void (*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger, MTLSize, id, NSUInteger, NSUInteger, MTLOrigin, NSUInteger))(self, _cmd, s, so, bpr, bpi, sz, d, dsl, dl, o, opt); LEAVE;
    if (OUTER) blit_bt(self, s, so, bpr, bpi, sz, d, dsl, dl);
}
static void blit_tb(id self, id s, NSUInteger ssl, NSUInteger sl, MTLSize sz, id d, NSUInteger dof, NSUInteger bpr, NSUInteger bpi)
{
    fp_node *n = cur_node(self);
    acc_tex(n, s, sl, 1, (uint32_t)ssl, 1, FP_RD, 0, 1, VIA_BLIT_SRC);
    acc_buf(n, d, dof, dof + img_bytes(bpr, bpi, sz), FP_WR, 0, 1, VIA_BLIT_DST);
}
static void h_b_tb(id self, SEL _cmd, id s, NSUInteger ssl, NSUInteger sl, MTLOrigin o, MTLSize sz, id d, NSUInteger dof, NSUInteger bpr, NSUInteger bpi)
{
    ENTER; ORIG(H_B_TB, void (*)(id, SEL, id, NSUInteger, NSUInteger, MTLOrigin, MTLSize, id, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, s, ssl, sl, o, sz, d, dof, bpr, bpi); LEAVE;
    if (OUTER) blit_tb(self, s, ssl, sl, sz, d, dof, bpr, bpi);
}
static void h_b_tb_o(id self, SEL _cmd, id s, NSUInteger ssl, NSUInteger sl, MTLOrigin o, MTLSize sz, id d, NSUInteger dof, NSUInteger bpr, NSUInteger bpi, NSUInteger opt)
{
    ENTER; ORIG(H_B_TB_O, void (*)(id, SEL, id, NSUInteger, NSUInteger, MTLOrigin, MTLSize, id, NSUInteger, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, s, ssl, sl, o, sz, d, dof, bpr, bpi, opt); LEAVE;
    if (OUTER) blit_tb(self, s, ssl, sl, sz, d, dof, bpr, bpi);
}
static void h_b_tt(id self, SEL _cmd, id s, NSUInteger ssl, NSUInteger sl, MTLOrigin so, MTLSize sz, id d, NSUInteger dsl, NSUInteger dl, MTLOrigin dor)
{
    ENTER; ORIG(H_B_TT, void (*)(id, SEL, id, NSUInteger, NSUInteger, MTLOrigin, MTLSize, id, NSUInteger, NSUInteger, MTLOrigin))(self, _cmd, s, ssl, sl, so, sz, d, dsl, dl, dor); LEAVE;
    if (!OUTER) return;
    fp_node *n = cur_node(self);
    acc_tex(n, s, sl, 1, (uint32_t)ssl, 1, FP_RD, 0, 1, VIA_BLIT_SRC);
    acc_tex(n, d, dl, 1, (uint32_t)dsl, 1, FP_WR, 0, 1, VIA_BLIT_DST);
}
static void h_b_tt_n(id self, SEL _cmd, id s, NSUInteger ssl, NSUInteger sl, id d, NSUInteger dsl, NSUInteger dl, NSUInteger nsl, NSUInteger nl)
{
    ENTER; ORIG(H_B_TT_N, void (*)(id, SEL, id, NSUInteger, NSUInteger, id, NSUInteger, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, s, ssl, sl, d, dsl, dl, nsl, nl); LEAVE;
    if (!OUTER) return;
    fp_node *n = cur_node(self);
    acc_tex(n, s, sl, nl, (uint32_t)ssl, (uint32_t)nsl, FP_RD, 0, 1, VIA_BLIT_SRC);
    acc_tex(n, d, dl, nl, (uint32_t)dsl, (uint32_t)nsl, FP_WR, 0, 1, VIA_BLIT_DST);
}
static void h_b_tt_all(id self, SEL _cmd, id s, id d)
{
    ENTER; ORIG(H_B_TT_ALL, void (*)(id, SEL, id, id))(self, _cmd, s, d); LEAVE;
    if (!OUTER) return;
    fp_node *n = cur_node(self);
    acc_res(n, s, FP_RD, 0, 1, VIA_BLIT_SRC);
    acc_res(n, d, FP_WR, 0, 1, VIA_BLIT_DST);
}
static void h_b_fill(id self, SEL _cmd, id b, NSRange r, uint8_t v)
{
    ENTER; ORIG(H_B_FILL, void (*)(id, SEL, id, NSRange, uint8_t))(self, _cmd, b, r, v); LEAVE;
    if (OUTER) acc_buf(cur_node(self), b, r.location, r.location + r.length, FP_WR, 0, 1, VIA_FILL);
}
static void h_b_mip(id self, SEL _cmd, id t)
{
    ENTER; ORIG(H_B_MIP, void (*)(id, SEL, id))(self, _cmd, t); LEAVE;
    if (OUTER) acc_res(cur_node(self), t, FP_RD | FP_WR, 0, 1, VIA_MIPGEN);
}
static void h_b_syncr(id self, SEL _cmd, id r)
{
    ENTER; ORIG(H_B_SYNCR, void (*)(id, SEL, id))(self, _cmd, r); LEAVE;
    if (OUTER) acc_res(cur_node(self), r, FP_RD, 0, 1, VIA_SYNC);
}
static void h_b_synct(id self, SEL _cmd, id t, NSUInteger slice, NSUInteger level)
{
    ENTER; ORIG(H_B_SYNCT, void (*)(id, SEL, id, NSUInteger, NSUInteger))(self, _cmd, t, slice, level); LEAVE;
    if (OUTER) acc_tex(cur_node(self), t, level, 1, (uint32_t)slice, 1, FP_RD, 0, 1, VIA_SYNC);
}

/* ---------------------------------------------------------------- installing hooks */

typedef struct { int h; const char *sel; IMP imp; } hook_def;
#define HD(h, s, f) { h, s, (IMP)f }

static const hook_def device_hooks[] = {
    HD(H_DEV_NEWQ, "newCommandQueue", h_dev_newq),
    HD(H_DEV_NEWQ_MAX, "newCommandQueueWithMaxCommandBufferCount:", h_dev_newq_max),
    HD(H_DEV_NEWQ_DESC, "newCommandQueueWithDescriptor:", h_dev_newq_desc),
    HD(H_DEV_NEWSEV, "newSharedEvent", h_dev_newsev),
    HD(H_DEV_NEWEV, "newEvent", h_dev_newev),
};
static const hook_def queue_hooks[] = {
    HD(H_Q_CB, "commandBuffer", h_q_cb),
    HD(H_Q_CBU, "commandBufferWithUnretainedReferences", h_q_cbu),
    HD(H_Q_CBD, "commandBufferWithDescriptor:", h_q_cbd),
};
static const hook_def cb_hooks[] = {
    HD(H_CB_RENC, "renderCommandEncoderWithDescriptor:", h_cb_renc),
    HD(H_CB_PRENC, "parallelRenderCommandEncoderWithDescriptor:", h_cb_prenc),
    HD(H_CB_CENC, "computeCommandEncoder", h_cb_cenc),
    HD(H_CB_CENC_DT, "computeCommandEncoderWithDispatchType:", h_cb_cenc_dt),
    HD(H_CB_CENC_D, "computeCommandEncoderWithDescriptor:", h_cb_cenc_d),
    HD(H_CB_BENC, "blitCommandEncoder", h_cb_benc),
    HD(H_CB_BENC_D, "blitCommandEncoderWithDescriptor:", h_cb_benc_d),
    HD(H_CB_SIG, "encodeSignalEvent:value:", h_cb_sig),
    HD(H_CB_WAIT, "encodeWaitForEvent:value:", h_cb_wait),
    HD(H_CB_COMMIT, "commit", h_cb_commit),
    HD(H_CB_ENQ, "enqueue", h_cb_enq),
};
static const hook_def event_hooks[] = {
    HD(H_EV_SET, "setSignaledValue:", h_ev_set),
};
static const hook_def render_hooks[] = {
    HD(H_END, "endEncoding", h_end),
    HD(H_USE, "useResource:usage:", h_use),
    HD(H_USES, "useResources:count:usage:", h_uses),
    HD(H_HEAP, "useHeap:", h_heap),
    HD(H_HEAPS, "useHeaps:count:", h_heaps),
    HD(H_R_USE_ST, "useResource:usage:stages:", h_r_use_st),
    HD(H_R_USES_ST, "useResources:count:usage:stages:", h_r_uses_st),
    HD(H_R_HEAP_ST, "useHeap:stages:", h_r_heap_st),
    HD(H_R_HEAPS_ST, "useHeaps:count:stages:", h_r_heaps_st),
    HD(H_R_UPD, "updateFence:afterStages:", h_r_upd),
    HD(H_R_WAIT, "waitForFence:beforeStages:", h_r_wait),
    HD(H_R_VTEX, "setVertexTexture:atIndex:", h_r_vtex),
    HD(H_R_FTEX, "setFragmentTexture:atIndex:", h_r_ftex),
    HD(H_R_VTEXS, "setVertexTextures:withRange:", h_r_vtexs),
    HD(H_R_FTEXS, "setFragmentTextures:withRange:", h_r_ftexs),
    HD(H_R_VBUF, "setVertexBuffer:offset:atIndex:", h_r_vbuf),
    HD(H_R_FBUF, "setFragmentBuffer:offset:atIndex:", h_r_fbuf),
    HD(H_R_OBUF, "setObjectBuffer:offset:atIndex:", h_r_obuf),
    HD(H_R_MBUF, "setMeshBuffer:offset:atIndex:", h_r_mbuf),
    HD(H_R_VBUFS, "setVertexBuffers:offsets:withRange:", h_r_vbufs),
    HD(H_R_FBUFS, "setFragmentBuffers:offsets:withRange:", h_r_fbufs),
    HD(H_R_DIDX, "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:", h_r_didx),
    HD(H_R_DIDX_I, "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:", h_r_didx_i),
    HD(H_R_DIDX_IBB, "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:", h_r_didx_ibb),
    HD(H_R_DIND, "drawPrimitives:indirectBuffer:indirectBufferOffset:", h_r_dind),
    HD(H_R_DIDXIND, "drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:", h_r_didxind),
    HD(H_R_MESHIND, "drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:", h_r_meshind),
    HD(H_R_CSA, "setColorStoreAction:atIndex:", h_r_csa),
    HD(H_R_DSA, "setDepthStoreAction:", h_r_dsa),
    HD(H_R_SSA, "setStencilStoreAction:", h_r_ssa),
};
static const hook_def compute_hooks[] = {
    HD(H_END, "endEncoding", h_end),
    HD(H_USE, "useResource:usage:", h_use),
    HD(H_USES, "useResources:count:usage:", h_uses),
    HD(H_HEAP, "useHeap:", h_heap),
    HD(H_HEAPS, "useHeaps:count:", h_heaps),
    HD(H_UPD, "updateFence:", h_upd),
    HD(H_WAIT, "waitForFence:", h_wait),
    HD(H_C_TEX, "setTexture:atIndex:", h_c_tex),
    HD(H_C_TEXS, "setTextures:withRange:", h_c_texs),
    HD(H_C_BUF, "setBuffer:offset:atIndex:", h_c_buf),
    HD(H_C_BUFS, "setBuffers:offsets:withRange:", h_c_bufs),
    HD(H_C_DISPIND, "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:", h_c_dispind),
};
static const hook_def blit_hooks[] = {
    HD(H_END, "endEncoding", h_end),
    HD(H_UPD, "updateFence:", h_upd),
    HD(H_WAIT, "waitForFence:", h_wait),
    HD(H_B_BB, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", h_b_bb),
    HD(H_B_BT, "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:", h_b_bt),
    HD(H_B_BT_O, "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:", h_b_bt_o),
    HD(H_B_TB, "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:", h_b_tb),
    HD(H_B_TB_O, "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:", h_b_tb_o),
    HD(H_B_TT, "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:", h_b_tt),
    HD(H_B_TT_N, "copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:", h_b_tt_n),
    HD(H_B_TT_ALL, "copyFromTexture:toTexture:", h_b_tt_all),
    HD(H_B_FILL, "fillBuffer:range:value:", h_b_fill),
    HD(H_B_MIP, "generateMipmapsForTexture:", h_b_mip),
    HD(H_B_SYNCR, "synchronizeResource:", h_b_syncr),
    HD(H_B_SYNCT, "synchronizeTexture:slice:level:", h_b_synct),
};

static struct { Class c; int kind; } g_hooked[64]; static int g_nhooked;

static void install(Class cls, int kind, const hook_def *defs, size_t n)
{
    if (!cls) return;
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_nhooked; i++)
        if (g_hooked[i].c == cls && g_hooked[i].kind == kind) { pthread_mutex_unlock(&g_lock); return; }
    if (g_nhooked < 64) { g_hooked[g_nhooked].c = cls; g_hooked[g_nhooked].kind = kind; g_nhooked++; }
    for (size_t i = 0; i < n; i++) hook(cls, defs[i].h, defs[i].sel, defs[i].imp);
    pthread_mutex_unlock(&g_lock);
}
#define N(a) (sizeof a / sizeof a[0])
static void hook_device(Class cls) { install(cls, 100, device_hooks, N(device_hooks)); }
static void hook_queue(Class cls) { install(cls, 101, queue_hooks, N(queue_hooks)); }
static void hook_cb(Class cls) { install(cls, 102, cb_hooks, N(cb_hooks)); }
static void hook_event(Class cls) { install(cls, 103, event_hooks, N(event_hooks)); }
static void hook_encoder(Class cls, int kind)
{
    if (kind == FPK_RENDER) install(cls, kind, render_hooks, N(render_hooks));
    else if (kind == FPK_COMPUTE) install(cls, kind, compute_hooks, N(compute_hooks));
    else install(cls, kind, blit_hooks, N(blit_hooks));
}

static void final_summary(void)
{
    static atomic_int once;
    if (atomic_exchange(&once, 1) || !g_st[ST_NODES]) return;   /* helper processes that never encoded: quiet */
    summary(1);
}

static void on_device(id dev)
{
    if (!dev) return;
    int first = 0;
    pthread_mutex_lock(&g_lock);
    if (!g_started) { g_started = 1; first = 1; }
    pthread_mutex_unlock(&g_lock);
    if (first) {
        fp_log("fencepost: watching Metal in pid %d (%s); reports go to %s\n", getpid(),
               [[NSProcessInfo processInfo] processName].UTF8String,
               g_out ? "FENCEPOST_LOG" : "stderr");
        g_last_summary = time(NULL);
        atexit(final_summary);
    }
    hook_device(object_getClass(dev));
}

/* ---------------------------------------------------------------- entry points */

static id<MTLDevice> fp_MTLCreateSystemDefaultDevice(void)
{
    id d = MTLCreateSystemDefaultDevice();
    on_device(d);
    return d;
}
static NSArray<id<MTLDevice>> *fp_MTLCopyAllDevices(void)
{
    NSArray *a = MTLCopyAllDevices();
    for (id d in a) on_device(d);
    return a;
}

#define INTERPOSE(repl, orig) \
    __attribute__((used)) static struct { const void *r; const void *o; } interpose_##orig \
    __attribute__((section("__DATA,__interpose"))) = { (const void *)(uintptr_t)&repl, (const void *)(uintptr_t)&orig }
INTERPOSE(fp_MTLCreateSystemDefaultDevice, MTLCreateSystemDefaultDevice);
INTERPOSE(fp_MTLCopyAllDevices, MTLCopyAllDevices);

__attribute__((constructor)) static void fp_init(void)
{
    const char *p = getenv("FENCEPOST_LOG");
    if (p && *p) {   /* %p in the name becomes the process id: one file per process */
        char path[1024], *o = path;
        for (; *p && o < path + sizeof path - 16; p++) {
            if (p[0] == '%' && p[1] == 'p') { o += snprintf(o, 16, "%d", getpid()); p++; }
            else *o++ = *p;
        }
        *o = 0;
        g_out = fopen(path, "a");
    }
    if ((p = getenv("FENCEPOST_MAX_REPORTS"))) g_max_reports = atol(p);
    if ((p = getenv("FENCEPOST_POSSIBLE"))) g_show_possible = atoi(p);
    if ((p = getenv("FENCEPOST_SUMMARY_SECS"))) g_summary_secs = atoi(p);
    if ((p = getenv("FENCEPOST_BIND_MAYWRITE"))) g_bind_maywrite = atoi(p);
}

/* For tests and tools: dlsym these. out[] gets ST_COUNT counters (see the enum). */
__attribute__((visibility("default"))) int fencepost_stats(long *out, int n)
{
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < n && i < ST_COUNT; i++) out[i] = g_st[i];
    pthread_mutex_unlock(&g_lock);
    return ST_COUNT;
}
/* Declare a barrier at the current point of cb: every encoder submitted to cb's queue
 * before this point must be ordered (by fences or events) before every encoder after it.
 * A runtime calls this where its API demands full ordering (a D3D12 ResourceBarrier, a
 * Vulkan pipeline barrier), and fencepost checks the synchronisation it encoded. */
__attribute__((visibility("default"))) void fencepost_barrier(id cb)
{
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    if (c) { fp_item it = { 3, 0, NULL, (uint64_t)c->nenc }; PUSH(c->it, c->nit, c->cit, it); }
    pthread_mutex_unlock(&g_lock);
}

/* Leave an encoder out of declared-barrier checks as an earlier encoder: the runtime
 * orders it by other means, e.g. it touches only tracked resources it names itself
 * (a present blit), which Metal orders against every later encoder naming them. Call it
 * while the encoder is open. Its resource accesses are still checked. */
__attribute__((visibility("default"))) void fencepost_exempt(id encoder)
{
    fp_node *n = cur_node(encoder);
    if (n) n->exempt = 1;
}

/* Analyse everything committed so far, settling event waits that are still open. */
__attribute__((visibility("default"))) void fencepost_flush(void)
{
    pthread_mutex_lock(&g_lock);
    process_pending(1);
    pthread_mutex_unlock(&g_lock);
}
