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
#include <Block.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/mount.h>
#include "../tools/trigger/trigger_write.h"
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
#define FP_OBJBITS      18
#define FP_OBJMAP       (1 << FP_OBJBITS)
#define FP_OBJPROBE     4096            /* give up a lookup after this many slots */
#define FP_SWEEP        2048            /* shadow slots cleaned per commit */
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
       ST_ORDER_UNIQ, ST_ORDER_HITS, ST_PARTIAL_UNIQ, ST_PARTIAL_HITS, ST_BARRIERS,
       ST_OBJS, ST_OBJ_DROPS, ST_ANALYSIS_NS,
       ST_FRAMES, ST_FRAME_ENC, ST_FRAME_CHAIN, ST_GPU_IDLE,   /* last finished frame's encoders and chain */
       ST_COUNT };

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
    uint8_t uncertain, exempt, mtl4;
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
    uint64_t frame, queue_seq, chain_prev;   /* chain_prev: earlier node on the longest ordered chain */
    uint32_t depth;                 /* encoders on that chain within the frame, this one included */
    fp_bits anc[2];                 /* what is ordered before this node's vertex / fragment work */
} fp_node;

typedef struct { uint8_t type; uint64_t uid; void *ev; uint64_t val; } fp_item;   /* 0 encoder, 1 signal, 2 wait, 3 barrier, 4 frame end */
typedef struct {
    uint64_t uid, enq, commit_tick;
    void *cb;
    struct fp_queue *q;
    int state, nenc, mtl4;
    fp_item *it; int nit, cit;
    char label[48];
    struct fp_trace_cb *trace;
} fp_cb;

typedef struct fp_queue {
    void *q;
    int idx, mtl4;
    uint64_t enq_assign, enq_next;
    uint64_t recent[FP_QRECENT]; uint64_t nrecent;
    uint64_t bar_mark, bar_cb; int bar_idx;   /* the last declared barrier */
    uint64_t *held; int nheld, cheld;
    fp_evwait *mtl4_waits;
    int mtl4_nwaits, mtl4_cwaits;
    struct fp_trace_cb *mtl4_last_trace;
    uint64_t mtl4_signal_limit, mtl4_signals_done;
} fp_queue;

typedef struct { fp_acc a; uint64_t uid, aseq; } fp_ent;
typedef struct { void *key; fp_ent *w; int nw, cw; fp_ent *r; int nr, cr; } fp_obj;   /* writes (incl. may-write) and pure reads */
typedef struct { void *ev; int known, gpu; uint64_t cpu_max; struct { uint64_t val, uid; } h[FP_EVHIST]; unsigned nh; } fp_ev;

#define PUSH(arr, n, cap, val) do { if ((n) == (cap)) { (cap) = (cap) ? (cap) * 2 : 8; \
    (arr) = realloc((arr), (size_t)(cap) * sizeof *(arr)); } (arr)[(n)++] = (val); } while (0)

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_mtl4_options_lock = PTHREAD_MUTEX_INITIALIZER;
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
static long g_mtl4_ml_encoders;
static FILE *g_out;
static long g_max_reports = 200, g_printed;
static int g_show_possible = 1, g_summary_secs = 10, g_started, g_bind_maywrite;
static time_t g_last_summary;
static long g_last_summary_hits = -1, g_last_summary_frames;

/* Frames: a present (or fencepost_frame) ends one. For each, the encoders in it and the
 * longest chain of them that synchronisation forces to run one after another. */
#define FP_FRAMES 16             /* frames still being analysed */
#define FP_FHIST  1024           /* finished frames the summary takes medians over */
typedef struct { uint64_t key; long queued, analysed, nenc, idle; uint32_t depth; uint64_t tail; } fp_frame;
static fp_frame g_fr[FP_FRAMES];
static uint64_t g_frame_cur, g_frame_next;   /* frame phase 1 is filling; next frame to finish */
static struct { long nenc, idle; uint32_t depth; } g_fhist[FP_FHIST]; static uint64_t g_nfhist;
static long g_inflight;                      /* command buffers committed and not completed */
static long g_chain_frame;                   /* FENCEPOST_CHAIN: the frame whose chain to print */
static int g_frame_api;                      /* fencepost_frame was called: presents no longer end frames */

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
    n->mtl4 = 0;
    n->req_q = NULL; n->req_mark = 0; n->bar_cb = 0; n->bar_idx = 0;
    n->frame = 0; n->queue_seq = 0; n->chain_prev = 0; n->depth = 0;
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

static fp_queue *queue_get_kind(void *q, int mtl4)   /* lock held */
{
    for (int i = 0; i < g_nq; i++) if (g_q[i].q == q && g_q[i].mtl4 == mtl4) return &g_q[i];
    if (g_nq == FP_MAXQ) return NULL;
    fp_queue *fq = &g_q[g_nq];
    memset(fq, 0, sizeof *fq);
    fq->q = q; fq->idx = g_nq++; fq->enq_next = 1; fq->mtl4 = mtl4;
    return fq;
}

static fp_queue *queue_get(void *q) { return queue_get_kind(q, 0); }

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
    unsigned h = ptr_hash(key, FP_OBJBITS);
    for (unsigned i = 0; i < FP_OBJPROBE; i++) {
        fp_obj *o = &g_obj[(h + i) & (FP_OBJMAP - 1)];
        if (o->key == key) return o;
        if (!o->key) { o->key = key; o->nw = o->nr = 0; g_st[ST_OBJS]++; return o; }
    }
    g_st[ST_OBJ_DROPS]++;
    return NULL;
}

/* Remove slot i (linear probing: later entries of its run move back so lookups still find them). */
static void obj_del(unsigned i)
{
    free(g_obj[i].w); free(g_obj[i].r);
    memset(&g_obj[i], 0, sizeof g_obj[i]);
    g_st[ST_OBJS]--;
    for (unsigned j = (i + 1) & (FP_OBJMAP - 1); g_obj[j].key; j = (j + 1) & (FP_OBJMAP - 1)) {
        unsigned k = ptr_hash(g_obj[j].key, FP_OBJBITS);
        int stays = i <= j ? (k > i && k <= j) : (k > i || k <= j);
        if (stays) continue;
        g_obj[i] = g_obj[j];
        memset(&g_obj[j], 0, sizeof g_obj[j]);
        i = j;
    }
}

/* ---------------------------------------------------------------- reporting */

static uint64_t fnv(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = p;
    while (n--) { h ^= *b++; h *= 0x100000001b3ull; }
    return h;
}

/* Labels often carry a running count ("R#15 clear"): leave out "#<digits>" so every
 * instance of the same pass groups into one report. */
static uint64_t fnv_label(uint64_t h, const char *l)
{
    for (; *l; l++) {
        if (l[0] == '#' && l[1] >= '0' && l[1] <= '9') { while (l[1] >= '0' && l[1] <= '9') l++; continue; }
        h ^= (unsigned char)*l; h *= 0x100000001b3ull;
    }
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

static fp_frame *frame_get(uint64_t f);

/* ---------------------------------------------------------------- timeline (lock held unless noted) */

#define FP_TRACE_SAMPLES 4096     /* 32 KiB of timestamps, the Apple GPU limit */
#define FP_TRACE_MARKERS 20000
#define FP_TRACE_RESERVE 4096
#define FP_TRACE_ATTACHMENTS 4
#define FP_SPIKE_FACTOR 4
#define FP_SPIKE_MIN_NS 50e6

enum { TR_CB, TR_GPU, TR_CPU, TR_MTL4_EVENT, TR_QUEUE_TRACKS };
enum { TR_FENCE, TR_EVENT, TR_COMPLETED, TR_RACE, TR_POSSIBLE, TR_ORDER, TR_FRAGMENT };
static const char *const trace_categories[] = { "fence", "event", "completed", "race", "race", "order", "order" };

typedef struct fp_trace_link fp_trace_link;
typedef struct {
    uint64_t start, end, frame;
    int tid, ready, encoder, gpu, sampled, counted;
    fp_trace_link *waiting;
} fp_trace_span;

typedef struct {
    uint64_t start, end, frame;
} fp_trace_interval;

typedef struct fp_trace_frame {
    uint64_t frame, start, end;
    struct fp_trace_frame *next;
} fp_trace_frame;

typedef struct {
    uint64_t floor;
    int tid, n, cap;
    fp_trace_interval *intervals;
} fp_trace_lane;

typedef struct {
    fp_trace_lane *lanes;
    int n, cap;
} fp_trace_track;

typedef struct fp_trace_device {
    id<MTLDevice> device;
    id<MTLCounterSampleBuffer> buffer;
    struct { uint64_t cpu, gpu; } first, last;
    uint64_t calibration;
    uint64_t previous_end[FP_TRACE_SAMPLES / 2];
    unsigned free_slots[FP_TRACE_SAMPLES / 2], nfree;
    struct fp_trace_device *next;
} fp_trace_device;

typedef struct fp_trace_encoder {
    fp_trace_span span;            /* also the value in g_trace_nodes */
    struct fp_trace_encoder *next;
    struct fp_trace_cb *cb;
    uint64_t uid, start, end, thread, gpu_start, gpu_end;
    int sample, selected, kind, args_dropped;
    char label[64], mtl4_cb_label[48];
    char *args;
    fp_evwait *waits;
    int nwaits;
} fp_trace_encoder;

@interface FPTraceOwner : NSObject {
@public
    struct fp_trace_cb *record;
}
@end

typedef struct { uint64_t uid; fp_trace_span span; } fp_trace_endpoint;

typedef struct fp_trace_cb {
    fp_trace_span span;
    fp_trace_encoder *encoders, **encoder_tail;
    fp_trace_device *device;
    FPTraceOwner *owner;           /* owned by the Metal command buffer */
    uint64_t uid;
    fp_trace_endpoint *completed_before;
    uint64_t *signals;
    int nsignals, csignals, refs;
    char *gpu_error;
    uint64_t error_time;
    uint64_t gpu_start, gpu_end;
    int queue, phased, completed, selected, has_encoder, mtl4;
    char label[48];
} fp_trace_cb;

struct fp_trace_link {
    fp_trace_link *next;
    uint64_t from, to;
    unsigned kind;
    char *title, *resource, *first, *second, *why;
};

static FILE *g_trace;
static int g_trace_enabled, g_trace_gpu = 1, g_trace_stopped, g_trace_sampling_supported;
static int g_trace_pid, g_trace_tid;
static size_t g_trace_bytes, g_trace_limit = 512u * 1024u * 1024u;
static uint64_t g_trace_origin, g_trace_first = 1, g_trace_last = UINT64_MAX, g_trace_frame_start;
static uint64_t g_trace_encoders, g_trace_sampled, g_trace_skipped, g_trace_flows, g_trace_dropped;
static uint64_t g_trace_pool_exhausted;
static uint64_t g_trace_alloc_dropped;
static uint64_t g_trace_markers, g_trace_marker_candidates, g_trace_marker_hits[4], g_trace_marker_counts[4];
static fp_trace_endpoint g_trace_completed[FP_MAXQ];
static char g_trace_queue_labels[FP_MAXQ][64];
static fp_trace_track g_trace_tracks[FP_MAXQ][TR_QUEUE_TRACKS];
static fp_trace_frame *g_trace_frames, **g_trace_frame_tail = &g_trace_frames;
static CFMutableDictionaryRef g_trace_nodes, g_trace_threads;
static fp_trace_device *g_trace_devices;
static const char g_trace_owner_key;

#define TRACE_KEY(uid) ((const void *)(uintptr_t)(uid))
#define TRACE_U64(n) ((unsigned long long)(n))

static uint64_t trace_clock(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

enum { CAP_RACE = 1, CAP_GPU_ERROR = 2, CAP_SPIKE = 4 };
static const char *g_capture_prefix, *g_capture_file;
static struct timespec g_capture_mtime;
static int g_capture_owner_warned;
static unsigned g_capture_on, g_capture_frames, g_capture_max, g_capture_free, g_capture_count;
static unsigned g_capture_max_secs, g_capture_idle_secs;
static uint64_t g_capture_deadline;
static double g_capture_cooldown;
static uint64_t g_capture_frame, g_capture_opened, g_capture_until, g_capture_started, g_capture_stopped;
static uint64_t g_capture_boundary, g_capture_durations[120], g_capture_duration_count;
static char g_capture_reason[33];
static int g_capture_available;              /* set once before the device hooks are installed */
static int g_capture_busy;                   /* starting, active or stopping (g_lock) */
static unsigned g_capture_commits;           /* tracked commits still in Metal (g_lock) */
static uint64_t g_capture_last_commit;        /* most recent submission activity (g_lock) */
static NSString *g_capture_path;
/* Gate command-buffer entry during capture transitions, with tl_depth set so
 * Metal's nested hooks pass through. Never hold this across allocation or
 * commit: a full command queue can make allocation wait for another commit.
 * Lock order is capture lock then detector lock. */
static pthread_mutex_t g_capture_lock = PTHREAD_MUTEX_INITIALIZER;

/* Called with g_lock held. Keep the first request for the next unopened frame. */
static void capture_request(const char *reason)
{
    if (!g_capture_prefix || g_capture_busy || g_capture_reason[0]) return;
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)(reason ?: "");
         *p && *p != '\n' && *p != '\r' && n < sizeof g_capture_reason - 1; p++) {
        unsigned c = *p;
        if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        g_capture_reason[n++] = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ? c : '-';
    }
    g_capture_reason[n] = 0;
    if (!n) strcpy(g_capture_reason, "manual");
}

__attribute__((visibility("default"))) void fencepost_capture(const char *reason)
{
    if (!g_capture_prefix) return;
    pthread_mutex_lock(&g_lock);
    capture_request(reason);
    pthread_mutex_unlock(&g_lock);
}

static int trace_frame_selected(uint64_t frame)
{
    return frame >= g_trace_first - 1 && frame < g_trace_last;
}

/* Existing labels are truncated by byte count. Preserve valid UTF-8 and replace a clipped sequence. */
static void trace_string(FILE *f, const char *s)
{
    fputc('"', f);
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        unsigned c = *p;
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c < 32) fprintf(f, "\\u%04x", c);
        else if (c < 128) fputc(c, f);
        else {
            int n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
            int i = 1;
            while (i < n && (p[i] & 0xc0) == 0x80) i++;
            if (n && i == n && !(c == 0xe0 && p[1] < 0xa0) && !(c == 0xed && p[1] >= 0xa0)
                    && !(c == 0xf0 && p[1] < 0x90) && !(c == 0xf4 && p[1] >= 0x90)) {
                fwrite(p, 1, (size_t)n, f);
                p += n;
                continue;
            }
            fputs("\\ufffd", f);
        }
        p++;
    }
    fputc('"', f);
}

/* Format ns as fractional microseconds, independent of the app's locale. */
static void trace_us(FILE *f, int64_t ns)
{
    uint64_t v = ns < 0 ? (uint64_t)-ns : (uint64_t)ns;
    fprintf(f, "%s%llu.%03llu", ns < 0 ? "-" : "", TRACE_U64(v / 1000), TRACE_U64(v % 1000));
}

/* A record can contain multiple inseparable events (e.g. a flow pair).
 * Frames are assigned in submission order, but analysis and GPU completion
 * can append to an earlier frame. All chunk operations run under g_lock. */
#define TRACE_METADATA UINT64_MAX
typedef struct fp_trace_record {
    struct fp_trace_record *next;
    char *json;
    size_t length;
    uint64_t source;
    unsigned type;
} fp_trace_record;

enum { REC_PLAIN, REC_ENCODER, REC_SAMPLED, REC_FLOW, REC_MARKER }; /* REC_MARKER + kind offset */

typedef struct fp_trace_chunk {
    struct fp_trace_chunk *next;
    uint64_t frame;
    fp_trace_record *first, *last;
    size_t bytes;
    unsigned outstanding;
    atomic_uint refs;
    uint64_t marker_hits[4], start, end;
} fp_trace_chunk;

static fp_trace_chunk g_trace_metadata, *g_trace_chunks;
static int g_trace_has_output, g_frames_tid;
static unsigned g_ring, g_ring_after = 2;
static size_t g_ring_bytes, g_ring_limit = 64u * 1024u * 1024u;
static uint64_t g_ring_floor, g_ring_latest, g_ring_evictions, g_ring_cap_evictions;
static fp_trace_endpoint *g_ring_history;
#ifdef FENCEPOST_TEST_HOOKS
static long g_mtl4_trace_live, g_mtl4_trace_peak, g_mtl4_signals_retired;
#endif
static char *g_ring_prefix;
static int g_ring_encoded, g_ring_race = 1, g_ring_gpuerr = 1;
static int g_spike_enabled = 1;
static double g_spike_factor = FP_SPIKE_FACTOR, g_spike_min_ns = FP_SPIKE_MIN_NS, g_cooldown_ns = 5e9;
static uint64_t g_durations[120], g_duration_count;
static uint64_t g_last_poll, g_suppressed;
enum { DUMP_MANUAL, DUMP_AUTO, DUMP_KINDS };
static unsigned g_max_dumps[DUMP_KINDS], g_dumps[DUMP_KINDS], g_pending_dumps[DUMP_KINDS];
static uint64_t g_last_fire[DUMP_KINDS];     /* cooldown per kind: automatic dumps never hold off a manual one */
static dispatch_queue_t g_ring_writer;
static char *g_trigger_file;
static struct timespec g_file_mtime;
static int g_trigger_owner_warned;

typedef struct {
    int active;
    unsigned kind;
    uint64_t frame, first, last, timestamp;
    char reason[33];
    const char *source;
    char *error;
} fp_trigger;

static fp_trigger g_trigger;
/* One manual request that arrived while another dump was still collecting frames; it fires
 * as soon as that dump is snapshotted (g_lock). */
static struct { int active; uint64_t frame, timestamp; char reason[33]; const char *source; } g_queued;
static void ring_update(int final);
static void ring_trigger(unsigned kind, const char *reason, const char *source, uint64_t frame,
                         uint64_t timestamp, const char *error);


static fp_trace_chunk *trace_chunk(uint64_t frame)
{
    if (frame == TRACE_METADATA) return &g_trace_metadata;
    if (g_ring && frame < g_ring_floor) return NULL;
    fp_trace_chunk **slot = &g_trace_chunks;
    while (*slot && (*slot)->frame < frame) slot = &(*slot)->next;
    if (!*slot || (*slot)->frame != frame) {
        fp_trace_chunk *chunk = calloc(1, sizeof *chunk);
        if (!chunk) {
            g_trace_alloc_dropped++;
            return NULL;
        }
        atomic_init(&chunk->refs, 1);
        chunk->frame = frame;
        chunk->next = *slot;
        *slot = chunk;
        if (g_ring) g_ring_bytes += sizeof *chunk;
    }
    return *slot;
}

static int trace_emit(FILE *out, int *started, const fp_trace_record *r)
{
    if (fputs(*started ? ",\n" : "\n", out) == EOF || fwrite(r->json, 1, r->length, out) != r->length) return 0;
    *started = 1;
    return 1;
}

static void trace_drain(fp_trace_chunk *chunk, FILE *out)
{
    while (chunk->first) {
        fp_trace_record *r = chunk->first;
        if (out) trace_emit(out, &g_trace_has_output, r);
        chunk->first = r->next;
        free(r->json);
        free(r);
    }
    chunk->last = NULL;
}

static void trace_release_chunk(fp_trace_chunk *chunk)
{
    if (atomic_fetch_sub(&chunk->refs, 1) == 1) {
        trace_drain(chunk, NULL);
        free(chunk);
    }
}

static void trace_stream_flush(void)
{
    trace_drain(&g_trace_metadata, g_trace);
    while (g_trace_chunks) {
        fp_trace_chunk *chunk = g_trace_chunks;
        g_trace_chunks = chunk->next;
        trace_drain(chunk, g_trace);
        free(chunk);
    }
}

static void ring_evict(int cap)
{
    fp_trace_chunk *chunk = g_trace_chunks;
    g_trace_chunks = chunk->next;
    g_ring_bytes -= chunk->bytes + sizeof *chunk;
    g_ring_floor = chunk->frame + 1;
    g_ring_evictions++;
    if (cap) g_ring_cap_evictions++;
    trace_release_chunk(chunk);
}

static void trace_drop_record(unsigned events)
{
    g_trace_alloc_dropped++;
    g_trace_dropped += events;
}

static int trace_close_record(FILE *f)
{
    if (!f) return 0;
    int failed = ferror(f);
    return !fclose(f) && !failed;
}

/* Budget whole records, including their separator, before publishing them. */
static int trace_store(uint64_t frame, FILE *f, char **record, size_t *length, unsigned events,
                       unsigned type, uint64_t source)
{
    if (!trace_close_record(f)) {
        trace_drop_record(events);
        free(*record);
        return 0;
    }
    if (!g_ring && !g_trace_stopped && g_trace_bytes + *length + 2 + FP_TRACE_RESERVE > g_trace_limit) {
        g_trace_stopped = 1;
        fp_log("fencepost: trace size limit reached (FENCEPOST_TRACE_MB); recording stopped\n");
        trace_stream_flush();
        fflush(g_trace);
    }
    if (!g_trace_stopped) {
        fp_trace_chunk *chunk = trace_chunk(frame);
        if (!chunk) {
            free(*record);
            g_trace_dropped += events;
            return 0;
        }
        size_t bytes = sizeof(fp_trace_record) + *length + 1;
        if (g_ring) {
            while (g_trace_chunks && g_ring_bytes + bytes > g_ring_limit) ring_evict(1);
            if ((frame != TRACE_METADATA && frame < g_ring_floor) || g_ring_bytes + bytes > g_ring_limit) {
                if (frame == TRACE_METADATA) g_trace_stopped = 1;
                free(*record);
                g_trace_dropped += events;
                return 0;
            }
        }
        fp_trace_record *r = calloc(1, sizeof *r);
        if (!r) {
            trace_drop_record(events);
            free(*record);
            return 0;
        }
        if (g_ring) {
            g_ring_bytes += bytes;
            chunk->bytes += bytes;
        }
        r->source = source;
        r->type = type;
        r->json = *record;
        r->length = *length;
        if (chunk->last) chunk->last->next = r;
        else chunk->first = r;
        chunk->last = r;
        g_trace_bytes += *length + (g_trace_bytes == 1 ? 1 : 2);
        return 1;
    }
    g_trace_dropped += events;
    free(*record);
    return 0;
}

static int trace_write(uint64_t frame, FILE *f, char **record, size_t *length, unsigned events)
{
    return trace_store(frame, f, record, length, events, REC_PLAIN, 0);
}

static void trace_header_at(FILE *f, char phase, int tid, uint64_t ts, const char *cat, const char *name,
                            uint64_t origin)
{
    fprintf(f, "{\"ph\":\"%c\",\"pid\":%d,\"tid\":%d,\"name\":", phase, g_trace_pid, tid);
    trace_string(f, name);
    if (cat) { fputs(",\"cat\":", f); trace_string(f, cat); }
    if (phase != 'M') {
        fputs(",\"ts\":", f);
        trace_us(f, (int64_t)ts - (int64_t)origin);
    }
}

static void trace_header(FILE *f, char phase, int tid, uint64_t ts, const char *cat, const char *name)
{
    if (phase != 'M' && !g_trace_origin) g_trace_origin = ts;
    trace_header_at(f, phase, tid, ts, cat, name, g_trace_origin);
}

static void trace_start(void)
{
    /* Completions emit earlier CPU and GPU times. Anchor the trace before
     * any of that work begins, since viewers can discard negative times. */
    g_trace_origin = g_trace_frame_start = trace_clock();
    char *record = NULL;
    size_t length = 0;
    FILE *f = open_memstream(&record, &length);
    if (!f) { trace_drop_record(1); return; }
    trace_header(f, 'M', 0, 0, NULL, "process_name");
    fputs(",\"ts\":0.000,\"args\":{\"name\":", f);
    trace_string(f, [[NSProcessInfo processInfo] processName].UTF8String);
    fputs("}}", f);
    trace_write(TRACE_METADATA, f, &record, &length, 1);
}

static int trace_name_track(const char *name)
{
    int tid = ++g_trace_tid;
    char *record = NULL;
    size_t length = 0;
    FILE *f = open_memstream(&record, &length);
    if (!f) { trace_drop_record(1); return tid; }
    trace_header(f, 'M', tid, 0, NULL, "thread_name");
    fputs(",\"args\":{\"name\":", f);
    trace_string(f, name);
    fputs("}}", f);
    trace_write(TRACE_METADATA, f, &record, &length, 1);
    return tid;
}

/* Completions can arrive out of order, so find the first lane with a gap
 * at this timestamp. Most inserts append to the end of a lane. */
static int trace_lane(fp_trace_track *track, const char *name, const fp_trace_span *span)
{
    for (int i = 0; ; i++) {
        if (i == track->n) {
            char title[160];
            if (i) snprintf(title, sizeof title, "%s lane %d", name, i + 1);
            else snprintf(title, sizeof title, "%s", name);
            fp_trace_lane fresh = { .tid = trace_name_track(title) };
            PUSH(track->lanes, track->n, track->cap, fresh);
        }
        fp_trace_lane *lane = &track->lanes[i];
        if (g_ring && lane->floor != g_ring_floor) {
            int kept = 0;
            for (int j = 0; j < lane->n; j++)
                if (lane->intervals[j].frame >= g_ring_floor) lane->intervals[kept++] = lane->intervals[j];
            lane->n = kept;
            lane->floor = g_ring_floor;
        }
        int lo = lane->n, hi = lane->n;
        if (lo && lane->intervals[lo - 1].end > span->start) lo = 0;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (lane->intervals[mid].end <= span->start) lo = mid + 1;
            else hi = mid;
        }
        if (lo < lane->n && lane->intervals[lo].start < span->end) continue;
        fp_trace_interval interval = { span->start, span->end, span->frame };
        PUSH(lane->intervals, lane->n, lane->cap, interval);
        if (lo < lane->n - 1) {
            memmove(lane->intervals + lo + 1, lane->intervals + lo,
                    (size_t)(lane->n - lo - 1) * sizeof *lane->intervals);
            lane->intervals[lo] = interval;
        }
        return lane->tid;
    }
}

static int trace_queue_lane(int queue, int type, const fp_trace_span *span)
{
    char name[128];
    if (type == TR_CB) {
        const char *label = g_trace_queue_labels[queue];
        snprintf(name, sizeof name, "Queue %d%s%s - command buffers", queue, *label ? " " : "", label);
    } else if (type == TR_MTL4_EVENT) {
        snprintf(name, sizeof name, "Queue %d %s - events", queue, g_trace_queue_labels[queue]);
    } else if (g_q[queue].mtl4) {
        snprintf(name, sizeof name, "Queue %d %s - encoders%s", queue, g_trace_queue_labels[queue],
                 type == TR_CPU ? " (CPU encode time)" : "");
    } else snprintf(name, sizeof name, "Queue %d - encoders%s", queue, type == TR_CPU ? " (CPU encode time)" : "");
    return trace_lane(&g_trace_tracks[queue][type], name, span);
}

static void trace_label(FILE *f, const char *key, const char *label)
{
    fprintf(f, ",\"%s\":", key);
    trace_string(f, label);
}

static void trace_slice(FILE *f, const fp_trace_span *span, const char *kind, const char *name)
{
    trace_header(f, 'X', span->tid, span->start, kind, name);
    fputs(",\"dur\":", f);
    trace_us(f, (int64_t)(span->end - span->start));
}

static fp_trace_span *trace_node(uint64_t uid)
{
    fp_trace_span *live = (fp_trace_span *)CFDictionaryGetValue(g_trace_nodes, TRACE_KEY(uid));
    if (live || !g_ring_history || !uid) return live;
    fp_trace_endpoint *old = &g_ring_history[uid % FP_POOL];
    return old->uid == uid ? &old->span : NULL;
}

static int trace_sampled(const fp_trace_encoder *e)
{
    return e->gpu_start && e->gpu_end > e->gpu_start;
}

static int trace_estimated(const fp_trace_span *span)
{
    return span->encoder && span->gpu && !span->sampled;
}

static void trace_flow_event(FILE *f, char phase, const fp_trace_span *span, uint64_t ts, const char *cat,
                             uint64_t id, const fp_trace_link *link)
{
    trace_header(f, phase, span->tid, ts, cat, cat);
    if (link->kind >= TR_RACE) fputs(",\"cname\":\"terrible\"", f);
    fprintf(f, ",\"bp\":\"e\",\"id\":%llu,\"args\":{\"from_uid\":%llu,\"to_uid\":%llu}}",
            TRACE_U64(id), TRACE_U64(link->from), TRACE_U64(link->to));
}

static void trace_flow(const fp_trace_link *link, const fp_trace_span *from, const fp_trace_span *to)
{
    if (from->gpu != to->gpu) return;
    if (g_trace_stopped) { g_trace_dropped += 2; return; }
    const char *cat = trace_categories[link->kind];
    uint64_t id = g_trace_flows + 1;
    /* At a shared boundary, Perfetto otherwise attaches the start to the next slice. */
    uint64_t start = from->end > from->start ? from->end - 1 : from->end;
    /* Clamping an estimate to the CB span can put it just inside a sampled
     * span. Anchor the arrow inside the source at their shared timestamp. */
    if ((link->kind == TR_FENCE || link->kind == TR_EVENT) && from->start <= to->start
            && to->start < start && (trace_estimated(from) || trace_estimated(to)))
        start = to->start;
    char *record = NULL;
    size_t length = 0;
    FILE *f = open_memstream(&record, &length);
    if (!f) { trace_drop_record(2); return; }
    trace_flow_event(f, 's', from, start, cat, id, link);
    fputs(",\n", f);
    trace_flow_event(f, 'f', to, to->start, cat, id, link);
    if (trace_store(to->frame, f, &record, &length, 2, REC_FLOW, from->frame)) g_trace_flows++;
}

static void trace_marker_record(FILE *f, const fp_trace_link *link, const fp_trace_span *to,
                                const fp_trace_span *from)
{
    trace_header(f, 'i', to->tid, to->start, trace_categories[link->kind], link->title);
    fprintf(f, ",\"s\":\"t\",\"cname\":\"terrible\",\"args\":{\"from_uid\":%llu,\"to_uid\":%llu",
            TRACE_U64(link->from), TRACE_U64(link->to));
    trace_label(f, "resource", link->resource);
    trace_label(f, "first", link->first);
    trace_label(f, "second", link->second);
    trace_label(f, "why", link->why);
    const char *overlap = "null", *order = "null";
    if (from && from->sampled && to->sampled) {
        int both = from->start < to->end && to->start < from->end;
        overlap = both ? "true" : "false";
        /* the second ran wholly before the first: the ordering was inverted on this run, not merely absent */
        order = both ? "\"overlapped\"" : to->end <= from->start ? "\"reversed\"" : "\"in order\"";
    }
    fprintf(f, ",\"overlapped_on_gpu\":%s,\"gpu_order\":%s}}", overlap, order);
}

static void trace_marker_clear(fp_trace_link *link)
{
    free(link->title); free(link->resource); free(link->first); free(link->second); free(link->why);
    link->title = NULL;
}

static void trace_marker(fp_trace_link *link, const fp_trace_span *to, const fp_trace_span *from)
{
    if (g_trace_stopped) { g_trace_dropped++; return; }
    char *record = NULL;
    size_t length = 0;
    FILE *f = open_memstream(&record, &length);
    if (!f) { trace_drop_record(1); return; }
    trace_marker_record(f, link, to, from);
    if (trace_store(to->frame, f, &record, &length, 1, REC_MARKER + link->kind - TR_RACE,
                    from ? from->frame : UINT64_MAX)) {
        g_trace_markers++;
        g_trace_marker_counts[link->kind - TR_RACE]++;
    }
}

static void trace_link_settle(fp_trace_link *link)
{
    fp_trace_span *to = trace_node(link->to), *from = trace_node(link->from);
    if (to && !to->ready) { link->next = to->waiting; to->waiting = link; return; }
    if (to && to->tid) {
        if (from && !from->ready) { link->next = from->waiting; from->waiting = link; return; }
        if (link->title) {
            trace_marker(link, to, from);
            trace_marker_clear(link);
        }
        if (from && from->tid && from != to) trace_flow(link, from, to);
    } else if (link->title) {
        trace_marker_clear(link);
        g_trace_dropped++;
    }
    free(link);
}

static void trace_ready(fp_trace_span *span)
{
    span->ready = 1;
    while (span->waiting) {
        fp_trace_link *link = span->waiting;
        span->waiting = link->next;
        trace_link_settle(link);
    }
}

static void trace_ordering(uint64_t from, uint64_t to, unsigned kind)
{
    if (g_trace_stopped || !trace_node(from) || !trace_node(to)) return;
    fp_trace_link *link = calloc(1, sizeof *link);
    link->from = from; link->to = to; link->kind = kind;
    trace_link_settle(link);
}

static int trace_marker_hit(const fp_node *to, unsigned kind)
{
    if (!trace_frame_selected(to->frame)) return 0;
    g_trace_marker_hits[kind - TR_RACE]++;
    if (g_ring) {
        fp_trace_chunk *chunk = trace_chunk(to->frame);
        if (!chunk) return 0;
        chunk->marker_hits[kind - TR_RACE]++;
    }
    if ((!g_ring && g_trace_marker_candidates++ >= FP_TRACE_MARKERS) || g_trace_stopped || !trace_node(to->uid)) {
        g_trace_dropped++;
        return 0;
    }
    return 1;
}

static void trace_diagnostic(unsigned kind, uint64_t from, uint64_t to, const char *title,
                             const char *resource, const char *first, const char *second, const char *why)
{
    fp_trace_link *link = calloc(1, sizeof *link);
    link->from = from; link->to = to; link->kind = kind;
    link->title = strdup(title); link->resource = strdup(resource);
    link->first = strdup(first); link->second = strdup(second); link->why = strdup(why);
    trace_link_settle(link);
}

static fp_trace_device *trace_device(id<MTLDevice> device)
{
    for (fp_trace_device *d = g_trace_devices; d; d = d->next) if (d->device == device) return d;
    fp_trace_device *d = calloc(1, sizeof *d);
    d->device = [device retain]; d->next = g_trace_devices; g_trace_devices = d;
    if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return d;
    for (id<MTLCounterSet> set in device.counterSets) {
        if (![set.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
        MTLCounterSampleBufferDescriptor *desc = [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet = set;
        desc.sampleCount = FP_TRACE_SAMPLES;
        desc.storageMode = MTLStorageModeShared;
        desc.label = @"fencepost timestamps";
        d->buffer = [device newCounterSampleBufferWithDescriptor:desc error:NULL];
        [desc release];
        break;
    }
    if (d->buffer) {
        g_trace_sampling_supported = 1;
        for (unsigned i = 0; i < FP_TRACE_SAMPLES / 2; i++) d->free_slots[d->nfree++] = FP_TRACE_SAMPLES - 2 - 2 * i;
    }
    return d;
}

static void trace_calibrate(fp_trace_device *d)
{
    uint64_t before = trace_clock();
    if (d->first.cpu && before - d->calibration < (d->last.cpu ? 250000000 : 0)) return;
    MTLTimestamp cpu = 0, gpu = 0;
    [d->device sampleTimestamps:&cpu gpuTimestamp:&gpu];
    uint64_t after = trace_clock(), middle = before + (after - before) / 2;
    if (!cpu || !gpu || gpu == MTLCounterErrorValue) return;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t scaled = (uint64_t)((long double)cpu * timebase.numer / timebase.denom);
    if (llabs((int64_t)scaled - (int64_t)middle) < llabs((int64_t)cpu - (int64_t)middle)) cpu = scaled;
    if (d->first.cpu && (cpu <= d->first.cpu || gpu <= d->first.gpu)) return;
    if (!d->first.cpu) { d->first.cpu = cpu; d->first.gpu = gpu; }
    else if (cpu - d->first.cpu >= 10000) {
        if (d->last.cpu) d->first = d->last;
        d->last.cpu = cpu; d->last.gpu = gpu;
    }
    d->calibration = before;
}

static void trace_release_sample(fp_trace_encoder *e)
{
    if (e->sample < 0) return;
    fp_trace_device *d = e->cb->device;
    d->free_slots[d->nfree++] = (unsigned)e->sample;
    e->sample = -1;
}

static void trace_resolve(fp_trace_encoder *e)
{
    fp_trace_device *d = e->cb->device;
    NSData *data = [d->buffer resolveCounterRange:NSMakeRange((NSUInteger)e->sample, 2)];
    if (data.length != 2 * sizeof(MTLCounterResultTimestamp)) return;
    const MTLCounterResultTimestamp *r = data.bytes;
    if (!r[0].timestamp || r[1].timestamp == MTLCounterErrorValue || r[1].timestamp <= r[0].timestamp) return;
    uint64_t *previous = &d->previous_end[e->sample / 2];
    if (r[0].timestamp < *previous) return;   /* an empty pass can leave old slot contents untouched */
    *previous = r[1].timestamp;
    if (!d->last.cpu) return;
    double scale = (double)(d->last.cpu - d->first.cpu) / (double)(d->last.gpu - d->first.gpu);
    double start = d->first.cpu + ((int64_t)r[0].timestamp - (int64_t)d->first.gpu) * scale;
    double end = d->first.cpu + ((int64_t)r[1].timestamp - (int64_t)d->first.gpu) * scale;
    /* Reject stale samples and counter clock discontinuities. Leave room for calibration error. */
    if (start < (double)e->cb->gpu_start - 1000000 || end > (double)e->cb->gpu_end + 1000000
            || start <= 0 || end <= start || !e->cb->gpu_start) return;
    e->gpu_start = (uint64_t)start; e->gpu_end = (uint64_t)end;
}

static void trace_discard_cb(fp_trace_cb *c);

static fp_trace_cb *trace_cb_new(id cb, uint64_t uid, int queue, int mtl4)
{
    FPTraceOwner *owner = mtl4 ? objc_getAssociatedObject(cb, &g_trace_owner_key) : nil;
    if (owner && owner->record) {
        trace_discard_cb(owner->record);
        owner->record = NULL;
    }
    if (g_trace_stopped || g_frame_cur >= g_trace_last) return NULL;
    fp_trace_cb *c = calloc(1, sizeof *c);
    c->refs = 1; c->uid = uid; c->queue = queue; c->encoder_tail = &c->encoders; c->mtl4 = mtl4;
    if (mtl4 && !g_ring_history) g_ring_history = calloc(FP_POOL, sizeof *g_ring_history);
#ifdef FENCEPOST_TEST_HOOKS
    if (mtl4) {
        g_mtl4_trace_live++;
        g_mtl4_trace_peak = MAX(g_mtl4_trace_peak, g_mtl4_trace_live);
    }
#endif
    if (g_trace_gpu && !mtl4) c->device = trace_device([cb device]);
    if (!g_trace_origin) trace_start();
    if (!owner) {
        owner = [[FPTraceOwner alloc] init];
        objc_setAssociatedObject(cb, &g_trace_owner_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [owner release];
    }
    owner->record = c; c->owner = owner;
    return c;
}

static void trace_discard_encoder(fp_trace_encoder *e)
{
    if (e->uid) CFDictionaryRemoveValue(g_trace_nodes, TRACE_KEY(e->uid));
    trace_ready(&e->span);
    trace_release_sample(e);
    free(e->args); free(e->waits); free(e);
}

static void trace_free_cb(fp_trace_cb *c)
{
#ifdef FENCEPOST_TEST_HOOKS
    if (c->mtl4) g_mtl4_trace_live--;
#endif
    free(c);
}

static void trace_discard_cb(fp_trace_cb *c)
{
    fp_trace_encoder *e = c->encoders;
    while (e) {
        fp_trace_encoder *next = e->next;
        trace_discard_encoder(e);
        e = next;
    }
    trace_free_cb(c);
}

@implementation FPTraceOwner
- (void)dealloc
{
    if (record) {
        pthread_mutex_lock(&g_lock);
        trace_discard_cb(record);
        pthread_mutex_unlock(&g_lock);
    }
    [super dealloc];
}
@end

static fp_trace_encoder *trace_encoder_begin(id cb)
{
    uint64_t start = trace_clock();
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    fp_trace_encoder *e = NULL;
    if (c && c->trace && !g_trace_stopped && g_frame_cur < g_trace_last) {
        e = calloc(1, sizeof *e);
        e->start = start; e->sample = -1; e->cb = c->trace; e->span.encoder = 1;
        pthread_threadid_np(NULL, &e->thread);
        *e->cb->encoder_tail = e; e->cb->encoder_tail = &e->next;
    }
    pthread_mutex_unlock(&g_lock);
    return e;
}

static int trace_take_sample(fp_trace_encoder *e)
{
    pthread_mutex_lock(&g_lock);
    fp_trace_device *d = e->cb->device;
    if (!g_trace_stopped) {
        if (d->nfree) {
            if (!d->first.cpu) trace_calibrate(d);
            e->sample = (int)d->free_slots[--d->nfree];
        } else g_trace_pool_exhausted++;
    }
    pthread_mutex_unlock(&g_lock);
    return e->sample >= 0;
}

static void trace_encoder_register(fp_trace_encoder *e, fp_node *n)
{
    e->uid = n->uid; e->kind = n->kind;
    CFDictionarySetValue(g_trace_nodes, TRACE_KEY(e->uid), &e->span);
}

static void trace_encoder_end(const fp_node *n)
{
    fp_trace_encoder *e = (fp_trace_encoder *)trace_node(n->uid);
    if (e) e->end = trace_clock();
}

/* Lay out the selected encoders in encode order before any links are settled. */
static void trace_layout_encoders(fp_trace_cb *c)
{
    if (!c->gpu_start || c->gpu_end <= c->gpu_start) return;
    uint64_t prev_end = c->gpu_start;
    uint64_t limit = 0;
    for (fp_trace_encoder *e = c->encoders; e; e = e->next) {
        if (!e->selected) continue;
        e->span.gpu = 1;
        if (trace_sampled(e)) {
            e->span.sampled = 1;
            e->span.start = e->gpu_start;
            e->span.end = e->gpu_end;
            limit = 0;
        } else {
            if (!limit) {
                fp_trace_encoder *next = e->next;
                while (next && (!next->selected || !trace_sampled(next))) next = next->next;
                limit = next ? MIN(next->gpu_start, c->gpu_end) : c->gpu_end;
            }
            uint64_t start = MAX(c->gpu_start, MIN(prev_end, c->gpu_end));
            e->span.start = start;
            e->span.end = MAX(start, MIN(start + 1000, limit));
        }
        prev_end = e->span.end;
    }
}

/* Keep resolved endpoints only as long as the detector can refer to their
 * UIDs. An encoder waiting for analysis still owns a reference to its CB. */
static void trace_archive(uint64_t uid, fp_trace_span *span)
{
    CFDictionaryRemoveValue(g_trace_nodes, TRACE_KEY(uid));
    if (!g_ring_history) return;
    fp_trace_endpoint *slot = &g_ring_history[uid % FP_POOL];
    if (slot->uid < uid) *slot = (fp_trace_endpoint){ uid, *span };
}

static void trace_release_cb(fp_trace_cb *c)
{
    if (!--c->refs) trace_free_cb(c);
}

static void trace_place_encoder_impl(fp_trace_encoder *e, int finalizing)
{
    if ((!e->args && !e->args_dropped) || e->span.ready) return;
    if (e->args_dropped) goto done;
    int gpu = e->span.gpu && trace_sampled(e);
    if (!e->span.gpu) {
        e->span.start = e->start;
        e->span.end = e->end > e->start ? e->end : e->start + 1;
    }
    if (!g_trace_stopped) {
        e->span.tid = trace_queue_lane(e->cb->queue, e->span.gpu ? TR_GPU : TR_CPU, &e->span);
        const char *kind = kind_name[e->kind];
        char name[96];
        if (e->label[0]) snprintf(name, sizeof name, "%s", e->label);
        else snprintf(name, sizeof name, "%s #%llu", kind, TRACE_U64(e->uid));
        char *record = NULL;
        size_t length = 0;
        FILE *f = open_memstream(&record, &length);
        if (!f) {
            trace_drop_record(1);
            e->span.tid = 0;
            goto done;
        }
        /* At an equal timestamp Perfetto binds to the most recent slice. Put
         * known markers before their encoder so arrows bind to the encoder. */
        unsigned markers[4] = {0}, count = 0;
        for (fp_trace_link *link = e->span.waiting; link; link = link->next) {
            if (link->to != e->uid || !link->title) continue;
            fp_trace_span *from = trace_node(link->from);
            /* A laid-out source has its timing even before its slice is written. */
            if (from && !from->ready && !from->start) continue;
            if (g_ring) {
                trace_marker(link, &e->span, from);
                trace_marker_clear(link);
                continue;
            }
            trace_marker_record(f, link, &e->span, from);
            fputs(",\n", f);
            markers[link->kind - TR_RACE]++;
            count++;
            trace_marker_clear(link);
        }
        trace_slice(f, &e->span, kind, name);
        fprintf(f, ",\"args\":{%s,\"gpu_time\":%s%s}}", e->args, gpu ? "true" : "false",
                trace_estimated(&e->span) ? ",\"estimated\":true" : "");
        if (trace_store(e->span.frame, f, &record, &length, count + 1, gpu ? REC_SAMPLED : REC_ENCODER, 0)) {
            g_trace_markers += count;
            for (int i = 0; i < 4; i++) g_trace_marker_counts[i] += markers[i];
            g_trace_encoders++;
            if (gpu) g_trace_sampled++;
            else if (g_trace_gpu) g_trace_skipped++;
            fp_trace_track *track = (fp_trace_track *)CFDictionaryGetValue(g_trace_threads, TRACE_KEY(e->thread));
            if (!track) {
                track = calloc(1, sizeof *track);
                CFDictionarySetValue(g_trace_threads, TRACE_KEY(e->thread), track);
            }
            char thread[64];
            snprintf(thread, sizeof thread, "CPU encode %llu", TRACE_U64(e->thread));
            fp_trace_span cpu = { .frame = e->span.frame, .start = e->start, .end = e->end > e->start ? e->end : e->start + 1 };
            cpu.tid = trace_lane(track, thread, &cpu);
            char *cpu_record = NULL;
            size_t cpu_length = 0;
            f = open_memstream(&cpu_record, &cpu_length);
            if (f) {
                trace_slice(f, &cpu, "cpu encode", name);
                fprintf(f, ",\"args\":{\"uid\":%llu}}", TRACE_U64(e->uid));
            }
            trace_write(e->span.frame, f, &cpu_record, &cpu_length, 1);
        } else if (!g_ring || e->span.frame >= g_ring_floor) e->span.tid = 0;
    } else g_trace_dropped++;
done:
    free(e->args); e->args = NULL;
    trace_ready(&e->span);
    if (!finalizing && (g_ring || e->cb->mtl4)) {
        fp_trace_chunk *chunk = e->span.counted ? trace_chunk(e->span.frame) : NULL;
        if (chunk) chunk->outstanding--;
        trace_archive(e->uid, &e->span);
        trace_release_cb(e->cb);
        free(e);
    }
}

static void trace_place_encoder(fp_trace_encoder *e)
{
    trace_place_encoder_impl(e, 0);
}

static const char *trace_stage(const fp_node *n, int stage, const char *nonrender)
{
    return n->kind == FPK_RENDER ? (stage ? "fragment" : "vertex") : nonrender;
}

static void trace_analyse(const fp_node *n)
{
    if (n->kind == FPK_SIGNAL) return;
    fp_trace_encoder *e = (fp_trace_encoder *)trace_node(n->uid);
    if (!e || !e->selected) {
        if (g_trace_stopped && trace_frame_selected(n->frame)) g_trace_dropped++;
        return;
    }
    if (g_trace_stopped) { g_trace_dropped++; trace_ready(&e->span); return; }
    size_t length = 0;
    memcpy(e->label, n->label, sizeof e->label);
    FILE *f = open_memstream(&e->args, &length);
    if (!f) goto drop;
    fprintf(f, "\"uid\":%llu,\"cb\":%llu,\"queue\":%d,\"frame\":%llu,\"chain_depth\":%u,\"accesses\":%d,"
            "\"after_declared_barrier\":%s", TRACE_U64(n->uid),
            TRACE_U64(e->cb->mtl4 ? e->cb->uid : n->cb_uid), e->cb->queue,
            TRACE_U64(n->frame + 1), n->depth, n->nacc, n->req_mark ? "true" : "false");
    if (e->cb->mtl4) fprintf(f, ",\"logical_cb\":%llu", TRACE_U64(n->cb_uid));
    trace_label(f, "kind", kind_name[n->kind]);
    trace_label(f, "label", n->label);
    trace_label(f, "cb_label", e->cb->mtl4 ? e->mtl4_cb_label : e->cb->label);
    fputs(",\"fence_waits\":[", f);
    for (int i = 0; i < n->nfw; i++)
        fprintf(f, "%s{\"fence\":\"%p\",\"last_updater_uid\":%llu,\"stage\":\"%s\"}", i ? "," : "",
                n->fw[i].f, TRACE_U64(n->fw[i].src), trace_stage(n, n->fw[i].st, "start"));
    fputs("],\"fence_updates\":[", f);
    for (int i = 0; i < n->nfu; i++)
        fprintf(f, "%s{\"fence\":\"%p\",\"stage\":\"%s\"}", i ? "," : "",
                n->fu[i].f, trace_stage(n, n->fu[i].st, "end"));
    fputs("],\"event_waits\":[", f);
    for (int i = 0; i < e->nwaits; i++)
        fprintf(f, "%s{\"event\":\"%p\",\"value\":%llu}", i ? "," : "", e->waits[i].ev, TRACE_U64(e->waits[i].val));
    fputc(']', f);
    if (!trace_close_record(f)) goto drop;
    free(e->waits); e->waits = NULL;
    int fence_edges = 0;
    for (int i = 0; i < n->nfw; i++) if (n->fw[i].src) fence_edges++;
    for (int i = 0; i < n->ned; i++)
        trace_ordering(n->ed[i].src, n->uid, i < fence_edges ? TR_FENCE : TR_EVENT);
    if (e->cb->completed) trace_place_encoder(e);
    return;

drop:
    trace_drop_record(1);
    e->args_dropped = 1;
    free(e->waits); e->waits = NULL;
    if (e->cb->completed) trace_place_encoder(e);
}

static void trace_phase_node(fp_trace_cb *c, const fp_node *n, const fp_evwait *waits, int count)
{
    if (g_ring) g_ring_latest = n->frame;
    if (n->kind != FPK_SIGNAL && !c->has_encoder) {
        c->has_encoder = 1;
        c->span.frame = n->frame;
    }
    if (!trace_frame_selected(n->frame) || g_trace_stopped) return;
    c->selected = 1;
    if (n->kind == FPK_SIGNAL) {
        CFDictionarySetValue(g_trace_nodes, TRACE_KEY(n->uid), &c->span);
        if (g_ring) PUSH(c->signals, c->nsignals, c->csignals, n->uid);
        return;
    }
    fp_trace_encoder *e = (fp_trace_encoder *)trace_node(n->uid);
    if (!e) return;
    if (g_ring) {
        fp_trace_chunk *chunk = trace_chunk(n->frame);
        if (chunk) {
            chunk->outstanding++;
            e->span.counted = 1;
        }
    }
    if (g_ring || c->mtl4) c->refs++;
    e->span.frame = n->frame;
    e->selected = 1;
    if (count) {
        e->waits = malloc((size_t)count * sizeof *waits);
        memcpy(e->waits, waits, (size_t)count * sizeof *waits);
        e->nwaits = count;
    }
}

/* Both sides can arrive first: an enqueue-order hold can outlive GPU completion. */
static void trace_cb_finish(fp_trace_cb *c)
{
    if (!c->phased || !c->completed) return;
    if (c->gpu_error) {
        ring_trigger(DUMP_AUTO, "gpu-error", "gpu-error", c->span.frame, c->error_time, c->gpu_error);
        free(c->gpu_error);
        c->gpu_error = NULL;
    }
    trace_layout_encoders(c);
    if (c->selected) {
        c->span.start = c->gpu_start; c->span.end = c->gpu_end;
        if (c->gpu_start && c->gpu_end > c->gpu_start && !g_trace_stopped) {
            c->span.gpu = 1;
            c->span.tid = trace_queue_lane(c->queue, TR_CB, &c->span);
            char name[96];
            snprintf(name, sizeof name, "%s #%llu%s%s", c->mtl4 ? "MTL4 commit" : "CB",
                     TRACE_U64(c->uid), c->label[0] ? " " : "", c->label);
            char *record = NULL;
            size_t length = 0;
            FILE *f = open_memstream(&record, &length);
            if (f) {
                trace_slice(f, &c->span, "command buffer", name);
                fprintf(f, ",\"args\":{\"uid\":%llu,\"queue\":%d}}", TRACE_U64(c->uid), c->queue);
            }
            if (!trace_write(c->span.frame, f, &record, &length, 1) &&
                (!g_ring || c->span.frame >= g_ring_floor)) c->span.tid = 0;
        }
        trace_ready(&c->span);
    }
    if (c->span.tid)
        for (int i = 0; i < FP_MAXQ; i++) {
            fp_trace_endpoint *from = &c->completed_before[i];
            if (!from->uid) continue;
            fp_trace_link link = { .from = from->uid, .to = c->uid, .kind = TR_COMPLETED };
            trace_flow(&link, &from->span, &c->span);
        }
    free(c->completed_before);
    c->completed_before = NULL;
    fp_trace_encoder *e = c->encoders;
    c->encoders = NULL;
    while (e) {
        fp_trace_encoder *next = e->next;
        if (e->selected) trace_place_encoder(e);
        else trace_discard_encoder(e);
        e = next;
    }
    g_trace_completed[c->queue] = (fp_trace_endpoint){ c->span.tid ? c->uid : 0, c->span };
    if (g_ring || c->mtl4) {
        for (int i = 0; i < c->nsignals; i++) trace_archive(c->signals[i], &c->span);
        free(c->signals);
        fp_trace_chunk *chunk = c->span.counted ? trace_chunk(c->span.frame) : NULL;
        if (chunk) chunk->outstanding--;
        trace_release_cb(c);
    } else if (!c->selected) trace_release_cb(c);
}

static void trace_cb_completed(fp_trace_cb *c, id<MTLCommandBuffer> cb)
{
    if (!c) return;
    @autoreleasepool {
        if (g_ring && g_ring_gpuerr && (cb.status == MTLCommandBufferStatusError || cb.error != nil)) {
            c->error_time = trace_clock();
            c->gpu_error = strdup(cb.error.localizedDescription.UTF8String ?: "command buffer error");
        }
        double start = cb.GPUStartTime, end = cb.GPUEndTime;
        if (cb.status == MTLCommandBufferStatusCompleted && start > 0 && end >= start
                && end <= (double)trace_clock() / 1e9 + 1) {
            c->gpu_start = (uint64_t)(start * 1e9); c->gpu_end = (uint64_t)(end * 1e9);
        }
        if (c->device && c->device->buffer) trace_calibrate(c->device);
        for (fp_trace_encoder *e = c->encoders; e; e = e->next) {
            if (e->sample < 0) continue;
            if (!g_trace_stopped) trace_resolve(e);
            trace_release_sample(e);
        }
    }
    c->completed = 1;
    trace_cb_finish(c);
}

static int compare_duration(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static int frame_duration_spiked(uint64_t duration, const uint64_t history[120],
                                 uint64_t count, double factor, double min_ns)
{
    if (count < 120) return 0;
    uint64_t sorted[120];
    memcpy(sorted, history, sizeof sorted);
    qsort(sorted, 120, sizeof *sorted, compare_duration);
    double median = 0.5 * sorted[59] + 0.5 * sorted[60];
    return duration > MAX(factor * median, median + min_ns);
}

static void capture_frame_boundary(void)
{
    if (!g_capture_available || !g_spike_enabled || !(g_capture_on & CAP_SPIKE)) return;
    uint64_t now = trace_clock();
    if (g_capture_boundary) {
        uint64_t duration = now - g_capture_boundary;
        if (frame_duration_spiked(duration, g_capture_durations, g_capture_duration_count, FP_SPIKE_FACTOR, FP_SPIKE_MIN_NS))
            capture_request("spike");
        g_capture_durations[g_capture_duration_count++ % 120] = duration;
    }
    g_capture_boundary = now;
}

static void ring_frame_duration(uint64_t frame, uint64_t timestamp)
{
    if (!frame || !g_spike_enabled || !g_spike_factor) return; /* frame 1 starts at first CB, not at a boundary */
    uint64_t duration = timestamp - g_trace_frame_start;
    if (frame_duration_spiked(duration, g_durations, g_duration_count, g_spike_factor, g_spike_min_ns))
        ring_trigger(DUMP_AUTO, "spike", "spike", frame, timestamp, NULL);
    g_durations[g_duration_count++ % 120] = duration;
}

static void trace_frame_boundary(uint64_t frame, uint64_t timestamp, fp_trace_cb *c)
{
    if (g_ring) g_ring_latest = frame;
    if (!g_trace_frame_start) g_trace_frame_start = timestamp;
    if (timestamp < g_trace_frame_start) timestamp = g_trace_frame_start;
    if (g_ring) ring_frame_duration(frame, timestamp);
    if (trace_frame_selected(frame) && !g_trace_stopped) {
        fp_trace_frame *item = malloc(sizeof *item);
        *item = (fp_trace_frame){ frame, g_trace_frame_start, timestamp, NULL };
        *g_trace_frame_tail = item;
        g_trace_frame_tail = &item->next;
        if (c) c->selected = 1;
    }
    g_trace_frame_start = timestamp;
}

static void trace_frame_done(uint64_t frame, const fp_frame *r)
{
    fp_trace_frame *item = g_trace_frames;
    if (!item || item->frame != frame) return;
    g_trace_frames = item->next;
    if (!g_trace_frames) g_trace_frame_tail = &g_trace_frames;
    if (!g_frames_tid) g_frames_tid = trace_name_track("Frames");
    if (g_ring) {
        fp_trace_chunk *chunk = trace_chunk(frame);
        if (chunk) {
            chunk->start = item->start;
            chunk->end = item->end;
        }
    }
    char name[40], *record = NULL;
    size_t length = 0;
    snprintf(name, sizeof name, "Frame %llu", TRACE_U64(frame + 1));
    fp_trace_span span = { .start = item->start, .end = item->end, .tid = g_frames_tid };
    free(item);
    FILE *f = open_memstream(&record, &length);
    if (!f) { trace_drop_record(1); return; }
    trace_slice(f, &span, "frame", name);
    fprintf(f, ",\"args\":{\"frame\":%llu,\"encoders\":%ld,\"longest_chain\":%u,\"parallelism\":",
            TRACE_U64(frame + 1), r->nenc, r->depth);
    /* Ratio with six fractional digits, without a locale-dependent decimal point. */
    uint64_t ratio = r->depth ? (uint64_t)r->nenc * 1000000 / r->depth : 0;
    fprintf(f, "%llu.%06llu,\"gpu_idle_commits\":%ld}}", TRACE_U64(ratio / 1000000), TRACE_U64(ratio % 1000000), r->idle);
    trace_write(frame, f, &record, &length, 1);
}

static void trace_final_node(const void *key, const void *value, void *context)
{
    (void)key; (void)context;
    fp_trace_span *span = (fp_trace_span *)value;
    if (!span->ready && span->encoder) trace_place_encoder_impl((fp_trace_encoder *)span, 1);
    if (!span->ready) trace_ready(span);
}

typedef struct {
    uint64_t encoders, sampled, skipped, flows, markers, dropped, hits[4], counts[4];
    uint64_t outside_flows, outside_markers;
} fp_trace_counts;

typedef struct {
    long stats[ST_COUNT];
    uint64_t pool_exhausted, suppressed, evictions, cap_evictions, alloc_dropped;
    uint64_t origin;
    int stopped, sampling_supported;
} fp_trace_summary;

static fp_trace_summary trace_summary_state(void)
{
    fp_trace_summary state = {
        .pool_exhausted = g_trace_pool_exhausted, .stopped = g_trace_stopped,
        .origin = g_trace_origin, .sampling_supported = g_trace_sampling_supported,
        .suppressed = g_suppressed, .evictions = g_ring_evictions, .cap_evictions = g_ring_cap_evictions,
        .alloc_dropped = g_trace_alloc_dropped
    };
    memcpy(state.stats, g_st, sizeof state.stats);
    return state;
}

static void trace_summary(FILE *f, fp_trace_counts totals, const fp_trace_summary *state, uint64_t timestamp)
{
    static const char *const stats[] = {
        "race_unique", "possible_unique", "race_hits", "possible_hits", "nodes", "accesses", "cbs",
        "heap_uses", "parallel", "tracked", "timeouts", "uncertain", "window", "lost",
        "order_unique", "order_hits", "partial_unique", "partial_hits", "barriers",
        "objs", "obj_drops", "analysis_ns", "frames", "frame_enc", "frame_chain", "gpu_idle"
    };
    _Static_assert(sizeof stats / sizeof *stats == ST_COUNT, "trace summary must include all detector counters");
    trace_header_at(f, 'i', 0, timestamp, "fencepost", "fencepost summary", state->origin ?: timestamp);
    fprintf(f, ",\"s\":\"p\",\"args\":{\"encoders_traced\":%llu,\"sampled\":%llu,\"sampling_skipped\":%llu,\"pool_exhausted\":%llu,"
            "\"flows\":%llu,\"markers\":%llu,\"dropped\":%llu,\"sampling_supported\":%s,\"size_limit_reached\":%s",
            TRACE_U64(totals.encoders), TRACE_U64(totals.sampled), TRACE_U64(totals.skipped),
            TRACE_U64(state->pool_exhausted), TRACE_U64(totals.flows),
            TRACE_U64(totals.markers), TRACE_U64(totals.dropped), state->sampling_supported ? "true" : "false",
            state->stopped ? "true" : "false");
    for (int i = 0; i < ST_COUNT; i++) fprintf(f, ",\"%s\":%ld", stats[i], state->stats[i]);
    static const char *const types[] = { "race", "possible", "order", "partial" };
    for (int i = 0; i < 4; i++)
        fprintf(f, ",\"%s_marker_hits\":%llu,\"%s_markers\":%llu", types[i], TRACE_U64(totals.hits[i]),
                types[i], TRACE_U64(totals.counts[i]));
    if (g_ring) {
        fprintf(f, ",\"suppressed_triggers\":%llu", TRACE_U64(state->suppressed));
        fprintf(f, ",\"evicted_frames\":%llu,\"memory_evictions\":%llu,\"outside_flows\":%llu,\"outside_markers\":%llu",
                TRACE_U64(state->evictions), TRACE_U64(state->cap_evictions),
                TRACE_U64(totals.outside_flows), TRACE_U64(totals.outside_markers));
    }
    if (state->alloc_dropped)
        fprintf(f, ",\"allocation_drops\":%llu", TRACE_U64(state->alloc_dropped));
    fputs("}}", f);
}

static void ring_trigger(unsigned kind, const char *reason, const char *source, uint64_t frame,
                         uint64_t timestamp, const char *error)
{
    if (!g_ring_encoded || g_trace_stopped || !trace_frame_selected(frame)) return;
    uint64_t now = trace_clock();
    const char *refusal = NULL;
    const char *shown = reason && *reason && *reason != '\n' && *reason != '\r' ? reason : "manual";
    int shown_len = (int)MIN(strcspn(shown, "\r\n"), 32);
    if (g_dumps[kind] + g_pending_dumps[kind] >= g_max_dumps[kind]) refusal = "dump limit reached";
    else if (g_last_fire[kind] && now - g_last_fire[kind] < g_cooldown_ns) refusal = "cooldown in effect";
    else if (g_trigger.active && kind == DUMP_MANUAL && !g_queued.active) {
        /* Don't drop a request someone made by hand: save it as the next dump. */
        g_queued.active = 1;
        g_queued.frame = frame;
        g_queued.timestamp = timestamp;
        g_queued.source = source;
        snprintf(g_queued.reason, sizeof g_queued.reason, "%s", reason ? reason : "");
        g_last_fire[kind] = now;
        fp_log("fencepost: dump request (%.*s) queued behind the pending %s dump\n", shown_len, shown, g_trigger.reason);
        return;
    } else if (g_trigger.active) refusal = "another dump is pending";
    if (refusal) {
        g_suppressed++;
        if (kind == DUMP_MANUAL)
            fp_log("fencepost: dump request (%.*s) refused: %s\n", shown_len, shown, refusal);
        return;
    }
    g_last_fire[kind] = now;
    fp_trigger trigger = { .active = 1, .kind = kind, .frame = frame, .timestamp = timestamp, .source = source };
    trigger.first = frame >= g_ring ? frame - g_ring + 1 : 0;
    trigger.last = frame + g_ring_after;
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)(reason ? reason : "");
         *p && *p != '\n' && *p != '\r' && n < sizeof trigger.reason - 1; p++) {
        unsigned c = *p;
        if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        trigger.reason[n++] = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ? c : '-';
    }
    if (!n) strcpy(trigger.reason, "manual");
    if (error) trigger.error = strdup(error);
    g_trigger = trigger;
}

__attribute__((visibility("default"))) void fencepost_trigger(const char *reason)
{
    if (!g_ring) return;
    pthread_mutex_lock(&g_lock);
    ring_trigger(DUMP_MANUAL, reason, "api", g_frame_cur, trace_clock(), NULL);
    pthread_mutex_unlock(&g_lock);
}

typedef struct {
    fp_trace_chunk *owner;
    const fp_trace_record *first, *last;
    uint64_t marker_hits[4];
} fp_dump_chunk;

typedef struct {
    fp_trigger trigger;
    fp_trace_summary summary;
    int frames_tid;
    uint64_t first, timestamp;
    size_t count;
    fp_dump_chunk chunks[];
} fp_ring_dump;

/* Snapshot the record boundaries. The live ring can append to a chunk or
 * evict it while the writer reads the pinned records through the saved tail. */
static fp_ring_dump *ring_snapshot(void)
{
    if (!g_frames_tid) g_frames_tid = trace_name_track("Frames");
    uint64_t first = MAX(g_trigger.first, g_ring_floor);
    fp_trace_chunk *start = g_trace_chunks;
    while (start && start->frame < first) start = start->next;
    size_t count = 1;
    for (fp_trace_chunk *c = start; c && c->frame <= g_trigger.last; c = c->next) count++;
    fp_ring_dump *dump = malloc(sizeof *dump + count * sizeof *dump->chunks);
    if (!dump) return NULL;
    dump->trigger = g_trigger;
    g_trigger.error = NULL;
    dump->frames_tid = g_frames_tid;
    dump->first = first;
    dump->timestamp = g_trigger.timestamp;
    dump->count = count;
    dump->chunks[0] = (fp_dump_chunk){ .first = g_trace_metadata.first, .last = g_trace_metadata.last };
    size_t i = 1;
    for (fp_trace_chunk *c = start; c && c->frame <= g_trigger.last; c = c->next) {
        if (c->frame == g_trigger.frame && c->end)
            dump->timestamp = MAX(c->start, MIN(dump->timestamp, c->end));
        atomic_fetch_add(&c->refs, 1);
        fp_dump_chunk *chunk = &dump->chunks[i++];
        *chunk = (fp_dump_chunk){ .owner = c, .first = c->first, .last = c->last };
        memcpy(chunk->marker_hits, c->marker_hits, sizeof chunk->marker_hits);
    }
    if (!g_trace_origin) g_trace_origin = dump->timestamp;
    dump->summary = trace_summary_state();
    return dump;
}

static int ring_write_dump(FILE *out, const fp_ring_dump *dump)
{
    int started = 0;
    fp_trace_counts totals = {0};
    const fp_trigger *trigger = &dump->trigger;
    if (fputc('[', out) == EOF) return 0;
    for (size_t index = 0; index < dump->count; index++) {
        const fp_dump_chunk *c = &dump->chunks[index];
        for (int i = 0; i < 4; i++) totals.hits[i] += c->marker_hits[i];
        for (const fp_trace_record *r = c->first; r; r = r == c->last ? NULL : r->next) {
            if (r->type >= REC_FLOW && (r->source < dump->first || r->source > trigger->last)) {
                if (r->type == REC_FLOW) totals.outside_flows++;
                else totals.outside_markers++;
                continue;
            }
            if (!trace_emit(out, &started, r)) return 0;
            if (r->type == REC_FLOW) totals.flows++;
            if (r->type >= REC_MARKER) {
                totals.markers++;
                totals.counts[r->type - REC_MARKER]++;
            }
            if (r->type == REC_SAMPLED) totals.sampled++;
            if (r->type == REC_ENCODER || r->type == REC_SAMPLED) totals.encoders++;
        }
    }
    totals.skipped = g_trace_gpu ? totals.encoders - totals.sampled : 0;
    totals.dropped = totals.outside_flows * 2;
    for (int i = 0; i < 4; i++) totals.dropped += totals.hits[i] - totals.counts[i];

    char title[48];
    snprintf(title, sizeof title, "TRIGGER %s", trigger->reason);
    fputs(started ? ",\n" : "\n", out);
    trace_header_at(out, 'i', dump->frames_tid, dump->timestamp, "frame", title, dump->summary.origin);
    fprintf(out, ",\"s\":\"t\",\"args\":{\"frame\":%llu", TRACE_U64(trigger->frame + 1));
    trace_label(out, "reason", trigger->reason);
    trace_label(out, "source", trigger->source);
    if (trigger->error) trace_label(out, "error", trigger->error);
    fputs("}},\n", out);
    trace_summary(out, totals, &dump->summary, trace_clock());
    return fputs("\n]\n", out) != EOF && !ferror(out);
}

static int ring_dump(const fp_ring_dump *dump)
{
    char *path = NULL;
    if (asprintf(&path, "%s-%d-f%llu-%s.json", g_ring_prefix, g_trace_pid,
                 TRACE_U64(dump->trigger.frame + 1), dump->trigger.reason) < 0) return 0;
    char *temp = NULL;
    int ok = 0;
    if (asprintf(&temp, "%s.tmp.XXXXXX", path) >= 0) {
        int fd = mkstemp(temp);
        if (fd >= 0) {
            FILE *out = fdopen(fd, "w");
            if (out) {
                ok = ring_write_dump(out, dump);
                if (fclose(out)) ok = 0;
            } else close(fd);
            /* Publish only a complete file, without replacing an earlier dump. */
            if (ok) ok = renamex_np(temp, path, RENAME_EXCL) == 0;
            int saved = errno;
            if (!ok) unlink(temp);
            errno = saved;
        }
        free(temp);
    }
    if (ok) fp_log("fencepost: timeline dump %s\n", path);
    else fp_log("fencepost: could not write timeline dump %s: %s\n", path, strerror(errno));
    free(path);
    return ok;
}

/* Wait for both analysis and Metal completion in the captured window. A
 * busy queue can complete several frames behind the CPU. Never block it. */
static void ring_update(int final)
{
    if (g_trigger.active && (final || g_frame_next > g_trigger.last)) {
        int ready = 1;
        for (fp_trace_chunk *c = g_trace_chunks; !final && c; c = c->next)
            if (c->frame >= g_trigger.first && c->frame <= g_trigger.last && c->outstanding) ready = 0;
        if (ready) {
            fp_ring_dump *dump = ring_snapshot();
            free(g_trigger.error);
            g_trigger = (fp_trigger){0};
            if (g_queued.active) {
                g_queued.active = 0;
                g_last_fire[DUMP_MANUAL] = 0;   /* already charged when it was queued */
                ring_trigger(DUMP_MANUAL, g_queued.reason, g_queued.source, g_queued.frame, g_queued.timestamp, NULL);
            }
            if (dump) {
                unsigned kind = dump->trigger.kind;
                g_pending_dumps[kind]++;
                dispatch_async(g_ring_writer, ^{
                    /* The producer has released the detector lock before we start writing. */
                    pthread_mutex_lock(&g_lock);
                    pthread_mutex_unlock(&g_lock);
                    int ok = ring_dump(dump);
                    for (size_t i = 1; i < dump->count; i++) trace_release_chunk(dump->chunks[i].owner);
                    free(dump->trigger.error);
                    free(dump);
                    pthread_mutex_lock(&g_lock);
                    g_pending_dumps[kind]--;
                    if (ok) g_dumps[kind]++;
                    pthread_mutex_unlock(&g_lock);
                });
            }
        }
    }
    while (g_trace_chunks && g_ring_bytes > g_ring_limit) ring_evict(1);
    uint64_t first = g_ring_latest >= g_ring ? g_ring_latest - g_ring + 1 : 0;
    if (g_trigger.active) first = MIN(first, g_trigger.first);
    while (g_trace_chunks && g_trace_chunks->frame < first) ring_evict(0);
    g_ring_floor = MAX(first, g_ring_floor);
}

static void trace_final(void)
{
    while (g_trace_frames) {
        uint64_t frame = g_trace_frames->frame;
        trace_frame_done(frame, frame_get(frame));
    }
    if (g_ring) { ring_update(1); return; }
    CFDictionaryApplyFunction(g_trace_nodes, trace_final_node, NULL);
    fp_trace_counts totals = { g_trace_encoders, g_trace_sampled, g_trace_skipped, g_trace_flows,
                               g_trace_markers, g_trace_dropped, {0}, {0}, 0, 0 };
    memcpy(totals.hits, g_trace_marker_hits, sizeof totals.hits);
    memcpy(totals.counts, g_trace_marker_counts, sizeof totals.counts);
    fp_trace_summary state = trace_summary_state();
    uint64_t timestamp = trace_clock();
    trace_stream_flush();
    fputs(g_trace_has_output ? ",\n" : "\n", g_trace);
    trace_summary(g_trace, totals, &state, timestamp);
    fputs("\n]\n", g_trace);
    fclose(g_trace);
    g_trace = NULL;
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
    h = fnv(h, &f->kind, sizeof f->kind); h = fnv(h, &fa->via, 1); h = fnv_label(h, f->label);
    h = fnv(h, &s->kind, sizeof s->kind); h = fnv(h, &sa->via, 1); h = fnv_label(h, s->label);
    unsigned kind = certain ? TR_RACE : TR_POSSIBLE;
    g_st[certain ? ST_RACE_HITS : ST_POSS_HITS]++;
    int marker = g_trace_enabled && trace_marker_hit(s, kind);
    int print = !sig_seen(h);
    if (print) {
        g_st[certain ? ST_RACE_UNIQ : ST_POSS_UNIQ]++;
        print = certain || g_show_possible;
        if (print && g_printed++ >= g_max_reports) {
            if (g_printed - 1 == g_max_reports) fp_log("fencepost: report limit reached (FENCEPOST_MAX_REPORTS); counting only\n");
            print = 0;
        }
    }
    if (g_capture_on & CAP_RACE) capture_request("race");
    if (g_ring && g_ring_race && (print || g_ring_race == 2))
        ring_trigger(DUMP_AUTO, "race", "race", s->frame, trace_clock(), NULL);
    if (!print && !marker) return;

    const char *prefix = certain ? "RACE" : "possible race";
    char d1[256], d2[256], st[320], first[384], second[384], why[512];
    describe_node(d1, sizeof d1, f);
    describe_node(d2, sizeof d2, s);
    describe_store(st, sizeof st, ya, y->done_tick == 0);
    snprintf(first, sizeof first, "%s %s it: %s%s", d1, rw_verb(fa->rw), via_name[fa->via], stage_name(f, fa->sl));
    snprintf(second, sizeof second, "%s %s it: %s%s", d2, rw_verb(sa->rw), via_name[sa->via], stage_name(s, sa->sf));

    /* why: look for the fence that was meant to order them */
    int said = 0;
    for (int i = 0; i < s->nfw && !said; i++)
        for (int j = 0; j < f->nfu && !said; j++) {
            if (s->fw[i].f != f->fu[j].f) continue;
            said = 1;
            if (s->fw[i].src && s->fw[i].src != f->uid)
                snprintf(why, sizeof why, "the second waits on fence %p, but encoder #%llu updated that fence after the first "
                         "did, so the wait pairs with the later update (fence reused too early)",
                         s->fw[i].f, (unsigned long long)s->fw[i].src);
            else if (!s->fw[i].src)
                snprintf(why, sizeof why, "the second waits on fence %p, which the first updates, but the wait was submitted "
                         "first (the update must be submitted before the wait)", s->fw[i].f);
            else
                snprintf(why, sizeof why, "fence %p is updated %s and waited for %s, which does not cover these accesses",
                         s->fw[i].f, f->kind != FPK_RENDER ? "at the end of the first" :
                         f->fu[j].st ? "after the first's fragment stage" : "after the first's vertex stage only",
                         s->kind != FPK_RENDER ? "before the second starts" :
                         s->fw[i].st ? "before the second's fragment stage only" : "before the second's vertex stage");
        }
    if (!said) {
        if (f->cb_uid == s->cb_uid)
            snprintf(why, sizeof why, "both are in command buffer #%llu and no fence orders them; untracked resources need "
                     "updateFence in the first and waitForFence in the second", (unsigned long long)f->cb_uid);
        else if (!f->done_tick || f->done_tick >= s->commit_tick)
            snprintf(why, sizeof why, "the first's command buffer had not completed when the second's was committed, and "
                     "no fence or event orders them");
        else
            snprintf(why, sizeof why, "no fence or event orders them");
    }
    const char *note = "one side waited on an event fencepost never saw signalled; the ordering may exist outside its view";
    if (print) {
        fp_log("fencepost: %s %s on %s\n", prefix, what, st);
        fp_log("  first : %s\n", first);
        fp_log("  second: %s\n", second);
        fp_log("  why   : %s\n", why);
        if (!said && s->nfw) fp_log("          (the second waits on %d fence(s), none updated by the first or after it)\n", s->nfw);
        if (x->uncertain || y->uncertain) fp_log("  note  : %s\n", note);
    }
    if (marker) {
        char title[96];
        snprintf(title, sizeof title, "%s %s", prefix, what);
        size_t used = strlen(why);
        if (!said && s->nfw)
            snprintf(why + used, sizeof why - used,
                     "\n(the second waits on %d fence(s), none updated by the first or after it)", s->nfw);
        used = strlen(why);
        if (x->uncertain || y->uncertain) snprintf(why + used, sizeof why - used, "\n%s", note);
        trace_diagnostic(kind, f->uid, s->uid, title, st, first, second, why);
    }
}

/* ---------------------------------------------------------------- frames */

static fp_frame *frame_get(uint64_t f)
{
    fp_frame *r = &g_fr[f % FP_FRAMES];
    if (r->key != f + 1) { memset(r, 0, sizeof *r); r->key = f + 1; }
    return r;
}
static fp_frame *frame_of(uint64_t f)   /* NULL once the frame is finished */
{
    return f < g_frame_next || f + FP_FRAMES <= g_frame_cur ? NULL : frame_get(f);
}

static uint64_t run_hash(const fp_node *n)
{
    return fnv_label(fnv(0xcbf29ce484222325ull, &n->kind, sizeof n->kind), n->label);
}

/* The longest chain of frame f, oldest first, with runs of the same pass folded. */
static void print_chain(uint64_t f, const fp_frame *r)
{
    int cap = (int)r->depth, n = 0, via_ev = 0;
    fp_node **st = calloc((size_t)cap + 1, sizeof *st);
    uint8_t *ev_in = calloc((size_t)cap + 1, 1);   /* ev_in[i]: st[i] follows st[i + 1] through an event */
    for (fp_node *x = node_get(r->tail); x && x->frame == f && n < cap; x = node_get(x->chain_prev)) {
        if (x->kind == FPK_SIGNAL) { via_ev = 1; continue; }
        if (n) ev_in[n - 1] = (uint8_t)via_ev;
        st[n++] = x; via_ev = 0;
    }
    fp_log("fencepost: CHAIN frame %llu: %ld encoders; the longest chain the synchronisation forces to run one after "
           "another has %u (parallelism %.2f)%s\n", (unsigned long long)f + 1, r->nenc, r->depth,
           r->depth ? (double)r->nenc / r->depth : 0.0, n < (int)r->depth ? "; its oldest steps were recycled" : "");
    int lines = 0, step = (int)r->depth - n + 1;
    for (int i = n - 1; i >= 0; lines++) {
        uint64_t h = run_hash(st[i]);
        int j = i, fences = 0, events = 0;
        while (j > 0 && run_hash(st[j - 1]) == h) { j--; if (ev_in[j]) events++; else fences++; }
        if (lines < 100) {
            char steps[32], links[64] = "";
            if (i == j) snprintf(steps, sizeof steps, "%d", step);
            else snprintf(steps, sizeof steps, "%d-%d", step, step + i - j);
            if (i > j) snprintf(links, sizeof links, ", one after another through %s",
                                fences && events ? "fences and events" : events ? "events" : "fences");
            fp_log("  %-11s %-7s \"%s\"%s%s\n", steps, kind_name[st[i]->kind], st[i]->label[0] ? st[i]->label : "(no label)",
                   i < n - 1 && ev_in[i] ? " (after an event wait)" : "", links);
        } else if (lines == 100) fp_log("  ... more steps not shown\n");
        step += i - j + 1;
        i = j - 1;
    }
    free(st); free(ev_in);
}

static void frame_done(uint64_t f, const fp_frame *r)
{
    if (g_trace_enabled) trace_frame_done(f, r);
    if (!r->nenc) return;
    g_st[ST_FRAMES]++; g_st[ST_FRAME_ENC] = r->nenc; g_st[ST_FRAME_CHAIN] = r->depth;
    uint64_t k = g_nfhist++ % FP_FHIST;
    g_fhist[k].nenc = r->nenc; g_fhist[k].depth = r->depth; g_fhist[k].idle = r->idle;
    if (g_chain_frame && (uint64_t)g_chain_frame == f + 1) print_chain(f, r);
}

/* Finish frames whose nodes are all analysed (force: whatever is left). */
static void frames_settle(int force)   /* lock held */
{
    while (g_frame_next < g_frame_cur) {
        fp_frame *r = frame_get(g_frame_next);
        if (!force && r->analysed < r->queued && g_frame_cur - g_frame_next < FP_FRAMES / 2) break;
        frame_done(g_frame_next, r);
        g_frame_next++;
    }
}

static int cmp_long(const void *a, const void *b) { long x = *(const long *)a, y = *(const long *)b; return (x > y) - (x < y); }
static int cmp_dbl(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void frame_line(void)   /* lock held */
{
    int m = g_nfhist < FP_FHIST ? (int)g_nfhist : FP_FHIST;
    if (!m) return;
    static long enc[FP_FHIST], ch[FP_FHIST]; static double par[FP_FHIST];
    long idle = 0;
    for (int i = 0; i < m; i++) {
        enc[i] = g_fhist[i].nenc; ch[i] = g_fhist[i].depth; idle += g_fhist[i].idle;
        par[i] = ch[i] ? (double)enc[i] / ch[i] : 0;
    }
    qsort(enc, m, sizeof *enc, cmp_long); qsort(ch, m, sizeof *ch, cmp_long); qsort(par, m, sizeof *par, cmp_dbl);
    fp_log("fencepost:   frames: %ld; median of the last %d: %ld encoders, longest ordered chain %ld (parallelism "
           "%.2f); the GPU had run out of work at %.1f commits per frame\n", g_st[ST_FRAMES], m, enc[m / 2], ch[m / 2],
           par[m / 2], (double)idle / m);
}

static void summary(int final)
{
    pthread_mutex_lock(&g_lock);
    long hits = g_st[ST_RACE_HITS] + g_st[ST_POSS_HITS];
    if (!final && hits == g_last_summary_hits) {   /* nothing new to report: just the frame line, if it moved */
        if (g_st[ST_FRAMES] != g_last_summary_frames) {
            g_last_summary_frames = g_st[ST_FRAMES]; frame_line();
            if (g_trace) fflush(g_trace);
        }
        pthread_mutex_unlock(&g_lock);
        return;
    }
    g_last_summary_hits = hits;
    g_last_summary_frames = g_st[ST_FRAMES];
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
    fp_log("fencepost:   cost: %.2f s analysing at commit; %ld resources with live history%s\n",
           g_st[ST_ANALYSIS_NS] / 1e9, g_st[ST_OBJS], g_st[ST_OBJ_DROPS] ? " (TABLE FULL: some accesses not checked)" : "");
    frame_line();
    if (g_st[ST_HEAP_USES] || g_st[ST_PARALLEL] || g_st[ST_TIMEOUTS] || g_st[ST_UNCERTAIN] || g_st[ST_WINDOW] || g_st[ST_LOST])
        fp_log("fencepost:   blind spots: %ld useHeap calls (resources inside heaps not seen), %ld parallel render "
               "encoders (not followed), %ld event waits timed out, %ld nodes with unresolved event waits, "
               "%ld checks past the %d-encoder window, %ld nodes dropped\n",
               g_st[ST_HEAP_USES], g_st[ST_PARALLEL], g_st[ST_TIMEOUTS], g_st[ST_UNCERTAIN], g_st[ST_WINDOW],
               FP_W, g_st[ST_LOST]);
    if (g_trace) fflush(g_trace);
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

/* Forget history that can no longer race: entries of encoders that are gone, finished
 * (any later commit is ordered after them) or out of the window. A resource left with no
 * history leaves the table, so resources that come and go (streamed textures, transient
 * buffers) do not fill it. A slice per commit. */
static unsigned g_sweep;
static int stale(const fp_ent *e)
{
    fp_node *x = node_get(e->uid);
    return !x || (x->done_tick && x->done_tick < g_tick) || g_aseq - e->aseq >= FP_W;
}
static void sweep(void)   /* lock held */
{
    for (unsigned n = 0; n < FP_SWEEP; n++) {
        unsigned i = g_sweep++ & (FP_OBJMAP - 1);
        fp_obj *o = &g_obj[i];
        if (!o->key) continue;
        int j = 0;
        for (int k = 0; k < o->nw; k++) if (!stale(&o->w[k])) o->w[j++] = o->w[k];
        o->nw = j;
        j = 0;
        for (int k = 0; k < o->nr; k++) if (!stale(&o->r[k])) o->r[j++] = o->r[k];
        o->nr = j;
        if (!o->nw && !o->nr) { obj_del(i); g_sweep--; }   /* a later entry may have moved into i */
    }
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
    h = fnv(h, &y->kind, sizeof y->kind); h = fnv_label(h, y->label);
    h = fnv(h, &x->kind, sizeof x->kind); h = fnv_label(h, x->label);
    h ^= bad ? 0x5bd1e995ull : 0;
    unsigned kind = bad ? TR_ORDER : TR_FRAGMENT;
    g_st[bad ? ST_ORDER_HITS : ST_PARTIAL_HITS]++;
    int marker = g_trace_enabled && trace_marker_hit(y, kind);
    int print = !sig_seen(h);
    if (print) {
        g_st[bad ? ST_ORDER_UNIQ : ST_PARTIAL_UNIQ]++;
        print = g_printed < g_max_reports;
        if (print) g_printed++;
    }
    if (g_capture_on & CAP_RACE) capture_request("race");
    if (g_ring && g_ring_race && (print || g_ring_race == 2))
        ring_trigger(DUMP_AUTO, "race", "race", y->frame, trace_clock(), NULL);
    if (!print && !marker) return;
    char d1[256], d2[256], barrier[96], second[384];
    describe_node(d1, sizeof d1, x);
    describe_node(d2, sizeof d2, y);
    snprintf(barrier, sizeof barrier, "command buffer #%llu, after its encoder %d",
             (unsigned long long)y->bar_cb, y->bar_idx);
    if (bad)
        snprintf(second, sizeof second, "%s is not ordered after it%s", d2,
                 nbad > 1 ? " (nor after other encoders from before the barrier)" : "");
    else
        snprintf(second, sizeof second, "%s waits for it only before its fragment stage; its vertex stage can overlap", d2);
    if (print) {
        if (bad) fp_log("fencepost: ORDER a declared barrier is not honoured\n");
        else fp_log("fencepost: ORDER a declared barrier is honoured for the fragment stage only\n");
        fp_log("  barrier: %s\n", barrier);
        fp_log("  before : %s\n", d1);
        fp_log("  after  : %s\n", second);
    }
    char *why = NULL;
    size_t why_length = 0;
    FILE *details = marker ? open_memstream(&why, &why_length) : NULL;
    if (details) fputs(bad ? "a declared barrier is not honoured" : "a declared barrier is honoured for the fragment stage only", details);
    for (int i = 0; i < y->nfw; i++) {
        fp_node *u = node_get(y->fw[i].src);
        char wait[160];
        snprintf(wait, sizeof wait, "fence %p before its %s, last updated by %s #%llu", y->fw[i].f,
                 y->kind != FPK_RENDER ? "start" : y->fw[i].st ? "fragment stage" : "vertex stage",
                 u ? kind_name[u->kind] : "an old or unknown encoder", (unsigned long long)y->fw[i].src);
        if (print) fp_log("  waits  : %s\n", wait);
        if (details) fprintf(details, "\nwaits: %s", wait);
    }
    if (!y->nfw && !y->ned) {
        if (print) fp_log("  waits  : nothing\n");
        if (details) fputs("\nwaits: nothing", details);
    }
    if (marker) {
        if (trace_close_record(details))
            trace_diagnostic(kind, x->uid, y->uid, bad ? "ORDER barrier not honoured" : "ORDER fragment-only",
                             barrier, d1, second, why);
        else trace_drop_record(1);
        free(why);
    }
}

static void analyse(fp_node *y)   /* lock held; every edge source is analysed or gone */
{
    y->aseq = ++g_aseq;
    memset(y->anc, 0, sizeof y->anc);
    uint32_t best = 0;
    for (int i = 0; i < y->ned; i++) {
        const fp_edge *e = &y->ed[i];
        fp_node *x = node_get(e->src);
        if (!x || !x->aseq || x->aseq >= y->aseq) continue;
        if (x->frame == y->frame && x->depth > best) { best = x->depth; y->chain_prev = x->uid; }
        uint64_t d = y->aseq - x->aseq;
        if (d >= FP_W) continue;
        fp_bits *t = &y->anc[e->before];
        bit_set(t, 2 * d);
        if (e->after) bit_set(t, 2 * d + 1);
        bits_or_shift(t, &x->anc[e->after], 2 * d);
    }
    for (int i = 0; i < FP_WORDS; i++) y->anc[1].w[i] |= y->anc[0].w[i];   /* fragment work starts after vertex work */
    y->state = 4;
    y->depth = best + (y->kind != FPK_SIGNAL);   /* an event signal passes its chain on */
    fp_frame *r = frame_of(y->frame);
    if (r) {
        r->analysed++;
        if (y->kind != FPK_SIGNAL) { r->nenc++; if (y->depth > r->depth) { r->depth = y->depth; r->tail = y->uid; } }
    }
    if (y->kind != FPK_SIGNAL) g_st[ST_NODES]++;
    if (!y->mtl4) {
        if (y->kind != FPK_SIGNAL && y->req_q && y->req_mark) check_barrier(y);
        for (int i = 0; i < y->nacc; i++) check_access(y, &y->acc[i]);
    }
    if (g_trace_enabled) trace_analyse(y);
}

/* force 1 clears waits with no candidate, leaving pending signal dependencies
 * for another analysis pass. Force 2 is the final fallback. */
static int ready(fp_node *n, int force)
{
    int i, j;
    for (i = 0, j = 0; i < n->nev; i++) {
        fp_evwait w = n->ew[i];
        fp_ev *ev = ev_get(w.ev, 0);
        int done = 0;
        if (!ev || !ev->known) done = 1;   /* event made outside our view (e.g. from a mach port): CPU-side */
        else {
            int pending_signal = 0;
            uint64_t source = 0, first_settled = UINT64_MAX;
            unsigned first = ev->nh > FP_EVHIST ? ev->nh - FP_EVHIST : 0;
            for (unsigned k = first; k < ev->nh; k++) {
                if (ev->h[k % FP_EVHIST].val < w.val) continue;
                uint64_t uid = ev->h[k % FP_EVHIST].uid;
                fp_node *signal = node_get(uid);
                if (signal && signal->req_q == n->req_q && signal->queue_seq >= n->queue_seq) continue;
                /* An unresolved signal may itself be blocked by this wait,
                 * including through another queue. Only use settled sources. */
                if (signal && !signal->aseq) { pending_signal = 1; continue; }
                /* Prefer the first settled signal that reaches the requested
                 * value. A dependent signal can be registered before its source. */
                uint64_t aseq = signal ? signal->aseq : 0;
                if (aseq < first_settled) { first_settled = aseq; source = uid; }
            }
            if (source) {
                fp_edge e = { source, 1, w.before };
                PUSH(n->ed, n->ned, n->ced, e);
                done = 1;
            }
            if (!done && ev->cpu_max >= w.val) done = 1;
            if (!done && (force == 2 || (force && !pending_signal) || g_tick - n->pend_since > FP_PEND_TIMEOUT)) {
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
        for (int i = 0; i < g_npend; i++) {
            fp_node *n = node_get(g_pend[i]);
            if (n && n->state == 3) { (void)ready(n, 2); analyse(n); }
        }
        g_npend = 0;
    }
    frames_settle(force);
    if (g_ring) ring_update(0);
}

static void queue_recent(fp_queue *q, uint64_t uid)
{
    q->recent[q->nrecent++ % FP_QRECENT] = uid;
}

static void frame_boundary(uint64_t timestamp, fp_trace_cb *trace)
{
    if (g_trace_enabled) trace_frame_boundary(g_frame_cur, timestamp, trace);
    capture_frame_boundary();
    g_frame_cur++;
}

static fp_node *phase_signal(fp_queue *q, uint64_t cb_uid, uint64_t tick, fp_item *it)
{
    fp_node *n = node_new(FPK_SIGNAL);
    it->uid = n->uid;
    n->cb_uid = cb_uid; n->state = 2;
    unsigned first = q->nrecent > FP_QRECENT ? q->nrecent - FP_QRECENT : 0;
    for (unsigned k = first; k < q->nrecent; k++) {   /* the signal follows earlier work on this queue */
        fp_node *p = node_get(q->recent[k % FP_QRECENT]);
        if (!p || (p->done_tick && p->done_tick < tick)) continue;
        fp_edge e = { p->uid, 1, 0 };
        PUSH(n->ed, n->ned, n->ced, e);
    }
    fp_ev *ev = ev_get(it->ev, 1);
    if (ev) {
        ev->gpu = 1;
        if (q->mtl4) ev->known = 1;
        ev->h[ev->nh % FP_EVHIST].val = it->val;
        ev->h[ev->nh % FP_EVHIST].uid = n->uid;
        ev->nh++;
    }
    return n;
}

static void phase_node(fp_queue *q, fp_node *n, uint64_t tick, fp_trace_cb *trace,
                       const fp_evwait *waits, int nwaits)
{
    for (int k = 0; k < nwaits; k++) PUSH(n->ew, n->nev, n->cev, waits[k]);
    n->commit_tick = tick;
    n->pend_since = g_tick;
    n->req_q = q; n->req_mark = q->bar_mark; n->bar_cb = q->bar_cb; n->bar_idx = q->bar_idx;
    n->frame = g_frame_cur; frame_get(g_frame_cur)->queued++;
    if (trace) trace_phase_node(trace, n, waits, nwaits);
    n->state = 3;
    n->queue_seq = q->nrecent;
    queue_recent(q, n->uid);
    PUSH(g_pend, g_npend, g_cpend, n->uid);
}

/* Phase 1, in submission order: resolve fence waits against the latest update, record
 * fence updates and event signals, queue the nodes for analysis. */
static void phase1(fp_cb *c)
{
    fp_evwait active[32]; int nactive = 0;
    fp_queue *q = c->q;
    fp_trace_cb *trace = g_trace_enabled ? c->trace : NULL;
    if (trace && !c->mtl4) {
        trace->span.frame = g_frame_cur;  /* fallback for an encoderless CB */
        if (trace_frame_selected(g_frame_cur) && !g_trace_stopped) trace->selected = 1;
    }
    g_st[ST_CBS]++;
    for (int i = 0; i < c->nit; i++) {
        fp_item *it = &c->it[i];
        if (it->type == 3) {
            q->bar_mark = q->nrecent; q->bar_cb = c->uid; q->bar_idx = it->val > 0 ? (int)it->val : 0;
            g_st[ST_BARRIERS]++;
            continue;
        }
        if (it->type == 4) {
            frame_boundary(it->val, trace);
            continue;
        }
        if (it->type == 2) {
            if (nactive < 32) { active[nactive].ev = it->ev; active[nactive].val = it->val; active[nactive].before = 0; nactive++; }
            continue;
        }
        fp_node *n;
        if (it->type == 1) {
            n = phase_signal(q, c->uid, c->commit_tick, it);
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
        const fp_evwait *waits = c->mtl4 ? q->mtl4_waits : active;
        int nwaits = c->mtl4 ? q->mtl4_nwaits : nactive;
        phase_node(q, n, c->commit_tick, trace, waits, nwaits);
    }
    if (trace && !c->mtl4) {
        if (g_ring && trace->selected) {
            fp_trace_chunk *chunk = trace_chunk(trace->span.frame);
            if (chunk) {
                chunk->outstanding++;
                trace->span.counted = 1;
            }
        }
        trace->phased = 1;
        trace_cb_finish(trace);
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

static void cb_mark_done(uint64_t uid)
{
    fp_cb *c = cb_get(uid);
    if (c)
        for (int i = 0; i < c->nit; i++) {
            fp_node *n = c->it[i].type != 2 ? node_get(c->it[i].uid) : NULL;
            if (n && !n->done_tick) n->done_tick = g_tick;
        }
}

static void cb_done(uint64_t uid, fp_trace_cb *trace, id<MTLCommandBuffer> b)
{
    pthread_mutex_lock(&g_lock);
    g_inflight--;
    cb_mark_done(uid);
    if ((g_capture_on & CAP_GPU_ERROR) && b.status == MTLCommandBufferStatusError)
        capture_request("gpu-error");
    if (trace) trace_cb_completed(trace, b);
    if (g_ring) ring_update(0);
    if (g_trace) trace_stream_flush();
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
    H_CB_SIG, H_CB_WAIT, H_CB_COMMIT, H_CB_ENQ, H_CB_PRES, H_CB_PRES_T, H_CB_PRES_D,
    H_EV_SET,
    H_END, H_USE, H_USES, H_HEAP, H_HEAPS, H_UPD, H_WAIT,
    H_R_USE_ST, H_R_USES_ST, H_R_HEAP_ST, H_R_HEAPS_ST, H_R_UPD, H_R_WAIT,
    H_R_VTEX, H_R_FTEX, H_R_VTEXS, H_R_FTEXS, H_R_VBUF, H_R_FBUF, H_R_OBUF, H_R_MBUF, H_R_VBUFS, H_R_FBUFS,
    H_R_DIDX, H_R_DIDX_I, H_R_DIDX_IBB, H_R_DIND, H_R_DIDXIND, H_R_MESHIND,
    H_R_CSA, H_R_DSA, H_R_SSA,
    H_C_TEX, H_C_TEXS, H_C_BUF, H_C_BUFS, H_C_DISPIND,
    H_B_BB, H_B_BT, H_B_BT_O, H_B_TB, H_B_TB_O, H_B_TT, H_B_TT_N, H_B_TT_ALL, H_B_FILL, H_B_MIP, H_B_SYNCR, H_B_SYNCT,
    H_4_DEV_Q, H_4_DEV_Q_DESC, H_4_DEV_CB, H_4_BEGIN, H_4_BEGIN_OPTS, H_4_END,
    H_4_RENDER, H_4_RENDER_OPTS, H_4_COMPUTE, H_4_ML,
    H_4_COMMIT, H_4_COMMIT_OPTS, H_4_SIGNAL, H_4_WAIT, H_4_DRAWABLE_SIGNAL, H_4_DRAWABLE_WAIT,
    H_4_OPTIONS_INIT, H_4_ADD_FEEDBACK,
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
static void hook_mtl4_cb(Class cls);
static void hook_mtl4_encoder(Class cls);
static void hook_mtl4_queue(Class cls);

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

static fp_cb *cb_record_new(id cb, fp_queue *q, int mtl4)   /* g_lock held */
{
    uint64_t uid = ++g_cbuid;
    fp_cb *c = &g_cbs[uid % FP_CBPOOL];
    c->uid = uid; c->cb = (void *)cb; c->q = q; c->state = 1; c->nenc = 0; c->nit = 0; c->mtl4 = mtl4;
    c->enq = 0; c->commit_tick = 0; c->label[0] = 0;
    c->trace = g_trace_enabled ? trace_cb_new(cb, uid, q ? q->idx : -1, mtl4) : NULL;
    unsigned slot = FP_OPENCB;
    for (unsigned i = 0; i < FP_OPENCB; i++) {
        if (g_opencb[i].cb == (void *)cb) { slot = i; break; }   /* stale entry for a reused address */
        if (slot == FP_OPENCB && !g_opencb[i].cb) slot = i;
    }
    if (slot == FP_OPENCB) slot = g_opencb_rr++ % FP_OPENCB;
    g_opencb[slot].cb = (void *)cb; g_opencb[slot].uid = uid;
    return c;
}

static void on_cb(id q, id cb)
{
    if (!cb) return;
    pthread_mutex_lock(&g_lock);
    fp_queue *fq = queue_get((void *)q);
    if (fq) cb_record_new(cb, fq, 0);
    pthread_mutex_unlock(&g_lock);
    hook_cb(object_getClass(cb));
}
static int trigger_file_owned(const struct stat *st, const char *path, int *warned)
{
    if (st->st_uid == geteuid()) return 1;
    if (!*warned) {
        *warned = 1;
        fp_log("fencepost: ignoring request file owned by another user: %s\n", path);
    }
    return 0;
}

/* Keep the existing reason limit, but find line two even after a long reason. */
static int trigger_read_request(int fd, char *reason, size_t capacity,
                                struct timespec stamp, struct timespec *mtime)
{
    char buf[256];
    ssize_t n = read(fd, buf, sizeof buf);
    if (n <= 0) return 0;
    *mtime = stamp;
    size_t length = MIN((size_t)n, capacity - 1);
    memcpy(reason, buf, length);
    reason[length] = 0;

    for (off_t offset = 0; n > 0; offset += n, n = read(fd, buf, sizeof buf)) {
        const char *newline = memchr(buf, '\n', (size_t)n);
        if (!newline) continue;
        char target[32];
        n = pread(fd, target, sizeof target - 1, offset + (newline - buf) + 1);
        if (n < 0) return 0;
        if (n < 4 || memcmp(target, "pid=", 4)) return 1;
        target[n] = 0;
        char *end;
        long pid = strtol(target + 4, &end, 10);
        if (*end == '\r') end++;
        return pid == getpid() && (*end == '\n' || (end == target + n && (size_t)n < sizeof target - 1));
    }
    return n == 0;
}

/* Called under the capture lock. Acknowledge the timestamp of the inode read,
 * including file requests received while a capture is running. */
static int capture_poll_file(char reason[static sizeof g_capture_reason])
{
    if (!g_capture_file) return 0;
    struct stat st;
    if (stat(g_capture_file, &st) || !S_ISREG(st.st_mode) ||
        !trigger_time_newer(st.st_mtimespec, g_capture_mtime)) return 0;
    int fd = open(g_capture_file, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return 0;
    int changed = 0;
    if (!fstat(fd, &st) && S_ISREG(st.st_mode) &&
        trigger_file_owned(&st, g_capture_file, &g_capture_owner_warned) &&
        trigger_time_newer(st.st_mtimespec, g_capture_mtime)) {
        changed = trigger_read_request(fd, reason, sizeof g_capture_reason, st.st_mtimespec, &g_capture_mtime);
    }
    close(fd);
    return changed;
}

static void capture_watchdog(uint64_t check_at);

static void capture_start(id<MTLCommandQueue> queue, uint64_t frame, const char *reason)
{
    @autoreleasepool {
        NSString *prefix = [NSFileManager.defaultManager stringWithFileSystemRepresentation:g_capture_prefix
                                                                                      length:strlen(g_capture_prefix)];
        NSString *path = prefix ? [prefix stringByAppendingFormat:@"-%d-f%llu-%s.gputrace",
                                   getpid(), frame + 1, reason] : nil;
        if (path && !path.isAbsolutePath)
            path = [NSFileManager.defaultManager.currentDirectoryPath stringByAppendingPathComponent:path];
        const char *refusal = NULL;
        uint64_t now = trace_clock();
        struct statfs volume;
        id<MTLDevice> device = queue.device;
        MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
        if (g_capture_count >= g_capture_max) refusal = "capture limit reached";
        else if (g_capture_stopped && (now - g_capture_stopped) / 1e9 < g_capture_cooldown)
            refusal = "cooldown in effect";
        else if (!device) refusal = "command queue has no Metal device";
        else if (!path.fileSystemRepresentation) refusal = "invalid output path";
        else if (manager.isCapturing) refusal = "Metal capture already running";
        else if (statfs(path.stringByDeletingLastPathComponent.fileSystemRepresentation, &volume))
            refusal = "cannot check free space on output volume";
        else if ((double)volume.f_bavail * volume.f_bsize < g_capture_free * 1073741824.0)
            refusal = "insufficient free space on output volume";
        if (refusal) {
            fp_log("fencepost: capture refused: %s\n", refusal);
            return;
        }

        MTLCaptureDescriptor *desc = [[[MTLCaptureDescriptor alloc] init] autorelease];
        desc.captureObject = device;
        desc.destination = MTLCaptureDestinationGPUTraceDocument;
        desc.outputURL = [NSURL fileURLWithPath:path];
        NSError *error = nil;
        if (![manager startCaptureWithDescriptor:desc error:&error]) {
            fp_log("fencepost: capture failed: %s\n", error.localizedDescription.UTF8String ?: "unknown error");
            return;
        }

        g_capture_path = [path copy];
        g_capture_started = trace_clock();
        pthread_mutex_lock(&g_lock);
        g_capture_last_commit = g_capture_started;
        pthread_mutex_unlock(&g_lock);
        g_capture_until = frame + g_capture_frames;
        g_capture_count++;
        g_capture_deadline = g_capture_started + (uint64_t)g_capture_max_secs * NSEC_PER_SEC;
        uint64_t idle_deadline = g_capture_started + (uint64_t)g_capture_idle_secs * NSEC_PER_SEC;
        capture_watchdog(MIN(g_capture_deadline, idle_deadline));
        fp_log("fencepost: capturing frame %llu (%s) -> %s\n", frame + 1, reason, path.fileSystemRepresentation);
    }
}

static NSString *capture_resource_name(NSString *entry)
{
    NSString *name = entry.lastPathComponent;
    size_t prefix;
    if ([name hasPrefix:@"MTLBuffer-"]) prefix = sizeof "MTLBuffer-" - 1;
    else if ([name hasPrefix:@"MTLTexture-"]) prefix = sizeof "MTLTexture-" - 1;
    else if ([name hasPrefix:@"MTLHeap-"]) prefix = sizeof "MTLHeap-" - 1;
    else return nil;

    /* Metal uses names like MTLBuffer-4-0, with the snapshot index last. */
    const char *id_start = name.UTF8String + prefix;
    size_t digits = strspn(id_start, "0123456789");
    if (!digits || (id_start[digits] && id_start[digits] != '-')) return nil;
    return [name substringToIndex:prefix + digits];
}

/* Keep command-buffer allocation gated until Metal has stopped recording.
 * A buffer allocated during this transition can enter the preceding capture. */
static void capture_stop(const char *limit)
{
    if (!g_capture_path) return;
    g_capture_deadline = 0;
    @autoreleasepool {
        MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
        if (manager.isCapturing) [manager stopCapture];
        g_capture_stopped = trace_clock();
        // Measure files visible when stopCapture returns. Metal may still be writing the document.
        uint64_t size = 0;
        NSFileManager *files = NSFileManager.defaultManager;
        NSDirectoryEnumerator *entries = [files enumeratorAtPath:g_capture_path];
        NSMutableDictionary<NSString *, NSNumber *> *sizes = [NSMutableDictionary dictionary];
        NSMutableDictionary<NSString *, NSNumber *> *counts = [NSMutableDictionary dictionary];
        NSString *entry;
        while ((entry = [entries nextObject])) {
            NSDictionary *attrs = entries.fileAttributes;
            if (![attrs.fileType isEqualToString:NSFileTypeRegular]) continue;
            uint64_t bytes = attrs.fileSize;
            size += bytes;
            NSString *name = capture_resource_name(entry);
            if (name) {
                sizes[name] = @(sizes[name].unsignedLongLongValue + bytes);
                counts[name] = @(counts[name].unsignedLongLongValue + 1);
            }
        }
        fp_log("fencepost: capture written %s, %.2f MB, %.1f s%s\n", g_capture_path.fileSystemRepresentation,
               size / 1048576.0, (g_capture_stopped - g_capture_started) / 1e9,
               limit ?: "");
        NSArray<NSString *> *largest = [[sizes allKeys] sortedArrayUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
            NSComparisonResult order = [sizes[b] compare:sizes[a]];
            return order == NSOrderedSame ? [a compare:b] : order;
        }];
        for (NSUInteger i = 0; i < MIN(largest.count, 3); i++) {
            NSString *name = largest[i];
            fp_log("fencepost: largest: %s %.0f MB in %llu snapshots\n", name.UTF8String,
                   sizes[name].unsignedLongLongValue / 1048576.0, counts[name].unsignedLongLongValue);
        }
        if (size > 2 * 1073741824ULL)
            fp_log("fencepost: big captures come from large buffers the frame touches. "
                   "A timeline dump (⌃⌥⌘F) is the cheap alternative.\n");
        [g_capture_path release];
        g_capture_path = nil;
        char ignored[sizeof g_capture_reason];
        capture_poll_file(ignored);
        pthread_mutex_lock(&g_lock);
        g_capture_busy = 0;
        pthread_mutex_unlock(&g_lock);
    }
}

/* Keep one timer per capture. Commits refresh the idle deadline without
 * queuing a timer for each submission. The hard deadline identifies the capture. */
static void capture_watchdog(uint64_t check_at)
{
    uint64_t deadline = g_capture_deadline;
    uint64_t now = trace_clock();
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, check_at > now ? check_at - now : 0),
                   dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        pthread_mutex_lock(&g_capture_lock);
        if (g_capture_deadline == deadline) {
            pthread_mutex_lock(&g_lock);
            uint64_t now = trace_clock();
            /* Metal can spend time inside commit taking snapshots. */
            uint64_t last = g_capture_commits ? now : g_capture_last_commit;
            pthread_mutex_unlock(&g_lock);
            uint64_t idle_deadline = last + (uint64_t)g_capture_idle_secs * NSEC_PER_SEC;
            uint64_t next = MIN(idle_deadline, deadline);
            if (now >= next) {
                tl_depth++;
                capture_stop(now >= deadline ? " (time limit)" : " (idle limit)");
                tl_depth--;
            } else capture_watchdog(next);
        }
        pthread_mutex_unlock(&g_capture_lock);
    });
}

static void capture_before_cb(id queue)
{
    pthread_mutex_lock(&g_capture_lock);
    uint64_t frame = g_capture_frame;
    if (frame > g_capture_opened) {
        g_capture_opened = frame;
        char reason[sizeof g_capture_reason];
        int changed = capture_poll_file(reason);
        pthread_mutex_lock(&g_lock);
        if (changed) capture_request(reason);
        memcpy(reason, g_capture_reason, sizeof reason);
        g_capture_reason[0] = 0;
        if (reason[0]) g_capture_busy = 1;
        pthread_mutex_unlock(&g_lock);
        if (reason[0]) {
            capture_start(queue, frame, reason);
            if (!g_capture_path) {
                pthread_mutex_lock(&g_lock);
                g_capture_busy = 0;
                pthread_mutex_unlock(&g_lock);
            }
        }
    }
    pthread_mutex_unlock(&g_capture_lock);
}

static void capture_after_commit(void)
{
    pthread_mutex_lock(&g_capture_lock);
    pthread_mutex_lock(&g_lock);
    /* Phase 1 can advance across enqueue-held buffers. Publish the new frame
     * only after those Metal commits have returned, so the ending submission
     * is inside the capture and the next allocation sees the transition. */
    if (--g_capture_commits == 0) g_capture_frame = g_frame_cur;
    uint64_t now = g_capture_path ? trace_clock() : 0;
    if (now) g_capture_last_commit = now;
    pthread_mutex_unlock(&g_lock);
    if (g_capture_path) {
        if (now >= g_capture_deadline) capture_stop(" (time limit)");
        else if (g_capture_frame >= g_capture_until) capture_stop(NULL);
    }
    pthread_mutex_unlock(&g_capture_lock);
}

static id h_q_cb(id self, SEL _cmd)
{
    ENTER;
    if (OUTER && g_capture_available) capture_before_cb(self);
    id cb = ORIG(H_Q_CB, id (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}
static id h_q_cbu(id self, SEL _cmd)
{
    ENTER;
    if (OUTER && g_capture_available) capture_before_cb(self);
    id cb = ORIG(H_Q_CBU, id (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}
static id h_q_cbd(id self, SEL _cmd, id d)
{
    ENTER;
    if (OUTER && g_capture_available) capture_before_cb(self);
    id cb = ORIG(H_Q_CBD, id (*)(id, SEL, id))(self, _cmd, d);
    LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}

/* command buffer */

static fp_node *on_encoder_kind(id cb, id enc, int kind, fp_trace_encoder *trace, int mtl4)
{
    if (!enc) return NULL;
    if (mtl4) hook_mtl4_encoder(object_getClass(enc));
    else hook_encoder(object_getClass(enc), kind);
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    fp_node *n = NULL;
    if (c) {
        n = node_new(kind);
        n->mtl4 = mtl4;
        n->enc = (void *)enc; n->cb_uid = c->uid; n->idx = ++c->nenc;
        if (g_ring) g_ring_encoded = 1;
        if (trace) trace_encoder_register(trace, n);
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

static fp_node *on_encoder(id cb, id enc, int kind, fp_trace_encoder *trace)
{
    return on_encoder_kind(cb, enc, kind, trace, 0);
}

/* The descriptor selectors may themselves call a hooked selector. Keep tl_depth
 * raised around the call and register once, in the application's outer hook. */
static MTLRenderPassDescriptor *trace_render_pass(fp_trace_encoder *e, MTLRenderPassDescriptor *app)
{
    if (!e || !app || !e->cb->device || !e->cb->device->buffer) return nil;
    NSUInteger slot = FP_TRACE_ATTACHMENTS;
    for (NSUInteger i = 0; i < FP_TRACE_ATTACHMENTS; i++) {
        MTLRenderPassSampleBufferAttachmentDescriptor *a = app.sampleBufferAttachments[i];
        if (!a.sampleBuffer) { if (slot == FP_TRACE_ATTACHMENTS) slot = i; }
        /* Apple GPUs can write only one attachment at each sampling boundary. */
        else if (a.startOfVertexSampleIndex != MTLCounterDontSample || a.endOfFragmentSampleIndex != MTLCounterDontSample)
            return nil;
    }
    if (slot == FP_TRACE_ATTACHMENTS || !trace_take_sample(e)) return nil;
    MTLRenderPassDescriptor *d = [app copy];
    MTLRenderPassSampleBufferAttachmentDescriptor *a = d.sampleBufferAttachments[slot];
    a.sampleBuffer = e->cb->device->buffer;
    a.startOfVertexSampleIndex = (NSUInteger)e->sample;
    a.endOfVertexSampleIndex = MTLCounterDontSample;
    a.startOfFragmentSampleIndex = MTLCounterDontSample;
    a.endOfFragmentSampleIndex = (NSUInteger)e->sample + 1;
    return d;
}

static MTLComputePassDescriptor *trace_compute_pass(fp_trace_encoder *e, MTLComputePassDescriptor *app,
                                                   MTLDispatchType type)
{
    if (!e || !e->cb->device || !e->cb->device->buffer) return nil;
    NSUInteger slot = app ? FP_TRACE_ATTACHMENTS : 0;
    for (NSUInteger i = 0; app && i < FP_TRACE_ATTACHMENTS; i++) {
        MTLComputePassSampleBufferAttachmentDescriptor *a = app.sampleBufferAttachments[i];
        if (!a.sampleBuffer) { if (slot == FP_TRACE_ATTACHMENTS) slot = i; }
        else if (a.startOfEncoderSampleIndex != MTLCounterDontSample || a.endOfEncoderSampleIndex != MTLCounterDontSample)
            return nil;
    }
    if (slot == FP_TRACE_ATTACHMENTS || !trace_take_sample(e)) return nil;
    MTLComputePassDescriptor *d = app ? [app copy] : [[MTLComputePassDescriptor alloc] init];
    if (!app) d.dispatchType = type;
    MTLComputePassSampleBufferAttachmentDescriptor *a = d.sampleBufferAttachments[slot];
    a.sampleBuffer = e->cb->device->buffer;
    a.startOfEncoderSampleIndex = (NSUInteger)e->sample;
    a.endOfEncoderSampleIndex = (NSUInteger)e->sample + 1;
    return d;
}

static MTLBlitPassDescriptor *trace_blit_pass(fp_trace_encoder *e, MTLBlitPassDescriptor *app)
{
    if (!e || !e->cb->device || !e->cb->device->buffer) return nil;
    NSUInteger slot = app ? FP_TRACE_ATTACHMENTS : 0;
    for (NSUInteger i = 0; app && i < FP_TRACE_ATTACHMENTS; i++) {
        MTLBlitPassSampleBufferAttachmentDescriptor *a = app.sampleBufferAttachments[i];
        if (!a.sampleBuffer) { if (slot == FP_TRACE_ATTACHMENTS) slot = i; }
        else if (a.startOfEncoderSampleIndex != MTLCounterDontSample || a.endOfEncoderSampleIndex != MTLCounterDontSample)
            return nil;
    }
    if (slot == FP_TRACE_ATTACHMENTS || !trace_take_sample(e)) return nil;
    MTLBlitPassDescriptor *d = app ? [app copy] : [[MTLBlitPassDescriptor alloc] init];
    MTLBlitPassSampleBufferAttachmentDescriptor *a = d.sampleBufferAttachments[slot];
    a.sampleBuffer = e->cb->device->buffer;
    a.startOfEncoderSampleIndex = (NSUInteger)e->sample;
    a.endOfEncoderSampleIndex = (NSUInteger)e->sample + 1;
    return d;
}

#define TRACE_ENCODER() (OUTER && g_trace_enabled ? trace_encoder_begin(self) : NULL)

static id h_cb_renc(id self, SEL _cmd, MTLRenderPassDescriptor *d)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    MTLRenderPassDescriptor *sampled = trace ? trace_render_pass(trace, d) : nil;
    id enc = ORIG(H_CB_RENC, id (*)(id, SEL, id))(self, _cmd, sampled ?: d);
    if (sampled) [sampled release];
    LEAVE;
    if (!OUTER) return enc;
    fp_node *n = on_encoder(self, enc, FPK_RENDER, trace);
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
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    MTLComputePassDescriptor *d = trace ? trace_compute_pass(trace, nil, MTLDispatchTypeSerial) : nil;
    id enc = d ? [self computeCommandEncoderWithDescriptor:d] : ORIG(H_CB_CENC, id (*)(id, SEL))(self, _cmd);
    if (d) [d release];
    LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_COMPUTE, trace);
    return enc;
}
static id h_cb_cenc_dt(id self, SEL _cmd, NSUInteger dt)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    MTLComputePassDescriptor *d = trace ? trace_compute_pass(trace, nil, dt) : nil;
    id enc = d ? [self computeCommandEncoderWithDescriptor:d] : ORIG(H_CB_CENC_DT, id (*)(id, SEL, NSUInteger))(self, _cmd, dt);
    if (d) [d release];
    LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_COMPUTE, trace);
    return enc;
}
static id h_cb_cenc_d(id self, SEL _cmd, id d)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    MTLComputePassDescriptor *sampled = trace && d ? trace_compute_pass(trace, d, MTLDispatchTypeSerial) : nil;
    id enc = ORIG(H_CB_CENC_D, id (*)(id, SEL, id))(self, _cmd, sampled ?: d);
    if (sampled) [sampled release];
    LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_COMPUTE, trace);
    return enc;
}
static id h_cb_benc(id self, SEL _cmd)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    MTLBlitPassDescriptor *d = trace ? trace_blit_pass(trace, nil) : nil;
    id enc = d ? [self blitCommandEncoderWithDescriptor:d] : ORIG(H_CB_BENC, id (*)(id, SEL))(self, _cmd);
    if (d) [d release];
    LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_BLIT, trace);
    return enc;
}
static id h_cb_benc_d(id self, SEL _cmd, id d)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    MTLBlitPassDescriptor *sampled = trace && d ? trace_blit_pass(trace, d) : nil;
    id enc = ORIG(H_CB_BENC_D, id (*)(id, SEL, id))(self, _cmd, sampled ?: d);
    if (sampled) [sampled release];
    LEAVE;
    if (OUTER) on_encoder(self, enc, FPK_BLIT, trace);
    return enc;
}

/* Metal 4 buffers are recorded independently of the queue that submits them. */
static void on_mtl4_queue(id queue)
{
    if (!queue) return;
    /* A Metal 4 buffer has no queue during encoding. The Metal 3 capture
     * path starts on the intended queue before a command buffer is opened. */
    static atomic_int capture_warned;
    if (g_capture_prefix && !atomic_exchange(&capture_warned, 1))
        fp_log("fencepost: capture not yet supported for Metal 4\n");
    const char *label = [[queue label] UTF8String];
    pthread_mutex_lock(&g_lock);
    fp_queue *q = queue_get_kind((void *)queue, 1);
    if (q) {
        snprintf(g_trace_queue_labels[q->idx], sizeof g_trace_queue_labels[0], "MTL4%s%s",
                 label && *label ? " " : "", label ?: "");
    }
    pthread_mutex_unlock(&g_lock);
    hook_mtl4_queue(object_getClass(queue));
}

static id h_mtl4_dev_q(id self, SEL _cmd)
{
    ENTER; id queue = ORIG(H_4_DEV_Q, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER) on_mtl4_queue(queue);
    return queue;
}

static id h_mtl4_dev_q_desc(id self, SEL _cmd, id desc, NSError **error)
{
    ENTER; id queue = ORIG(H_4_DEV_Q_DESC, id (*)(id, SEL, id, NSError **))(self, _cmd, desc, error); LEAVE;
    if (OUTER) on_mtl4_queue(queue);
    return queue;
}

static id h_mtl4_dev_cb(id self, SEL _cmd)
{
    ENTER; id cb = ORIG(H_4_DEV_CB, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER && cb) hook_mtl4_cb(object_getClass(cb));
    return cb;
}

static void on_mtl4_begin(id cb)
{
    pthread_mutex_lock(&g_lock);
    cb_record_new(cb, NULL, 1);
    pthread_mutex_unlock(&g_lock);
}

static void h_mtl4_begin(id self, SEL _cmd, id allocator)
{
    ENTER; ORIG(H_4_BEGIN, void (*)(id, SEL, id))(self, _cmd, allocator); LEAVE;
    if (OUTER) on_mtl4_begin(self);
}

static void h_mtl4_begin_opts(id self, SEL _cmd, id allocator, id options)
{
    ENTER; ORIG(H_4_BEGIN_OPTS, void (*)(id, SEL, id, id))(self, _cmd, allocator, options); LEAVE;
    if (OUTER) on_mtl4_begin(self);
}

static void h_mtl4_end(id self, SEL _cmd)
{
    ENTER; ORIG(H_4_END, void (*)(id, SEL))(self, _cmd); LEAVE;
    if (!OUTER) return;
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)self);
    if (c) c->state = 2;
    pthread_mutex_unlock(&g_lock);
}

static id h_mtl4_render(id self, SEL _cmd, id desc)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    id enc = ORIG(H_4_RENDER, id (*)(id, SEL, id))(self, _cmd, desc);
    LEAVE;
    if (OUTER) on_encoder_kind(self, enc, FPK_RENDER, trace, 1);
    return enc;
}

static id h_mtl4_render_opts(id self, SEL _cmd, id desc, NSUInteger options)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    id enc = ORIG(H_4_RENDER_OPTS, id (*)(id, SEL, id, NSUInteger))(self, _cmd, desc, options);
    LEAVE;
    if (OUTER) on_encoder_kind(self, enc, FPK_RENDER, trace, 1);
    return enc;
}

static id h_mtl4_compute(id self, SEL _cmd)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    id enc = ORIG(H_4_COMPUTE, id (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (OUTER) on_encoder_kind(self, enc, FPK_COMPUTE, trace, 1);
    return enc;
}

static id h_mtl4_ml(id self, SEL _cmd)
{
    ENTER; id enc = ORIG(H_4_ML, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (OUTER && enc) {
        pthread_mutex_lock(&g_lock);
        g_mtl4_ml_encoders++;
        pthread_mutex_unlock(&g_lock);
    }
    return enc;
}

static void cb_item(id cb, int type, id ev, uint64_t val)
{
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    if (c) {
        if (g_trace_enabled && type == 4) val = trace_clock();
        fp_item it = { (uint8_t)type, 0, (void *)ev, val };
        PUSH(c->it, c->nit, c->cit, it);
    }
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

/* a present ends a frame, unless the program marks frames itself (fencepost_frame) */
static void h_cb_pres(id self, SEL _cmd, id d)
{
    ENTER; ORIG(H_CB_PRES, void (*)(id, SEL, id))(self, _cmd, d); LEAVE;
    if (OUTER && !g_frame_api) cb_item(self, 4, nil, 0);
}
static void h_cb_pres_t(id self, SEL _cmd, id d, CFTimeInterval t)
{
    ENTER; ORIG(H_CB_PRES_T, void (*)(id, SEL, id, CFTimeInterval))(self, _cmd, d, t); LEAVE;
    if (OUTER && !g_frame_api) cb_item(self, 4, nil, 0);
}
static void h_cb_pres_d(id self, SEL _cmd, id d, CFTimeInterval t)
{
    ENTER; ORIG(H_CB_PRES_D, void (*)(id, SEL, id, CFTimeInterval))(self, _cmd, d, t); LEAVE;
    if (OUTER && !g_frame_api) cb_item(self, 4, nil, 0);
}

static void maybe_summary(void)
{
    time_t now = time(NULL);
    if (g_summary_secs > 0 && now - g_last_summary >= g_summary_secs) {
        g_last_summary = now;
        summary(0);
    }
}

static int ring_new_mtime(struct timespec stamp)
{
    return stamp.tv_sec > g_file_mtime.tv_sec ||
           (stamp.tv_sec == g_file_mtime.tv_sec && stamp.tv_nsec > g_file_mtime.tv_nsec);
}

/* The file is shared by every interested process. Read the request from
 * the same inode whose timestamp we acknowledge, even on rename. */
static void ring_poll_file(void)
{
    if (!g_trigger_file || !g_ring_encoded) return;
    uint64_t now = trace_clock();
    if (now - g_last_poll < 250000000) return;
    g_last_poll = now;
    struct stat st;
    if (stat(g_trigger_file, &st) || !S_ISREG(st.st_mode) || !ring_new_mtime(st.st_mtimespec)) return;
    int fd = open(g_trigger_file, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return;
    if (!fstat(fd, &st) && S_ISREG(st.st_mode) &&
        trigger_file_owned(&st, g_trigger_file, &g_trigger_owner_warned) && ring_new_mtime(st.st_mtimespec)) {
        char reason[33];
        if (trigger_read_request(fd, reason, sizeof reason, st.st_mtimespec, &g_file_mtime)) {
            ring_trigger(DUMP_MANUAL, reason, "file", g_frame_cur, now, NULL);
        }
    }
    close(fd);
}

static void h_cb_commit(id self, SEL _cmd)
{
    ENTER;
    int capture_commit = 0;
    if (OUTER) {
        uint64_t uid = 0;
        fp_trace_cb *trace = NULL;
        NSString *l = [self label];
        pthread_mutex_lock(&g_lock);
        if (g_capture_busy) g_capture_last_commit = trace_clock();
        fp_cb *c = cb_find_open((void *)self);
        if (c && c->state == 1) {
            uid = c->uid;
            if (g_capture_available) {
                g_capture_commits++;
                capture_commit = 1;
            }
            if (l) snprintf(c->label, sizeof c->label, "%s", l.UTF8String);
            if (g_trace_enabled && (trace = c->trace)) {
                trace->owner->record = NULL;
                memcpy(trace->label, c->label, sizeof trace->label);
                trace->completed_before = malloc(sizeof g_trace_completed);
                memcpy(trace->completed_before, g_trace_completed, sizeof g_trace_completed);
                if (!g_trace_tracks[c->q->idx][TR_CB].n) {
                    const char *label = [(id<MTLCommandBuffer>)self commandQueue].label.UTF8String;
                    if (label) snprintf(g_trace_queue_labels[c->q->idx], sizeof g_trace_queue_labels[0], "%s", label);
                }
            }
            cb_close((void *)self);
            c->state = 2;
            if (!g_inflight && g_tick) { frame_get(g_frame_cur)->idle++; g_st[ST_GPU_IDLE]++; }   /* the GPU ran dry */
            g_inflight++;
            c->commit_tick = ++g_tick;
            if (!c->enq) c->enq = ++c->q->enq_assign;
            uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            if (g_ring) ring_poll_file();
            queue_submit(c->q, c);
            process_pending(0);
            if (g_trace) trace_stream_flush();
            sweep();
            g_st[ST_ANALYSIS_NS] += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
        }
        pthread_mutex_unlock(&g_lock);
        if (uid) {
            [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> b) { cb_done(uid, trace, b); }];
            maybe_summary();
        }
    }
    ORIG(H_CB_COMMIT, void (*)(id, SEL))(self, _cmd);
    if (capture_commit) capture_after_commit();
    LEAVE;
}

typedef struct {
    fp_trace_cb *trace;
    fp_queue *queue;
    uint64_t signal_limit;
    NSUInteger ncbs;
    uint64_t cbs[];
} fp_mtl4_submission;

static void trace_mtl4_join(fp_trace_cb **batch, fp_cb *c)
{
    fp_trace_cb *trace = c->trace;
    if (!trace) return;
    trace->owner->record = NULL;
    trace->owner = nil;
    if (!*batch) {
        *batch = trace;
        trace->queue = c->q->idx;
        trace->span.frame = g_frame_cur;
        trace->selected = trace_frame_selected(g_frame_cur) && !g_trace_stopped;
        memcpy(trace->label, c->label, sizeof trace->label);
    }
    for (fp_trace_encoder *e = trace->encoders; e; e = e->next) {
        e->cb = *batch;
        memcpy(e->mtl4_cb_label, c->label, sizeof e->mtl4_cb_label);
    }
    if (trace != *batch) {
        if (trace->encoders) {
            *(*batch)->encoder_tail = trace->encoders;
            (*batch)->encoder_tail = trace->encoder_tail;
        }
        trace_free_cb(trace);
        c->trace = *batch;
    }
}

static void trace_mtl4_submit(fp_queue *q, fp_trace_cb *trace)
{
    if (q->mtl4_last_trace) trace_release_cb(q->mtl4_last_trace);
    q->mtl4_last_trace = trace;
    if (!trace) return;
    trace->refs++;   /* A queue signal can refer to the commit after feedback arrives. */
    trace->completed_before = malloc(sizeof g_trace_completed);
    memcpy(trace->completed_before, g_trace_completed, sizeof g_trace_completed);
    if (g_ring && trace->selected) {
        fp_trace_chunk *chunk = trace_chunk(trace->span.frame);
        if (chunk) {
            chunk->outstanding++;
            trace->span.counted = 1;
        }
    }
    trace->phased = 1;
}

static fp_mtl4_submission *mtl4_record_commit(id queue, const id *buffers, NSUInteger count)
{
    char local_labels[4][sizeof ((fp_cb *)0)->label];
    if (!count || count > MIN((SIZE_MAX - sizeof(fp_mtl4_submission)) / sizeof(uint64_t),
                             SIZE_MAX / sizeof local_labels[0])) return NULL;
    fp_mtl4_submission *submission = calloc(1, sizeof *submission + count * sizeof submission->cbs[0]);
    if (!submission) return NULL;
    char (*labels)[sizeof local_labels[0]] = count <= 4 ? local_labels : malloc(count * sizeof local_labels[0]);
    if (!labels) { free(submission); return NULL; }
    for (NSUInteger i = 0; i < count; i++) {
        const char *label = [[buffers[i] label] UTF8String];
        snprintf(labels[i], sizeof labels[i], "%s", label ?: "");
    }
    pthread_mutex_lock(&g_lock);
    fp_queue *q = queue_get_kind((void *)queue, 1);
    if (q) {
        submission->queue = q;
        submission->signal_limit = q->mtl4_signal_limit;
        if (!g_inflight && g_tick) { frame_get(g_frame_cur)->idle++; g_st[ST_GPU_IDLE]++; }
        uint64_t tick = ++g_tick;
        int nenc = 0;
        if (g_ring) ring_poll_file();
        for (NSUInteger i = 0; i < count; i++) {
            id cb = buffers[i];
            fp_cb *c = cb_find_open((void *)cb);
            if (!c || !c->mtl4 || c->state != 2) continue;
            snprintf(c->label, sizeof c->label, "%s", labels[i]);
            c->q = q;
            c->commit_tick = tick;
            submission->cbs[submission->ncbs++] = c->uid;
            nenc += c->nenc;
            cb_close((void *)cb);
            trace_mtl4_join(&submission->trace, c);
            phase1(c);
        }
        trace_mtl4_submit(q, submission->trace);
        if (submission->ncbs) g_inflight++;
        if (nenc) q->mtl4_nwaits = 0;
        process_pending(0);
        if (g_trace) trace_stream_flush();
    }
    pthread_mutex_unlock(&g_lock);
    if (labels != local_labels) free(labels);
    if (submission->ncbs) return submission;
    free(submission);
    return NULL;
}

/* Successful completion proves that the queue passed every signal scheduled
 * before this submission. Earlier submissions cannot retire trailing signals. */
static void mtl4_retire_signals(fp_queue *q, uint64_t limit)
{
    uint64_t floor = q->nrecent > FP_QRECENT ? q->nrecent - FP_QRECENT : 0;
    for (uint64_t k = MAX(floor, q->mtl4_signals_done); k < limit; k++) {
        fp_node *n = node_get(q->recent[k % FP_QRECENT]);
        if (n && n->kind == FPK_SIGNAL && !n->done_tick) {
            n->done_tick = g_tick;
#ifdef FENCEPOST_TEST_HOOKS
            g_mtl4_signals_retired++;
#endif
        }
    }
    q->mtl4_signals_done = MAX(q->mtl4_signals_done, limit);
}

API_AVAILABLE(macos(26.0))
static void mtl4_completed(fp_mtl4_submission *submission, id<MTL4CommitFeedback> feedback)
{
    NSError *error = feedback.error;
    double start = feedback.GPUStartTime, end = feedback.GPUEndTime;
    pthread_mutex_lock(&g_lock);
    g_inflight--;
    for (NSUInteger i = 0; i < submission->ncbs; i++) cb_mark_done(submission->cbs[i]);
    if (!error) mtl4_retire_signals(submission->queue, submission->signal_limit);
    fp_trace_cb *trace = submission->trace;
    if (trace) {
        if (error && g_ring && g_ring_gpuerr) @autoreleasepool {
            trace->error_time = trace_clock();
            trace->gpu_error = strdup(error.localizedDescription.UTF8String ?: "Metal 4 commit error");
        }
        if (!error && start > 0 && end >= start && end <= (double)trace_clock() / 1e9 + 1) {
            trace->gpu_start = (uint64_t)(start * 1e9);
            trace->gpu_end = (uint64_t)(end * 1e9);
        }
        trace->completed = 1;
        trace_cb_finish(trace);
    }
    if (g_ring) ring_update(0);
    if (g_trace) trace_stream_flush();
    pthread_mutex_unlock(&g_lock);
    free(submission);
}

@interface FPMTL4Callbacks : NSObject {
@public
    NSMutableArray *blocks;
}
@end

@implementation FPMTL4Callbacks
- (void)dealloc
{
    [blocks release];
    [super dealloc];
}
@end

static Class g_mtl4_options_class;
static char g_mtl4_callbacks_key;

static id h_mtl4_options_init(id self, SEL _cmd)
{
    id options = ORIG(H_4_OPTIONS_INIT, id (*)(id, SEL))(self, _cmd);
    if (!tl_depth && object_getClass(options) == g_mtl4_options_class) {
        FPMTL4Callbacks *callbacks = [[FPMTL4Callbacks alloc] init];
        objc_setAssociatedObject(options, &g_mtl4_callbacks_key, callbacks, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [callbacks release];
    }
    return options;
}

static void h_mtl4_app_feedback(id self, SEL _cmd, void (^handler)(id))
{
    FPMTL4Callbacks *callbacks = objc_getAssociatedObject(self, &g_mtl4_callbacks_key);
    if (!callbacks || !handler) {
        ORIG(H_4_ADD_FEEDBACK, void (*)(id, SEL, void (^)(id)))(self, _cmd, handler);
        return;
    }
    void (^copied)(id) = Block_copy(handler);
    pthread_mutex_lock(&g_mtl4_options_lock);
    if (!callbacks->blocks) callbacks->blocks = [[NSMutableArray alloc] init];
    [callbacks->blocks addObject:copied];
    pthread_mutex_unlock(&g_mtl4_options_lock);
    Block_release(copied);
}

API_AVAILABLE(macos(26.0))
static MTL4CommitOptions *mtl4_submit_options(id queue, const id *buffers, NSUInteger count, id app_options)
{
    FPMTL4Callbacks *callbacks = app_options ? objc_getAssociatedObject(app_options, &g_mtl4_callbacks_key) : nil;
    if (app_options && !callbacks) return nil;
    MTL4CommitOptions *opts = [[g_mtl4_options_class alloc] init];
    if (!opts) return nil;

    /* Metal consumes the app's registrations once per submission. Take that
     * list under a short lock and replay it on private, per-commit options.
     * Native submission can block and must never hold a fencepost lock. */
    if (callbacks) {
        pthread_mutex_lock(&g_mtl4_options_lock);
        NSArray *blocks = callbacks->blocks;
        callbacks->blocks = nil;
        pthread_mutex_unlock(&g_mtl4_options_lock);
        for (MTL4CommitFeedbackHandler handler in blocks) [opts addFeedbackHandler:handler];
        [blocks release];
    }
    fp_mtl4_submission *submission = mtl4_record_commit(queue, buffers, count);
    if (submission)
        [opts addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            mtl4_completed(submission, feedback);
        }];
    return opts;
}

static void h_mtl4_commit(id self, SEL _cmd, const id *buffers, NSUInteger count)
{
    ENTER;
    int submitted = 0;
    if (OUTER) {
        if (@available(macOS 26.0, *)) {
            if (count && [self respondsToSelector:@selector(commit:count:options:)]) {
                MTL4CommitOptions *opts = mtl4_submit_options(self, buffers, count, nil);
                if (opts) {
                    [(id<MTL4CommandQueue>)self commit:(const id<MTL4CommandBuffer> *)buffers count:count options:opts];
                    submitted = 1;
                    [opts release];
                }
            }
        }
    }
    if (!submitted) ORIG(H_4_COMMIT, void (*)(id, SEL, const id *, NSUInteger))(self, _cmd, buffers, count);
    LEAVE;
    if (submitted) maybe_summary();
}

static void h_mtl4_commit_opts(id self, SEL _cmd, const id *buffers, NSUInteger count, id options)
{
    ENTER;
    id opts = nil;
    if (OUTER && options) {
        if (@available(macOS 26.0, *)) opts = mtl4_submit_options(self, buffers, count, options);
    }
    ORIG(H_4_COMMIT_OPTS, void (*)(id, SEL, const id *, NSUInteger, id))(self, _cmd, buffers, count, opts ?: options);
    [opts release];
    LEAVE;
    if (opts) maybe_summary();
}

static void trace_mtl4_signal(fp_queue *q, uint64_t uid, uint64_t value)
{
    if (!g_trace_enabled || g_trace_stopped || !trace_frame_selected(g_frame_cur)) return;
    fp_trace_cb *last = q->mtl4_last_trace;
    if (last && last->selected && (!last->completed || last->span.tid)) {
        if (last->completed) trace_archive(uid, &last->span);
        else {
            CFDictionarySetValue(g_trace_nodes, TRACE_KEY(uid), &last->span);
            PUSH(last->signals, last->nsignals, last->csignals, uid);
        }
        return;
    }
    /* A queue can signal before its first commit. The scheduling timestamp is
     * a lower bound for when that signal reaches the GPU. */
    if (!g_trace_origin) trace_start();
    uint64_t now = trace_clock();
    fp_trace_span span = { .start = now, .end = now, .frame = g_frame_cur, .ready = 1, .gpu = 1 };
    span.tid = trace_queue_lane(q->idx, TR_MTL4_EVENT, &span);
    char *record = NULL;
    size_t length = 0;
    FILE *f = open_memstream(&record, &length);
    if (f) {
        trace_slice(f, &span, "event signal", "MTL4 event signal");
        fprintf(f, ",\"args\":{\"uid\":%llu,\"value\":%llu,\"estimated\":true}}", TRACE_U64(uid), TRACE_U64(value));
    }
    if (!trace_write(span.frame, f, &record, &length, 1)) return;
    if (!g_ring_history) g_ring_history = calloc(FP_POOL, sizeof *g_ring_history);
    trace_archive(uid, &span);
}

static void mtl4_record_event(id queue, fp_item item)
{
    pthread_mutex_lock(&g_lock);
    fp_queue *q = queue_get_kind((void *)queue, 1);
    if (q && item.type == 1) {
        fp_node *n = phase_signal(q, 0, ++g_tick, &item);
        n->mtl4 = 1;
        phase_node(q, n, g_tick, NULL, q->mtl4_waits, q->mtl4_nwaits);
        q->mtl4_signal_limit = q->nrecent;
        trace_mtl4_signal(q, n->uid, item.val);
        process_pending(0);
    } else if (q) {
        fp_evwait wait = { item.ev, item.val, 0 };
        PUSH(q->mtl4_waits, q->mtl4_nwaits, q->mtl4_cwaits, wait);
    }
    pthread_mutex_unlock(&g_lock);
}

/* Record ordered calls before entering Metal. Calls that overlap on the
 * same queue have no app-defined order; the detector uses hook entry order. */
static void h_mtl4_signal(id self, SEL _cmd, id event, uint64_t value)
{
    ENTER;
    if (OUTER) mtl4_record_event(self, (fp_item){ 1, 0, (void *)event, value });
    ORIG(H_4_SIGNAL, void (*)(id, SEL, id, uint64_t))(self, _cmd, event, value);
    LEAVE;
}

static void h_mtl4_wait(id self, SEL _cmd, id event, uint64_t value)
{
    ENTER;
    if (OUTER) mtl4_record_event(self, (fp_item){ 2, 0, (void *)event, value });
    ORIG(H_4_WAIT, void (*)(id, SEL, id, uint64_t))(self, _cmd, event, value);
    LEAVE;
}

static void h_mtl4_drawable_signal(id self, SEL _cmd, id drawable)
{
    ENTER;
    if (OUTER) {
        pthread_mutex_lock(&g_lock);
        if (!g_frame_api) {
            frame_boundary(g_trace_enabled ? trace_clock() : 0, NULL);
            process_pending(0);
        }
        pthread_mutex_unlock(&g_lock);
    }
    ORIG(H_4_DRAWABLE_SIGNAL, void (*)(id, SEL, id))(self, _cmd, drawable);
    LEAVE;
}

static void h_mtl4_drawable_wait(id self, SEL _cmd, id drawable)
{
    /* This acquires the drawable for upcoming work, which ends at signalDrawable:. */
    tl_depth++;
    ORIG(H_4_DRAWABLE_WAIT, void (*)(id, SEL, id))(self, _cmd, drawable);
    tl_depth--;
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
    if (n->mtl4) snprintf(n->label, sizeof n->label, "MTL4 %s", l.length ? l.UTF8String : kind_name[n->kind]);
    else if (l) snprintf(n->label, sizeof n->label, "%s", l.UTF8String);
    if (g_trace_enabled) trace_encoder_end(n);
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
static const hook_def mtl4_device_hooks[] = {
    HD(H_4_DEV_Q, "newMTL4CommandQueue", h_mtl4_dev_q),
    HD(H_4_DEV_Q_DESC, "newMTL4CommandQueueWithDescriptor:error:", h_mtl4_dev_q_desc),
    HD(H_4_DEV_CB, "newCommandBuffer", h_mtl4_dev_cb),
};
static const hook_def mtl4_cb_hooks[] = {
    HD(H_4_BEGIN, "beginCommandBufferWithAllocator:", h_mtl4_begin),
    HD(H_4_BEGIN_OPTS, "beginCommandBufferWithAllocator:options:", h_mtl4_begin_opts),
    HD(H_4_END, "endCommandBuffer", h_mtl4_end),
    HD(H_4_RENDER, "renderCommandEncoderWithDescriptor:", h_mtl4_render),
    HD(H_4_RENDER_OPTS, "renderCommandEncoderWithDescriptor:options:", h_mtl4_render_opts),
    HD(H_4_COMPUTE, "computeCommandEncoder", h_mtl4_compute),
    HD(H_4_ML, "machineLearningCommandEncoder", h_mtl4_ml),
};
static const hook_def mtl4_encoder_hooks[] = {
    HD(H_END, "endEncoding", h_end),
};
static const hook_def mtl4_queue_hooks[] = {
    HD(H_4_COMMIT, "commit:count:", h_mtl4_commit),
    HD(H_4_COMMIT_OPTS, "commit:count:options:", h_mtl4_commit_opts),
    HD(H_4_SIGNAL, "signalEvent:value:", h_mtl4_signal),
    HD(H_4_WAIT, "waitForEvent:value:", h_mtl4_wait),
    HD(H_4_DRAWABLE_SIGNAL, "signalDrawable:", h_mtl4_drawable_signal),
    HD(H_4_DRAWABLE_WAIT, "waitForDrawable:", h_mtl4_drawable_wait),
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
    HD(H_CB_PRES, "presentDrawable:", h_cb_pres),
    HD(H_CB_PRES_T, "presentDrawable:atTime:", h_cb_pres_t),
    HD(H_CB_PRES_D, "presentDrawable:afterMinimumDuration:", h_cb_pres_d),
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
static void hook_device(Class cls)
{
    install(cls, 100, device_hooks, N(device_hooks));
    if (@available(macOS 26.0, *)) install(cls, 104, mtl4_device_hooks, N(mtl4_device_hooks));
}
static void hook_queue(Class cls) { install(cls, 101, queue_hooks, N(queue_hooks)); }
static void hook_cb(Class cls) { install(cls, 102, cb_hooks, N(cb_hooks)); }
static void hook_event(Class cls) { install(cls, 103, event_hooks, N(event_hooks)); }
static void hook_mtl4_cb(Class cls) { install(cls, 105, mtl4_cb_hooks, N(mtl4_cb_hooks)); }
static void hook_mtl4_encoder(Class cls) { install(cls, 106, mtl4_encoder_hooks, N(mtl4_encoder_hooks)); }
static void hook_mtl4_queue(Class cls) { install(cls, 107, mtl4_queue_hooks, N(mtl4_queue_hooks)); }
static void hook_encoder(Class cls, int kind)
{
    if (kind == FPK_RENDER) install(cls, kind, render_hooks, N(render_hooks));
    else if (kind == FPK_COMPUTE) install(cls, kind, compute_hooks, N(compute_hooks));
    else install(cls, kind, blit_hooks, N(blit_hooks));
}

static void final_summary(void)
{
    static atomic_int once;
    if (atomic_exchange(&once, 1)) return;
    if (g_capture_available && !pthread_mutex_trylock(&g_capture_lock)) {
        tl_depth++;
        capture_stop(NULL);
        tl_depth--;
        pthread_mutex_unlock(&g_capture_lock);
    }
    if (g_st[ST_NODES]) summary(1);   /* helper processes that never encoded: quiet */
    if (g_mtl4_ml_encoders) fp_log("fencepost: Metal 4 machine-learning encoders: %ld\n", g_mtl4_ml_encoders);
    if (g_trace_enabled) {
        pthread_mutex_lock(&g_lock);
        trace_final();
        g_trace_stopped = 1;
        if (g_trace_alloc_dropped)
            fp_log("fencepost: final: %llu trace records dropped while allocating or formatting\n",
                   TRACE_U64(g_trace_alloc_dropped));
        pthread_mutex_unlock(&g_lock);
        if (g_ring_writer) dispatch_sync(g_ring_writer, ^{});
    }
}

static void capture_check_destination(void)
{
    /* Metal can report no destinations until the first device is initialized. */
    @autoreleasepool {
        g_capture_available = [MTLCaptureManager.sharedCaptureManager
                              supportsDestination:MTLCaptureDestinationGPUTraceDocument];
        if (!g_capture_available)
            fp_log("fencepost: Metal does not support GPU trace documents; capture disabled\n");
    }
}

static void on_device(id dev)
{
    if (!dev) return;
    static pthread_once_t capture_check = PTHREAD_ONCE_INIT;
    if (g_capture_prefix) pthread_once(&capture_check, capture_check_destination);
    int first = 0;
    pthread_mutex_lock(&g_lock);
    if (!g_started) { g_started = 1; first = 1; }
    pthread_mutex_unlock(&g_lock);
    if (first) {
        fp_log("fencepost: watching Metal in pid %d (%s); reports go to %s\n", getpid(),
               [[NSProcessInfo processInfo] processName].UTF8String,
               g_out ? "FENCEPOST_LOG" : "stderr");
        g_last_summary = time(NULL);
        if (!g_trace_enabled) atexit(final_summary);
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

static FILE *fp_open_path(const char *p, const char *mode)
{
    char path[1024], *o = path;
    for (; *p && o < path + sizeof path - 16; p++) {
        if (p[0] == '%' && p[1] == 'p') { o += snprintf(o, 16, "%d", getpid()); p++; }
        else *o++ = *p;
    }
    *o = 0;
    return fopen(path, mode);
}

static unsigned trace_unsigned(const char *name, unsigned fallback)
{
    const char *s = getenv(name);
    if (!s || *s < '0' || *s > '9') return fallback;
    char *end;
    unsigned long long n = strtoull(s, &end, 10);
    return *end || n > UINT_MAX ? fallback : (unsigned)n;
}

static void capture_init(const char *prefix)
{
    @autoreleasepool {
        if (strcmp(getenv("MTL_CAPTURE_ENABLED") ?: "", "1")) {
            fp_log("fencepost: FENCEPOST_CAPTURE needs MTL_CAPTURE_ENABLED=1 at launch; capture disabled\n");
            return;
        }
        g_capture_prefix = strdup(prefix);
        const char *file = getenv("FENCEPOST_CAPTURE_FILE");
        char default_path[64];
        if (!file) file = capture_default_path(default_path);
        struct stat st;
        if (*file) {
            g_capture_file = strdup(file);
            if (!stat(file, &st) && st.st_uid == geteuid()) g_capture_mtime = st.st_mtimespec;
        }
        g_capture_frames = MAX(1u, trace_unsigned("FENCEPOST_CAPTURE_FRAMES", 1));
        g_capture_idle_secs = MAX(1u, trace_unsigned("FENCEPOST_CAPTURE_IDLE_SECS", 10));
        g_capture_max_secs = MAX(1u, trace_unsigned("FENCEPOST_CAPTURE_MAX_SECS", 120));
        g_capture_max = trace_unsigned("FENCEPOST_CAPTURE_MAX", 3);
        g_capture_free = trace_unsigned("FENCEPOST_CAPTURE_MIN_FREE_GB", 10);
        g_capture_cooldown = 10;
        const char *value = getenv("FENCEPOST_CAPTURE_COOLDOWN");
        double seconds;
        char extra;
        if (value && sscanf(value, "%lf%c", &seconds, &extra) == 1 && isfinite(seconds) && seconds >= 0)
            g_capture_cooldown = seconds;
        const char *on = getenv("FENCEPOST_CAPTURE_ON");
        if (on) {
            for (NSString *source in [[NSString stringWithUTF8String:on] componentsSeparatedByString:@","]) {
                if ([source isEqualToString:@"race"]) g_capture_on |= CAP_RACE;
                if ([source isEqualToString:@"gpu-error"]) g_capture_on |= CAP_GPU_ERROR;
                if ([source isEqualToString:@"spike"]) g_capture_on |= CAP_SPIKE;
            }
        }
    }
}

static void ring_init_triggers(void)
{
    const char *p = getenv("FENCEPOST_TRIGGER_RACE");
    if (p && !strcmp(p, "all")) g_ring_race = 2;
    else if (p && !strcmp(p, "0")) g_ring_race = 0;
    g_ring_gpuerr = trace_unsigned("FENCEPOST_TRIGGER_GPUERR", 1) != 0;
    unsigned max_dumps = trace_unsigned("FENCEPOST_MAX_DUMPS", 20);
    g_max_dumps[DUMP_MANUAL] = trace_unsigned("FENCEPOST_MAX_MANUAL_DUMPS", max_dumps);
    g_max_dumps[DUMP_AUTO] = trace_unsigned("FENCEPOST_MAX_AUTO_DUMPS", max_dumps);
    char extra;
    double factor, ms, seconds;
    p = getenv("FENCEPOST_TRIGGER_SPIKE");
    if (p && !strcmp(p, "0")) g_spike_factor = 0;
    else if (p && sscanf(p, "%lf:%lf%c", &factor, &ms, &extra) == 2 &&
             isfinite(factor) && factor > 0 && isfinite(ms) && ms >= 0) {
        g_spike_factor = factor;
        g_spike_min_ns = ms * 1e6;
    } else if (!p && !getenv("FENCEPOST_SPIKE") && !strcmp(getenv("MTL_CAPTURE_ENABLED") ?: "", "1")) {
        g_spike_factor = 0;
        fp_log("fencepost: timeline spike triggers disabled by default with MTL_CAPTURE_ENABLED=1\n");
    }
    p = getenv("FENCEPOST_TRIGGER_COOLDOWN");
    if (p && sscanf(p, "%lf%c", &seconds, &extra) == 1 && isfinite(seconds) && seconds >= 0)
        g_cooldown_ns = seconds * 1e9;
    p = getenv("FENCEPOST_TRIGGER_FILE");
    if (!p) asprintf(&g_trigger_file, "/tmp/fencepost-trigger-%u", getuid());
    else if (*p) g_trigger_file = strdup(p);
    struct stat st;
    if (g_trigger_file && !stat(g_trigger_file, &st) && st.st_uid == geteuid()) g_file_mtime = st.st_mtimespec;
}

static void trace_init(const char *path)
{
    g_ring = trace_unsigned("FENCEPOST_RING", 0);
    if (g_ring) {
        g_ring_after = trace_unsigned("FENCEPOST_RING_AFTER", 2);
        unsigned mb = trace_unsigned("FENCEPOST_RING_MB", 64);
        g_ring_limit = (size_t)(mb ? mb : 64) * 1024 * 1024;
        ring_init_triggers();
        g_ring_writer = dispatch_queue_create("fencepost.ring-writer", DISPATCH_QUEUE_SERIAL);
        g_ring_prefix = strdup(path);
        g_ring_history = calloc(FP_POOL, sizeof *g_ring_history);
    } else g_trace = fp_open_path(path, "w");
    if (!g_ring && !g_trace) {
        fp_log("fencepost: could not open FENCEPOST_TRACE: %s\n", strerror(errno));
        return;
    }
    const char *p = getenv("FENCEPOST_TRACE_FRAMES");
    unsigned long long first, last;
    char tail;
    if (p && sscanf(p, "%llu:%llu%c", &first, &last, &tail) == 2 && first > 0 && last >= first) {
        g_trace_first = first; g_trace_last = last;
    }
    if ((p = getenv("FENCEPOST_TRACE_MB"))) {
        char *end;
        unsigned long long mb = strtoull(p, &end, 10);
        if (*p != '-' && !*end && mb > 0 && mb <= SIZE_MAX / (1024 * 1024)) g_trace_limit = (size_t)mb * 1024 * 1024;
    }
    if ((p = getenv("FENCEPOST_TRACE_GPU"))) g_trace_gpu = atoi(p) != 0;
    g_trace_pid = getpid();
    g_trace_nodes = CFDictionaryCreateMutable(NULL, 0, NULL, NULL);
    g_trace_threads = CFDictionaryCreateMutable(NULL, 0, NULL, NULL);
    if (g_trace) {
        setvbuf(g_trace, NULL, _IOFBF, 64 * 1024);
        fputc('[', g_trace);
    }
    g_trace_bytes = 1;
    g_trace_enabled = 1;
    atexit(final_summary);
}

__attribute__((constructor)) static void fp_init(void)
{
    if (@available(macOS 26.0, *)) {
        g_mtl4_options_class = objc_lookUpClass("MTL4CommitOptions");
        if (g_mtl4_options_class) {
            pthread_mutex_lock(&g_lock);
            hook(g_mtl4_options_class, H_4_OPTIONS_INIT, "init", (IMP)h_mtl4_options_init);
            hook(g_mtl4_options_class, H_4_ADD_FEEDBACK, "addFeedbackHandler:", (IMP)h_mtl4_app_feedback);
            pthread_mutex_unlock(&g_lock);
        }
    }
    g_spike_enabled = strcmp(getenv("FENCEPOST_SPIKE") ?: "", "0") != 0;
    const char *p = getenv("FENCEPOST_LOG");
    if (p && *p) g_out = fp_open_path(p, "a");
    if ((p = getenv("FENCEPOST_CAPTURE")) && *p) capture_init(p);
    if ((p = getenv("FENCEPOST_TRACE")) && *p) trace_init(p);
    if ((p = getenv("FENCEPOST_MAX_REPORTS"))) g_max_reports = atol(p);
    if ((p = getenv("FENCEPOST_POSSIBLE"))) g_show_possible = atoi(p);
    if ((p = getenv("FENCEPOST_SUMMARY_SECS"))) g_summary_secs = atoi(p);
    if ((p = getenv("FENCEPOST_BIND_MAYWRITE"))) g_bind_maywrite = atoi(p);
    if ((p = getenv("FENCEPOST_CHAIN"))) g_chain_frame = atol(p);
}

/* For tests and tools: dlsym these. out[] gets ST_COUNT counters (see the enum). */
__attribute__((visibility("default"))) int fencepost_stats(long *out, int n)
{
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < n && i < ST_COUNT; i++) out[i] = g_st[i];
    pthread_mutex_unlock(&g_lock);
    return ST_COUNT;
}

#ifdef FENCEPOST_TEST_HOOKS
__attribute__((visibility("default"))) long fencepost_test_trace_cbs(long *peak)
{
    pthread_mutex_lock(&g_lock);
    long live = g_mtl4_trace_live;
    *peak = g_mtl4_trace_peak;
    pthread_mutex_unlock(&g_lock);
    return live;
}

__attribute__((visibility("default"))) long fencepost_test_retired_signals(void)
{
    pthread_mutex_lock(&g_lock);
    long retired = g_mtl4_signals_retired;
    pthread_mutex_unlock(&g_lock);
    return retired;
}
#endif
/* Declare a barrier at the current point of cb: every encoder submitted to cb's queue
 * before this point must be ordered (by fences or events) before every encoder after it.
 * A runtime calls this where its API demands full ordering (a D3D12 ResourceBarrier, a
 * Vulkan pipeline barrier), and fencepost checks the synchronisation it encoded. */
__attribute__((visibility("default"))) void fencepost_barrier(id cb)
{
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    if (c && !c->mtl4) { fp_item it = { 3, 0, NULL, (uint64_t)c->nenc }; PUSH(c->it, c->nit, c->cit, it); }
    pthread_mutex_unlock(&g_lock);
}

/* End a frame at the current point of cb. Presents end frames by themselves; a program
 * that presents some other way, or wants its own frame boundaries, calls this instead, and
 * from its first call presents no longer count. */
__attribute__((visibility("default"))) void fencepost_frame(id cb)
{
    pthread_mutex_lock(&g_lock);
    g_frame_api = 1;
    fp_cb *c = cb_find_open((void *)cb);
    if (c) {
        fp_item it = { 4, 0, NULL, g_trace_enabled ? trace_clock() : 0 };
        PUSH(c->it, c->nit, c->cit, it);
    }
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
