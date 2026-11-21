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
#import <objc/message.h>
#import <objc/runtime.h>
#include <Block.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <os/lock.h>
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
#define FP_QRECENT      1024
#define FP_MAXQ         32
#define FP_OBJBITS      18
#define FP_OBJMAP       (1 << FP_OBJBITS)
#define FP_OBJPROBE     4096            /* give up a lookup after this many slots */
#define FP_SWEEP        2048            /* shadow slots cleaned per commit */
#define FP_FENCEMAP     8192
#define FP_EVHIST       64
#define FP_SIGMAP       (1 << 14)
#define FP_ENTCAP       512
#define FP_PEND_TIMEOUT 256             /* commits a node may wait for an event signal */
#define FP_HOOKCLS      8

enum { FP_RD = 1, FP_WR = 2, FP_MAYBE = 4 };    /* FP_MAYBE: declared read+write, may not write */
enum { FPK_RENDER = 1, FPK_COMPUTE, FPK_BLIT, FPK_SIGNAL, FPK_ML };
static const char *const kind_name[] = { "?", "render", "compute", "blit", "event signal", "machine learning" };
enum { FP4_VERTEX, FP4_FRAGMENT, FP4_TILE, FP4_OBJECT, FP4_MESH,
       FP4_DISPATCH, FP4_BLIT, FP4_ACCEL, FP4_ML, FP4_RESOURCE_STATE, FP4_STAGE_COUNT };
static const char *const fp4_stage_names[] = {
    "Vertex", "Fragment", "Tile", "Object", "Mesh", "Dispatch", "Blit",
    "AccelerationStructure", "MachineLearning", "ResourceState"
};
#define FP4_STAGE(i) (1u << (i))
#define FP4_ALL (FP4_STAGE(FP4_STAGE_COUNT) - 1)
enum { FP4_TABLES = FP4_DISPATCH + 1 };
enum { FP4_SUSPENDING = 1, FP4_RESUMING = 2 };
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
       ST_EVENT_LOOKUP_MISSES, ST_EVENT_UNKNOWN_SIGNALS,
       ST_NODE_POOL_EXHAUSTED, ST_ABANDONED_ENCODERS, ST_COUNT };

typedef struct { uint64_t w[FP_WORDS]; } fp_bits;

typedef struct {
    void *store;            /* root storage: texture (views resolved) or buffer */
    uint64_t a0, a1;        /* texture: mip levels [a0,a1); buffer: bytes [a0,a1) */
    uint32_t s0, s1;        /* texture: slices [s0,s1) */
    uint8_t is_tex, rw, sf, sl, via;   /* sf/sl: first and last stage the access happens in */
    uint16_t mtl4_stage;
    uint32_t first_epoch, last_epoch;
    union { uint64_t gpu_address; int description; };
} fp_acc;

typedef struct {
    const void *address;
    NSUInteger size, height, format, levels;  /* buffer length or texture width and shape */
    char label[96];
} fp_store_description;

typedef struct { uint64_t src; uint8_t after, before; } fp_edge;
typedef struct { void *ev; uint64_t val, queue_seq; uint8_t before; uint64_t ticket; } fp_evwait;
typedef struct { void *f; uint64_t src; uint8_t st; } fp_fwait;
typedef struct { void *f; uint8_t st; } fp_fupd;
typedef struct { uint64_t uid, seq; } fp_queue_signal;
enum { FP4_CONSUMER, FP4_PRODUCER, FP4_INTRA, FP4_FENCE_UPDATE, FP4_FENCE_WAIT };
typedef struct {
    void *fence;
    uint32_t epoch;
    uint16_t after, before;
    uint8_t kind;
} fp4_sync;
typedef struct { uint64_t uid, seq; uint32_t epoch; } fp4_prefix;
typedef struct { fp4_prefix cut[FP4_STAGE_COUNT][FP4_STAGE_COUNT]; } fp4_prefixes;
typedef struct { uint32_t epoch; uint16_t mask; } fp4_work_epoch;
typedef struct {
    MTLRenderPassAttachmentDescriptor *desc;
    NSUInteger layers;
    MTLStoreAction action;
    int slot;
} fp_deferred_att;

struct fp_queue;
typedef struct fp_node {
    uint64_t uid, aseq;             /* aseq: analysis order, 0 = not analysed yet */
    uint64_t cb_uid, commit_tick, done_tick, pend_since;
    int kind, state, idx;           /* state: 1 encoding, 2 ended, 3 committed, 4 analysed */
    _Atomic uint64_t open_uid;      /* nonzero while an encoder owns the slot */
    uint8_t uncertain, exempt, mtl4, mtl4_drawn;   /* mtl4_drawn: a draw or dispatch was encoded */
    uint8_t mtl4_has_attachment, mtl4_render_options;
    fp_deferred_att *deferred_att; int ndeferred_att, cdeferred_att;
    char label[64];
    fp_acc *acc;     int nacc, cacc;
    fp_store_description *descriptions; int ndescriptions, cdescriptions;
    fp_fwait *fw;    int nfw, cfw;
    fp_fupd *fu;     int nfu, cfu;
    fp_edge *ed;     int ned, ced;
    fp_evwait *ew;   int nev, cev;
    struct fp_queue *req_q;         /* declared barrier: must follow req_q's nodes before req_mark */
    uint64_t req_mark, bar_cb;
    int bar_idx;
    uint64_t frame, queue_seq, chain_prev;   /* chain_prev: earlier node on the longest ordered chain */
    uint32_t depth;                 /* encoders on that chain within the frame, this one included */
    uint32_t mtl4_epoch, mtl4_work, mtl4_store_epoch, mtl4_draw_epoch;
    uint32_t mtl4_first[FP4_STAGE_COUNT], mtl4_last[FP4_STAGE_COUNT];
    uint64_t mtl4_previous, mtl4_resume_from;
    fp4_sync *mtl4_sync; int mtl4_nsync, mtl4_csync;
    fp4_work_epoch *mtl4_work_epochs; int mtl4_nwork_epochs, mtl4_cwork_epochs;
    fp4_prefixes *mtl4_prefixes;
    id mtl4_tables[FP4_TABLES];
    uint64_t mtl4_versions[FP4_TABLES];
    id mtl4_pipeline;               /* FP4PipelineTable of the bound pipeline, nil = access unknown */
    void *mtl4_pipeline_object;     /* the bound pipeline itself, compared only */
    uint32_t mtl4_pipeline_serial;
    fp_bits anc[2];                 /* what is ordered before this node's vertex / fragment work */
} fp_node;

typedef struct { uint8_t type; uint64_t uid; void *ev; uint64_t val; } fp_item;   /* 0 encoder, 1 signal, 2 wait, 3 barrier, 4 frame end */
typedef struct {
    uint64_t uid, enq, commit_tick;
    void *cb;
    struct fp_queue *q;
    int state, nenc, mtl4, mtl4_resident;
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
    uint64_t mtl4_wait_ticket, mtl4_last_node, mtl4_event_completed_through;
    fp4_prefixes *mtl4_prefixes;
    struct fp_trace_cb *mtl4_last_trace;
    void **mtl4_residency;
    int mtl4_nresidency, mtl4_cresidency;
    fp_queue_signal *mtl4_signals;
    int mtl4_sig_first, mtl4_nsignals, mtl4_csignals;
} fp_queue;

typedef struct { fp_acc a; uint64_t uid, aseq; } fp_ent;
typedef struct { void *key; fp_ent *w; int nw, cw; fp_ent *r; int nr, cr; } fp_obj;   /* writes (incl. may-write) and pure reads */
typedef struct {
    int known, gpu;
    uint64_t cpu_max, nh, rotated_max;
    struct { uint64_t val, uid; } h[FP_EVHIST];
} fp_ev;

#define PUSH(arr, n, cap, val) do { if ((n) == (cap)) { (cap) = (cap) ? (cap) * 2 : 8; \
    (arr) = realloc((arr), (size_t)(cap) * sizeof *(arr)); } (arr)[(n)++] = (val); } while (0)

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_mtl4_options_lock = PTHREAD_MUTEX_INITIALIZER;
static fp_node g_pool[FP_POOL];
static fp_cb g_cbs[FP_CBPOOL];
static fp_queue g_q[FP_MAXQ]; static int g_nq;
static struct { void *cb; uint64_t uid; } g_opencb[FP_OPENCB]; static unsigned g_opencb_rr;
static struct { void *f; uint64_t uid; uint8_t after; } g_fence[FP_FENCEMAP];
static CFMutableDictionaryRef g_events;   /* event lifetime key -> fp_ev owned by its association */
static uintptr_t g_next_event_key;
static char g_event_owner_key;
static fp_obj g_obj[FP_OBJMAP];
static uint64_t g_sig[FP_SIGMAP];
static uint64_t *g_pend; static int g_npend, g_cpend;
static uint64_t g_uid, g_cbuid, g_aseq, g_tick;
static long g_st[ST_COUNT];
static long g_mtl4_ml_encoders, g_mtl4_gpu_errors;
static long g_mtl4_residency_only, g_mtl4_accel_builds, g_mtl4_partial_hits;
static atomic_long g_fp4_address_misses, g_fp4_id_misses, g_fp4_unknown_pipelines;
static atomic_int g_fp4_active;   /* the process made a Metal 4 object: request pipeline reflection */
#ifdef FENCEPOST_TEST_HOOKS
static unsigned g_mtl4_test_error_commit;
static int g_mtl4_test_reflection_fail, g_mtl4_test_reflection_failed;
#endif
static FILE *g_out;
static long g_max_reports = 200, g_printed;
static int g_show_possible = 1, g_summary_secs = 10, g_started, g_bind_maywrite, g_mtl4_rw_possible;
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
static __thread struct {
    void *enc;
    uint64_t uid;
    void *table, *table_state;
    uint64_t table_generation;
} tl_cache;

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

/* Producers can hold g_lock. Only the drain touches the output fd, and it
 * never holds the log mutex while writing. Keep the amount of queued data bounded. */
#define FP_LOG_LINES 2048
#define FP_LOG_LINE_BYTES 1536
static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;
static dispatch_queue_t g_log_queue;
static dispatch_group_t g_log_group;
static char g_log_lines[FP_LOG_LINES][FP_LOG_LINE_BYTES];
static unsigned g_log_head, g_log_count;
static unsigned long long g_log_dropped;
static int g_log_active, g_log_fd = STDERR_FILENO;
static __thread FILE *tl_log_final;
static char *g_log_final;

static int fp_log_write(int fd, const char *text)
{
    size_t left = strlen(text);
    while (left) {
        ssize_t written = write(fd, text, left);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return 0;
        text += written;
        left -= (size_t)written;
    }
    return 1;
}

static void fp_log_drain(void *unused)
{
    for (;;) {
        char line[FP_LOG_LINE_BYTES];
        pthread_mutex_lock(&g_log_lock);
        if (g_log_dropped) {
            snprintf(line, sizeof line, "fencepost: %llu log lines dropped while output was backed up\n", g_log_dropped);
            g_log_dropped = 0;
        } else if (g_log_count) {
            memcpy(line, g_log_lines[g_log_head], sizeof line);
            g_log_head = (g_log_head + 1) % FP_LOG_LINES;
            g_log_count--;
        } else {
            g_log_active = 0;
            dispatch_group_leave(g_log_group);
            pthread_mutex_unlock(&g_log_lock);
            return;
        }
        pthread_mutex_unlock(&g_log_lock);
        fp_log_write(g_log_fd, line);
    }
}

static void fp_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void fp_log(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    if (tl_log_final) {
        vfprintf(tl_log_final, fmt, ap);
        va_end(ap);
        return;
    }
    pthread_mutex_lock(&g_log_lock);
    if (g_log_count == FP_LOG_LINES) g_log_dropped++;
    else {
        char *line = g_log_lines[(g_log_head + g_log_count) % FP_LOG_LINES];
        int length = vsnprintf(line, FP_LOG_LINE_BYTES, fmt, ap);
        if (length >= FP_LOG_LINE_BYTES) line[FP_LOG_LINE_BYTES - 2] = '\n';
        if (length >= 0) g_log_count++;
    }
    if (!g_log_active && g_log_count) {
        g_log_active = 1;
        dispatch_group_enter(g_log_group);
        dispatch_async_f(g_log_queue, NULL, fp_log_drain);
    }
    pthread_mutex_unlock(&g_log_lock);
    va_end(ap);
}

static void fp_log_flush(const char *required)
{
    /* A final summary or fatal diagnostic must have room even in a full FIFO. */
    char *copy = required && *required ? strdup(required) : NULL;
    __block int delivered = 0;
    if (copy) dispatch_group_async(g_log_group, g_log_queue, ^{
        delivered = fp_log_write(g_log_fd, copy);
        free(copy);
    });
    long pending = dispatch_group_wait(g_log_group, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC));
    if (required && *required && (pending || !delivered)) {
        /* Writing again to a full pipe could stall exit. Save the complete
         * diagnostic directly to a private file without using the log queue. */
        char path[96];
        snprintf(path, sizeof path, "/tmp/fencepost-final-%d-XXXXXX", getpid());
        int fd = mkstemp(path);
        if (fd >= 0) {
            fp_log_write(fd, required);
            close(fd);
        }
    }
}

static void fp_log_finish(void)
{
    fp_log_flush(g_log_final);
    free(g_log_final);
}

static fp_node *node_get(uint64_t uid)
{
    fp_node *n = &g_pool[uid % FP_POOL];
    return uid && n->uid == uid ? n : NULL;
}

static void clear_deferred_attachments(fp_node *n)
{
    for (int i = 0; i < n->ndeferred_att; i++) [n->deferred_att[i].desc release];
    n->ndeferred_att = 0;
}

static void trace_recycle_node(uint64_t uid);

static fp_node *node_new(int kind)   /* lock held */
{
    uint64_t uid = g_uid + 1;
    fp_node *n = &g_pool[uid % FP_POOL];
    /* Open encoders use their slot without g_lock and still own its tables and attachments. */
    while (n->state == 1) {
        if (uid - g_uid == FP_POOL) {
            g_st[ST_NODE_POOL_EXHAUSTED]++;
            return NULL;
        }
        n = &g_pool[++uid % FP_POOL];
    }
    g_uid = uid;
    if (n->uid) trace_recycle_node(n->uid);
    if (n->mtl4)
        for (int i = 0; i < FP4_TABLES; i++) [n->mtl4_tables[i] release];
    if (n->mtl4) [n->mtl4_pipeline release];
    memset(n->mtl4_tables, 0, sizeof n->mtl4_tables);
    n->mtl4_pipeline = nil; n->mtl4_pipeline_object = NULL; n->mtl4_pipeline_serial = 0;
    memset(n->mtl4_versions, 0, sizeof n->mtl4_versions);
    n->mtl4_epoch = 1; n->mtl4_work = 0; n->mtl4_store_epoch = 0; n->mtl4_draw_epoch = 0;
    n->mtl4_previous = n->mtl4_resume_from = 0; n->mtl4_nsync = 0; n->mtl4_nwork_epochs = 0;
    free(n->mtl4_prefixes); n->mtl4_prefixes = NULL;
    memset(n->mtl4_first, 0, sizeof n->mtl4_first);
    memset(n->mtl4_last, 0, sizeof n->mtl4_last);
    if (n->uid && n->state && n->state < 4) g_st[ST_LOST]++;
    n->uid = uid; n->aseq = 0; n->cb_uid = 0; n->commit_tick = 0; n->done_tick = 0; n->pend_since = 0;
    n->kind = kind; n->state = 1; n->idx = 0; n->uncertain = 0; n->exempt = 0; n->label[0] = 0;
    n->mtl4 = 0; n->mtl4_drawn = 0; n->mtl4_has_attachment = n->mtl4_render_options = 0;
    n->req_q = NULL; n->req_mark = 0; n->bar_cb = 0; n->bar_idx = 0;
    n->frame = 0; n->queue_seq = 0; n->chain_prev = 0; n->depth = 0;
    n->ndescriptions = n->nacc = n->nfw = n->nfu = n->ned = n->nev = 0;
    clear_deferred_attachments(n);
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

@interface FPEventOwner : NSObject {
@public
    void *key;
    fp_ev history;
}
@end

@implementation FPEventOwner
- (void)dealloc
{
    if (key) {
        pthread_mutex_lock(&g_lock);
        CFDictionaryRemoveValue(g_events, key);
        pthread_mutex_unlock(&g_lock);
    }
    [super dealloc];
}
@end

static void *event_key(id event, int known)
{
    if (!event) return NULL;
    FPEventOwner *owner = objc_getAssociatedObject(event, &g_event_owner_key);
    if (owner && !known) return owner->key;
    FPEventOwner *created = nil;
    pthread_mutex_lock(&g_lock);
    if (!owner) owner = objc_getAssociatedObject(event, &g_event_owner_key);
    if (!owner) {
        if (!g_events) g_events = CFDictionaryCreateMutable(NULL, 0, NULL, NULL);
        if (g_events) owner = created = [[FPEventOwner alloc] init];
        if (owner) {
            /* An outstanding wait must never bind to an object that reused
             * this event's address. */
            owner->key = (void *)++g_next_event_key;
            CFDictionarySetValue(g_events, owner->key, &owner->history);
            objc_setAssociatedObject(event, &g_event_owner_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        }
    }
    if (owner && known) owner->history.known = 1;
    void *key = owner ? owner->key : NULL;
    pthread_mutex_unlock(&g_lock);
    [created release];
    return key;
}

static fp_ev *ev_get(void *key)   /* lock held */
{
    return key && g_events ? (fp_ev *)CFDictionaryGetValue(g_events, key) : NULL;
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

static void describe_store(char *buf, size_t len, const fp_node *n, const fp_acc *a)
{
    if (a->mtl4_stage) {
        if (a->is_tex)
            snprintf(buf, len, "Metal 4 texture 0x%llx, levels [%llu,%llu), slices [%u,%u)",
                     (unsigned long long)a->gpu_address, (unsigned long long)a->a0, (unsigned long long)a->a1,
                     a->s0, a->s1);
        else
            snprintf(buf, len, "Metal 4 buffer 0x%llx, bytes [%llu,%llu)", (unsigned long long)a->gpu_address,
                     (unsigned long long)a->a0, (unsigned long long)a->a1);
        return;
    }
    const fp_store_description *d = &n->descriptions[a->description];
    const char *quote = d->label[0] ? "\"" : "", *end_quote = d->label[0] ? "\" " : "";
    char what[160];
    if (a->is_tex) {
        snprintf(what, sizeof what, "texture %s%s%s%lux%lu, pixel format %lu, %lu levels",
                 quote, d->label, end_quote, (unsigned long)d->size, (unsigned long)d->height,
                 (unsigned long)d->format, (unsigned long)d->levels);
        char lv[48], sl[48];
        if (a->a1 - a->a0 > 1) snprintf(lv, sizeof lv, "levels %llu-%llu", (unsigned long long)a->a0, (unsigned long long)(a->a1 - 1));
        else snprintf(lv, sizeof lv, "level %llu", (unsigned long long)a->a0);
        if (a->s1 - a->s0 > 1) snprintf(sl, sizeof sl, "slices %u-%u", a->s0, a->s1 - 1);
        else snprintf(sl, sizeof sl, "slice %u", a->s0);
        snprintf(buf, len, "%s %p, %s, %s", what, d->address, lv, sl);
    } else {
        snprintf(what, sizeof what, "buffer %s%s%s%lu bytes", quote, d->label, end_quote, (unsigned long)d->size);
        snprintf(buf, len, "%s %p, bytes [%llu, %llu)", what, d->address,
                 (unsigned long long)a->a0, (unsigned long long)a->a1);
    }
}

static fp_frame *frame_get(uint64_t f);

/* ---------------------------------------------------------------- timeline (lock held unless noted) */

#define FP_TRACE_SAMPLES 4096     /* 32 KiB of timestamps, the Apple GPU limit */
#define FP_TRACE_GPU_SLACK_NS 1000000
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

typedef struct { uint64_t cpu, gpu; } fp_trace_clock_pair;
typedef struct { fp_trace_clock_pair pair; uint64_t at; } fp_trace_calibration;

typedef struct fp_trace_device {
    id<MTLDevice> device;
    id<MTLCounterSampleBuffer> buffer;
    id heap;                       /* MTL4CounterHeap, initialized only on Metal 4 */
    uint64_t frequency;
    fp_trace_clock_pair first, last;
    uint64_t calibration;
    uint64_t previous_end[FP_TRACE_SAMPLES / 2];
    unsigned free_slots[FP_TRACE_SAMPLES / 2], nfree;
    int mtl4;
    struct fp_trace_device *next;
} fp_trace_device;

typedef struct fp_trace_encoder {
    fp_trace_span span;            /* also the value in g_trace_nodes */
    struct fp_trace_encoder *next;
    struct fp_trace_cb *cb;
    uint64_t uid, start, end, thread, gpu_start, gpu_end;
    uint64_t first_sample, last_sample;  /* written by the completion callback before it takes g_lock */
    int sample, sample_ended, selected, kind, args_dropped;
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
enum { CAP_MTL4_ATTEMPTS = 3 };
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
static int g_capture_mtl4, g_capture_mtl4_unsafe;  /* Metal 4 observed / tracking lost (g_lock) */
static int g_capture_mtl4_failed;            /* a failed commit left tracked buffers (g_lock) */
static unsigned g_capture_mtl4_ended;       /* ended, unsubmitted buffers (g_lock) */
static int g_capture_starting, g_capture_stop_queued;
static uint64_t g_capture_generation, g_capture_feedback;
static int g_capture_feedback_unavailable;
static unsigned g_capture_attempts;          /* first begins tried for this request (capture lock) */
static pthread_cond_t g_capture_started_cond = PTHREAD_COND_INITIALIZER;
static NSString *g_capture_path;
static dispatch_queue_t g_capture_reporter;
/* Gate command-buffer entry during capture transitions, with tl_depth set so
 * Metal's nested hooks pass through. Never hold this across allocation or
 * commit: a full command queue can make allocation wait for another commit.
 * Lock order is capture lock then detector lock. Metal 4 end only waits for
 * start, via the condition under g_lock, so stop cannot block an end hook. */
static pthread_mutex_t g_capture_lock = PTHREAD_MUTEX_INITIALIZER;
static void capture_feedback_done(uint64_t generation, int failed);

typedef struct {
    NSMutableArray *ends;
    const char *refusal;
} fp_capture_start;
static _Thread_local fp_capture_start *tl_capture_start;

@interface FPCaptureBufferOwner : NSObject {
@public
    atomic_int ended;
}
@end

static char g_capture_buffer_key;

/* g_lock held. A failed submission can recover because its identities are
 * known. Lost hooks or ownership are process-wide: their lifetimes are unknown. */
static void capture_mtl4_recovered(void)
{
    if (!g_capture_mtl4_failed || g_capture_mtl4_unsafe || g_capture_mtl4_ended || g_capture_commits) return;
    g_capture_mtl4_failed = 0;
    fp_log("fencepost: Metal 4 capture gate recovered after failed commit\n");
}

static void capture_mtl4_mark_untracked(const char *reason)
{
    if (g_capture_mtl4_unsafe) return;
    g_capture_mtl4_unsafe = 1;
    fp_log("fencepost: Metal 4 capture disabled for this process: %s\n", reason);
}

@implementation FPCaptureBufferOwner
- (void)dealloc
{
    if (atomic_load_explicit(&ended, memory_order_relaxed)) {
        pthread_mutex_lock(&g_lock);
        g_capture_mtl4_ended--;
        capture_mtl4_recovered();
        pthread_mutex_unlock(&g_lock);
    }
    [super dealloc];
}
@end

static void capture_interrupt_start(const char *reason)
{
    tl_capture_start->refusal = reason;
    @throw [NSException exceptionWithName:@"FencepostCaptureStartInterrupted"
                                   reason:[NSString stringWithUTF8String:reason] userInfo:nil];
}

/* Returns true on the starting thread, where the caller must unwind instead
 * of waiting or allowing an unhooked object to escape the factory. */
static int capture_mtl4_tracking_lost(const char *reason)
{
    pthread_mutex_lock(&g_lock);
    capture_mtl4_mark_untracked(reason);
    /* A factory must finish the transition before returning an unhooked object. */
    while (g_capture_starting && !tl_capture_start)
        pthread_cond_wait(&g_capture_started_cond, &g_lock);
    pthread_mutex_unlock(&g_lock);
    return tl_capture_start != NULL;
}

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
static fp_trace_endpoint *g_trace_history;
#ifdef FENCEPOST_TEST_HOOKS
static long g_mtl4_trace_live, g_mtl4_trace_peak, g_mtl4_signals_retired;
static long g_mtl3_trace_live;
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

static fp_trace_encoder *trace_live_encoder(uint64_t uid)
{
    fp_trace_span *span = (fp_trace_span *)CFDictionaryGetValue(g_trace_nodes, TRACE_KEY(uid));
    return span && span->encoder ? (fp_trace_encoder *)span : NULL;
}

static fp_trace_span *trace_node(uint64_t uid)
{
    fp_trace_span *live = (fp_trace_span *)CFDictionaryGetValue(g_trace_nodes, TRACE_KEY(uid));
    if (live || !g_trace_history || !uid) return live;
    fp_trace_endpoint *old = &g_trace_history[uid % FP_POOL];
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

static fp_trace_device *trace_find_device(id<MTLDevice> device, int mtl4)   /* g_lock held */
{
    for (fp_trace_device *d = g_trace_devices; d; d = d->next)
        if (d->device == device && d->mtl4 == mtl4) return d;
    return NULL;
}

static void trace_init_device(fp_trace_device *d, id<MTLDevice> device, int mtl4)
{
    d->device = [device retain];
    d->mtl4 = mtl4;
    if (mtl4) {
        if (@available(macOS 26.0, *)) {
            if ([device sizeOfCounterHeapEntry:MTL4CounterHeapTypeTimestamp] != sizeof(MTL4TimestampHeapEntry))
                return;
            d->frequency = [device queryTimestampFrequency];
            if (!d->frequency) return;
            MTL4CounterHeapDescriptor *desc = [[MTL4CounterHeapDescriptor alloc] init];
            desc.type = MTL4CounterHeapTypeTimestamp;
            desc.count = FP_TRACE_SAMPLES;
            d->heap = [device newCounterHeapWithDescriptor:desc error:NULL];
            [desc release];
            [(id<MTL4CounterHeap>)d->heap setLabel:@"fencepost timestamps"];
        }
    } else if ([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
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
    }
    if (d->buffer || d->heap)
        for (unsigned i = 0; i < FP_TRACE_SAMPLES / 2; i++) d->free_slots[d->nfree++] = FP_TRACE_SAMPLES - 2 - 2 * i;
}

static fp_trace_device *trace_cb_device(id cb, int mtl4)
{
    if (!g_trace_enabled || !g_trace_gpu) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = [cb device];
        pthread_mutex_lock(&g_lock);
        int stopped = g_trace_stopped || g_frame_cur >= g_trace_last;
        fp_trace_device *d = stopped ? NULL : trace_find_device(device, mtl4);
        pthread_mutex_unlock(&g_lock);
        if (stopped || d) return d;

        fp_trace_device *candidate = calloc(1, sizeof *candidate);
        if (!candidate) return NULL;
        trace_init_device(candidate, device, mtl4);

        /* First buffers can arrive on several threads. Publish one fully initialized
         * set of counters, and release any duplicate Metal objects off the lock. */
        pthread_mutex_lock(&g_lock);
        d = trace_find_device(device, mtl4);
        if (!d) {
            d = candidate;
            candidate = NULL;
            d->next = g_trace_devices;
            g_trace_devices = d;
            if (d->buffer || d->heap) g_trace_sampling_supported = 1;
        }
        pthread_mutex_unlock(&g_lock);
        if (candidate) {
            [candidate->buffer release];
            [candidate->heap release];
            [candidate->device release];
            free(candidate);
        }
        return d;
    }
}

static int trace_calibration_due(const fp_trace_device *d)   /* g_lock held */
{
    return d && d->buffer && (!d->last.cpu || trace_clock() - d->calibration >= 250000000);
}

static fp_trace_calibration trace_read_calibration(const fp_trace_device *d)
{
    uint64_t before = trace_clock();
    MTLTimestamp cpu = 0, gpu = 0;
    [d->device sampleTimestamps:&cpu gpuTimestamp:&gpu];
    uint64_t after = trace_clock(), middle = before + (after - before) / 2;
    if (!cpu || !gpu || gpu == MTLCounterErrorValue) return (fp_trace_calibration){0};
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t scaled = (uint64_t)((long double)cpu * timebase.numer / timebase.denom);
    if (llabs((int64_t)scaled - (int64_t)middle) < llabs((int64_t)cpu - (int64_t)middle)) cpu = scaled;
    return (fp_trace_calibration){ { cpu, gpu }, before };
}

static void trace_record_calibration(fp_trace_device *d, fp_trace_calibration value)   /* g_lock held */
{
    fp_trace_clock_pair pair = value.pair;
    const fp_trace_clock_pair newest = d->last.cpu ? d->last : d->first;
    if (value.at <= d->calibration || pair.cpu <= newest.cpu || pair.gpu <= newest.gpu) return;
    if (!d->first.cpu) d->first = pair;
    else if (pair.cpu - d->first.cpu >= 10000) {
        if (d->last.cpu) d->first = d->last;
        d->last = pair;
    }
    d->calibration = value.at;
}

static void trace_release_sample(fp_trace_encoder *e)
{
    if (e->sample < 0) return;
    fp_trace_device *d = e->cb->device;
    d->free_slots[d->nfree++] = (unsigned)e->sample;
    e->sample = -1;
}

/* The committed list and sample indices are stable through feedback. The slots
 * stay reserved until the locked completion publishes or discards these results. */
static void trace_read_samples(fp_trace_cb *c)
{
    if (!c->device) return;
    for (fp_trace_encoder *e = c->encoders; e; e = e->next) {
        if (e->sample < 0) continue;
        NSRange range = NSMakeRange((NSUInteger)e->sample, 2);
        if (c->mtl4) {
            if (@available(macOS 26.0, *)) {
                if (!e->selected || !e->sample_ended) continue;
                NSData *data = [(id<MTL4CounterHeap>)c->device->heap resolveCounterRange:range];
                if (data.length != 2 * sizeof(MTL4TimestampHeapEntry)) continue;
                const MTL4TimestampHeapEntry *r = data.bytes;
                e->first_sample = r[0].timestamp;
                e->last_sample = r[1].timestamp;
            }
        } else {
            NSData *data = [c->device->buffer resolveCounterRange:range];
            if (data.length != 2 * sizeof(MTLCounterResultTimestamp)) continue;
            const MTLCounterResultTimestamp *r = data.bytes;
            e->first_sample = r[0].timestamp;
            e->last_sample = r[1].timestamp;
        }
    }
}

static int trace_valid_samples(const fp_trace_encoder *e)
{
    return e->first_sample && e->last_sample != MTLCounterErrorValue && e->last_sample > e->first_sample;
}

static void trace_resolve(fp_trace_encoder *e)   /* g_lock held */
{
    fp_trace_device *d = e->cb->device;
    if (!trace_valid_samples(e)) return;
    uint64_t *previous = &d->previous_end[e->sample / 2];
    if (e->first_sample < *previous) return;  /* an empty pass can leave old slot contents untouched */
    *previous = e->last_sample;
    if (!d->last.cpu) return;
    double scale = (double)(d->last.cpu - d->first.cpu) / (double)(d->last.gpu - d->first.gpu);
    double start = d->first.cpu + ((int64_t)e->first_sample - (int64_t)d->first.gpu) * scale;
    double end = d->first.cpu + ((int64_t)e->last_sample - (int64_t)d->first.gpu) * scale;
    /* Reject stale samples and counter clock discontinuities. Leave room for calibration error. */
    if (start < (double)e->cb->gpu_start - FP_TRACE_GPU_SLACK_NS
            || end > (double)e->cb->gpu_end + FP_TRACE_GPU_SLACK_NS
            || start <= 0 || end <= start || !e->cb->gpu_start) return;
    e->gpu_start = (uint64_t)start; e->gpu_end = (uint64_t)end;
}

static void trace_mtl4_resolve(fp_trace_encoder *e)   /* g_lock held */
{
    fp_trace_cb *c = e->cb;
    if (!trace_valid_samples(e) || c->gpu_end <= c->gpu_start) return;

    /* The heap ticks share the commit feedback clock. A recycled pair can
     * contain old writes if this pass failed to produce either endpoint.
     * Those writes predate CPU encoding. Don't invalidate a heap that may
     * still be in use by another submission. */
    double ns_per_tick = 1e9 / (double)c->device->frequency;
    double start = (double)e->first_sample * ns_per_tick;
    double end = (double)e->last_sample * ns_per_tick;
    if (start < (double)e->start - ns_per_tick
            || start < (double)c->gpu_start - FP_TRACE_GPU_SLACK_NS
            || end > (double)c->gpu_end + FP_TRACE_GPU_SLACK_NS) return;
    uint64_t clipped_start = (uint64_t)MAX(start, (double)c->gpu_start);
    uint64_t clipped_end = (uint64_t)MIN(end, (double)c->gpu_end);
    if (clipped_end <= clipped_start) return;
    e->gpu_start = clipped_start;
    e->gpu_end = clipped_end;
}

static void trace_discard_cb(fp_trace_cb *c);

static fp_trace_cb *trace_cb_new(id cb, uint64_t uid, int queue, int mtl4, fp_trace_device *device)
{
    FPTraceOwner *owner = mtl4 ? objc_getAssociatedObject(cb, &g_trace_owner_key) : nil;
    if (owner && owner->record) {
        trace_discard_cb(owner->record);
        owner->record = NULL;
    }
    if (g_trace_stopped || g_frame_cur >= g_trace_last) return NULL;
    fp_trace_cb *c = calloc(1, sizeof *c);
    c->refs = 1; c->uid = uid; c->queue = queue; c->encoder_tail = &c->encoders; c->mtl4 = mtl4;
#ifdef FENCEPOST_TEST_HOOKS
    if (mtl4) {
        g_mtl4_trace_live++;
        g_mtl4_trace_peak = MAX(g_mtl4_trace_peak, g_mtl4_trace_live);
    } else g_mtl3_trace_live++;
#endif
    c->device = device;
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
    else g_mtl3_trace_live--;
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
    fp_trace_device *d = e->cb->device;
    int calibrate = 0;
    pthread_mutex_lock(&g_lock);
    if (!g_trace_stopped) {
        if (d->nfree) {
            calibrate = !d->mtl4 && !d->first.cpu;
            e->sample = (int)d->free_slots[--d->nfree];
        } else g_trace_pool_exhausted++;
    }
    pthread_mutex_unlock(&g_lock);
    if (calibrate) {
        fp_trace_calibration value = trace_read_calibration(d);
        pthread_mutex_lock(&g_lock);
        trace_record_calibration(d, value);
        pthread_mutex_unlock(&g_lock);
    }
    return e->sample >= 0;
}

API_AVAILABLE(macos(26.0))
static void trace_mtl4_stamp(id enc, fp_trace_encoder *e, int end)
{
    id<MTL4CounterHeap> heap = e->cb->device->heap;
    NSUInteger index = (NSUInteger)e->sample + end;
    if (e->kind == FPK_RENDER) {
        [(id<MTL4RenderCommandEncoder>)enc writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
            afterStage:end ? MTLRenderStageFragment : MTLRenderStageVertex intoHeap:heap atIndex:index];
    } else {
        [(id<MTL4ComputeCommandEncoder>)enc writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
            intoHeap:heap atIndex:index];
    }
}

static void trace_encoder_register(fp_trace_encoder *e, fp_node *n)
{
    e->uid = n->uid; e->kind = n->kind;
    CFDictionarySetValue(g_trace_nodes, TRACE_KEY(e->uid), &e->span);
}

static void trace_encoder_end(const fp_node *n)
{
    fp_trace_encoder *e = trace_live_encoder(n->uid);
    if (e) e->end = trace_clock();
}

/* Lay out estimates in encode order before any links are settled. Keep
 * sampled spans at their measured times so overlapping work gets extra lanes. */
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
    if (!g_trace_history) return;
    fp_trace_endpoint *slot = &g_trace_history[uid % FP_POOL];
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
    if (!finalizing) {
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

/* A selected encoder can outlive GPU completion while analysis waits for an
 * event. If its detector node is recycled, retire the now-unreportable record. */
static void trace_recycle_node(uint64_t uid)
{
    if (!g_trace_enabled) return;
    fp_trace_encoder *e = trace_live_encoder(uid);
    if (!e || !e->selected || !e->cb->phased || !e->cb->completed) return;
    fp_trace_cb *c = e->cb;
    fp_trace_chunk *chunk = e->span.counted ? trace_chunk(e->span.frame) : NULL;
    if (chunk) chunk->outstanding--;
    g_trace_dropped++;
    trace_discard_encoder(e);
    trace_release_cb(c);
}

static const char *trace_stage(const fp_node *n, int stage, const char *nonrender)
{
    return n->kind == FPK_RENDER ? (stage ? "fragment" : "vertex") : nonrender;
}

static void trace_analyse(const fp_node *n)
{
    if (n->kind == FPK_SIGNAL) return;
    fp_trace_encoder *e = trace_live_encoder(n->uid);
    if (!e || !e->selected) {
        if (g_trace_stopped && trace_frame_selected(n->frame)) g_trace_dropped++;
        return;
    }
    if (g_trace_stopped) {
        g_trace_dropped++;
        e->args_dropped = 1;
        if (e->cb->completed) trace_place_encoder(e);
        return;
    }
    size_t length = 0;
    memcpy(e->label, n->label, sizeof e->label);
    FILE *f = open_memstream(&e->args, &length);
    if (!f) goto drop;
    fprintf(f, "\"uid\":%llu,\"cb\":%llu,\"queue\":%d,\"frame\":%llu,\"chain_depth\":%u,\"accesses\":%d,"
            "\"after_declared_barrier\":%s", TRACE_U64(n->uid),
            TRACE_U64(e->cb->mtl4 ? e->cb->uid : n->cb_uid), e->cb->queue,
            TRACE_U64(n->frame + 1), n->depth, n->nacc, n->req_mark ? "true" : "false");
    if (e->cb->mtl4) {
        fprintf(f, ",\"logical_cb\":%llu,\"mtl4_sync\":[", TRACE_U64(n->cb_uid));
        for (int i = 0; i < n->mtl4_nsync; i++) {
            const fp4_sync *sync = &n->mtl4_sync[i];
            static const char *const names[] = { "consumer", "producer", "intra-pass", "fence update", "fence wait" };
            fprintf(f, "%s{\"kind\":\"%s\",\"after_mask\":%u,\"before_mask\":%u}", i ? "," : "",
                    names[sync->kind], sync->after, sync->before);
        }
        fputc(']', f);
    }
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
        PUSH(c->signals, c->nsignals, c->csignals, n->uid);
        return;
    }
    fp_trace_encoder *e = trace_live_encoder(n->uid);
    if (!e) return;
    if (g_ring) {
        fp_trace_chunk *chunk = trace_chunk(n->frame);
        if (chunk) {
            chunk->outstanding++;
            e->span.counted = 1;
        }
    }
    c->refs++;
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
        if (e->selected && trace_node(e->uid)) {
            /* Finished args survive detector-slot reuse. A pending encoder
             * whose slot is gone cannot ever finish analysis. */
            if (!e->args && !e->args_dropped && !node_get(e->uid)) {
                g_trace_dropped++;
                e->args_dropped = 1;
            }
            trace_place_encoder(e);
        }
        else {
            if (e->selected) {
                if (!e->args_dropped) g_trace_dropped++;
                fp_trace_chunk *chunk = e->span.counted ? trace_chunk(e->span.frame) : NULL;
                if (chunk) chunk->outstanding--;
                trace_release_cb(c);
            }
            trace_discard_encoder(e);
        }
        e = next;
    }
    g_trace_completed[c->queue] = (fp_trace_endpoint){ c->span.tid ? c->uid : 0, c->span };
    for (int i = 0; i < c->nsignals; i++) trace_archive(c->signals[i], &c->span);
    free(c->signals);
    fp_trace_chunk *chunk = c->span.counted ? trace_chunk(c->span.frame) : NULL;
    if (chunk) chunk->outstanding--;
    trace_release_cb(c);
}

static void trace_cb_completed(fp_trace_cb *c, double start, double end, char *error,
                               fp_trace_calibration calibration)
{
    if (error) {
        c->error_time = trace_clock();
        c->gpu_error = error;
    }
    if (start > 0 && end >= start && end <= (double)trace_clock() / 1e9 + 1) {
        c->gpu_start = (uint64_t)(start * 1e9); c->gpu_end = (uint64_t)(end * 1e9);
    }
    if (calibration.at) trace_record_calibration(c->device, calibration);
    for (fp_trace_encoder *e = c->encoders; e; e = e->next) {
        if (e->sample < 0) continue;
        if (!g_trace_stopped) trace_resolve(e);
        trace_release_sample(e);
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
    long mtl4_gpu_errors, mtl4_address_misses, mtl4_id_misses, mtl4_residency_only, mtl4_accel_builds;
    uint64_t pool_exhausted, suppressed, evictions, cap_evictions, alloc_dropped;
    uint64_t origin;
    int stopped, sampling_supported;
} fp_trace_summary;

static fp_trace_summary trace_summary_state(void)
{
    fp_trace_summary state = {
        .pool_exhausted = g_trace_pool_exhausted, .stopped = g_trace_stopped,
        .origin = g_trace_origin, .sampling_supported = g_trace_sampling_supported,
        .mtl4_gpu_errors = g_mtl4_gpu_errors,
        .mtl4_address_misses = atomic_load_explicit(&g_fp4_address_misses, memory_order_relaxed),
        .mtl4_id_misses = atomic_load_explicit(&g_fp4_id_misses, memory_order_relaxed),
        .mtl4_residency_only = g_mtl4_residency_only,
        .mtl4_accel_builds = __atomic_load_n(&g_mtl4_accel_builds, __ATOMIC_RELAXED),
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
        "objs", "obj_drops", "analysis_ns", "frames", "frame_enc", "frame_chain", "gpu_idle",
        "event_wait_lookup_misses", "event_wait_unknown_signals", "node_pool_exhausted", "abandoned_encoders"
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
    if (state->mtl4_gpu_errors) fprintf(f, ",\"mtl4_gpu_errors\":%ld", state->mtl4_gpu_errors);
    if (state->mtl4_address_misses || state->mtl4_id_misses)
        fprintf(f, ",\"mtl4_address_misses\":%ld,\"mtl4_resource_id_misses\":%ld",
                state->mtl4_address_misses, state->mtl4_id_misses);
    if (state->mtl4_residency_only)
        fprintf(f, ",\"mtl4_residency_only_encoders\":%ld", state->mtl4_residency_only);
    if (state->mtl4_accel_builds)
        fprintf(f, ",\"mtl4_acceleration_structure_builds\":%ld", state->mtl4_accel_builds);
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

static void ring_trigger_impl(unsigned kind, const char *reason, const char *source, uint64_t frame,
                              uint64_t timestamp, const char *error, int allow_untraced)
{
    if (!allow_untraced && (!g_ring_encoded || g_trace_stopped || !trace_frame_selected(frame))) return;
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

static void ring_trigger(unsigned kind, const char *reason, const char *source, uint64_t frame,
                         uint64_t timestamp, const char *error)
{
    ring_trigger_impl(kind, reason, source, frame, timestamp, error, 0);
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

static void fp4_mask_text(char *text, size_t capacity, unsigned mask)
{
    if (mask == FP4_ALL) { snprintf(text, capacity, "All"); return; }
    if (!mask) { snprintf(text, capacity, "None"); return; }
    size_t used = 0;
    for (unsigned bits = mask; bits; bits &= bits - 1) {
        int n = snprintf(text + used, capacity - used, "%s%s", used ? "|" : "",
                         fp4_stage_names[__builtin_ctz(bits)]);
        if (n < 0 || (size_t)n >= capacity - used) break;
        used += n;
    }
}

static void fp4_explain(char *why, size_t size, const fp_node *f, const fp_acc *fa,
                        const fp_node *s, const fp_acc *sa)
{
    const char *after = fp4_stage_names[__builtin_ctz(fa->mtl4_stage)];
    const char *before = fp4_stage_names[__builtin_ctz(sa->mtl4_stage)];
    for (int i = 0; i < s->ned; i++) {
        fp_node *signal = node_get(s->ed[i].src);
        if (signal && signal->kind == FPK_SIGNAL && signal->req_q == f->req_q &&
            signal->queue_seq < f->queue_seq) {
            snprintf(why, size, "event wait matched a signal before the producer's %s work", after);
            return;
        }
    }
    if (f->req_q == s->req_q) {
        for (int side = 0; side < 2; side++) {
            const fp_node *n = side ? f : s;
            for (int i = 0; i < n->mtl4_nsync; i++) {
                const fp4_sync *op = &n->mtl4_sync[i];
                if (op->kind != (side ? FP4_PRODUCER : FP4_CONSUMER)) continue;
                char a[160], b[160];
                fp4_mask_text(a, sizeof a, op->after);
                fp4_mask_text(b, sizeof b, op->before);
                snprintf(why, size, "queue barrier %s -> %s does not order %s -> %s", a, b, after, before);
                return;
            }
        }
    }
    snprintf(why, size, "no Metal 4 ordering covers %s -> %s", after, before);
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
    describe_store(st, sizeof st, y, ya);
    if (fa->mtl4_stage && sa->mtl4_stage) {
        snprintf(first, sizeof first, "%s %s it: %s, %s stage", d1, rw_verb(fa->rw), via_name[fa->via],
                 fp4_stage_names[__builtin_ctz(fa->mtl4_stage)]);
        snprintf(second, sizeof second, "%s %s it: %s, %s stage", d2, rw_verb(sa->rw), via_name[sa->via],
                 fp4_stage_names[__builtin_ctz(sa->mtl4_stage)]);
    } else {
        snprintf(first, sizeof first, "%s %s it: %s%s", d1, rw_verb(fa->rw), via_name[fa->via], stage_name(f, fa->sl));
        snprintf(second, sizeof second, "%s %s it: %s%s", d2, rw_verb(sa->rw), via_name[sa->via], stage_name(s, sa->sf));
    }

    /* why: look for the fence that was meant to order them */
    int said = 0;
    if (fa->mtl4_stage && sa->mtl4_stage) { fp4_explain(why, sizeof why, f, fa, s, sa); said = 1; }
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
        fp_log("fencepost:   %ld declared barriers: %ld encoder(s) not ordered after them (%ld hits), %ld ordered for "
               "%s only (%ld hits)\n", g_st[ST_BARRIERS], g_st[ST_ORDER_UNIQ], g_st[ST_ORDER_HITS],
               g_st[ST_PARTIAL_UNIQ], g_mtl4_partial_hits ? "some stages" : "the fragment stage", g_st[ST_PARTIAL_HITS]);
    if (g_st[ST_TRACKED])
        fp_log("fencepost:   %ld uses of tracked resources skipped (Metal orders those itself)\n", g_st[ST_TRACKED]);
    if (g_mtl4_gpu_errors)
        fp_log("fencepost:   Metal 4 GPU commit errors: %ld\n", g_mtl4_gpu_errors);
    long address_misses = atomic_load_explicit(&g_fp4_address_misses, memory_order_relaxed);
    long id_misses = atomic_load_explicit(&g_fp4_id_misses, memory_order_relaxed);
    if (address_misses || id_misses)
        fp_log("fencepost:   Metal 4 unknown bindings: %ld GPU addresses, %ld resource IDs\n", address_misses, id_misses);
    long unknown_pipelines = atomic_load_explicit(&g_fp4_unknown_pipelines, memory_order_relaxed);
    if (unknown_pipelines)
        fp_log("fencepost:   Metal 4 pipelines without binding reflection (access unknown): %ld\n", unknown_pipelines);
    if (g_mtl4_residency_only)
        fp_log("fencepost:   Metal 4 encoders with residency sets and no visible accesses: %ld\n", g_mtl4_residency_only);
    long accel_builds = __atomic_load_n(&g_mtl4_accel_builds, __ATOMIC_RELAXED);
    if (accel_builds) fp_log("fencepost:   Metal 4 acceleration-structure builds: %ld\n", accel_builds);
    fp_log("fencepost:   cost: %.2f s analysing at commit; %ld resources with live history%s\n",
           g_st[ST_ANALYSIS_NS] / 1e9, g_st[ST_OBJS], g_st[ST_OBJ_DROPS] ? " (TABLE FULL: some accesses not checked)" : "");
    frame_line();
    if (g_st[ST_HEAP_USES] || g_st[ST_PARALLEL] || g_st[ST_TIMEOUTS] || g_st[ST_UNCERTAIN] || g_st[ST_WINDOW] || g_st[ST_LOST])
        fp_log("fencepost:   blind spots: %ld useHeap calls (resources inside heaps not seen), %ld parallel render "
               "encoders (not followed), %ld event waits timed out, %ld nodes with unresolved event waits, "
               "%ld checks past the %d-encoder window, %ld nodes dropped\n",
               g_st[ST_HEAP_USES], g_st[ST_PARALLEL], g_st[ST_TIMEOUTS], g_st[ST_UNCERTAIN], g_st[ST_WINDOW],
               FP_W, g_st[ST_LOST]);
    if (g_st[ST_NODE_POOL_EXHAUSTED])
        fp_log("fencepost:   node pool exhausted: %ld nodes skipped while all slots were open\n",
               g_st[ST_NODE_POOL_EXHAUSTED]);
    if (g_st[ST_ABANDONED_ENCODERS])
        fp_log("fencepost:   %ld encoders released without endEncoding\n", g_st[ST_ABANDONED_ENCODERS]);
    if (g_st[ST_EVENT_LOOKUP_MISSES])
        fp_log("fencepost:   event waits with missing records: %ld\n", g_st[ST_EVENT_LOOKUP_MISSES]);
    if (g_st[ST_EVENT_UNKNOWN_SIGNALS])
        fp_log("fencepost:   event waits with unavailable signal history: %ld\n", g_st[ST_EVENT_UNKNOWN_SIGNALS]);
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

typedef struct { fp_node *node; uint32_t epoch; uint8_t stage, stores_only; } fp4_visit;
static fp4_visit *g_fp4_worklist;
static int g_fp4_nwork, g_fp4_cwork;
static uint64_t g_fp4_query;
static struct { uint64_t query; uint32_t epoch; } g_fp4_seen[FP_POOL][FP4_STAGE_COUNT + 1];

static void fp4_visit_work(fp_node *n, int stage, uint32_t epoch, int stores_only)
{
    unsigned slot = n->uid % FP_POOL, lane = stores_only ? FP4_STAGE_COUNT : stage;
    if (g_fp4_seen[slot][lane].query == g_fp4_query && g_fp4_seen[slot][lane].epoch >= epoch) return;
    g_fp4_seen[slot][lane].query = g_fp4_query;
    g_fp4_seen[slot][lane].epoch = epoch;
    fp4_visit visit = { n, epoch, stage, stores_only };
    PUSH(g_fp4_worklist, g_fp4_nwork, g_fp4_cwork, visit);
}

static uint32_t fp4_last_work_before(const fp_node *n, int stage, uint32_t epoch)
{
    if (n->mtl4_first[stage] > epoch) return 0;
    if (n->mtl4_last[stage] <= epoch) return n->mtl4_last[stage];
    for (int i = n->mtl4_nwork_epochs - 1; i >= 0; i--) {
        const fp4_work_epoch *work = &n->mtl4_work_epochs[i];
        if (work->epoch <= epoch && (work->mask & FP4_STAGE(stage))) return work->epoch;
    }
    return 0;
}

/* A Fragment update or producer barrier covers a render pass's attachment stores only
 * when no draw is encoded after it: the stores run at the end of the pass. Sync
 * operations are work for chaining, so only draws set mtl4_draw_epoch. */
static int fp4_store_after(const fp_node *n, uint32_t epoch)
{
    return n->mtl4_store_epoch > epoch && n->mtl4_draw_epoch <= epoch;
}

static void fp4_push_visit(fp_node *n, unsigned mask, uint32_t epoch, int include_stores)
{
    if (n->kind == FPK_SIGNAL) {
        fp4_visit_work(n, 0, epoch, 0);
        return;
    }
    /* A barrier alone cannot bridge a stage. Visit only recorded work at or before the cut. */
    mask &= n->mtl4_work;
    /* A render update/producer barrier covers attachment stores at pass end without
     * moving its cutoff for ordinary commands encoded after it. */
    if (include_stores && (mask & FP4_STAGE(FP4_FRAGMENT)) && fp4_store_after(n, epoch))
        fp4_visit_work(n, FP4_FRAGMENT, n->mtl4_store_epoch, 1);
    while (mask) {
        int stage = __builtin_ctz(mask);
        mask &= mask - 1;
        uint32_t last = fp4_last_work_before(n, stage, epoch);
        if (last) fp4_visit_work(n, stage, last, 0);
    }
}

static int fp4_walk_prefix(const fp_node *dst, fp4_prefix prefix, unsigned after,
                           const fp_node *from, int from_stage, uint32_t from_epoch)
{
    if (!prefix.uid) return 0;
    if (dst->req_q == from->req_q && (after & FP4_STAGE(from_stage)) &&
        (from->queue_seq < prefix.seq || (from->queue_seq == prefix.seq &&
         (from_epoch <= prefix.epoch || (from_stage == FP4_FRAGMENT &&
          from_epoch == from->mtl4_store_epoch && fp4_store_after(from, prefix.epoch))))))
        return 1;

    uint32_t epoch = prefix.epoch;
    for (fp_node *p = node_get(prefix.uid); p && p->aseq >= from->aseq; p = node_get(p->mtl4_previous)) {
        /* A queue barrier names work stages, not queue signal operations.
         * The signal's own prefix is entered via an event edge instead. */
        if (p->kind != FPK_SIGNAL) fp4_push_visit(p, after, epoch, 1);
        epoch = UINT32_MAX;
    }
    return 0;
}

static fp4_prefix fp4_previous_prefix(const fp_node *n)
{
    return (fp4_prefix){ n->mtl4_previous, n->queue_seq ? n->queue_seq - 1 : 0, UINT32_MAX };
}

/* Queue barriers use a cut per pair of stages. Only a conflict query walks
 * the older work behind a cut, so recording a barrier never builds N edges.
 * Epochs separate commands before and after each synchronization operation. */
static int fp4_reaches(fp_node *from, int from_stage, uint32_t from_epoch,
                       fp_node *to, int to_stage, uint32_t to_epoch)
{
    if (from->done_tick && from->done_tick < to->commit_tick) return 1;
    g_fp4_query++;
    g_fp4_nwork = 0;
    if (to->mtl4) fp4_push_visit(to, FP4_STAGE(to_stage), to_epoch, 0);
    else fp4_visit_work(to, to_stage, 0, 0);
    while (g_fp4_nwork) {
        fp4_visit v = g_fp4_worklist[--g_fp4_nwork];
        fp_node *n = v.node;
        if (from == n && (n->mtl4 ? v.stage == from_stage : v.stage >= from_stage) &&
            (v.stores_only ? from_epoch == v.epoch : from_epoch <= v.epoch)) return 1;
        if (from->done_tick && from->done_tick < n->commit_tick) return 1;
        if (v.stage == FP4_FRAGMENT && n->mtl4_resume_from) {
            /* Skip drawless middle segments without adding work to an empty stage. */
            for (fp_node *p = node_get(n->mtl4_resume_from); p && p->aseq >= from->aseq;
                 p = node_get(p->mtl4_resume_from)) {
                if (!p->mtl4_last[FP4_FRAGMENT]) continue;
                fp4_visit_work(p, FP4_FRAGMENT, p->mtl4_last[FP4_FRAGMENT], 0);
                break;
            }
        }
        if (n->kind == FPK_SIGNAL && n->mtl4 &&
            fp4_walk_prefix(n, fp4_previous_prefix(n), FP4_ALL, from, from_stage, from_epoch))
            return 1;
        for (int i = 0; i < n->ned; i++) {
            const fp_edge *edge = &n->ed[i];
            if (!n->mtl4 && edge->before > v.stage) continue;
            fp_node *p = node_get(edge->src);
            if (!p || p->aseq < from->aseq) continue;
            /* Event edges cover all Metal 4 stages. On Metal 3 the edge's
             * vertex/fragment slots also preserve intervening fence scope. */
            if (p->mtl4) fp4_push_visit(p, FP4_ALL, UINT32_MAX, 0);
            else fp4_visit_work(p, edge->after, 0, 0);
        }
        if (n->mtl4_prefixes) {
            for (int a = 0; a < FP4_STAGE_COUNT; a++)
                if (fp4_walk_prefix(n, n->mtl4_prefixes->cut[a][v.stage], FP4_STAGE(a), from, from_stage, from_epoch))
                    return 1;
        }
        for (int i = 0; i < n->mtl4_nsync; i++) {
            const fp4_sync *sync = &n->mtl4_sync[i];
            if (sync->epoch >= v.epoch || !(sync->before & FP4_STAGE(v.stage))) continue;
            if (sync->kind == FP4_CONSUMER) {
                if (fp4_walk_prefix(n, fp4_previous_prefix(n), sync->after, from, from_stage, from_epoch))
                    return 1;
            } else if (sync->kind == FP4_FENCE_WAIT) {
                for (fp_node *p = node_get(n->mtl4_previous); p && p->aseq >= from->aseq;
                     p = node_get(p->mtl4_previous)) {
                    for (int j = 0; j < p->mtl4_nsync; j++) {
                        const fp4_sync *update = &p->mtl4_sync[j];
                        if (update->kind == FP4_FENCE_UPDATE && update->fence == sync->fence)
                            fp4_push_visit(p, update->after, update->epoch, 1);
                    }
                }
            }
        }
    }
    return 0;
}

static void fp4_publish_prefixes(fp_queue *q, fp_node *n)
{
    if (q->mtl4_prefixes && n->kind != FPK_SIGNAL) {
        n->mtl4_prefixes = malloc(sizeof *n->mtl4_prefixes);
        if (n->mtl4_prefixes) *n->mtl4_prefixes = *q->mtl4_prefixes;
    }
    for (int i = 0; i < n->mtl4_nsync; i++) {
        const fp4_sync *sync = &n->mtl4_sync[i];
        fp4_prefix prefix;
        if (sync->kind == FP4_PRODUCER) prefix = (fp4_prefix){ n->uid, n->queue_seq, sync->epoch };
        else if (sync->kind == FP4_CONSUMER) {
            /* Apple's online docs and GPU timing agree that the before stages extend to
             * later encoders, although the SDK header only says current encoder. */
            prefix = fp4_previous_prefix(n);
            if (!prefix.uid) continue;
        } else continue;
        if (!q->mtl4_prefixes) q->mtl4_prefixes = calloc(1, sizeof *q->mtl4_prefixes);
        if (!q->mtl4_prefixes) continue;
        for (unsigned after = sync->after; after; after &= after - 1) {
            for (unsigned before = sync->before; before; before &= before - 1) {
                fp4_prefix *cut = &q->mtl4_prefixes->cut[__builtin_ctz(after)][__builtin_ctz(before)];
                /* A producer cut on this encoder already covers the previous prefix. */
                if (sync->kind == FP4_PRODUCER || cut->uid != n->uid) *cut = prefix;
            }
        }
    }
}

static int fp4_barrier_stages(fp_node *x, fp_node *y)
{
    unsigned ordered = 0;
    for (unsigned before = y->mtl4_work; before; before &= before - 1) {
        int b = __builtin_ctz(before);
        int covered = 1;
        for (unsigned after = x->mtl4_work; after; after &= after - 1) {
            int a = __builtin_ctz(after);
            uint32_t last = x->mtl4_last[a];
            uint32_t before_store = a == FP4_FRAGMENT && last == x->mtl4_store_epoch
                ? fp4_last_work_before(x, a, last - 1) : 0;
            if (!fp4_reaches(x, a, last, y, b, y->mtl4_first[b]) ||
                (before_store && !fp4_reaches(x, a, before_store, y, b, y->mtl4_first[b]))) {
                covered = 0;
                break;
            }
        }
        if (covered) ordered |= FP4_STAGE(b);
    }
    return ordered;
}

static int ordered(const fp_ent *e, fp_node *x, fp_node *y, const fp_acc *a)
{
    if (!x) return 1;
    if (x->done_tick && x->done_tick < y->commit_tick) return 1;   /* completed before y was committed */
    uint64_t d = y->aseq - e->aseq;
    if (d >= FP_W) { g_st[ST_WINDOW]++; return 1; }
    if (!e->a.mtl4_stage && !a->mtl4_stage) {
        if (bit_test(&y->anc[a->sf], 2 * d + e->a.sl)) return 1;
        if (!atomic_load_explicit(&g_fp4_active, memory_order_relaxed)) return 0;
    }
    int from_stage = e->a.mtl4_stage ? __builtin_ctz(e->a.mtl4_stage) : e->a.sl;
    int to_stage = a->mtl4_stage ? __builtin_ctz(a->mtl4_stage) : a->sf;
    return fp4_reaches(x, from_stage, e->a.last_epoch, y, to_stage, a->first_epoch);
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
                if (merged && same_range(&e->a, a) && !a->mtl4_stage) {
                    e->a.rw |= a->rw;
                    if (a->sl > e->a.sl) e->a.sl = a->sl;
                    *merged = 1;
                }
            } else if ((e->a.rw | a->rw) & (FP_WR | FP_MAYBE)) {
                if (!ordered(e, x, y, a)) report(x, &e->a, y, a);
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
    else if (o->nr && o->r[o->nr - 1].uid == y->uid && !a->mtl4_stage &&
             same_range(&o->r[o->nr - 1].a, a)) merged = 1;
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
    return !x || (!x->mtl4 && x->done_tick && x->done_tick < g_tick) || g_aseq - e->aseq >= FP_W;
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
    unsigned partial_stages = 0;
    long nbad = 0;
    for (uint64_t k = hi; k-- > lo;) {
        fp_node *x = node_get(q->recent[k % FP_QRECENT]);
        if (!x || x == y || x->kind == FPK_SIGNAL || x->exempt || !x->aseq || x->aseq >= y->aseq) continue;
        if (x->done_tick && x->done_tick < y->commit_tick) continue;
        uint64_t d = y->aseq - x->aseq;
        if (d >= FP_W) continue;
        if (y->mtl4) {
            if (!x->mtl4_work || !y->mtl4_work) continue;
            unsigned covered = fp4_barrier_stages(x, y);
            if (covered == y->mtl4_work) continue;
            if (covered) {
                if (!part) { part = x; partial_stages = covered; }
                continue;
            }
        } else {
            if (bit_test(&y->anc[0], 2 * d + 1)) continue;
            if (bit_test(&y->anc[1], 2 * d + 1)) { if (!part) part = x; continue; }
        }
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
    if (!bad && y->mtl4) g_mtl4_partial_hits++;
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
    if (y->mtl4) {
        char source[160], target[160], covered[160];
        fp4_mask_text(source, sizeof source, x->mtl4_work);
        fp4_mask_text(target, sizeof target, y->mtl4_work);
        fp4_mask_text(covered, sizeof covered, partial_stages);
        snprintf(second, sizeof second, "%s: %s -> %s is %s%s", d2, source, target,
                 bad ? "unordered" : "ordered only before ", bad ? "" : covered);
    } else if (bad)
        snprintf(second, sizeof second, "%s is not ordered after it%s", d2,
                 nbad > 1 ? " (nor after other encoders from before the barrier)" : "");
    else
        snprintf(second, sizeof second, "%s waits for it only before its fragment stage; its vertex stage can overlap", d2);
    const char *detail = bad ? "a declared barrier is not honoured"
        : y->mtl4 ? "a declared barrier is honoured for only some stages"
                  : "a declared barrier is honoured for the fragment stage only";
    if (print) {
        fp_log("fencepost: ORDER %s\n", detail);
        fp_log("  barrier: %s\n", barrier);
        fp_log("  before : %s\n", d1);
        fp_log("  after  : %s\n", second);
    }
    char *why = NULL;
    size_t why_length = 0;
    FILE *details = marker ? open_memstream(&why, &why_length) : NULL;
    if (details) fputs(detail, details);
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
            trace_diagnostic(kind, x->uid, y->uid,
                             bad ? "ORDER barrier not honoured" : y->mtl4 ? "ORDER stage partial" : "ORDER fragment-only",
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
    if (y->mtl4 && y->kind == FPK_SIGNAL) {
        for (fp_node *x = node_get(y->mtl4_previous); x && x->frame == y->frame; x = node_get(x->mtl4_previous)) {
            if (x->done_tick && x->done_tick < y->commit_tick) continue;
            if (x->depth > best) { best = x->depth; y->chain_prev = x->uid; }
        }
    }
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
    if (y->kind != FPK_SIGNAL && y->req_q && y->req_mark) check_barrier(y);
    for (int i = 0; i < y->nacc; i++) check_access(y, &y->acc[i]);
    if (g_trace_enabled) trace_analyse(y);
}

static void record_event_signal(fp_ev *ev, uint64_t value, uint64_t uid)
{
    unsigned slot = ev->nh % FP_EVHIST;
    if (ev->nh >= FP_EVHIST) ev->rotated_max = MAX(ev->rotated_max, ev->h[slot].val);
    ev->h[slot].val = value;
    ev->h[slot].uid = uid;
    ev->nh++;
    ev->gpu = ev->known = 1;
}

/* Find the earliest eligible signal. A recycled node or a rotated-out value
 * that could satisfy the wait leaves its source unknown. Substituting a newer
 * signal could incorrectly order the waiter after unrelated work. */
static fp_node *first_event_signal(const fp_ev *ev, const fp_queue *queue, fp_evwait wait, int *gone)
{
    if (gone) *gone = 0;
    if (!ev || !ev->known) return NULL;
    uint64_t first = ev->nh > FP_EVHIST ? ev->nh - FP_EVHIST : 0;
    if (first && ev->rotated_max >= wait.val) {
        /* The discarded values no longer have queue and position metadata. */
        if (gone) *gone = 1;
        return NULL;
    }
    for (uint64_t k = first; k < ev->nh; k++) {
        if (ev->h[k % FP_EVHIST].val < wait.val) continue;
        fp_node *signal = node_get(ev->h[k % FP_EVHIST].uid);
        if (!signal) {
            if (gone) *gone = 1;
            return NULL;
        }
        if (signal->req_q != queue || signal->queue_seq < wait.queue_seq) return signal;
    }
    return NULL;
}

/* force 1 clears waits with no candidate, leaving pending signal dependencies
 * for another analysis pass. Force 2 is the final fallback. */
static int ready(fp_node *n, int force)
{
    int i, j;
    for (i = 0, j = 0; i < n->nev; i++) {
        fp_evwait w = n->ew[i];
        fp_ev *ev = ev_get(w.ev);
        int done = 0;
        if (!ev) g_st[ST_EVENT_LOOKUP_MISSES]++;
        if (!ev || !ev->known) done = 1;   /* event made outside our view (e.g. from a mach port): CPU-side */
        else {
            int gone;
            fp_node *signal = first_event_signal(ev, n->req_q, w, &gone);
            int pending_signal = signal && !signal->aseq;
            if (gone) {
                g_st[ST_EVENT_UNKNOWN_SIGNALS]++;
                done = 1;
            }
            if (signal && !pending_signal) {
                fp_edge e = { signal->uid, 1, w.before };
                PUSH(n->ed, n->ned, n->ced, e);
                done = 1;
            }
            if (!done && !pending_signal && ev->cpu_max >= w.val) done = 1;
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
    /* Process Metal 4 nodes in queue order for analysis. This allows a
     * signal to settle after its prefix without adding implicit GPU edges. */
    if (n->mtl4) {
        fp_node *previous = node_get(n->mtl4_previous);
        if (previous && !previous->aseq) return 0;
    }
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
    if (!n) return NULL;
    it->uid = n->uid;
    n->cb_uid = cb_uid; n->state = 2;
    unsigned first = q->nrecent > FP_QRECENT ? q->nrecent - FP_QRECENT : 0;
    for (unsigned k = first; !q->mtl4 && k < q->nrecent; k++) {   /* Metal 3 signal edges */
        fp_node *p = node_get(q->recent[k % FP_QRECENT]);
        if (!p || (p->done_tick && p->done_tick < tick)) continue;
        fp_edge e = { p->uid, 1, 0 };
        PUSH(n->ed, n->ned, n->ced, e);
    }
    fp_ev *ev = ev_get(it->ev);
    if (ev) record_event_signal(ev, it->val, n->uid);
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
    if (n->mtl4) {
        n->mtl4_previous = q->mtl4_last_node;
        q->mtl4_last_node = n->uid;
        fp4_publish_prefixes(q, n);
    }
    queue_recent(q, n->uid);
    PUSH(g_pend, g_npend, g_cpend, n->uid);
}

/* Phase 1, in submission order: resolve fence waits against the latest update, record
 * fence updates and event signals, queue the nodes for analysis. */
static void phase1(fp_cb *c, fp_node **suspended)
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
            if (nactive < 32) active[nactive++] = (fp_evwait){ it->ev, it->val, q->nrecent, 0, 0 };
            continue;
        }
        fp_node *n;
        if (it->type == 1) {
            n = phase_signal(q, c->uid, c->commit_tick, it);
            if (!n) continue;
        } else {
            n = node_get(it->uid);
            if (!n) {
                if (suspended) *suspended = NULL;
                continue;
            }
            if (c->mtl4 && !n->nacc && (c->mtl4_resident || q->mtl4_nresidency)) g_mtl4_residency_only++;
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
        if (suspended) {
            /* Metal stitches the same pass in commit-array order. CPU
             * encoding can happen concurrently or in a different order. */
            if (*suspended && (n->mtl4_render_options & FP4_RESUMING) && (*suspended)->cb_uid != n->cb_uid)
                n->mtl4_resume_from = (*suspended)->uid;
            *suspended = n->mtl4_render_options & FP4_SUSPENDING ? n : NULL;
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
        phase1(h, NULL);
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

static void cb_done(uint64_t uid, fp_trace_cb *trace, id<MTLCommandBuffer> b, uint64_t capture_generation)
{
    MTLCommandBufferStatus status = MTLCommandBufferStatusNotEnqueued;
    double start = 0, end = 0;
    char *error = NULL;
    fp_trace_calibration calibration = {0};
    if (trace || (g_capture_on & CAP_GPU_ERROR)) @autoreleasepool {
        status = b.status;
        if (trace) {
            if (g_ring && g_ring_gpuerr) {
                NSError *failure = b.error;
                if (status == MTLCommandBufferStatusError || failure)
                    error = strdup(failure.localizedDescription.UTF8String ?: "command buffer error");
            }
            if (status == MTLCommandBufferStatusCompleted) {
                start = b.GPUStartTime; end = b.GPUEndTime;
            }
            if (trace->device && trace->device->buffer) {
                pthread_mutex_lock(&g_lock);
                int calibrate = trace_calibration_due(trace->device);
                int sample = !g_trace_stopped;
                pthread_mutex_unlock(&g_lock);
                if (calibrate) calibration = trace_read_calibration(trace->device);
                if (sample) trace_read_samples(trace);
            }
        }
    }
    pthread_mutex_lock(&g_lock);
    g_inflight--;
    cb_mark_done(uid);
    if ((g_capture_on & CAP_GPU_ERROR) && status == MTLCommandBufferStatusError)
        capture_request("gpu-error");
    if (trace) trace_cb_completed(trace, start, end, error, calibration);
    if (g_ring) ring_update(0);
    if (g_trace) trace_stream_flush();
    pthread_mutex_unlock(&g_lock);
    if (capture_generation) capture_feedback_done(capture_generation, 0);
}

/* ---------------------------------------------------------------- recording */

@interface FPEncoderOwner : NSObject {
@public
    uint64_t uid;
}
@end

static char g_encoder_owner_key;

static fp_node *open_node(uint64_t uid)
{
    fp_node *n = &g_pool[uid % FP_POOL];
    return uid && atomic_load_explicit(&n->open_uid, memory_order_acquire) == uid ? n : NULL;
}

@implementation FPEncoderOwner
- (void)dealloc
{
    if (open_node(uid)) {
        id tables[FP4_TABLES] = {0}, pipeline = nil;
        fp_deferred_att *attachments = NULL;
        int count = 0;
        pthread_mutex_lock(&g_lock);
        fp_node *n = open_node(uid);
        if (n) {
            memcpy(tables, n->mtl4_tables, sizeof tables);
            memset(n->mtl4_tables, 0, sizeof n->mtl4_tables);
            pipeline = n->mtl4_pipeline;
            n->mtl4_pipeline = nil;
            count = n->ndeferred_att;
            if (count) {
                attachments = n->deferred_att;
                n->deferred_att = NULL;
                n->ndeferred_att = n->cdeferred_att = 0;
            }
            /* Invalidate the command buffer's item too. Its trace record
             * still owns any sample slots until the buffer settles. */
            atomic_store_explicit(&n->open_uid, 0, memory_order_release);
            n->uid = 0;
            n->state = 0;
            g_st[ST_ABANDONED_ENCODERS]++;
        }
        pthread_mutex_unlock(&g_lock);
        for (int i = 0; i < FP4_TABLES; i++) [tables[i] release];
        [pipeline release];
        for (int i = 0; i < count; i++) [attachments[i].desc release];
        free(attachments);
    }
    [super dealloc];
}
@end

static fp_node *cur_node(id enc)
{
    if (tl_cache.enc == (void *)enc) {
        fp_node *n = open_node(tl_cache.uid);
        if (n) return n;
    }
    FPEncoderOwner *owner = objc_getAssociatedObject(enc, &g_encoder_owner_key);
    fp_node *n = owner ? open_node(owner->uid) : NULL;
    if (n) { tl_cache.enc = (void *)enc; tl_cache.uid = owner->uid; }
    return n;
}

static int push_acc(fp_node *n, const fp_acc *a)
{
    for (int i = n->nacc - 1; i >= 0 && i >= n->nacc - 32; i--) {
        fp_acc *p = &n->acc[i];
        if (p->store == a->store && p->a0 == a->a0 && p->a1 == a->a1 && p->s0 == a->s0 && p->s1 == a->s1 &&
            p->via == a->via && p->mtl4_stage == a->mtl4_stage && (!a->mtl4_stage || p->rw == a->rw)) {
            p->rw |= a->rw;
            if (a->sf < p->sf) p->sf = a->sf;
            if (a->sl > p->sl) p->sl = a->sl;
            if (a->last_epoch > p->last_epoch) p->last_epoch = a->last_epoch;
            return i;
        }
    }
    PUSH(n->acc, n->nacc, n->cacc, *a);
    __atomic_fetch_add(&g_st[ST_ACCESSES], 1, __ATOMIC_RELAXED);
    return n->nacc - 1;
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

/* Resource metadata outlives a lookup without retaining an application allocation. */
typedef struct {
    void *store;
    uint64_t address, end, offset;
    uint32_t level, levels, slice, slices;
    int texture, shader_write;
} fp4_resource;

@interface FP4ResourceOwner : NSObject {
@public
    fp4_resource info;
    uintptr_t device;
    uint64_t resource_id;
    int indexed;
}
@end

typedef struct {
    FP4ResourceOwner **items;
    int n, cap, is_id;
} fp4_resource_index;

static fp4_resource_index g_fp4_addresses, g_fp4_ids = { .is_id = 1 };
static pthread_mutex_t g_fp4_resource_lock = PTHREAD_MUTEX_INITIALIZER;
static uintptr_t g_fp4_store_id;
static atomic_uint_fast64_t g_fp4_resource_generation;
static char g_fp4_resource_key;

static uint64_t fp4_index_key(const fp4_resource_index *index, const FP4ResourceOwner *owner)
{
    return index->is_id ? owner->resource_id : owner->info.address;
}

static int fp4_index_lower(const fp4_resource_index *index, uintptr_t device, uint64_t key)
{
    int lo = 0, hi = index->n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        const FP4ResourceOwner *owner = index->items[mid];
        if (owner->device < device || (owner->device == device && fp4_index_key(index, owner) < key)) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static void fp4_index_insert(fp4_resource_index *index, FP4ResourceOwner *owner)
{
    uint64_t start = fp4_index_key(index, owner);
    int low = fp4_index_lower(index, owner->device, start), high = low;
    if (index->is_id) {
        if (high < index->n && index->items[high]->device == owner->device && index->items[high]->resource_id == start)
            high++;
    } else {
        if (low && index->items[low - 1]->device == owner->device && index->items[low - 1]->info.end > start) low--;
        high = fp4_index_lower(index, owner->device, owner->info.end);
    }
    if (high > low) {
        memmove(index->items + low + 1, index->items + high,
                (size_t)(index->n - high) * sizeof *index->items);
        index->n -= high - low - 1;
    } else {
        PUSH(index->items, index->n, index->cap, owner);
        memmove(index->items + low + 1, index->items + low,
                (size_t)(index->n - low - 1) * sizeof *index->items);
    }
    index->items[low] = owner;
    owner->indexed = 1;
    atomic_fetch_add_explicit(&g_fp4_resource_generation, 1, memory_order_relaxed);
}

@implementation FP4ResourceOwner
- (void)dealloc
{
    if (indexed) {
        pthread_mutex_lock(&g_fp4_resource_lock);
        fp4_resource_index *index = resource_id ? &g_fp4_ids : &g_fp4_addresses;
        int i = fp4_index_lower(index, device, fp4_index_key(index, self));
        if (i < index->n && index->items[i] == self) {
            index->n--;
            atomic_fetch_add_explicit(&g_fp4_resource_generation, 1, memory_order_relaxed);
            memmove(index->items + i, index->items + i + 1, (size_t)(index->n - i) * sizeof *index->items);
        }
        pthread_mutex_unlock(&g_fp4_resource_lock);
    }
    [super dealloc];
}
@end

static fp4_resource fp4_install_resource(id resource, FP4ResourceOwner *owner, int index_address)
{
    pthread_mutex_lock(&g_fp4_resource_lock);
    FP4ResourceOwner *existing = objc_getAssociatedObject(resource, &g_fp4_resource_key);
    if (!existing) {
        /* Both APIs share a key for this allocation's lifetime. */
        if (!owner->info.store) owner->info.store = (void *)((++g_fp4_store_id << 4) | 1);
        if (owner->resource_id) fp4_index_insert(&g_fp4_ids, owner);
        else if (index_address && owner->info.address && owner->info.end > owner->info.address)
            fp4_index_insert(&g_fp4_addresses, owner);
        objc_setAssociatedObject(resource, &g_fp4_resource_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        existing = owner;
    }
    fp4_resource info = existing->info;
    pthread_mutex_unlock(&g_fp4_resource_lock);
    [owner release];
    return info;
}

static fp4_resource fp4_buffer(id<MTLBuffer> b)
{
    FP4ResourceOwner *existing = objc_getAssociatedObject(b, &g_fp4_resource_key);
    if (existing) return existing->info;
    if (!b) return (fp4_resource){0};
    FP4ResourceOwner *owner = [[FP4ResourceOwner alloc] init];
    owner->device = (uintptr_t)b.device;
    if ([b respondsToSelector:@selector(gpuAddress)]) owner->info.address = b.gpuAddress;
    uint64_t size = b.length, address = owner->info.address;
    if (size <= UINT64_MAX - address) owner->info.end = address + size;
    return fp4_install_resource(b, owner, 1);
}

static fp4_resource fp4_texture(id<MTLTexture> t)
{
    FP4ResourceOwner *existing = objc_getAssociatedObject(t, &g_fp4_resource_key);
    if (existing) return existing->info;
    if (!t) return (fp4_resource){0};
    FP4ResourceOwner *owner = [[FP4ResourceOwner alloc] init];
    owner->device = (uintptr_t)t.device;
    fp4_resource *info = &owner->info;
    info->texture = 1;
    info->levels = MAX(1, t.mipmapLevelCount);
    info->slices = tex_slices(t);
    info->shader_write = (t.usage & MTLTextureUsageShaderWrite) != 0;
    if ([t respondsToSelector:@selector(gpuResourceID)]) owner->resource_id = t.gpuResourceID._impl;
    info->address = owner->resource_id;

    id<MTLTexture> parent = t;
    while (parent.parentTexture) {
        info->level += parent.parentRelativeLevel;
        info->slice += parent.parentRelativeSlice;
        parent = parent.parentTexture;
    }
    id<MTLBuffer> backing = parent.buffer;
    if (backing) {
        fp4_resource root = fp4_buffer(backing);
        info->store = root.store;
        info->address = root.address;
        info->end = root.end;
        info->offset = parent.bufferOffset;
        info->texture = 0;
    } else if (parent != t) info->store = fp4_texture(parent).store;
    return fp4_install_resource(t, owner, 0);
}

static fp4_resource fp4_lookup(uintptr_t device, uint64_t key, int by_id)
{
    if (!key) return (fp4_resource){0};
    pthread_mutex_lock(&g_fp4_resource_lock);
    const fp4_resource_index *index = by_id ? &g_fp4_ids : &g_fp4_addresses;
    int i = fp4_index_lower(index, device, key);
    if (!by_id && i && (i == index->n || index->items[i]->device != device || index->items[i]->info.address != key)) i--;
    FP4ResourceOwner *owner = i < index->n ? index->items[i] : nil;
    fp4_resource info = {0};
    if (owner && owner->device == device &&
        (by_id ? owner->resource_id == key : owner->info.address <= key && key < owner->info.end))
        info = owner->info;
    pthread_mutex_unlock(&g_fp4_resource_lock);
    if (!info.store)
        atomic_fetch_add_explicit(by_id ? &g_fp4_id_misses : &g_fp4_address_misses, 1, memory_order_relaxed);
    return info;
}

static void fp4_note_work(fp_node *n, int stage)
{
    if (n->mtl4_last[stage] == n->mtl4_epoch) return;
    unsigned mask = FP4_STAGE(stage);
    n->mtl4_work |= mask;
    if (!n->mtl4_first[stage]) n->mtl4_first[stage] = n->mtl4_epoch;
    n->mtl4_last[stage] = n->mtl4_epoch;
    if (n->mtl4_nwork_epochs && n->mtl4_work_epochs[n->mtl4_nwork_epochs - 1].epoch == n->mtl4_epoch)
        n->mtl4_work_epochs[n->mtl4_nwork_epochs - 1].mask |= mask;
    else {
        fp4_work_epoch work = { n->mtl4_epoch, mask };
        PUSH(n->mtl4_work_epochs, n->mtl4_nwork_epochs, n->mtl4_cwork_epochs, work);
    }
}

static void fp4_note_attachment_loads(fp_node *n)
{
    for (int i = 0; i < n->nacc; i++) {
        fp_acc *a = &n->acc[i];
        if (a->via != VIA_ATT_LOAD) continue;
        a->first_epoch = a->last_epoch = n->mtl4_epoch;
        fp4_note_work(n, FP4_FRAGMENT);
    }
}

static int fp4_record_access(fp_node *n, fp4_resource r, uint64_t off, uint64_t end, uint32_t slice, uint32_t slices,
                             int rw, int stage, int via)
{
    if (!r.store) return -1;
    fp_acc a = {
        .store = r.store, .a0 = off, .a1 = end, .s0 = slice, .s1 = slice + slices,
        .is_tex = r.texture, .rw = rw, .via = via, .mtl4_stage = FP4_STAGE(stage),
        .first_epoch = n->mtl4_epoch, .last_epoch = n->mtl4_epoch, .gpu_address = r.address
    };
    if (r.texture) {
        a.a0 += r.level; a.a1 += r.level; a.s0 += r.slice; a.s1 += r.slice;
    } else {
        if (r.levels) { a.a0 = r.offset; a.a1 = UINT64_MAX; } /* buffer-backed texture */
        a.a1 = MIN(a.a1, r.end - r.address);
    }
    if (a.a0 >= a.a1) return -1;
    if (via != VIA_ATT_LOAD && via != VIA_ATT && via != VIA_RESOLVE) fp4_note_work(n, stage);
    return push_acc(n, &a);
}

typedef struct {
    uint64_t value, generation, sampled_uid;
    uint32_t sampled_epoch, sampled_pipeline;
    uint16_t sampled_stages;
    int by_id;
    fp4_resource resource;
} fp4_slot_cache;

typedef struct {
    atomic_uint_fast64_t value;
    atomic_int by_id;
    fp4_slot_cache cache[2];
} fp4_arg_slot;

@interface FP4ArgumentState : NSObject {
@public
    os_unfair_lock lock;
    atomic_uint_fast64_t version;
    uintptr_t device;
    NSUInteger buffers, textures;
    fp4_arg_slot *slots;
}
@end

@implementation FP4ArgumentState
- (instancetype)init
{
    self = [super init];
    if (self) {
        lock = OS_UNFAIR_LOCK_INIT;
        atomic_init(&version, 1);
    }
    return self;
}
- (void)dealloc
{
    free(slots);
    [super dealloc];
}
@end

@interface FP4ArgumentOwner : NSObject {
@public
    FP4ArgumentState *state;
}
@end

static atomic_uint_fast64_t g_fp4_table_generation;
static char g_fp4_argument_key;

@implementation FP4ArgumentOwner
- (void)dealloc
{
    atomic_fetch_add_explicit(&g_fp4_table_generation, 1, memory_order_release);
    [state release];
    [super dealloc];
}
@end

static FP4ArgumentState *fp4_argument_state(id table)
{
    if (!table) return nil;
    /* The generation invalidates cached identities when an Objective-C
     * table dies, even if an encoder still owns its recording state. */
    uint64_t generation = atomic_load_explicit(&g_fp4_table_generation, memory_order_acquire);
    if (tl_cache.table != (void *)table || tl_cache.table_generation != generation) {
        FP4ArgumentOwner *owner = objc_getAssociatedObject(table, &g_fp4_argument_key);
        tl_cache.table_generation = generation;
        tl_cache.table = (void *)table;
        tl_cache.table_state = owner ? owner->state : nil;
    }
    return tl_cache.table_state;
}

static void fp4_set_slot(id table, NSUInteger index, int texture, uint64_t value, int by_id)
{
    FP4ArgumentState *state = fp4_argument_state(table);
    if (!state || index >= (texture ? state->textures : state->buffers)) return;
    fp4_arg_slot *slot = &state->slots[index + (texture ? state->buffers : 0)];
    if (atomic_load_explicit(&slot->value, memory_order_relaxed) == value &&
        atomic_load_explicit(&slot->by_id, memory_order_relaxed) == by_id) return;
    /* The application's ordering of mutations and draws also orders the
     * native table snapshot. Atomics keep observation safe across threads. */
    atomic_store_explicit(&slot->by_id, by_id, memory_order_relaxed);
    atomic_store_explicit(&slot->value, value, memory_order_relaxed);
    atomic_fetch_add_explicit(&state->version, 1, memory_order_release);
}

static fp4_slot_cache *fp4_resolve_slot(FP4ArgumentState *state, fp4_arg_slot *slot,
                                        uint64_t value, int by_id, uint64_t generation)
{
    for (int i = 0; i < 2; i++)
        if (slot->cache[i].value == value && slot->cache[i].by_id == by_id &&
            slot->cache[i].generation == generation)
            return &slot->cache[i];
    slot->cache[1] = slot->cache[0];
    slot->cache[0] = (fp4_slot_cache){
        .value = value, .generation = generation, .by_id = by_id,
        .resource = fp4_lookup(state->device, value, by_id)
    };
    return &slot->cache[0];
}

/* Per-slot shader access of one pipeline, from binding reflection. A stage the
 * reflection describes lists every slot it declares and uses; any other slot has no
 * access in that stage. Stages it does not describe fall back to the binding policy. */
@interface FP4PipelineTable : NSObject {
@public
    uint16_t stages;
    uint16_t buffers[FP4_TABLES], textures[FP4_TABLES];
    uint8_t *access[FP4_TABLES];    /* FP_RD | FP_WR per slot: buffers, then textures */
}
@end

@implementation FP4PipelineTable
- (void)dealloc
{
    for (int i = 0; i < FP4_TABLES; i++) free(access[i]);
    [super dealloc];
}
@end

static char g_fp4_pipeline_key;
static FP4PipelineTable *g_fp4_unknown_table;   /* marks a pipeline whose access is unknown */
#define FP4_MAX_SLOT 4096

API_AVAILABLE(macos(13.0))
static int fp4_binding_space(id<MTLBinding> binding)   /* 0 buffer index, 1 texture index, -1 neither */
{
    switch (binding.type) {
    case MTLBindingTypeBuffer: case MTLBindingTypeVisibleFunctionTable: case MTLBindingTypeIntersectionFunctionTable:
    case MTLBindingTypePrimitiveAccelerationStructure: case MTLBindingTypeInstanceAccelerationStructure:
        return 0;
    case MTLBindingTypeTexture: return 1;
    default: return -1;
    }
}

API_AVAILABLE(macos(13.0))
static void fp4_table_stage(FP4PipelineTable *table, int stage, NSArray<id<MTLBinding>> *bindings)
{
    NSUInteger counts[2] = { 0, 0 };
    for (id<MTLBinding> binding in bindings) {
        int space = fp4_binding_space(binding);
        if (space >= 0 && binding.used && binding.index < FP4_MAX_SLOT)
            counts[space] = MAX(counts[space], binding.index + 1);
    }
    uint8_t *access = counts[0] + counts[1] ? calloc(counts[0] + counts[1], 1) : NULL;
    if (counts[0] + counts[1] && !access) return;   /* undescribed: the stage keeps the fallback */
    for (id<MTLBinding> binding in bindings) {
        int space = fp4_binding_space(binding);
        if (space < 0 || !binding.used || binding.index >= FP4_MAX_SLOT) continue;
        MTLBindingAccess mode = binding.access;
        access[binding.index + (space ? counts[0] : 0)] |= mode == MTLBindingAccessReadOnly ? FP_RD
            : mode == MTLBindingAccessWriteOnly ? FP_WR : (FP_RD | FP_WR);
    }
    table->stages |= FP4_STAGE(stage);
    table->buffers[stage] = (uint16_t)counts[0];
    table->textures[stage] = (uint16_t)counts[1];
    table->access[stage] = access;
}

/* nil when reflection carries no binding information. */
static FP4PipelineTable *fp4_table_new(id reflection)
{
    if (!reflection) return nil;
    FP4PipelineTable *table = [[FP4PipelineTable alloc] init];
    if (@available(macOS 13.0, *)) {
        if ([reflection isKindOfClass:[MTLComputePipelineReflection class]]) {
            fp4_table_stage(table, FP4_DISPATCH, [(MTLComputePipelineReflection *)reflection bindings]);
        } else if ([reflection isKindOfClass:[MTLRenderPipelineReflection class]]) {
            MTLRenderPipelineReflection *render = reflection;
            fp4_table_stage(table, FP4_VERTEX, render.vertexBindings);
            fp4_table_stage(table, FP4_FRAGMENT, render.fragmentBindings);
            fp4_table_stage(table, FP4_TILE, render.tileBindings);
            fp4_table_stage(table, FP4_OBJECT, render.objectBindings);
            fp4_table_stage(table, FP4_MESH, render.meshBindings);
        }
    }
    if (!table->stages) { [table release]; return nil; }
    return table;
}

static void fp4_attach_table(id pipeline, FP4PipelineTable *table)
{
    if (pipeline && table)
        objc_setAssociatedObject(pipeline, &g_fp4_pipeline_key, table, OBJC_ASSOCIATION_RETAIN);
}

static void fp4_attach_reflection(id pipeline, id reflection)
{
    if (!pipeline || !reflection) return;
    FP4PipelineTable *table = fp4_table_new(reflection);
    fp4_attach_table(pipeline, table);
    [table release];
}

/* A pipeline fencepost did not see built (binary archives, data sets, pipelines made
 * before the first Metal 4 object) is looked up once: its own reflection if Metal kept
 * one, otherwise it is marked access unknown. */
static FP4PipelineTable *fp4_pipeline_table(id pipeline)
{
    if (!pipeline) return nil;
    FP4PipelineTable *table = objc_getAssociatedObject(pipeline, &g_fp4_pipeline_key);
    if (!table) {
        id reflection = nil;
        if (@available(macOS 26.0, *))
            if ([pipeline respondsToSelector:@selector(reflection)]) reflection = [pipeline reflection];
        table = [fp4_table_new(reflection) autorelease];
        if (!table) {
            static dispatch_once_t once;
            dispatch_once(&once, ^{ g_fp4_unknown_table = [[FP4PipelineTable alloc] init]; });
            table = g_fp4_unknown_table;
            atomic_fetch_add_explicit(&g_fp4_unknown_pipelines, 1, memory_order_relaxed);
        }
        fp4_attach_table(pipeline, table);
    }
    return table->stages ? table : nil;
}

static void fp4_bind_pipeline(id encoder, id pipeline)
{
    fp_node *n = cur_node(encoder);
    if (!n || !n->mtl4 || n->mtl4_pipeline_object == (void *)pipeline) return;
    /* Reflection and the associated table may autorelease on the app's encoder thread. */
    @autoreleasepool {
        FP4PipelineTable *table = fp4_pipeline_table(pipeline);
        n->mtl4_pipeline_object = (void *)pipeline;
        if (table == n->mtl4_pipeline) return;
        [n->mtl4_pipeline release];
        n->mtl4_pipeline = [table retain];
        n->mtl4_pipeline_serial++;
        memset(n->mtl4_versions, 0, sizeof n->mtl4_versions);   /* resample every bound table */
    }
}

/* The access a draw or dispatch makes through slot i of a table bound at stage. */
static int fp4_slot_access(const fp_node *n, const FP4ArgumentState *state, NSUInteger i, int stage,
                           const fp4_resource *r)
{
    FP4PipelineTable *table = n->mtl4_pipeline;
    int texture = i >= state->buffers;
    if (!table || !(table->stages & FP4_STAGE(stage))) {
        int rw = FP_RD;
        if (g_bind_maywrite && (!texture || r->shader_write)) rw |= FP_MAYBE;
        return rw;
    }
    NSUInteger index = texture ? i - state->buffers : i;
    NSUInteger count = texture ? table->textures[stage] : table->buffers[stage];
    if (index >= count) return 0;
    int access = table->access[stage][index + (texture ? table->buffers[stage] : 0)];
    /* A non-const device pointer reflects as read-write even when the shader only reads it. */
    if (!texture && access == (FP_RD | FP_WR) && g_mtl4_rw_possible) return FP_RD | FP_MAYBE;
    return access;
}

static void fp4_sample_table(fp_node *n, FP4ArgumentState *state, unsigned stages)
{
    uint64_t version = atomic_load_explicit(&state->version, memory_order_acquire);
    unsigned changed = 0;
    for (unsigned bits = stages; bits; bits &= bits - 1) {
        int stage = __builtin_ctz(bits);
        if (n->mtl4_versions[stage] != version) changed |= FP4_STAGE(stage);
    }
    if (!changed) return;
    os_unfair_lock_lock(&state->lock);
    uint64_t generation = atomic_load_explicit(&g_fp4_resource_generation, memory_order_relaxed);
    for (unsigned bits = changed; bits; bits &= bits - 1) n->mtl4_versions[__builtin_ctz(bits)] = version;
    for (NSUInteger i = 0; i < state->buffers + state->textures; i++) {
        fp4_arg_slot *slot = &state->slots[i];
        uint64_t value = atomic_load_explicit(&slot->value, memory_order_relaxed);
        if (!value) continue;
        int by_id = atomic_load_explicit(&slot->by_id, memory_order_relaxed);
        fp4_slot_cache *cache = fp4_resolve_slot(state, slot, value, by_id, generation);
        if (!cache->resource.store) continue;
        if (cache->sampled_uid != n->uid || cache->sampled_epoch != n->mtl4_epoch ||
            cache->sampled_pipeline != n->mtl4_pipeline_serial) {
            cache->sampled_uid = n->uid;
            cache->sampled_epoch = n->mtl4_epoch;
            cache->sampled_pipeline = n->mtl4_pipeline_serial;
            cache->sampled_stages = 0;
        }
        unsigned pending = changed & ~cache->sampled_stages;
        if (!pending) continue;
        fp4_resource r = cache->resource;
        uint64_t off = by_id ? 0 : value - r.address;
        for (unsigned bits = pending; bits; bits &= bits - 1) {
            int stage = __builtin_ctz(bits);
            int rw = fp4_slot_access(n, state, i, stage, &r);
            if (rw) fp4_record_access(n, r, off, r.texture ? r.levels : UINT64_MAX, 0, r.slices, rw, stage, VIA_BIND);
        }
        cache->sampled_stages |= pending;
    }
    os_unfair_lock_unlock(&state->lock);
}

static void fp4_capture_tables(id encoder, unsigned stages)
{
    fp_node *n = cur_node(encoder);
    if (!n || !n->mtl4) return;
    if (!n->mtl4_drawn && n->kind == FPK_RENDER) fp4_note_attachment_loads(n);
    n->mtl4_drawn = 1;
    if (n->kind == FPK_RENDER) n->mtl4_draw_epoch = n->mtl4_epoch;   /* the last draw, not sync work */
    if (stages == (FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT))) {
        FP4ArgumentState *shared = n->mtl4_tables[FP4_VERTEX];
        if (shared == n->mtl4_tables[FP4_FRAGMENT]) {
            if (n->mtl4_last[FP4_VERTEX] != n->mtl4_epoch) fp4_note_work(n, FP4_VERTEX);
            if (n->mtl4_last[FP4_FRAGMENT] != n->mtl4_epoch) fp4_note_work(n, FP4_FRAGMENT);
            if (shared) fp4_sample_table(n, shared, stages);
            return;
        }
    }
    while (stages) {
        int stage = __builtin_ctz(stages);
        unsigned together = FP4_STAGE(stage);
        stages &= stages - 1;
        fp4_note_work(n, stage);
        FP4ArgumentState *state = n->mtl4_tables[stage];
        if (!state) continue;
        for (unsigned bits = stages; bits; bits &= bits - 1) {
            int other = __builtin_ctz(bits);
            if (n->mtl4_tables[other] == state) {
                together |= FP4_STAGE(other);
                fp4_note_work(n, other);
            }
        }
        stages &= ~together;
        fp4_sample_table(n, state, together);
    }
}

static void fp4_bind_table(id encoder, id table, unsigned stages)
{
    fp_node *n = cur_node(encoder);
    if (!n || !n->mtl4) return;
    FP4ArgumentState *state = fp4_argument_state(table);
    while (stages) {
        int stage = __builtin_ctz(stages);
        stages &= stages - 1;
        if (n->mtl4_tables[stage] == state) continue;
        [n->mtl4_tables[stage] release];
        n->mtl4_tables[stage] = [state retain];
        n->mtl4_versions[stage] = 0;
    }
}

/* Take the snapshot only for a new access. Later reports may outlive the Metal resource. */
static void record_store(fp_node *n, fp_acc *a, id<MTLResource> resource, NSUInteger buffer_length)
{
    int count = n->nacc;
    a->description = n->ndescriptions;
    if (push_acc(n, a) != count) return;
    fp_store_description d = { .address = resource, .size = buffer_length };
    @autoreleasepool {
        const char *label = resource.label.UTF8String;
        if (label) snprintf(d.label, sizeof d.label, "%s", label);
        if (a->is_tex) {
            id<MTLTexture> t = (id<MTLTexture>)resource;
            d.size = t.width; d.height = t.height; d.format = t.pixelFormat; d.levels = t.mipmapLevelCount;
        }
    }
    PUSH(n->descriptions, n->ndescriptions, n->cdescriptions, d);
}

static int acc_buf(fp_node *n, id<MTLBuffer> b, uint64_t off, uint64_t end, int rw, int sf, int sl, int via)
{
    if (!b || !n) return -1;
    if (n->mtl4)
        return fp4_record_access(n, fp4_buffer(b), off, end, 0, 0, rw, FP4_BLIT, via);
    if (b.hazardTrackingMode != MTLHazardTrackingModeUntracked) { __atomic_fetch_add(&g_st[ST_TRACKED], 1, __ATOMIC_RELAXED); return -1; }
    uint64_t len = b.length;
    if (end > len) end = len;
    if (off >= end) return -1;
    void *store = (void *)b;
    if (@available(macOS 26.0, *)) store = fp4_buffer(b).store;
    fp_acc a = { .store = store, .a0 = off, .a1 = end, .rw = rw, .sf = sf, .sl = sl, .via = via };
    record_store(n, &a, b, len);
    return n->nacc - 1;
}

/* level/slice ranges are in t's own terms; views are mapped onto their storage */
static int acc_tex(fp_node *n, id<MTLTexture> t, uint64_t level, uint64_t nlev, uint32_t slice, uint32_t nsl,
                   int rw, int sf, int sl, int via)
{
    if (!t || !n) return -1;
    if (n->mtl4) {
        int stage = n->kind == FPK_RENDER ? FP4_FRAGMENT : FP4_BLIT;
        return fp4_record_access(n, fp4_texture(t), level, level + MAX(1, nlev), slice, MAX(1, nsl), rw, stage, via);
    }
    while (t.parentTexture) {
        level += t.parentRelativeLevel;
        slice += (uint32_t)t.parentRelativeSlice;
        t = t.parentTexture;
    }
    if (t.buffer) return acc_buf(n, t.buffer, t.bufferOffset, UINT64_MAX, rw, sf, sl, via);
    if (t.hazardTrackingMode != MTLHazardTrackingModeUntracked) { __atomic_fetch_add(&g_st[ST_TRACKED], 1, __ATOMIC_RELAXED); return -1; }
    void *store = (void *)t;
    if (@available(macOS 26.0, *)) store = fp4_texture(t).store;
    fp_acc a = { .store = store, .a0 = level, .a1 = level + MAX(1, nlev), .s0 = slice, .s1 = slice + MAX(1, nsl),
                 .is_tex = 1, .rw = rw, .sf = sf, .sl = sl, .via = via };
    record_store(n, &a, t, 0);
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

static void record_attachment(fp_node *n, id<MTLTexture> t, MTLRenderPassAttachmentDescriptor *d,
                              NSUInteger layers, MTLStoreAction sa)
{
    MTLLoadAction la = d.loadAction;
    /* Tile contents survive between stitched segments. Only the endpoints
     * operate on attachment memory. */
    if (n->mtl4_render_options & FP4_RESUMING) la = MTLLoadActionDontCare;
    if (n->mtl4_render_options & FP4_SUSPENDING) sa = MTLStoreActionDontCare;
    int rw = la == MTLLoadActionLoad ? FP_RD : 0;
    if (sa == MTLStoreActionStore || sa == MTLStoreActionStoreAndMultisampleResolve ||
        sa == MTLStoreActionCustomSampleDepthStore) rw |= FP_WR;
    if (rw) {
        uint32_t slice = t.textureType == MTLTextureType3D ? 0 : (uint32_t)d.slice;
        acc_tex(n, t, d.level, 1, slice, (uint32_t)MAX(1, layers), rw, 1, 1,
                la == MTLLoadActionLoad ? VIA_ATT_LOAD : VIA_ATT);
    }
    if (sa == MTLStoreActionMultisampleResolve || sa == MTLStoreActionStoreAndMultisampleResolve)
        acc_tex(n, d.resolveTexture, d.resolveLevel, 1, (uint32_t)d.resolveSlice, 1, FP_WR, 1, 1, VIA_RESOLVE);
}

static void att(fp_node *n, int slot, MTLRenderPassAttachmentDescriptor *d, NSUInteger layers)
{
    id<MTLTexture> t = d.texture;
    if (!t) return;
    MTLStoreAction sa = d.storeAction;
    if (n->mtl4) n->mtl4_has_attachment = 1;
    if (!n->mtl4 && sa == MTLStoreActionUnknown) {
        /* Metal 3 finalizes Unknown with set*StoreAction: before endEncoding. */
        fp_deferred_att deferred = { [d copy], layers, sa, slot };
        PUSH(n->deferred_att, n->ndeferred_att, n->cdeferred_att, deferred);
    } else record_attachment(n, t, d, layers, sa);
}

static void store_action(id enc, int slot, MTLStoreAction sa)   /* a deferred store action arrives */
{
    fp_node *n = cur_node(enc);
    if (!n) return;
    for (int i = 0; i < n->ndeferred_att; i++)
        if (n->deferred_att[i].slot == slot) n->deferred_att[i].action = sa;
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
    H_4_NEW_HEAP, H_4_NEW_BUFFER, H_4_NEW_BYTES, H_4_NEW_BYTES_NOCOPY, H_4_NEW_SPARSE_BUFFER,
    H_4_NEW_TEXTURE, H_4_NEW_IOSURFACE, H_4_SHARED_TEXTURE, H_4_SHARED_HANDLE,
    H_4_HEAP_BUFFER_OFFSET, H_4_HEAP_TEXTURE_OFFSET, H_4_BUFFER_TEXTURE,
    H_4_VIEW_FORMAT, H_4_VIEW_RANGE, H_4_VIEW_SWIZZLE, H_4_VIEW_DESC,
    H_4_ARGUMENT_TABLE, H_4_ADDRESS, H_4_STRIDE, H_4_RESOURCE,
    H_4_TEXTURE, H_4_BIND_COMPUTE, H_4_BIND_RENDER, H_4_ACCEL_BUILD,
    H_4_DRAW, H_4_DRAW_I, H_4_DRAW_B, H_4_DRAW_IDX,
    H_4_DRAW_IDX_I, H_4_DRAW_IDX_B, H_4_DRAW_IND, H_4_DRAW_IDX_IND,
    H_4_DRAW_MESH_GROUPS, H_4_DRAW_MESH, H_4_DRAW_MESH_IND, H_4_DRAW_TILE,
    H_4_DISPATCH, H_4_DISPATCH_GROUPS, H_4_DISPATCH_IND, H_4_DISPATCH_GROUPS_IND,
    H_4_RES_ADD, H_4_RES_ADDS, H_4_RES_REMOVE, H_4_RES_REMOVES, H_4_RES_USE, H_4_RES_USES,
    H_4_CONSUMER_BARRIER, H_4_PRODUCER_BARRIER, H_4_INTRA_BARRIER, H_4_FENCE_UPDATE, H_4_FENCE_WAIT,
    H_4_SET_COMPUTE_PIPELINE, H_4_SET_RENDER_PIPELINE, H_4_NEW_COMPILER,
    H_P_CFN, H_P_CFN_R, H_P_CFN_A, H_P_CFN_AR, H_P_CDESC_R, H_P_CDESC_AR,
    H_P_RDESC, H_P_RDESC_R, H_P_RDESC_A, H_P_RDESC_AR, H_P_TILE_R, H_P_TILE_AR, H_P_MESH_R, H_P_MESH_AR,
    H_C4_COMPUTE, H_C4_COMPUTE_LINK, H_C4_RENDER, H_C4_RENDER_LINK, H_C4_SPECIALIZE,
    H_C4_COMPUTE_A, H_C4_COMPUTE_LINK_A, H_C4_RENDER_A, H_C4_RENDER_LINK_A, H_C4_SPECIALIZE_A,
    H_C4_REFLECTION,
    H_COUNT
};

static struct { Class c; IMP imp; } g_orig[H_COUNT][FP_HOOKCLS];
static atomic_int g_norig[H_COUNT];

static IMP orig_imp(int h, id self)
{
    int count = atomic_load_explicit(&g_norig[h], memory_order_acquire);
    for (Class c = object_getClass(self); c; c = class_getSuperclass(c))
        for (int i = 0; i < count; i++)
            if (g_orig[h][i].c == c) return g_orig[h][i].imp;
    char message[FP_LOG_LINE_BYTES];
    snprintf(message, sizeof message, "fencepost: internal error: no original for hook %d on %s\n",
             h, class_getName(object_getClass(self)));
    fp_log_flush(message);
    abort();
}

static void hook(Class cls, int h, const char *selname, IMP repl)   /* g_lock held */
{
    SEL sel = sel_registerName(selname);
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) return;
    IMP cur = method_getImplementation(m);
    if (cur == repl) return;   /* inherited from a class already hooked */
    int count = atomic_load_explicit(&g_norig[h], memory_order_relaxed);
    if (count == FP_HOOKCLS) { fp_log("fencepost: too many classes for %s\n", selname); return; }
    g_orig[h][count].c = cls;
    g_orig[h][count].imp = cur;
    atomic_store_explicit(&g_norig[h], count + 1, memory_order_release);
    if (!class_addMethod(cls, sel, repl, method_getTypeEncoding(m))) method_setImplementation(m, repl);
}

/* Called from the replacement for h itself, so the receiver has a class
 * registered for this hook. Most Metal draw and binding hooks have only one. */
static inline IMP fp4_orig_imp(int h, id self)
{
    if (atomic_load_explicit(&g_norig[h], memory_order_acquire) == 1) return g_orig[h][0].imp;
    return orig_imp(h, self);
}

#define ORIG(h, T) ((T)orig_imp((h), self))
#define FP4_ORIG(h, T) ((T)fp4_orig_imp((h), self))
#define ENTER int outer_ = !tl_depth; tl_depth++
#define LEAVE tl_depth--
#define OUTER outer_

static void hook_device(Class cls);
static void hook_queue(Class cls);
static void hook_cb(Class cls);
static void hook_encoder(Class cls, int kind);
static void hook_event(Class cls);
static void h_mtl4_end(id self, SEL _cmd);
static void hook_mtl4_cb(Class cls);
static void hook_mtl4_encoder(Class cls, int kind);
static void hook_mtl4_argument(Class cls);
static void hook_mtl4_queue(Class cls);
static void hook_mtl4_heap(Class cls);
static void hook_mtl4_buffer(Class cls);
static void hook_mtl4_texture(Class cls);

/* device */

static void on_mtl4_buffer(id buffer)
{
    if (!buffer) return;
    fp4_buffer(buffer);
    hook_mtl4_buffer(object_getClass(buffer));
}

static void on_mtl4_texture(id texture)
{
    if (!texture) return;
    fp4_texture(texture);
    hook_mtl4_texture(object_getClass(texture));
}

#define FP4_FACTORY1(fn, hook_id, type, observe) \
    static id fn(id self, SEL _cmd, type arg) { \
        ENTER; id result = ORIG(hook_id, id (*)(id, SEL, type))(self, _cmd, arg); LEAVE; \
        if (OUTER) observe(result); \
        return result; \
    }

static void on_mtl4_heap(id heap)
{
    if (heap) hook_mtl4_heap(object_getClass(heap));
}
FP4_FACTORY1(h_mtl4_new_heap, H_4_NEW_HEAP, id, on_mtl4_heap)
FP4_FACTORY1(h_mtl4_new_texture, H_4_NEW_TEXTURE, id, on_mtl4_texture)
FP4_FACTORY1(h_mtl4_shared_texture, H_4_SHARED_TEXTURE, id, on_mtl4_texture)
FP4_FACTORY1(h_mtl4_shared_handle, H_4_SHARED_HANDLE, id, on_mtl4_texture)
FP4_FACTORY1(h_mtl4_view_format, H_4_VIEW_FORMAT, NSUInteger, on_mtl4_texture)
FP4_FACTORY1(h_mtl4_view_desc, H_4_VIEW_DESC, id, on_mtl4_texture)
#undef FP4_FACTORY1

static id h_mtl4_new_buffer(id self, SEL _cmd, NSUInteger len, NSUInteger options)
{
    ENTER; id b = ORIG(H_4_NEW_BUFFER, id (*)(id, SEL, NSUInteger, NSUInteger))(self, _cmd, len, options); LEAVE;
    if (OUTER) on_mtl4_buffer(b);
    return b;
}

static id h_mtl4_new_bytes(id self, SEL _cmd, const void *bytes, NSUInteger len, NSUInteger options)
{
    ENTER;
    id b = ORIG(H_4_NEW_BYTES, id (*)(id, SEL, const void *, NSUInteger, NSUInteger))(self, _cmd, bytes, len, options);
    LEAVE;
    if (OUTER) on_mtl4_buffer(b);
    return b;
}

static id h_mtl4_new_bytes_nocopy(id self, SEL _cmd, void *bytes, NSUInteger len, NSUInteger options, id dealloc)
{
    ENTER;
    id b = ORIG(H_4_NEW_BYTES_NOCOPY, id (*)(id, SEL, void *, NSUInteger, NSUInteger, id))(self, _cmd,
                                                                                        bytes, len, options, dealloc);
    LEAVE;
    if (OUTER) on_mtl4_buffer(b);
    return b;
}

static id h_mtl4_sparse_buffer(id self, SEL _cmd, NSUInteger len, NSUInteger options, NSInteger page)
{
    ENTER;
    id b = ORIG(H_4_NEW_SPARSE_BUFFER, id (*)(id, SEL, NSUInteger, NSUInteger, NSInteger))(self, _cmd, len, options, page);
    LEAVE;
    if (OUTER) on_mtl4_buffer(b);
    return b;
}

static id h_mtl4_heap_buffer_offset(id self, SEL _cmd, NSUInteger len, NSUInteger options, NSUInteger offset)
{
    ENTER;
    id b = ORIG(H_4_HEAP_BUFFER_OFFSET, id (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, len, options, offset);
    LEAVE;
    if (OUTER) on_mtl4_buffer(b);
    return b;
}

static id h_mtl4_heap_texture_offset(id self, SEL _cmd, id desc, NSUInteger offset)
{
    ENTER; id t = ORIG(H_4_HEAP_TEXTURE_OFFSET, id (*)(id, SEL, id, NSUInteger))(self, _cmd, desc, offset); LEAVE;
    if (OUTER) on_mtl4_texture(t);
    return t;
}

static id h_mtl4_buffer_texture(id self, SEL _cmd, id desc, NSUInteger offset, NSUInteger row)
{
    ENTER; id t = ORIG(H_4_BUFFER_TEXTURE, id (*)(id, SEL, id, NSUInteger, NSUInteger))(self, _cmd, desc, offset, row); LEAVE;
    if (OUTER) on_mtl4_texture(t);
    return t;
}

static id h_mtl4_iosurface(id self, SEL _cmd, id desc, IOSurfaceRef surface, NSUInteger plane)
{
    ENTER; id t = ORIG(H_4_NEW_IOSURFACE, id (*)(id, SEL, id, IOSurfaceRef, NSUInteger))(self, _cmd, desc, surface, plane); LEAVE;
    if (OUTER) on_mtl4_texture(t);
    return t;
}

static id h_mtl4_view_range(id self, SEL _cmd, NSUInteger format, NSUInteger type, NSRange levels, NSRange slices)
{
    ENTER;
    id t = ORIG(H_4_VIEW_RANGE, id (*)(id, SEL, NSUInteger, NSUInteger, NSRange, NSRange))(self, _cmd,
                                                                                       format, type, levels, slices);
    LEAVE;
    if (OUTER) on_mtl4_texture(t);
    return t;
}

static id h_mtl4_view_swizzle(id self, SEL _cmd, NSUInteger format, NSUInteger type, NSRange levels, NSRange slices,
                             MTLTextureSwizzleChannels channels)
{
    ENTER;
    id t = ORIG(H_4_VIEW_SWIZZLE, id (*)(id, SEL, NSUInteger, NSUInteger, NSRange, NSRange, MTLTextureSwizzleChannels))
                (self, _cmd, format, type, levels, slices, channels);
    LEAVE;
    if (OUTER) on_mtl4_texture(t);
    return t;
}

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
    event_key(ev, 1);
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

static fp_cb *cb_record_new(id cb, fp_queue *q, int mtl4, fp_trace_device *device)   /* g_lock held */
{
    uint64_t uid = ++g_cbuid;
    fp_cb *c = &g_cbs[uid % FP_CBPOOL];
    c->uid = uid; c->cb = (void *)cb; c->q = q; c->state = 1; c->nenc = 0; c->nit = 0; c->mtl4 = mtl4; c->mtl4_resident = 0;
    c->enq = 0; c->commit_tick = 0; c->label[0] = 0;
    c->trace = g_trace_enabled ? trace_cb_new(cb, uid, q ? q->idx : -1, mtl4, device) : NULL;
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
    fp_trace_device *device = trace_cb_device(cb, 0);
    pthread_mutex_lock(&g_lock);
    fp_queue *fq = queue_get((void *)q);
    if (fq) cb_record_new(cb, fq, 0, device);
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
static void capture_stop(const char *limit);

/* 0 consumes the request, 1 defers it to another unopened frame. */
static int capture_start(id owner, uint64_t frame, const char *reason)
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
        id<MTLDevice> device = [owner device];
        MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
        if (g_capture_count >= g_capture_max) refusal = "capture limit reached";
        else if (g_capture_stopped && (now - g_capture_stopped) / 1e9 < g_capture_cooldown)
            refusal = "cooldown in effect";
        else if (!device) refusal = "command buffer or queue has no Metal device";
        else if (!path.fileSystemRepresentation) refusal = "invalid output path";
        else if (manager.isCapturing) refusal = "Metal capture already running";
        else if (statfs(path.stringByDeletingLastPathComponent.fileSystemRepresentation, &volume))
            refusal = "cannot check free space on output volume";
        else if ((double)volume.f_bavail * volume.f_bsize < g_capture_free * 1073741824.0)
            refusal = "insufficient free space on output volume";
        if (refusal) {
            fp_log("fencepost: capture refused: %s\n", refusal);
            return 0;
        }

        MTLCaptureDescriptor *desc = [[[MTLCaptureDescriptor alloc] init] autorelease];
        desc.captureObject = device;
        desc.destination = MTLCaptureDestinationGPUTraceDocument;
        desc.outputURL = [NSURL fileURLWithPath:path];
        NSError *error = nil;

        /* Reserve the transition under the same lock as end and commit.
         * A commit removes ended identities at entry but holds the in-Metal
         * count until it returns. End cannot enter Metal during this start. */
        pthread_mutex_lock(&g_lock);
        int unsafe = g_capture_mtl4_unsafe;
        int blocked = g_capture_mtl4_ended || (g_capture_mtl4 && g_capture_commits);
        if (!unsafe && !blocked) g_capture_starting = 1;
        pthread_mutex_unlock(&g_lock);
        if (unsafe) {
            fp_log("fencepost: capture refused: Metal 4 command buffer tracking unavailable\n");
            return 0;
        }
        if (blocked) return 1;

        BOOL started = NO;
        BOOL threw = NO;
        fp_capture_start start = {0};
        int depth = tl_depth;
        @try {
            tl_capture_start = &start;
            started = [manager startCaptureWithDescriptor:desc error:&error];
        } @catch (NSException *exception) {
            threw = YES;
            if (!start.refusal)
                fp_log("fencepost: capture failed: %s\n", exception.reason.UTF8String ?: exception.name.UTF8String);
        } @finally {
            tl_capture_start = NULL;
            tl_depth = depth;
            pthread_mutex_lock(&g_lock);
            if (started && !start.refusal) {
                g_capture_started = g_capture_last_commit = trace_clock();
                g_capture_until = frame + g_capture_frames;
                g_capture_generation = ++g_capture_count;
                g_capture_feedback = 0;
                g_capture_feedback_unavailable = 0;
            }
            g_capture_starting = 0;
            pthread_cond_broadcast(&g_capture_started_cond);
            pthread_mutex_unlock(&g_lock);
        }
        if (start.refusal) {
            fp_log("fencepost: capture refused: %s\n", start.refusal);
            if (started || manager.isCapturing) {
                g_capture_started = trace_clock();
                g_capture_count++;
                g_capture_path = [path copy];
                capture_stop(" (start interrupted)");
            }
            for (id cb in start.ends) {
                @try {
                    h_mtl4_end(cb, @selector(endCommandBuffer));
                } @catch (NSException *exception) {
                    fp_log("fencepost: deferred Metal 4 end failed: %s\n",
                           exception.reason.UTF8String ?: exception.name.UTF8String);
                } @finally {
                    tl_depth = depth;
                }
            }
            [start.ends release];
            return 0;
        }
        if (!started) {
            if (!threw) fp_log("fencepost: capture failed: %s\n", error.localizedDescription.UTF8String ?: "unknown error");
            return 0;
        }

        g_capture_path = [path copy];
        g_capture_deadline = g_capture_started + (uint64_t)g_capture_max_secs * NSEC_PER_SEC;
        uint64_t idle_deadline = g_capture_started + (uint64_t)g_capture_idle_secs * NSEC_PER_SEC;
        capture_watchdog(MIN(g_capture_deadline, idle_deadline));
        fp_log("fencepost: capturing frame %llu (%s) -> %s\n", frame + 1, reason, path.fileSystemRepresentation);
        return 0;
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

static NSDictionary<NSString *, NSNumber *> *capture_snapshot(NSString *path)
{
    NSMutableDictionary<NSString *, NSNumber *> *snapshot = [[NSMutableDictionary alloc] init];
    NSDirectoryEnumerator *entries = [NSFileManager.defaultManager enumeratorAtPath:path];
    NSString *entry;
    while ((entry = [entries nextObject])) {
        NSDictionary *attrs = entries.fileAttributes;
        if ([attrs.fileType isEqualToString:NSFileTypeRegular]) snapshot[entry] = @(attrs.fileSize);
    }
    return snapshot;
}

static void capture_report(NSString *path, NSDictionary<NSString *, NSNumber *> *snapshot,
                           uint64_t duration, const char *limit)
{
    @autoreleasepool {
        uint64_t size = 0;
        NSMutableDictionary<NSString *, NSNumber *> *sizes = [NSMutableDictionary dictionary];
        NSMutableDictionary<NSString *, NSNumber *> *counts = [NSMutableDictionary dictionary];
        for (NSString *entry in snapshot) {
            uint64_t bytes = snapshot[entry].unsignedLongLongValue;
            size += bytes;
            NSString *name = capture_resource_name(entry);
            if (name) {
                sizes[name] = @(sizes[name].unsignedLongLongValue + bytes);
                counts[name] = @(counts[name].unsignedLongLongValue + 1);
            }
        }
        fp_log("fencepost: capture written %s, %.2f MB, %.1f s%s\n", path.fileSystemRepresentation,
               size / 1048576.0, duration / 1e9,
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
    }
}

/* Keep command-buffer allocation gated until Metal has stopped recording.
 * A buffer allocated during this transition can enter the preceding capture. */
static void capture_stop(const char *limit)
{
    if (!g_capture_path) return;
    pthread_mutex_lock(&g_lock);
    g_capture_generation = 0;
    pthread_mutex_unlock(&g_lock);
    g_capture_deadline = 0;
    NSString *path = g_capture_path;
    NSDictionary *snapshot;
    @autoreleasepool {
        MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
        if (manager.isCapturing) [manager stopCapture];
        g_capture_stopped = trace_clock();
        /* Metal may keep writing after stopCapture. Take the visible file
         * sizes here, while this capture still owns the gate and path. */
        snapshot = capture_snapshot(path);
    }
    uint64_t duration = g_capture_stopped - g_capture_started;
    g_capture_path = nil;
    char ignored[sizeof g_capture_reason];
    capture_poll_file(ignored);
    pthread_mutex_lock(&g_lock);
    g_capture_busy = 0;
    pthread_mutex_unlock(&g_lock);

    /* The queued block owns the snapshot. It never takes the allocation gate. */
    dispatch_async(g_capture_reporter, ^{ capture_report(path, snapshot, duration, limit); });
    [snapshot release];
    [path release];
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

/* g_lock held. The boundary must have reached Metal. Only wait for feedback
 * when every tracked submission in the recording could provide it. */
static int capture_ready(void)
{
    return g_capture_generation && !g_capture_commits && g_capture_frame >= g_capture_until &&
           (!g_capture_mtl4 || !g_capture_feedback || g_capture_feedback_unavailable);
}

/* Called under the capture lock, never from Metal's feedback callback. */
static void capture_stop_at_boundary(void)
{
    pthread_mutex_lock(&g_lock);
    int ready = capture_ready();
    if (ready) g_capture_generation = 0;
    pthread_mutex_unlock(&g_lock);
    if (g_capture_path && trace_clock() >= g_capture_deadline) capture_stop(" (time limit)");
    else if (ready) capture_stop(NULL);
}

/* g_lock held. Feedback must never wait for stopCapture or the entry gate. */
static void capture_schedule_stop(void)
{
    if (!g_capture_mtl4 || g_capture_stop_queued || !capture_ready()) return;
    g_capture_stop_queued = 1;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        pthread_mutex_lock(&g_capture_lock);
        pthread_mutex_lock(&g_lock);
        g_capture_stop_queued = 0;
        pthread_mutex_unlock(&g_lock);
        tl_depth++;
        capture_stop_at_boundary();
        tl_depth--;
        pthread_mutex_unlock(&g_capture_lock);
    });
}

static void capture_feedback_done(uint64_t generation, int failed)
{
    pthread_mutex_lock(&g_lock);
    if (failed && (g_capture_on & CAP_GPU_ERROR)) capture_request("gpu-error");
    if (generation && generation == g_capture_generation) {
        g_capture_feedback--;
        g_capture_last_commit = trace_clock();
        capture_schedule_stop();
    }
    pthread_mutex_unlock(&g_lock);
}

/* Register after attaching the handler and before handing it to Metal. */
static void capture_feedback_attached(uint64_t generation)
{
    if (!generation) return;
    pthread_mutex_lock(&g_lock);
    if (generation == g_capture_generation) g_capture_feedback++;
    pthread_mutex_unlock(&g_lock);
}

static void capture_feedback_missing(uint64_t generation)
{
    if (!generation) return;
    pthread_mutex_lock(&g_lock);
    if (generation == g_capture_generation && !g_capture_feedback_unavailable) {
        g_capture_feedback_unavailable = 1;
        fp_log("fencepost: capture feedback unavailable for a submission; stopping at the frame boundary\n");
    }
    pthread_mutex_unlock(&g_lock);
}

static void capture_before_cb(id owner, int mtl4)
{
    pthread_mutex_lock(&g_capture_lock);
    capture_stop_at_boundary();
    pthread_mutex_lock(&g_lock);
    uint64_t frame = mtl4 ? g_frame_cur : g_capture_frame;
    /* Either API can open the frame. capture_start checks the shared Metal 4
     * gate before starting the device capture from either entry point. */
    int new_frame = frame > g_capture_opened;
    /* Do not hold a new frame hostage to feedback that could depend on its
     * work. If recording continues into this frame, wait for its end too. */
    if (g_capture_generation && g_capture_mtl4)
        g_capture_until = MAX(g_capture_until, g_frame_cur + 1);
    pthread_mutex_unlock(&g_lock);
    if (new_frame) {
        g_capture_opened = frame;
        char reason[sizeof g_capture_reason];
        int changed = capture_poll_file(reason);
        pthread_mutex_lock(&g_lock);
        if (changed) capture_request(reason);
        memcpy(reason, g_capture_reason, sizeof reason);
        if (reason[0]) g_capture_busy = 1;
        pthread_mutex_unlock(&g_lock);
        if (reason[0]) {
            int deferred = capture_start(owner, frame, reason);
            if (deferred && ++g_capture_attempts >= CAP_MTL4_ATTEMPTS) {
                fp_log("fencepost: capture refused: Metal 4 command buffers still awaiting submission after %u frame starts\n",
                       g_capture_attempts);
                deferred = 0;
            }
            if (deferred)
                fp_log("fencepost: capture deferred: Metal 4 command buffers awaiting submission\n");
            else g_capture_attempts = 0;
            pthread_mutex_lock(&g_lock);
            if (!deferred) g_capture_reason[0] = 0;
            if (!g_capture_path) g_capture_busy = 0;
            pthread_mutex_unlock(&g_lock);
        }
    }
    pthread_mutex_unlock(&g_capture_lock);
}

/* g_lock held. Explicit markers and presents can be processed before native
 * submission. Publish them only after the last native operation returns. */
static void capture_publish_frame(void)
{
    if (!g_capture_commits) g_capture_frame = g_frame_cur;
    capture_schedule_stop();
}

static void capture_after_commit(void)
{
    pthread_mutex_lock(&g_lock);
    g_capture_commits--;
    capture_mtl4_recovered();
    if (g_capture_generation) g_capture_last_commit = trace_clock();
    capture_publish_frame();
    int synchronous = !g_capture_mtl4;
    pthread_mutex_unlock(&g_lock);
    if (synchronous) {
        pthread_mutex_lock(&g_capture_lock);
        capture_stop_at_boundary();
        pthread_mutex_unlock(&g_capture_lock);
    }
}

static id h_q_cb(id self, SEL _cmd)
{
    ENTER;
    if (OUTER && g_capture_available) capture_before_cb(self, 0);
    id cb = ORIG(H_Q_CB, id (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}
static id h_q_cbu(id self, SEL _cmd)
{
    ENTER;
    if (OUTER && g_capture_available) capture_before_cb(self, 0);
    id cb = ORIG(H_Q_CBU, id (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}
static id h_q_cbd(id self, SEL _cmd, id d)
{
    ENTER;
    if (OUTER && g_capture_available) capture_before_cb(self, 0);
    id cb = ORIG(H_Q_CBD, id (*)(id, SEL, id))(self, _cmd, d);
    LEAVE;
    if (OUTER) on_cb(self, cb);
    return cb;
}

/* command buffer */

static fp_node *on_encoder_kind(id cb, id enc, int kind, fp_trace_encoder *trace, int mtl4)
{
    if (!enc) return NULL;
    if (mtl4) hook_mtl4_encoder(object_getClass(enc), kind);
    else hook_encoder(object_getClass(enc), kind);
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    fp_node *n = c ? node_new(kind) : NULL;
    if (n) {
        n->mtl4 = mtl4;
        n->cb_uid = c->uid; n->idx = ++c->nenc;
        if (g_ring) g_ring_encoded = 1;
        if (trace) trace_encoder_register(trace, n);
        fp_item it = { 0, n->uid, NULL, 0 };
        PUSH(c->it, c->nit, c->cit, it);
        atomic_store_explicit(&n->open_uid, n->uid, memory_order_release);
    }
    pthread_mutex_unlock(&g_lock);
    if (n) {
        FPEncoderOwner *owner = [[FPEncoderOwner alloc] init];
        owner->uid = n->uid;
        objc_setAssociatedObject(enc, &g_encoder_owner_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [owner release];
        tl_cache.enc = (void *)enc; tl_cache.uid = n->uid;
    }
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
    atomic_store_explicit(&g_fp4_active, 1, memory_order_relaxed);
    if (!queue) return;
    const char *label = [[queue label] UTF8String];
    pthread_mutex_lock(&g_lock);
    if (g_capture_available) g_capture_mtl4 = 1;
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
    atomic_store_explicit(&g_fp4_active, 1, memory_order_relaxed);
    ENTER; id cb = ORIG(H_4_DEV_CB, id (*)(id, SEL))(self, _cmd); LEAVE;
    if (cb && (OUTER || g_capture_available)) {
        Class cls = object_getClass(cb);
        hook_mtl4_cb(cls);
        if (g_capture_available) {
            IMP end = class_getMethodImplementation(cls, sel_registerName("endCommandBuffer"));
            if (end != (IMP)h_mtl4_end && capture_mtl4_tracking_lost("Metal 4 command buffer end hook unavailable")) {
                [cb release];
                capture_interrupt_start("Metal 4 command buffer tracking lost during capture start");
            }
            pthread_mutex_lock(&g_lock);
            g_capture_mtl4 = 1;
            pthread_mutex_unlock(&g_lock);
        }
    }
    return cb;
}

static void on_mtl4_begin(id cb)
{
    fp_trace_device *device = trace_cb_device(cb, 1);
    pthread_mutex_lock(&g_lock);
    cb_record_new(cb, NULL, 1, device);
    pthread_mutex_unlock(&g_lock);
}

static void h_mtl4_begin(id self, SEL _cmd, id allocator)
{
    ENTER;
    if (OUTER && g_capture_available) capture_before_cb(self, 1);
    ORIG(H_4_BEGIN, void (*)(id, SEL, id))(self, _cmd, allocator);
    LEAVE;
    if (OUTER) on_mtl4_begin(self);
}

static void h_mtl4_begin_opts(id self, SEL _cmd, id allocator, id options)
{
    ENTER;
    if (OUTER && g_capture_available) capture_before_cb(self, 1);
    ORIG(H_4_BEGIN_OPTS, void (*)(id, SEL, id, id))(self, _cmd, allocator, options);
    LEAVE;
    if (OUTER) on_mtl4_begin(self);
}

static void h_mtl4_end(id self, SEL _cmd)
{
    ENTER;
    if (g_capture_available) {
        FPCaptureBufferOwner *owner = objc_getAssociatedObject(self, &g_capture_buffer_key);
        if (!owner) {
            owner = [[FPCaptureBufferOwner alloc] init];
            objc_setAssociatedObject(self, &g_capture_buffer_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            [owner release];
        }
        pthread_mutex_lock(&g_lock);
        while (g_capture_starting && !tl_capture_start)
            pthread_cond_wait(&g_capture_started_cond, &g_lock);
        if (owner)
            g_capture_mtl4_ended += !atomic_exchange_explicit(&owner->ended, 1, memory_order_relaxed);
        else capture_mtl4_mark_untracked("could not track an ended Metal 4 command buffer");
        pthread_mutex_unlock(&g_lock);
        if (tl_capture_start) {
            /* The starting thread cannot wait for itself. Record the end above,
             * but unwind before native end: Metal may still be activating the
             * capture layer. If Metal catches this exception and starts anyway,
             * capture_start stops it before replaying any native ends. */
            if (!tl_capture_start->ends) tl_capture_start->ends = [[NSMutableArray alloc] init];
            if ([tl_capture_start->ends indexOfObjectIdenticalTo:self] == NSNotFound)
                [tl_capture_start->ends addObject:self];
            LEAVE;
            capture_interrupt_start("Metal 4 end reentered capture start");
        }
    }
    ORIG(H_4_END, void (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (!OUTER) return;
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)self);
    if (c) c->state = 2;
    pthread_mutex_unlock(&g_lock);
}

API_AVAILABLE(macos(26.0))
static void fp4_render_attachments(fp_node *n, MTL4RenderPassDescriptor *desc)
{
    if (!n || !desc) return;
    NSUInteger layers = desc.renderTargetArrayLength;
    for (int i = 0; i < 8; i++) att(n, i, desc.colorAttachments[i], layers);
    att(n, 8, desc.depthAttachment, layers);
    att(n, 9, desc.stencilAttachment, layers);
}

API_AVAILABLE(macos(26.0))
static void trace_mtl4_begin_encoder(id enc, fp_node *n, fp_trace_encoder *trace)
{
    if (n && trace && trace->cb->device && trace->cb->device->heap && trace_take_sample(trace))
        trace_mtl4_stamp(enc, trace, 0);
}

static id h_mtl4_render(id self, SEL _cmd, id desc)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    id enc = ORIG(H_4_RENDER, id (*)(id, SEL, id))(self, _cmd, desc);
    LEAVE;
    if (OUTER) {
        if (@available(macOS 26.0, *)) {
            fp_node *n = on_encoder_kind(self, enc, FPK_RENDER, trace, 1);
            fp4_render_attachments(n, desc);
            trace_mtl4_begin_encoder(enc, n, trace);
        }
    }
    return enc;
}

static id h_mtl4_render_opts(id self, SEL _cmd, id desc, NSUInteger options)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    id enc = ORIG(H_4_RENDER_OPTS, id (*)(id, SEL, id, NSUInteger))(self, _cmd, desc, options);
    LEAVE;
    if (OUTER) {
        if (@available(macOS 26.0, *)) {
            fp_node *n = on_encoder_kind(self, enc, FPK_RENDER, trace, 1);
            if (n) {
                if (options & MTL4RenderEncoderOptionSuspending) n->mtl4_render_options |= FP4_SUSPENDING;
                if (options & MTL4RenderEncoderOptionResuming) n->mtl4_render_options |= FP4_RESUMING;
                fp4_render_attachments(n, desc);
            }
            if (options == MTL4RenderEncoderOptionNone) trace_mtl4_begin_encoder(enc, n, trace);
        }
    }
    return enc;
}

static id h_mtl4_compute(id self, SEL _cmd)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    id enc = ORIG(H_4_COMPUTE, id (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (OUTER) {
        fp_node *n = on_encoder_kind(self, enc, FPK_COMPUTE, trace, 1);
        if (trace) {
            if (@available(macOS 26.0, *)) trace_mtl4_begin_encoder(enc, n, trace);
        }
    }
    return enc;
}

static id h_mtl4_ml(id self, SEL _cmd)
{
    ENTER;
    fp_trace_encoder *trace = TRACE_ENCODER();
    id enc = ORIG(H_4_ML, id (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (OUTER && enc) {
        pthread_mutex_lock(&g_lock);
        g_mtl4_ml_encoders++;
        pthread_mutex_unlock(&g_lock);
        if (trace) on_encoder_kind(self, enc, FPK_ML, trace, 1);
    }
    return enc;
}

API_AVAILABLE(macos(26.0))
static id h_mtl4_argument_table(id self, SEL _cmd, MTL4ArgumentTableDescriptor *desc, NSError **error)
{
    atomic_store_explicit(&g_fp4_active, 1, memory_order_relaxed);
    ENTER;
    id<MTL4ArgumentTable> table = ORIG(H_4_ARGUMENT_TABLE, id (*)(id, SEL, id, NSError **))(self, _cmd, desc, error);
    LEAVE;
    if (OUTER && table) {
        FP4ArgumentState *state = [[FP4ArgumentState alloc] init];
        state->device = (uintptr_t)table.device;
        state->buffers = desc.maxBufferBindCount;
        state->textures = desc.maxTextureBindCount;
        state->slots = calloc(state->buffers + state->textures, sizeof *state->slots);
        if (state->slots) {
            for (NSUInteger i = 0; i < state->buffers + state->textures; i++) {
                atomic_init(&state->slots[i].value, 0);
                atomic_init(&state->slots[i].by_id, 0);
            }
            FP4ArgumentOwner *owner = [[FP4ArgumentOwner alloc] init];
            owner->state = [state retain];
            objc_setAssociatedObject(table, &g_fp4_argument_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            [owner release];
            atomic_fetch_add_explicit(&g_fp4_table_generation, 1, memory_order_release);
            hook_mtl4_argument(object_getClass(table));
        }
        [state release];
    }
    return table;
}

/* Pipeline creation. fencepost asks Metal for binding reflection itself so that argument-table
 * slots get the read/write access the shader declares. Device-built pipelines return it only
 * through the reflection out-parameter or handler (pipeline.reflection stays nil); Metal 4
 * compiler pipelines keep it in pipeline.reflection. Device-path requests start only after the
 * process makes a Metal 4 object, so programs that use only Metal 3 build pipelines unchanged. */

typedef id (*fp4_reflect_imp)(id, SEL, id, NSUInteger, id *, NSError **);
typedef void (^fp4_pipeline_handler)(id, NSError *);
typedef void (^fp4_reflection_handler)(id, id, NSError *);

static int fp4_reflect_device(NSUInteger options)
{
    return atomic_load_explicit(&g_fp4_active, memory_order_relaxed) &&
           !(options & MTLPipelineOptionFailOnBinaryArchiveMiss);   /* an archive may hold no reflection */
}

#ifdef FENCEPOST_TEST_HOOKS
/* Fail fencepost's own BindingInfo request, so the retry path runs as if Metal refused it. */
static int fp4_test_fail_reflection(void)
{
    if (!g_mtl4_test_reflection_fail) return 0;
    g_mtl4_test_reflection_fail--;
    g_mtl4_test_reflection_failed++;
    return 1;
}
#endif

static id fp4_device_sync(int h, id self, SEL cmd, id desc, NSUInteger options, id *reflection, NSError **error)
{
    ENTER;
    fp4_reflect_imp orig = (fp4_reflect_imp)orig_imp(h, self);
    int want = OUTER && fp4_reflect_device(options);
    id local = nil, *out = reflection ? reflection : want ? &local : NULL;
    id pipeline = nil;
#ifdef FENCEPOST_TEST_HOOKS
    int force = want && !(options & MTLPipelineOptionBindingInfo) && fp4_test_fail_reflection();
    if (!force)
#endif
        pipeline = orig(self, cmd, desc, want ? options | MTLPipelineOptionBindingInfo : options, out, error);
    if (want && pipeline) fp4_attach_reflection(pipeline, *out);
    else if (want && !(options & MTLPipelineOptionBindingInfo)) {
        if (error) *error = nil;
        if (reflection) *reflection = nil;
        pipeline = orig(self, cmd, desc, options, reflection, error);
    }
    LEAVE;
    return pipeline;
}

static id fp4_device_plain(int h, id self, SEL cmd, SEL with_reflection, id desc, NSError **error)
{
    ENTER;
    id pipeline = nil;
    if (OUTER && fp4_reflect_device(0)) {
        id reflection = nil;
#ifdef FENCEPOST_TEST_HOOKS
        if (!fp4_test_fail_reflection())
#endif
            pipeline = ((fp4_reflect_imp)objc_msgSend)(self, with_reflection, desc, MTLPipelineOptionBindingInfo,
                                                       &reflection, error);
        fp4_attach_reflection(pipeline, reflection);
        if (!pipeline && error) *error = nil;
    }
    if (!pipeline) pipeline = ((id (*)(id, SEL, id, NSError **))orig_imp(h, self))(self, cmd, desc, error);
    LEAVE;
    return pipeline;
}

static void fp4_device_async(int h, id self, SEL cmd, id desc, NSUInteger options, fp4_reflection_handler handler)
{
    ENTER;
    void (*orig)(id, SEL, id, NSUInteger, fp4_reflection_handler) = (void *)orig_imp(h, self);
    if (OUTER && handler && fp4_reflect_device(options)) {
        void (^complete)(id, id, NSError *) = ^(id pipeline, id reflection, NSError *error) {
            if (!pipeline && !(options & MTLPipelineOptionBindingInfo)) {
                tl_depth++;
                orig(self, cmd, desc, options, handler);
                tl_depth--;
                return;
            }
            fp4_attach_reflection(pipeline, reflection);
            handler(pipeline, reflection, error);
        };
#ifdef FENCEPOST_TEST_HOOKS
        if (!(options & MTLPipelineOptionBindingInfo) && fp4_test_fail_reflection()) complete(nil, nil, nil);
        else
#endif
        orig(self, cmd, desc, options | MTLPipelineOptionBindingInfo, complete);
    } else orig(self, cmd, desc, options, handler);
    LEAVE;
}

static void fp4_device_async_plain(int h, id self, SEL cmd, SEL with_reflection, id desc, fp4_pipeline_handler handler)
{
    ENTER;
    void (*orig)(id, SEL, id, fp4_pipeline_handler) = (void *)orig_imp(h, self);
    if (OUTER && handler && fp4_reflect_device(0)) {
        ((void (*)(id, SEL, id, NSUInteger, fp4_reflection_handler))objc_msgSend)(self, with_reflection, desc,
            MTLPipelineOptionBindingInfo, ^(id pipeline, id reflection, NSError *error) {
                if (!pipeline) {
                    tl_depth++;
                    orig(self, cmd, desc, handler);
                    tl_depth--;
                    return;
                }
                fp4_attach_reflection(pipeline, reflection);
                handler(pipeline, error);
            });
    } else orig(self, cmd, desc, handler);
    LEAVE;
}

#define FP4_DEVICE_SYNC(fn, hook_id) \
    static id fn(id self, SEL _cmd, id desc, NSUInteger options, id *reflection, NSError **error) \
    { return fp4_device_sync(hook_id, self, _cmd, desc, options, reflection, error); }
#define FP4_DEVICE_ASYNC(fn, hook_id) \
    static void fn(id self, SEL _cmd, id desc, NSUInteger options, fp4_reflection_handler handler) \
    { fp4_device_async(hook_id, self, _cmd, desc, options, handler); }
FP4_DEVICE_SYNC(h_p_cfn_r, H_P_CFN_R)
FP4_DEVICE_SYNC(h_p_cdesc_r, H_P_CDESC_R)
FP4_DEVICE_SYNC(h_p_rdesc_r, H_P_RDESC_R)
FP4_DEVICE_SYNC(h_p_tile_r, H_P_TILE_R)
FP4_DEVICE_SYNC(h_p_mesh_r, H_P_MESH_R)
FP4_DEVICE_ASYNC(h_p_cfn_ar, H_P_CFN_AR)
FP4_DEVICE_ASYNC(h_p_cdesc_ar, H_P_CDESC_AR)
FP4_DEVICE_ASYNC(h_p_rdesc_ar, H_P_RDESC_AR)
FP4_DEVICE_ASYNC(h_p_tile_ar, H_P_TILE_AR)
FP4_DEVICE_ASYNC(h_p_mesh_ar, H_P_MESH_AR)
#undef FP4_DEVICE_SYNC
#undef FP4_DEVICE_ASYNC

static id h_p_cfn(id self, SEL _cmd, id function, NSError **error)
{
    return fp4_device_plain(H_P_CFN, self, _cmd, @selector(newComputePipelineStateWithFunction:options:reflection:error:),
                            function, error);
}
static id h_p_rdesc(id self, SEL _cmd, id desc, NSError **error)
{
    return fp4_device_plain(H_P_RDESC, self, _cmd, @selector(newRenderPipelineStateWithDescriptor:options:reflection:error:),
                            desc, error);
}
static void h_p_cfn_a(id self, SEL _cmd, id function, fp4_pipeline_handler handler)
{
    fp4_device_async_plain(H_P_CFN_A, self, _cmd, @selector(newComputePipelineStateWithFunction:options:completionHandler:),
                           function, handler);
}
static void h_p_rdesc_a(id self, SEL _cmd, id desc, fp4_pipeline_handler handler)
{
    fp4_device_async_plain(H_P_RDESC_A, self, _cmd, @selector(newRenderPipelineStateWithDescriptor:options:completionHandler:),
                           desc, handler);
}

/* A copy of the app's descriptor that also asks for binding reflection (+1). The app's
 * descriptor and its options object are never changed. */
static id fp4_reflect_descriptor(id desc)
{
    if (@available(macOS 26.0, *)) {
        if (![desc isKindOfClass:[MTL4PipelineDescriptor class]]) return [desc retain];
        MTL4PipelineDescriptor *app = desc;
        MTL4PipelineOptions *options = app.options;
        if (options.shaderReflection & MTL4ShaderReflectionBindingInfo) return [desc retain];
        MTL4PipelineDescriptor *copy = [app copy];
        MTL4PipelineOptions *mine = options ? [options copy] : [[MTL4PipelineOptions alloc] init];
        mine.shaderReflection |= MTL4ShaderReflectionBindingInfo;
        copy.options = mine;
        [mine release];
        return copy;
    }
    return [desc retain];
}

static char g_fp4_private_reflection_key;

static id h_fp4_reflection(id self, SEL cmd)
{
    if (objc_getAssociatedObject(self, &g_fp4_private_reflection_key)) return nil;
    return ORIG(H_C4_REFLECTION, id (*)(id, SEL))(self, cmd);
}

static int fp4_private_reflection(id desc, id copy, id base)
{
    if (@available(macOS 26.0, *)) {
        MTL4PipelineDescriptor *original = desc;
        return copy != desc && original.options.shaderReflection == MTL4ShaderReflectionNone &&
               (![base respondsToSelector:@selector(reflection)] || ![base reflection]);
    }
    return 0;
}

/* A specialized pipeline without its own reflection keeps the bindings of the one it specializes. */
static void fp4_compiled(id pipeline, id base, int private_reflection)
{
    if (!pipeline) return;
    id reflection = nil;
    if (@available(macOS 26.0, *))
        if ([pipeline respondsToSelector:@selector(reflection)]) reflection = [pipeline reflection];
    FP4PipelineTable *table = fp4_table_new(reflection);
    if (!table && base) table = [objc_getAssociatedObject(base, &g_fp4_pipeline_key) retain];
    fp4_attach_table(pipeline, table);
    [table release];
    if (private_reflection && reflection) {
        pthread_mutex_lock(&g_lock);
        hook(object_getClass(pipeline), H_C4_REFLECTION, "reflection", (IMP)h_fp4_reflection);
        pthread_mutex_unlock(&g_lock);
        objc_setAssociatedObject(pipeline, &g_fp4_private_reflection_key, @YES, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
}

static id fp4_compile(int h, id self, SEL cmd, id desc, id x, id y, NSError **error, int four, int specialize)
{
    ENTER;
    IMP imp = orig_imp(h, self);
#define FP4_CALL(d) (four ? ((id (*)(id, SEL, id, id, id, NSError **))imp)(self, cmd, d, x, y, error) \
                          : ((id (*)(id, SEL, id, id, NSError **))imp)(self, cmd, d, x, error))
    id pipeline = nil;
    if (OUTER) {
        id mine = fp4_reflect_descriptor(desc), base = specialize ? x : nil;
        int private_reflection = fp4_private_reflection(desc, mine, base);
#ifdef FENCEPOST_TEST_HOOKS
        int force = mine != desc && fp4_test_fail_reflection();
        if (!force)
#endif
            pipeline = FP4_CALL(mine);
        if (!pipeline && mine != desc) {
            if (error) *error = nil;
            pipeline = FP4_CALL(desc);
            private_reflection = 0;
        }
        [mine release];
        fp4_compiled(pipeline, base, private_reflection);
    } else pipeline = FP4_CALL(desc);
#undef FP4_CALL
    LEAVE;
    return pipeline;
}

/* Both Metal tasks belong to one app request. A waiter must stay blocked through a retry. */
API_AVAILABLE(macos(26.0))
@interface FP4CompilerTask : NSObject <MTL4CompilerTask> {
    id<MTL4Compiler> _compiler;
    id<MTL4CompilerTask> _initial;
    dispatch_group_t _pending;
    os_unfair_lock _lock;
}
- (instancetype)initWithCompiler:(id<MTL4Compiler>)compiler;
- (void)setInitialTask:(id<MTL4CompilerTask>)task;
- (void)finish;
@end

@implementation FP4CompilerTask
- (instancetype)initWithCompiler:(id<MTL4Compiler>)compiler
{
    if (!(self = [super init])) return nil;
    _compiler = [compiler retain];
    _pending = dispatch_group_create();
    dispatch_group_enter(_pending);
    _lock = OS_UNFAIR_LOCK_INIT;
    return self;
}
- (void)dealloc
{
    [_initial release];
    [_compiler release];
    dispatch_release(_pending);
    [super dealloc];
}
- (id<MTL4Compiler>)compiler { return _compiler; }
- (void)setInitialTask:(id<MTL4CompilerTask>)task
{
    os_unfair_lock_lock(&_lock);
    if (dispatch_group_wait(_pending, DISPATCH_TIME_NOW)) _initial = [task retain];
    os_unfair_lock_unlock(&_lock);
}
- (void)finish
{
    os_unfair_lock_lock(&_lock);
    id initial = _initial;
    _initial = nil;
    dispatch_group_leave(_pending);
    os_unfair_lock_unlock(&_lock);
    [initial release];
}
- (MTL4CompilerTaskStatus)status
{
    os_unfair_lock_lock(&_lock);
    id<MTL4CompilerTask> initial = [_initial retain];
    BOOL pending = dispatch_group_wait(_pending, DISPATCH_TIME_NOW) != 0;
    os_unfair_lock_unlock(&_lock);
    MTL4CompilerTaskStatus status = initial ? initial.status : MTL4CompilerTaskStatusScheduled;
    [initial release];
    if (!pending) return MTL4CompilerTaskStatusFinished;
    return status == MTL4CompilerTaskStatusFinished ? MTL4CompilerTaskStatusCompiling : status;
}
- (void)waitUntilCompleted { dispatch_group_wait(_pending, DISPATCH_TIME_FOREVER); }
@end

API_AVAILABLE(macos(26.0))
static id fp4_compile_async(int h, id self, SEL cmd, id desc, id x, id y, fp4_pipeline_handler handler,
                            int four, int specialize)
{
    ENTER;
    IMP imp = orig_imp(h, self);
#define FP4_CALL(d, b) (four ? ((id (*)(id, SEL, id, id, id, fp4_pipeline_handler))imp)(self, cmd, d, x, y, b) \
                             : ((id (*)(id, SEL, id, id, fp4_pipeline_handler))imp)(self, cmd, d, x, b))
    id task;
    if (OUTER && handler) {
        id mine = fp4_reflect_descriptor(desc), base = specialize ? x : nil;
        if (mine != desc) {
            int private_reflection = fp4_private_reflection(desc, mine, base);
            FP4CompilerTask *request = [[FP4CompilerTask alloc] initWithCompiler:self];
            void (^complete)(id, NSError *) = ^(id pipeline, NSError *error) {
                if (pipeline) {
                    fp4_compiled(pipeline, base, private_reflection);
                    [request finish];
                    handler(pipeline, error);
                } else {
                    tl_depth++;
                    id retry = FP4_CALL(desc, ^(id plain, NSError *plainError) {
                        fp4_compiled(plain, base, 0);
                        [request finish];
                        handler(plain, plainError);
                    });
                    tl_depth--;
                    [retry release];
                }
            };
            id initial = nil;
#ifdef FENCEPOST_TEST_HOOKS
            if (fp4_test_fail_reflection()) complete(nil, nil);
            else
#endif
            initial = FP4_CALL(mine, complete);
            [request setInitialTask:initial];
            [initial release];
            task = request;
        } else {
            task = FP4_CALL(mine, ^(id pipeline, NSError *error) {
                fp4_compiled(pipeline, base, 0);
                handler(pipeline, error);
            });
        }
        [mine release];
    } else task = FP4_CALL(desc, handler);
#undef FP4_CALL
    LEAVE;
    return task;
}

static id h_c4_compute(id self, SEL _cmd, id desc, id task, NSError **error)
{ return fp4_compile(H_C4_COMPUTE, self, _cmd, desc, task, nil, error, 0, 0); }
static id h_c4_compute_link(id self, SEL _cmd, id desc, id link, id task, NSError **error)
{ return fp4_compile(H_C4_COMPUTE_LINK, self, _cmd, desc, link, task, error, 1, 0); }
static id h_c4_render(id self, SEL _cmd, id desc, id task, NSError **error)
{ return fp4_compile(H_C4_RENDER, self, _cmd, desc, task, nil, error, 0, 0); }
static id h_c4_render_link(id self, SEL _cmd, id desc, id link, id task, NSError **error)
{ return fp4_compile(H_C4_RENDER_LINK, self, _cmd, desc, link, task, error, 1, 0); }
static id h_c4_specialize(id self, SEL _cmd, id desc, id pipeline, NSError **error)
{ return fp4_compile(H_C4_SPECIALIZE, self, _cmd, desc, pipeline, nil, error, 0, 1); }
API_AVAILABLE(macos(26.0))
static id h_c4_compute_a(id self, SEL _cmd, id desc, id task, fp4_pipeline_handler handler)
{ return fp4_compile_async(H_C4_COMPUTE_A, self, _cmd, desc, task, nil, handler, 0, 0); }
API_AVAILABLE(macos(26.0))
static id h_c4_compute_link_a(id self, SEL _cmd, id desc, id link, id task, fp4_pipeline_handler handler)
{ return fp4_compile_async(H_C4_COMPUTE_LINK_A, self, _cmd, desc, link, task, handler, 1, 0); }
API_AVAILABLE(macos(26.0))
static id h_c4_render_a(id self, SEL _cmd, id desc, id task, fp4_pipeline_handler handler)
{ return fp4_compile_async(H_C4_RENDER_A, self, _cmd, desc, task, nil, handler, 0, 0); }
API_AVAILABLE(macos(26.0))
static id h_c4_render_link_a(id self, SEL _cmd, id desc, id link, id task, fp4_pipeline_handler handler)
{ return fp4_compile_async(H_C4_RENDER_LINK_A, self, _cmd, desc, link, task, handler, 1, 0); }
API_AVAILABLE(macos(26.0))
static id h_c4_specialize_a(id self, SEL _cmd, id desc, id pipeline, fp4_pipeline_handler handler)
{ return fp4_compile_async(H_C4_SPECIALIZE_A, self, _cmd, desc, pipeline, nil, handler, 0, 1); }

static void hook_mtl4_compiler(Class cls);
static id h_mtl4_new_compiler(id self, SEL _cmd, id desc, NSError **error)
{
    atomic_store_explicit(&g_fp4_active, 1, memory_order_relaxed);
    ENTER; id compiler = ORIG(H_4_NEW_COMPILER, id (*)(id, SEL, id, NSError **))(self, _cmd, desc, error); LEAVE;
    if (OUTER && compiler) hook_mtl4_compiler(object_getClass(compiler));
    return compiler;
}

static void h_mtl4_set_compute_pipeline(id self, SEL _cmd, id pipeline)
{
    ENTER; ORIG(H_4_SET_COMPUTE_PIPELINE, void (*)(id, SEL, id))(self, _cmd, pipeline); LEAVE;
    if (OUTER) fp4_bind_pipeline(self, pipeline);
}

static void h_mtl4_set_render_pipeline(id self, SEL _cmd, id pipeline)
{
    ENTER; ORIG(H_4_SET_RENDER_PIPELINE, void (*)(id, SEL, id))(self, _cmd, pipeline); LEAVE;
    if (OUTER) fp4_bind_pipeline(self, pipeline);
}

static void h_mtl4_address(id self, SEL _cmd, uint64_t address, NSUInteger index)
{
    FP4_ORIG(H_4_ADDRESS, void (*)(id, SEL, uint64_t, NSUInteger))(self, _cmd, address, index);
    fp4_set_slot(self, index, 0, address, 0);
}

static void h_mtl4_stride(id self, SEL _cmd, uint64_t address, NSUInteger stride, NSUInteger index)
{
    ENTER;
    FP4_ORIG(H_4_STRIDE, void (*)(id, SEL, uint64_t, NSUInteger, NSUInteger))(self, _cmd, address, stride, index);
    LEAVE;
    if (OUTER) fp4_set_slot(self, index, 0, address, 0);
}

static void h_mtl4_resource(id self, SEL _cmd, MTLResourceID resource, NSUInteger index)
{
    ENTER;
    FP4_ORIG(H_4_RESOURCE, void (*)(id, SEL, MTLResourceID, NSUInteger))(self, _cmd, resource, index);
    LEAVE;
    if (OUTER) fp4_set_slot(self, index, 0, resource._impl, 1);
}

static void h_mtl4_texture(id self, SEL _cmd, MTLResourceID resource, NSUInteger index)
{
    FP4_ORIG(H_4_TEXTURE, void (*)(id, SEL, MTLResourceID, NSUInteger))(self, _cmd, resource, index);
    fp4_set_slot(self, index, 1, resource._impl, 1);
}

static void h_mtl4_bind_compute(id self, SEL _cmd, id table)
{
    ENTER; ORIG(H_4_BIND_COMPUTE, void (*)(id, SEL, id))(self, _cmd, table); LEAVE;
    if (OUTER) fp4_bind_table(self, table, FP4_STAGE(FP4_DISPATCH));
}

static void h_mtl4_bind_render(id self, SEL _cmd, id table, NSUInteger stages)
{
    ENTER; ORIG(H_4_BIND_RENDER, void (*)(id, SEL, id, NSUInteger))(self, _cmd, table, stages); LEAVE;
    if (OUTER) fp4_bind_table(self, table, (unsigned)stages & (FP4_STAGE(FP4_MESH + 1) - 1));
}

static void fp4_index_access(id enc, uint64_t address, NSUInteger length)
{
    fp_node *n = cur_node(enc);
    if (!n || !n->mtl4 || !address || !length) return;
    /* The encoder's device is the same one used to create its command buffer. */
    fp4_resource resource = fp4_lookup((uintptr_t)[enc device], address, 0);
    if (!resource.store) return;
    uint64_t offset = address - resource.address;
    fp4_record_access(n, resource, offset, offset + MIN(length, UINT64_MAX - offset),
                      0, 0, FP_RD, FP4_VERTEX, VIA_INDEX);
}

API_AVAILABLE(macos(26.0))
static void h_mtl4_accel_build(id self, SEL _cmd, id accel, id descriptor, MTL4BufferRange scratch)
{
    ENTER;
    ORIG(H_4_ACCEL_BUILD, void (*)(id, SEL, id, id, MTL4BufferRange))(self, _cmd, accel, descriptor, scratch);
    LEAVE;
    if (OUTER) {
        __atomic_fetch_add(&g_mtl4_accel_builds, 1, __ATOMIC_RELAXED);
        fp_node *n = cur_node(self);
        if (n && n->mtl4) fp4_note_work(n, FP4_ACCEL);
    }
}

static void h_mtl4_draw(id self, SEL _cmd, NSUInteger primitive, NSUInteger start, NSUInteger count)
{
    fp4_capture_tables(self, FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT));
    FP4_ORIG(H_4_DRAW, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, primitive, start, count);
}

static void h_mtl4_draw_i(id self, SEL _cmd, NSUInteger primitive, NSUInteger start, NSUInteger count, NSUInteger instances)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT));
    FP4_ORIG(H_4_DRAW_I, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, primitive, start, count, instances);
    LEAVE;
}

static void h_mtl4_draw_b(id self, SEL _cmd, NSUInteger primitive, NSUInteger start, NSUInteger count, NSUInteger instances, NSUInteger base)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT));
    FP4_ORIG(H_4_DRAW_B, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, primitive, start, count, instances, base);
    LEAVE;
}

static void h_mtl4_draw_idx(id self, SEL _cmd, NSUInteger primitive, NSUInteger count, NSUInteger index_type, uint64_t index_address, NSUInteger index_bytes)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT));
    if (OUTER) fp4_index_access(self, index_address, index_bytes);
    FP4_ORIG(H_4_DRAW_IDX, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, uint64_t, NSUInteger))(self, _cmd, primitive, count, index_type, index_address, index_bytes);
    LEAVE;
}

static void h_mtl4_draw_idx_i(id self, SEL _cmd, NSUInteger primitive, NSUInteger count, NSUInteger index_type, uint64_t index_address, NSUInteger index_bytes, NSUInteger instances)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT));
    if (OUTER) fp4_index_access(self, index_address, index_bytes);
    FP4_ORIG(H_4_DRAW_IDX_I, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, uint64_t, NSUInteger, NSUInteger))(self, _cmd, primitive, count, index_type, index_address, index_bytes, instances);
    LEAVE;
}

static void h_mtl4_draw_idx_b(id self, SEL _cmd, NSUInteger primitive, NSUInteger count, NSUInteger index_type, uint64_t index_address, NSUInteger index_bytes, NSUInteger instances, NSInteger base_vertex, NSUInteger base_instance)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT));
    if (OUTER) fp4_index_access(self, index_address, index_bytes);
    FP4_ORIG(H_4_DRAW_IDX_B, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, uint64_t, NSUInteger, NSUInteger, NSInteger, NSUInteger))(self, _cmd, primitive, count, index_type, index_address, index_bytes, instances, base_vertex, base_instance);
    LEAVE;
}

static void h_mtl4_draw_ind(id self, SEL _cmd, NSUInteger primitive, uint64_t indirect)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT));
    FP4_ORIG(H_4_DRAW_IND, void (*)(id, SEL, NSUInteger, uint64_t))(self, _cmd, primitive, indirect);
    LEAVE;
}

static void h_mtl4_draw_idx_ind(id self, SEL _cmd, NSUInteger primitive, NSUInteger index_type, uint64_t index_address, NSUInteger index_bytes, uint64_t indirect)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_VERTEX) | FP4_STAGE(FP4_FRAGMENT));
    if (OUTER) fp4_index_access(self, index_address, index_bytes);
    FP4_ORIG(H_4_DRAW_IDX_IND, void (*)(id, SEL, NSUInteger, NSUInteger, uint64_t, NSUInteger, uint64_t))(self, _cmd, primitive, index_type, index_address, index_bytes, indirect);
    LEAVE;
}

static void h_mtl4_draw_mesh_groups(id self, SEL _cmd, MTLSize grid, MTLSize object, MTLSize mesh)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, (FP4_STAGE(FP4_OBJECT) | FP4_STAGE(FP4_MESH) | FP4_STAGE(FP4_FRAGMENT)));
    FP4_ORIG(H_4_DRAW_MESH_GROUPS, void (*)(id, SEL, MTLSize, MTLSize, MTLSize))(self, _cmd, grid, object, mesh);
    LEAVE;
}

static void h_mtl4_draw_mesh(id self, SEL _cmd, MTLSize grid, MTLSize object, MTLSize mesh)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, (FP4_STAGE(FP4_OBJECT) | FP4_STAGE(FP4_MESH) | FP4_STAGE(FP4_FRAGMENT)));
    FP4_ORIG(H_4_DRAW_MESH, void (*)(id, SEL, MTLSize, MTLSize, MTLSize))(self, _cmd, grid, object, mesh);
    LEAVE;
}

static void h_mtl4_draw_mesh_ind(id self, SEL _cmd, uint64_t indirect, MTLSize object, MTLSize mesh)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, (FP4_STAGE(FP4_OBJECT) | FP4_STAGE(FP4_MESH) | FP4_STAGE(FP4_FRAGMENT)));
    FP4_ORIG(H_4_DRAW_MESH_IND, void (*)(id, SEL, uint64_t, MTLSize, MTLSize))(self, _cmd, indirect, object, mesh);
    LEAVE;
}

static void h_mtl4_draw_tile(id self, SEL _cmd, MTLSize threads)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_TILE));
    FP4_ORIG(H_4_DRAW_TILE, void (*)(id, SEL, MTLSize))(self, _cmd, threads);
    LEAVE;
}

static void h_mtl4_dispatch(id self, SEL _cmd, MTLSize grid, MTLSize threads)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_DISPATCH));
    FP4_ORIG(H_4_DISPATCH, void (*)(id, SEL, MTLSize, MTLSize))(self, _cmd, grid, threads);
    LEAVE;
}

static void h_mtl4_dispatch_groups(id self, SEL _cmd, MTLSize grid, MTLSize threads)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_DISPATCH));
    FP4_ORIG(H_4_DISPATCH_GROUPS, void (*)(id, SEL, MTLSize, MTLSize))(self, _cmd, grid, threads);
    LEAVE;
}

static void h_mtl4_dispatch_ind(id self, SEL _cmd, uint64_t indirect)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_DISPATCH));
    FP4_ORIG(H_4_DISPATCH_IND, void (*)(id, SEL, uint64_t))(self, _cmd, indirect);
    LEAVE;
}

static void h_mtl4_dispatch_groups_ind(id self, SEL _cmd, uint64_t indirect, MTLSize threads)
{
    ENTER;
    if (OUTER) fp4_capture_tables(self, FP4_STAGE(FP4_DISPATCH));
    FP4_ORIG(H_4_DISPATCH_GROUPS_IND, void (*)(id, SEL, uint64_t, MTLSize))(self, _cmd, indirect, threads);
    LEAVE;
}

API_AVAILABLE(macos(26.0))
static void mtl4_residency(id self, const id *sets, NSUInteger count, int add, int is_cb)
{
    for (NSUInteger at = 0; at < count; at++) {
        void *set = (void *)sets[at];
        if (add && ![(id<MTLResidencySet>)sets[at] allocationCount]) continue;
        pthread_mutex_lock(&g_lock);
        if (is_cb) {
            fp_cb *c = cb_find_open((void *)self);
            if (c && c->mtl4) c->mtl4_resident = 1;
        } else {
            fp_queue *q = queue_get_kind((void *)self, 1);
            if (q) {
                int i = 0;
                while (i < q->mtl4_nresidency && q->mtl4_residency[i] != set) i++;
                if (add && i == q->mtl4_nresidency)
                    PUSH(q->mtl4_residency, q->mtl4_nresidency, q->mtl4_cresidency, set);
                else if (!add && i < q->mtl4_nresidency)
                    q->mtl4_residency[i] = q->mtl4_residency[--q->mtl4_nresidency];
            }
        }
        pthread_mutex_unlock(&g_lock);
    }
}

#define FP4_RES_SINGLE(fn, hook_id, add, cb) \
    API_AVAILABLE(macos(26.0)) \
    static void fn(id self, SEL _cmd, id set) { \
        ENTER; ORIG(hook_id, void (*)(id, SEL, id))(self, _cmd, set); LEAVE; \
        if (OUTER) mtl4_residency(self, &set, 1, add, cb); \
    }
#define FP4_RES_ARRAY(fn, hook_id, add, cb) \
    API_AVAILABLE(macos(26.0)) \
    static void fn(id self, SEL _cmd, const id *sets, NSUInteger count) { \
        ENTER; ORIG(hook_id, void (*)(id, SEL, const id *, NSUInteger))(self, _cmd, sets, count); LEAVE; \
        if (OUTER) mtl4_residency(self, sets, count, add, cb); \
    }
FP4_RES_SINGLE(h_mtl4_res_add, H_4_RES_ADD, 1, 0)
FP4_RES_ARRAY(h_mtl4_res_adds, H_4_RES_ADDS, 1, 0)
FP4_RES_SINGLE(h_mtl4_res_remove, H_4_RES_REMOVE, 0, 0)
FP4_RES_ARRAY(h_mtl4_res_removes, H_4_RES_REMOVES, 0, 0)
FP4_RES_SINGLE(h_mtl4_res_use, H_4_RES_USE, 1, 1)
FP4_RES_ARRAY(h_mtl4_res_uses, H_4_RES_USES, 1, 1)
#undef FP4_RES_SINGLE
#undef FP4_RES_ARRAY

static unsigned fp4_stage_mask(NSUInteger mask)
{
    if (@available(macOS 26.0, *)) {
        unsigned stages = (unsigned)mask & (FP4_STAGE(FP4_MESH + 1) - 1);
        if (mask & MTLStageDispatch) stages |= FP4_STAGE(FP4_DISPATCH);
        if (mask & MTLStageBlit) stages |= FP4_STAGE(FP4_BLIT);
        if (mask & MTLStageAccelerationStructure) stages |= FP4_STAGE(FP4_ACCEL);
        if (mask & MTLStageMachineLearning) stages |= FP4_STAGE(FP4_ML);
        if (mask & MTLStageResourceState) stages |= FP4_STAGE(FP4_RESOURCE_STATE);
        return stages;
    }
    return 0;
}

/* Render pass attachments. Neither the headers nor Apple's synchronization articles say
 * whether a consumer barrier or fence wait at the start of a render encoder orders the
 * pass's attachment loads (the articles say only that loads happen at the start of the
 * pass and stores at its end). fencepost decides conservatively:
 * - a consumer queue barrier or fence wait encoded before the encoder's first draw covers
 *   the attachment loads and stores (in its before stages, attachments being Fragment);
 * - one encoded after a draw does not cover the loads, but does cover the stores.
 * Loads take the first draw's epoch (the final epoch on a drawless pass). Stores take the
 * final epoch. A Fragment update or producer barrier covers the stores only when it is
 * encoded after the pass's last draw; one before or between draws covers neither them nor
 * later draws. Fence operations are also work for chaining through a covered stage. */
static void fp4_record_sync(id encoder, int kind, id fence, NSUInteger after, NSUInteger before)
{
    fp_node *n = cur_node(encoder);
    if (!n || !n->mtl4) return;
    fp4_sync sync = { (void *)fence, n->mtl4_epoch, fp4_stage_mask(after), fp4_stage_mask(before), kind };
    if (kind == FP4_FENCE_UPDATE)
        for (unsigned mask = sync.after; mask; mask &= mask - 1) fp4_note_work(n, __builtin_ctz(mask));
    n->mtl4_epoch++;
    if (kind == FP4_FENCE_WAIT)
        for (unsigned mask = sync.before; mask; mask &= mask - 1) fp4_note_work(n, __builtin_ctz(mask));
    PUSH(n->mtl4_sync, n->mtl4_nsync, n->mtl4_csync, sync);
    memset(n->mtl4_versions, 0, sizeof n->mtl4_versions);
}

#define FP4_BARRIER(fn, hook_id, kind) \
    static void fn(id self, SEL _cmd, NSUInteger after, NSUInteger before, NSUInteger visibility) { \
        ENTER; ORIG(hook_id, void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger))(self, _cmd, after, before, visibility); LEAVE; \
        if (OUTER) fp4_record_sync(self, kind, nil, after, before); \
    }
FP4_BARRIER(h_mtl4_consumer_barrier, H_4_CONSUMER_BARRIER, FP4_CONSUMER)
FP4_BARRIER(h_mtl4_producer_barrier, H_4_PRODUCER_BARRIER, FP4_PRODUCER)
FP4_BARRIER(h_mtl4_intra_barrier, H_4_INTRA_BARRIER, FP4_INTRA)
#undef FP4_BARRIER

static void h_mtl4_fence_update(id self, SEL _cmd, id fence, NSUInteger stages)
{
    ENTER; ORIG(H_4_FENCE_UPDATE, void (*)(id, SEL, id, NSUInteger))(self, _cmd, fence, stages); LEAVE;
    if (OUTER) fp4_record_sync(self, FP4_FENCE_UPDATE, fence, stages, 0);
}

static void h_mtl4_fence_wait(id self, SEL _cmd, id fence, NSUInteger stages)
{
    ENTER; ORIG(H_4_FENCE_WAIT, void (*)(id, SEL, id, NSUInteger))(self, _cmd, fence, stages); LEAVE;
    if (OUTER) fp4_record_sync(self, FP4_FENCE_WAIT, fence, 0, stages);
}

static void cb_item(id cb, int type, void *ev, uint64_t val)
{
    pthread_mutex_lock(&g_lock);
    fp_cb *c = cb_find_open((void *)cb);
    if (c) {
        if (g_trace_enabled && type == 4) val = trace_clock();
        fp_item it = { (uint8_t)type, 0, ev, val };
        PUSH(c->it, c->nit, c->cit, it);
    }
    pthread_mutex_unlock(&g_lock);
}
static void h_cb_sig(id self, SEL _cmd, id ev, uint64_t v)
{
    ENTER; ORIG(H_CB_SIG, void (*)(id, SEL, id, uint64_t))(self, _cmd, ev, v); LEAVE;
    if (OUTER) cb_item(self, 1, event_key(ev, 0), v);
}
static void h_cb_wait(id self, SEL _cmd, id ev, uint64_t v)
{
    ENTER; ORIG(H_CB_WAIT, void (*)(id, SEL, id, uint64_t))(self, _cmd, ev, v); LEAVE;
    if (OUTER) cb_item(self, 2, event_key(ev, 0), v);
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
    uint64_t capture_generation = 0;
    uint64_t uid = 0;
    fp_trace_cb *trace = NULL;
    if (OUTER) {
        char cb_label[sizeof ((fp_cb *)0)->label] = "";
        char queue_label[sizeof g_trace_queue_labels[0]] = "";
        @autoreleasepool {
            const char *label = [[self label] UTF8String];
            if (label) snprintf(cb_label, sizeof cb_label, "%s", label);
            if (g_trace_enabled) {
                label = [(id<MTLCommandBuffer>)self commandQueue].label.UTF8String;
                if (label) snprintf(queue_label, sizeof queue_label, "%s", label);
            }
        }
        pthread_mutex_lock(&g_lock);
        if (g_capture_busy) g_capture_last_commit = trace_clock();
        fp_cb *c = cb_find_open((void *)self);
        if (c && c->state == 1) {
            uid = c->uid;
            if (g_capture_available) {
                g_capture_commits++;
                capture_commit = 1;
                capture_generation = g_capture_generation;
            }
            memcpy(c->label, cb_label, sizeof c->label);
            if (g_trace_enabled && (trace = c->trace)) {
                trace->owner->record = NULL;
                memcpy(trace->label, c->label, sizeof trace->label);
                trace->completed_before = malloc(sizeof g_trace_completed);
                memcpy(trace->completed_before, g_trace_completed, sizeof g_trace_completed);
                if (!g_trace_tracks[c->q->idx][TR_CB].n && queue_label[0])
                    memcpy(g_trace_queue_labels[c->q->idx], queue_label, sizeof queue_label);
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
    }
    int returned = 0;
    @try {
        if (uid) {
            [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> b) {
                cb_done(uid, trace, b, capture_generation);
            }];
            capture_feedback_attached(capture_generation);
            maybe_summary();
        }
        ORIG(H_CB_COMMIT, void (*)(id, SEL))(self, _cmd);
        returned = 1;
    } @finally {
        if (!returned) capture_feedback_missing(capture_generation);
        if (capture_commit) capture_after_commit();
        LEAVE;
    }
}

typedef struct {
    fp_trace_cb *trace;
    fp_queue *queue;
    uint64_t signal_limit, wait_limit, frame;
#ifdef FENCEPOST_TEST_HOOKS
    int inject_error;
#endif
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

/* Without a feedback hook we cannot establish when an encoded write retires.
 * Keep those pairs reserved for the lifetime of the heap. */
static void trace_mtl4_abandon_samples(id cb)
{
    FPTraceOwner *owner = objc_getAssociatedObject(cb, &g_trace_owner_key);
    if (!owner || !owner->record) return;
    for (fp_trace_encoder *e = owner->record->encoders; e; e = e->next) e->sample = -1;
}

static void trace_mtl4_untracked_commit(const id *buffers, NSUInteger count)
{
    pthread_mutex_lock(&g_lock);
    for (NSUInteger i = 0; i < count; i++) trace_mtl4_abandon_samples(buffers[i]);
    pthread_mutex_unlock(&g_lock);
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
        submission->signal_limit = q->nrecent;
        submission->frame = g_frame_cur;
        if (!g_inflight && g_tick) { frame_get(g_frame_cur)->idle++; g_st[ST_GPU_IDLE]++; }
        uint64_t tick = ++g_tick;
        int nenc = 0;
        fp_node *suspended = NULL;
        if (g_ring) ring_poll_file();
        for (NSUInteger i = 0; i < count; i++) {
            id cb = buffers[i];
            fp_cb *c = cb_find_open((void *)cb);
            if (!c || !c->mtl4 || c->state != 2) {
                suspended = NULL;
                if (g_trace_enabled && g_trace_gpu) trace_mtl4_abandon_samples(cb);
                continue;
            }
            snprintf(c->label, sizeof c->label, "%s", labels[i]);
            c->q = q;
            c->commit_tick = tick;
            submission->cbs[submission->ncbs++] = c->uid;
            nenc += c->nenc;
            cb_close((void *)cb);
            trace_mtl4_join(&submission->trace, c);
            phase1(c, &suspended);
            if (!c->nenc) suspended = NULL;
        }
        trace_mtl4_submit(q, submission->trace);
        if (submission->ncbs) {
            g_inflight++;
            if (submission->trace) submission->frame = submission->trace->span.frame;
#ifdef FENCEPOST_TEST_HOOKS
            submission->inject_error = g_mtl4_test_error_commit && --g_mtl4_test_error_commit == 0;
#endif
        }
        if (nenc) submission->wait_limit = q->mtl4_wait_ticket;
        process_pending(0);
        if ((g_tick & 15) == 0) sweep();
        if (g_trace) trace_stream_flush();
    } else if (g_trace_enabled && g_trace_gpu) {
        for (NSUInteger i = 0; i < count; i++) trace_mtl4_abandon_samples(buffers[i]);
    }
    pthread_mutex_unlock(&g_lock);
    if (labels != local_labels) free(labels);
    if (submission->ncbs) return submission;
    free(submission);
    return NULL;
}

/* Successful completion proves that the queue passed every signal scheduled
 * before this submission. Earlier submissions cannot retire trailing signals. */
static void mtl4_compact_signals(fp_queue *q)
{
    if (q->mtl4_sig_first < (q->mtl4_nsignals + 1) / 2) return;
    q->mtl4_nsignals -= q->mtl4_sig_first;
    if (q->mtl4_nsignals)
        memmove(q->mtl4_signals, q->mtl4_signals + q->mtl4_sig_first,
                (size_t)q->mtl4_nsignals * sizeof *q->mtl4_signals);
    q->mtl4_sig_first = 0;
}

static void mtl4_retire_signals(fp_queue *q, uint64_t limit)
{
    while (q->mtl4_sig_first < q->mtl4_nsignals && q->mtl4_signals[q->mtl4_sig_first].seq < limit) {
        fp_node *n = node_get(q->mtl4_signals[q->mtl4_sig_first++].uid);
        if (n && !n->done_tick) {
            n->done_tick = g_tick;
#ifdef FENCEPOST_TEST_HOOKS
            g_mtl4_signals_retired++;
#endif
        }
    }
    mtl4_compact_signals(q);
}

/* Keep queue waits active for every submission until GPU progress confirms
 * they passed. Completion also proves the resolved signal's prefix finished,
 * even when feedback delivery on the signaling queue lags this callback. */
static void mtl4_retire_waits(fp_queue *q, uint64_t limit)
{
    int retired = 0;
    while (retired < q->mtl4_nwaits && q->mtl4_waits[retired].ticket <= limit) {
        fp_evwait wait = q->mtl4_waits[retired++];
        fp_ev *event = ev_get(wait.ev);
        if (event && event->cpu_max >= wait.val) continue;
        fp_node *signal = first_event_signal(event, q, wait, NULL);
        if (!signal || !signal->mtl4) continue;
        fp_queue *producer = signal->req_q;
        for (fp_node *p = signal; p && p->queue_seq >= producer->mtl4_event_completed_through;
             p = node_get(p->mtl4_previous)) {
            if (p->kind == FPK_SIGNAL) continue;   /* signal retirement has its own completion checkpoint */
            if (!p->done_tick) p->done_tick = g_tick;
        }
        producer->mtl4_event_completed_through = MAX(producer->mtl4_event_completed_through, signal->queue_seq + 1);
    }
    if (retired) {
        q->mtl4_nwaits -= retired;
        memmove(q->mtl4_waits, q->mtl4_waits + retired, (size_t)q->mtl4_nwaits * sizeof *q->mtl4_waits);
    }
}

API_AVAILABLE(macos(26.0))
static void mtl4_completed(fp_mtl4_submission *submission, id<MTL4CommitFeedback> feedback)
{
    fp_trace_cb *trace = submission->trace;
    double start, end;
    int failed;
    char *message = NULL;
    @autoreleasepool {
        NSError *error = feedback.error;
        start = feedback.GPUStartTime; end = feedback.GPUEndTime;
        failed = error != nil;
#ifdef FENCEPOST_TEST_HOOKS
        failed |= submission->inject_error;
#endif
        if (failed && g_ring && g_ring_gpuerr) {
            message = strdup(error ? (error.localizedDescription.UTF8String ?: "Metal 4 commit error")
                                   : "debug-only simulated Metal 4 feedback error");
        }
        if (trace && trace->device && trace->device->heap && !failed && start > 0 && end > start) {
            pthread_mutex_lock(&g_lock);
            int sample = !g_trace_stopped;
            pthread_mutex_unlock(&g_lock);
            if (sample) trace_read_samples(trace);
        }
    }
    pthread_mutex_lock(&g_lock);
    g_inflight--;
    for (NSUInteger i = 0; i < submission->ncbs; i++) cb_mark_done(submission->cbs[i]);
    if (failed) {
        g_mtl4_gpu_errors++;
        if (g_ring && g_ring_gpuerr)
            ring_trigger_impl(DUMP_AUTO, "gpu-error", "gpu-error", submission->frame, trace_clock(), message, 1);
    } else {
        mtl4_retire_signals(submission->queue, submission->signal_limit);
        mtl4_retire_waits(submission->queue, submission->wait_limit);
    }
    if (trace) {
        if (!failed && start > 0 && end >= start && end <= (double)trace_clock() / 1e9 + 1) {
            trace->gpu_start = (uint64_t)(start * 1e9);
            trace->gpu_end = (uint64_t)(end * 1e9);
        }
        for (fp_trace_encoder *e = trace->encoders; e; e = e->next) {
            if (e->sample < 0) continue;
            if (e->selected && !g_trace_stopped) trace_mtl4_resolve(e);
            trace_release_sample(e);
        }
        trace->completed = 1;
        trace_cb_finish(trace);
    }
    if (g_ring) ring_update(0);
    if (g_trace) trace_stream_flush();
    pthread_mutex_unlock(&g_lock);
    free(message);
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

typedef struct {
    FPCaptureBufferOwner **owners;
    NSUInteger count;
} fp_capture_submit;

static int capture_mtl4_submit(const id *buffers, NSUInteger count, int outer, uint64_t *generation,
                               fp_capture_submit *submit)
{
    if (!g_capture_available || !buffers || !count) return 0;
    pthread_mutex_lock(&g_lock);
    int tracked = outer;
    if (g_capture_mtl4_ended) {
        submit->owners = calloc(MIN(count, g_capture_mtl4_ended), sizeof *submit->owners);
        /* Allocation failure leaves all ended identities registered. */
        for (NSUInteger i = 0; submit->owners && i < count; i++) {
            FPCaptureBufferOwner *owner = objc_getAssociatedObject(buffers[i], &g_capture_buffer_key);
            if (!owner || !atomic_exchange_explicit(&owner->ended, 0, memory_order_relaxed)) continue;
            submit->owners[submit->count++] = [owner retain];
            g_capture_mtl4_ended--;
            tracked = 1;
        }
    }
    if (tracked) {
        g_capture_mtl4 = 1;
        g_capture_commits++;
        *generation = g_capture_generation;
    }
    pthread_mutex_unlock(&g_lock);
    return tracked;
}

static void capture_mtl4_submit_done(fp_capture_submit *submit, int tracked, int returned)
{
    /* Restore only the identities this call removed, before unblocking starts.
     * Keep any newer end of a reused buffer registered as well. */
    if (!returned && submit->count) {
        pthread_mutex_lock(&g_lock);
        if (!g_capture_mtl4_failed)
            fp_log("fencepost: Metal 4 commit failed; capture gate held until pending buffers drain\n");
        g_capture_mtl4_failed = 1;
        for (NSUInteger i = 0; i < submit->count; i++)
            g_capture_mtl4_ended += !atomic_exchange_explicit(&submit->owners[i]->ended, 1, memory_order_relaxed);
        pthread_mutex_unlock(&g_lock);
    }
    for (NSUInteger i = 0; i < submit->count; i++) [submit->owners[i] release];
    free(submit->owners);
    if (tracked) capture_after_commit();
}

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
static MTL4CommitOptions *mtl4_submit_options(id queue, const id *buffers, NSUInteger count, id app_options,
                                             uint64_t capture_generation)
{
    FPMTL4Callbacks *callbacks = app_options ? objc_getAssociatedObject(app_options, &g_mtl4_callbacks_key) : nil;
    if (app_options && !callbacks) return nil;
    MTL4CommitOptions *opts = [[g_mtl4_options_class alloc] init];
    if (!opts) return nil;

    /* Metal consumes the app's registrations once per submission. Take that
     * list under a short lock and replay it on private, per-commit options.
     * Native submission can block and must never hold a fencepost lock. */
    NSArray *blocks = nil;
    if (callbacks) {
        pthread_mutex_lock(&g_mtl4_options_lock);
        blocks = callbacks->blocks;
        callbacks->blocks = nil;
        pthread_mutex_unlock(&g_mtl4_options_lock);
    }
    fp_mtl4_submission *submission = mtl4_record_commit(queue, buffers, count);
    /* Record completion before an app handler can submit dependent work. */
    if (submission || capture_generation || (g_capture_on & CAP_GPU_ERROR))
        [opts addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            if (submission) mtl4_completed(submission, feedback);
            if (capture_generation || (g_capture_on & CAP_GPU_ERROR))
                capture_feedback_done(capture_generation, feedback.error != nil);
        }];
    if (!submission && g_trace_enabled && g_trace_gpu) trace_mtl4_untracked_commit(buffers, count);
    for (MTL4CommitFeedbackHandler handler in blocks) [opts addFeedbackHandler:handler];
    [blocks release];
    capture_feedback_attached(capture_generation);
    return opts;
}

static void h_mtl4_commit(id self, SEL _cmd, const id *buffers, NSUInteger count)
{
    ENTER;
    uint64_t generation = 0;
    fp_capture_submit submit = {0};
    int tracked = capture_mtl4_submit(buffers, count, OUTER, &generation, &submit), returned = 0;
    id opts = nil;
    @try {
        if (OUTER) {
            if (@available(macOS 26.0, *)) {
                if (count && [self respondsToSelector:@selector(commit:count:options:)])
                    opts = mtl4_submit_options(self, buffers, count, nil, generation);
                if (opts)
                    [(id<MTL4CommandQueue>)self commit:(const id<MTL4CommandBuffer> *)buffers count:count options:opts];
            }
        }
        if (!opts) {
            if (OUTER && g_trace_enabled && g_trace_gpu) trace_mtl4_untracked_commit(buffers, count);
            ORIG(H_4_COMMIT, void (*)(id, SEL, const id *, NSUInteger))(self, _cmd, buffers, count);
        }
        returned = 1;
    } @finally {
        if (!opts || !returned) capture_feedback_missing(generation);
        [opts release];
        capture_mtl4_submit_done(&submit, tracked, returned);
        LEAVE;
    }
    if (opts) maybe_summary();
}

static void h_mtl4_commit_opts(id self, SEL _cmd, const id *buffers, NSUInteger count, id options)
{
    ENTER;
    uint64_t generation = 0;
    fp_capture_submit submit = {0};
    int tracked = capture_mtl4_submit(buffers, count, OUTER, &generation, &submit), returned = 0;
    id opts = nil;
    @try {
        if (OUTER && (options || generation)) {
            if (@available(macOS 26.0, *)) opts = mtl4_submit_options(self, buffers, count, options, generation);
        }
        if (OUTER && !opts && g_trace_enabled && g_trace_gpu) trace_mtl4_untracked_commit(buffers, count);
        ORIG(H_4_COMMIT_OPTS, void (*)(id, SEL, const id *, NSUInteger, id))(self, _cmd, buffers, count, opts ?: options);
        returned = 1;
    } @finally {
        if (!opts || !returned) capture_feedback_missing(generation);
        [opts release];
        capture_mtl4_submit_done(&submit, tracked, returned);
        LEAVE;
    }
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
    trace_archive(uid, &span);
}

static void mtl4_record_event(id queue, fp_item item)
{
    pthread_mutex_lock(&g_lock);
    fp_queue *q = queue_get_kind((void *)queue, 1);
    if (q && item.type == 1) {
        fp_node *n = phase_signal(q, 0, ++g_tick, &item);
        if (!n) {
            pthread_mutex_unlock(&g_lock);
            return;
        }
        n->mtl4 = 1;
        phase_node(q, n, g_tick, NULL, q->mtl4_waits, q->mtl4_nwaits);
        while (q->mtl4_sig_first < q->mtl4_nsignals && !node_get(q->mtl4_signals[q->mtl4_sig_first].uid))
            q->mtl4_sig_first++;
        mtl4_compact_signals(q);
        fp_queue_signal signal = { n->uid, n->queue_seq };
        PUSH(q->mtl4_signals, q->mtl4_nsignals, q->mtl4_csignals, signal);
        trace_mtl4_signal(q, n->uid, item.val);
        process_pending(0);
    } else if (q) {
        fp_evwait wait = { item.ev, item.val, q->nrecent, 0, ++q->mtl4_wait_ticket };
        PUSH(q->mtl4_waits, q->mtl4_nwaits, q->mtl4_cwaits, wait);
    }
    pthread_mutex_unlock(&g_lock);
}

/* Record queue events in hook entry order, before entering Metal. If an app
 * submits to the same queue from two threads without serializing its calls,
 * this order may differ from Metal's submission order. */
static void h_mtl4_signal(id self, SEL _cmd, id event, uint64_t value)
{
    ENTER;
    if (OUTER) mtl4_record_event(self, (fp_item){ 1, 0, event_key(event, 0), value });
    ORIG(H_4_SIGNAL, void (*)(id, SEL, id, uint64_t))(self, _cmd, event, value);
    LEAVE;
}

static void h_mtl4_wait(id self, SEL _cmd, id event, uint64_t value)
{
    ENTER;
    if (OUTER) mtl4_record_event(self, (fp_item){ 2, 0, event_key(event, 0), value });
    ORIG(H_4_WAIT, void (*)(id, SEL, id, uint64_t))(self, _cmd, event, value);
    LEAVE;
}

static void h_mtl4_drawable_signal(id self, SEL _cmd, id drawable)
{
    ENTER;
    int capture_boundary = 0;
    if (OUTER) {
        pthread_mutex_lock(&g_lock);
        if (!g_frame_api) {
            if (g_capture_available) {
                g_capture_commits++;
                capture_boundary = 1;
            }
            frame_boundary(g_trace_enabled ? trace_clock() : 0, NULL);
            process_pending(0);
        }
        pthread_mutex_unlock(&g_lock);
    }
    @try {
        ORIG(H_4_DRAWABLE_SIGNAL, void (*)(id, SEL, id))(self, _cmd, drawable);
    } @finally {
        if (capture_boundary) capture_after_commit();
        LEAVE;
    }
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
    FPEventOwner *owner = objc_getAssociatedObject(self, &g_event_owner_key);
    if (!owner) return;
    pthread_mutex_lock(&g_lock);
    if (v > owner->history.cpu_max) { owner->history.cpu_max = v; process_pending(0); }
    pthread_mutex_unlock(&g_lock);
}

/* every encoder */

API_AVAILABLE(macos(26.0))
static void trace_mtl4_end_encoder(id enc, const fp_node *n)
{
    pthread_mutex_lock(&g_lock);
    fp_trace_encoder *e = trace_live_encoder(n->uid);
    pthread_mutex_unlock(&g_lock);
    if (!e || e->sample < 0) return;

    int work = n->mtl4_drawn;
    if (n->kind == FPK_RENDER) {
        work |= n->mtl4_has_attachment;
    } else {
        work = [(id<MTL4ComputeCommandEncoder>)enc stages] != 0;
    }
    if (work) {
        trace_mtl4_stamp(enc, e, 1);
        e->sample_ended = 1;
    }
}

static void h_end(id self, SEL _cmd)
{
    ENTER;
    fp_node *n = NULL;
    if (OUTER && g_trace_enabled && g_trace_gpu) {
        n = cur_node(self);
        if (n && n->mtl4) {
            if (@available(macOS 26.0, *)) trace_mtl4_end_encoder(self, n);
        }
    }
    ORIG(H_END, void (*)(id, SEL))(self, _cmd);
    LEAVE;
    if (!OUTER) return;
    if (!n) n = cur_node(self);
    if (!n) return;
    for (int i = 0; i < n->ndeferred_att; i++) {
        fp_deferred_att *deferred = &n->deferred_att[i];
        record_attachment(n, deferred->desc.texture, deferred->desc, deferred->layers, deferred->action);
    }
    clear_deferred_attachments(n);
    if (n->mtl4) {
        if (!n->mtl4_drawn && n->kind == FPK_RENDER) fp4_note_attachment_loads(n);
        if (n->kind == FPK_RENDER) n->mtl4_epoch++;   /* distinguish stores from the last ordinary commands */
        for (int i = 0, count = n->nacc; i < count; i++) {
            fp_acc *access = &n->acc[i];
            if (access->via != VIA_ATT && access->via != VIA_ATT_LOAD && access->via != VIA_RESOLVE) continue;
            if (!(access->rw & FP_WR)) continue;
            n->mtl4_store_epoch = n->mtl4_epoch;
            fp4_note_work(n, FP4_FRAGMENT);
            /* The store happens at the end of the pass (see fp4_record_sync). */
            if (access->via == VIA_ATT_LOAD) {
                fp_acc store = *access;
                access->rw &= ~FP_WR;
                store.rw = FP_WR;
                store.via = VIA_ATT;
                store.first_epoch = store.last_epoch = n->mtl4_epoch;
                push_acc(n, &store);
            } else access->first_epoch = access->last_epoch = n->mtl4_epoch;
        }
        for (int i = 0; i < FP4_TABLES; i++) {
            [n->mtl4_tables[i] release];
            n->mtl4_tables[i] = nil;
        }
        [n->mtl4_pipeline release];
        n->mtl4_pipeline = nil;
        n->mtl4_pipeline_object = NULL;
    }
    @autoreleasepool {
        NSString *label = [self label];
        if (n->mtl4)
            snprintf(n->label, sizeof n->label, "MTL4 %s", label.length ? label.UTF8String : kind_name[n->kind]);
        else if (label) snprintf(n->label, sizeof n->label, "%s", label.UTF8String);
    }
    pthread_mutex_lock(&g_lock);
    if (g_trace_enabled) trace_encoder_end(n);
    n->state = 2;
    atomic_store_explicit(&n->open_uid, 0, memory_order_release);
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
static const hook_def mtl4_device_hooks[] API_AVAILABLE(macos(26.0)) = {
    HD(H_4_DEV_Q, "newMTL4CommandQueue", h_mtl4_dev_q),
    HD(H_4_DEV_Q_DESC, "newMTL4CommandQueueWithDescriptor:error:", h_mtl4_dev_q_desc),
    HD(H_4_DEV_CB, "newCommandBuffer", h_mtl4_dev_cb),
    HD(H_4_ARGUMENT_TABLE, "newArgumentTableWithDescriptor:error:", h_mtl4_argument_table),
    HD(H_4_NEW_HEAP, "newHeapWithDescriptor:", h_mtl4_new_heap),
    HD(H_4_NEW_BUFFER, "newBufferWithLength:options:", h_mtl4_new_buffer),
    HD(H_4_NEW_BYTES, "newBufferWithBytes:length:options:", h_mtl4_new_bytes),
    HD(H_4_NEW_BYTES_NOCOPY, "newBufferWithBytesNoCopy:length:options:deallocator:", h_mtl4_new_bytes_nocopy),
    HD(H_4_NEW_SPARSE_BUFFER, "newBufferWithLength:options:placementSparsePageSize:", h_mtl4_sparse_buffer),
    HD(H_4_NEW_TEXTURE, "newTextureWithDescriptor:", h_mtl4_new_texture),
    HD(H_4_NEW_IOSURFACE, "newTextureWithDescriptor:iosurface:plane:", h_mtl4_iosurface),
    HD(H_4_SHARED_TEXTURE, "newSharedTextureWithDescriptor:", h_mtl4_shared_texture),
    HD(H_4_SHARED_HANDLE, "newSharedTextureWithHandle:", h_mtl4_shared_handle),
    HD(H_4_NEW_COMPILER, "newCompilerWithDescriptor:error:", h_mtl4_new_compiler),
    HD(H_P_CFN, "newComputePipelineStateWithFunction:error:", h_p_cfn),
    HD(H_P_CFN_R, "newComputePipelineStateWithFunction:options:reflection:error:", h_p_cfn_r),
    HD(H_P_CFN_A, "newComputePipelineStateWithFunction:completionHandler:", h_p_cfn_a),
    HD(H_P_CFN_AR, "newComputePipelineStateWithFunction:options:completionHandler:", h_p_cfn_ar),
    HD(H_P_CDESC_R, "newComputePipelineStateWithDescriptor:options:reflection:error:", h_p_cdesc_r),
    HD(H_P_CDESC_AR, "newComputePipelineStateWithDescriptor:options:completionHandler:", h_p_cdesc_ar),
    HD(H_P_RDESC, "newRenderPipelineStateWithDescriptor:error:", h_p_rdesc),
    HD(H_P_RDESC_R, "newRenderPipelineStateWithDescriptor:options:reflection:error:", h_p_rdesc_r),
    HD(H_P_RDESC_A, "newRenderPipelineStateWithDescriptor:completionHandler:", h_p_rdesc_a),
    HD(H_P_RDESC_AR, "newRenderPipelineStateWithDescriptor:options:completionHandler:", h_p_rdesc_ar),
    HD(H_P_TILE_R, "newRenderPipelineStateWithTileDescriptor:options:reflection:error:", h_p_tile_r),
    HD(H_P_TILE_AR, "newRenderPipelineStateWithTileDescriptor:options:completionHandler:", h_p_tile_ar),
    HD(H_P_MESH_R, "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:", h_p_mesh_r),
    HD(H_P_MESH_AR, "newRenderPipelineStateWithMeshDescriptor:options:completionHandler:", h_p_mesh_ar),
};
API_AVAILABLE(macos(26.0))
static const hook_def mtl4_compiler_hooks[] = {
    HD(H_C4_COMPUTE, "newComputePipelineStateWithDescriptor:compilerTaskOptions:error:", h_c4_compute),
    HD(H_C4_COMPUTE_LINK, "newComputePipelineStateWithDescriptor:dynamicLinkingDescriptor:compilerTaskOptions:error:",
       h_c4_compute_link),
    HD(H_C4_RENDER, "newRenderPipelineStateWithDescriptor:compilerTaskOptions:error:", h_c4_render),
    HD(H_C4_RENDER_LINK, "newRenderPipelineStateWithDescriptor:dynamicLinkingDescriptor:compilerTaskOptions:error:",
       h_c4_render_link),
    HD(H_C4_SPECIALIZE, "newRenderPipelineStateBySpecializationWithDescriptor:pipeline:error:", h_c4_specialize),
    HD(H_C4_COMPUTE_A, "newComputePipelineStateWithDescriptor:compilerTaskOptions:completionHandler:", h_c4_compute_a),
    HD(H_C4_COMPUTE_LINK_A,
       "newComputePipelineStateWithDescriptor:dynamicLinkingDescriptor:compilerTaskOptions:completionHandler:",
       h_c4_compute_link_a),
    HD(H_C4_RENDER_A, "newRenderPipelineStateWithDescriptor:compilerTaskOptions:completionHandler:", h_c4_render_a),
    HD(H_C4_RENDER_LINK_A,
       "newRenderPipelineStateWithDescriptor:dynamicLinkingDescriptor:compilerTaskOptions:completionHandler:",
       h_c4_render_link_a),
    HD(H_C4_SPECIALIZE_A, "newRenderPipelineStateBySpecializationWithDescriptor:pipeline:completionHandler:",
       h_c4_specialize_a),
};
static const hook_def mtl4_heap_hooks[] = {
    HD(H_4_NEW_BUFFER, "newBufferWithLength:options:", h_mtl4_new_buffer),
    HD(H_4_HEAP_BUFFER_OFFSET, "newBufferWithLength:options:offset:", h_mtl4_heap_buffer_offset),
    HD(H_4_NEW_TEXTURE, "newTextureWithDescriptor:", h_mtl4_new_texture),
    HD(H_4_HEAP_TEXTURE_OFFSET, "newTextureWithDescriptor:offset:", h_mtl4_heap_texture_offset),
};
static const hook_def mtl4_buffer_hooks[] = {
    HD(H_4_BUFFER_TEXTURE, "newTextureWithDescriptor:offset:bytesPerRow:", h_mtl4_buffer_texture),
};
static const hook_def mtl4_texture_hooks[] = {
    HD(H_4_VIEW_FORMAT, "newTextureViewWithPixelFormat:", h_mtl4_view_format),
    HD(H_4_VIEW_RANGE, "newTextureViewWithPixelFormat:textureType:levels:slices:", h_mtl4_view_range),
    HD(H_4_VIEW_SWIZZLE, "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:", h_mtl4_view_swizzle),
    HD(H_4_VIEW_DESC, "newTextureViewWithDescriptor:", h_mtl4_view_desc),
};
static const hook_def mtl4_cb_hooks[] API_AVAILABLE(macos(26.0)) = {
    HD(H_4_BEGIN, "beginCommandBufferWithAllocator:", h_mtl4_begin),
    HD(H_4_BEGIN_OPTS, "beginCommandBufferWithAllocator:options:", h_mtl4_begin_opts),
    HD(H_4_END, "endCommandBuffer", h_mtl4_end),
    HD(H_4_RENDER, "renderCommandEncoderWithDescriptor:", h_mtl4_render),
    HD(H_4_RENDER_OPTS, "renderCommandEncoderWithDescriptor:options:", h_mtl4_render_opts),
    HD(H_4_COMPUTE, "computeCommandEncoder", h_mtl4_compute),
    HD(H_4_ML, "machineLearningCommandEncoder", h_mtl4_ml),
    HD(H_4_RES_USE, "useResidencySet:", h_mtl4_res_use),
    HD(H_4_RES_USES, "useResidencySets:count:", h_mtl4_res_uses),
};
static const hook_def mtl4_argument_hooks[] = {
    HD(H_4_ADDRESS, "setAddress:atIndex:", h_mtl4_address),
    HD(H_4_STRIDE, "setAddress:attributeStride:atIndex:", h_mtl4_stride),
    HD(H_4_RESOURCE, "setResource:atBufferIndex:", h_mtl4_resource),
    HD(H_4_TEXTURE, "setTexture:atIndex:", h_mtl4_texture),
};
static const hook_def mtl4_render_hooks[] = {
    HD(H_END, "endEncoding", h_end),
    HD(H_4_CONSUMER_BARRIER, "barrierAfterQueueStages:beforeStages:visibilityOptions:", h_mtl4_consumer_barrier),
    HD(H_4_PRODUCER_BARRIER, "barrierAfterStages:beforeQueueStages:visibilityOptions:", h_mtl4_producer_barrier),
    HD(H_4_INTRA_BARRIER, "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", h_mtl4_intra_barrier),
    HD(H_4_FENCE_UPDATE, "updateFence:afterEncoderStages:", h_mtl4_fence_update),
    HD(H_4_FENCE_WAIT, "waitForFence:beforeEncoderStages:", h_mtl4_fence_wait),
    HD(H_R_CSA, "setColorStoreAction:atIndex:", h_r_csa),
    HD(H_R_DSA, "setDepthStoreAction:", h_r_dsa),
    HD(H_R_SSA, "setStencilStoreAction:", h_r_ssa),
    HD(H_4_BIND_RENDER, "setArgumentTable:atStages:", h_mtl4_bind_render),
    HD(H_4_SET_RENDER_PIPELINE, "setRenderPipelineState:", h_mtl4_set_render_pipeline),
    HD(H_4_DRAW, "drawPrimitives:vertexStart:vertexCount:", h_mtl4_draw),
    HD(H_4_DRAW_I, "drawPrimitives:vertexStart:vertexCount:instanceCount:", h_mtl4_draw_i),
    HD(H_4_DRAW_B, "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:", h_mtl4_draw_b),
    HD(H_4_DRAW_IDX, "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferLength:", h_mtl4_draw_idx),
    HD(H_4_DRAW_IDX_I, "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferLength:instanceCount:", h_mtl4_draw_idx_i),
    HD(H_4_DRAW_IDX_B, "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferLength:instanceCount:baseVertex:baseInstance:", h_mtl4_draw_idx_b),
    HD(H_4_DRAW_IND, "drawPrimitives:indirectBuffer:", h_mtl4_draw_ind),
    HD(H_4_DRAW_IDX_IND, "drawIndexedPrimitives:indexType:indexBuffer:indexBufferLength:indirectBuffer:", h_mtl4_draw_idx_ind),
    HD(H_4_DRAW_MESH_GROUPS, "drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:", h_mtl4_draw_mesh_groups),
    HD(H_4_DRAW_MESH, "drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:", h_mtl4_draw_mesh),
    HD(H_4_DRAW_MESH_IND, "drawMeshThreadgroupsWithIndirectBuffer:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:", h_mtl4_draw_mesh_ind),
    HD(H_4_DRAW_TILE, "dispatchThreadsPerTile:", h_mtl4_draw_tile),
};
static const hook_def mtl4_compute_hooks[] API_AVAILABLE(macos(26.0)) = {
    HD(H_END, "endEncoding", h_end),
    HD(H_4_CONSUMER_BARRIER, "barrierAfterQueueStages:beforeStages:visibilityOptions:", h_mtl4_consumer_barrier),
    HD(H_4_PRODUCER_BARRIER, "barrierAfterStages:beforeQueueStages:visibilityOptions:", h_mtl4_producer_barrier),
    HD(H_4_INTRA_BARRIER, "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", h_mtl4_intra_barrier),
    HD(H_4_FENCE_UPDATE, "updateFence:afterEncoderStages:", h_mtl4_fence_update),
    HD(H_4_FENCE_WAIT, "waitForFence:beforeEncoderStages:", h_mtl4_fence_wait),
    HD(H_4_BIND_COMPUTE, "setArgumentTable:", h_mtl4_bind_compute),
    HD(H_4_SET_COMPUTE_PIPELINE, "setComputePipelineState:", h_mtl4_set_compute_pipeline),
    HD(H_4_ACCEL_BUILD, "buildAccelerationStructure:descriptor:scratchBuffer:", h_mtl4_accel_build),
    HD(H_4_DISPATCH, "dispatchThreads:threadsPerThreadgroup:", h_mtl4_dispatch),
    HD(H_4_DISPATCH_GROUPS, "dispatchThreadgroups:threadsPerThreadgroup:", h_mtl4_dispatch_groups),
    HD(H_4_DISPATCH_IND, "dispatchThreadsWithIndirectBuffer:", h_mtl4_dispatch_ind),
    HD(H_4_DISPATCH_GROUPS_IND, "dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:", h_mtl4_dispatch_groups_ind),
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
};
static const hook_def mtl4_queue_hooks[] API_AVAILABLE(macos(26.0)) = {
    HD(H_4_COMMIT, "commit:count:", h_mtl4_commit),
    HD(H_4_COMMIT_OPTS, "commit:count:options:", h_mtl4_commit_opts),
    HD(H_4_SIGNAL, "signalEvent:value:", h_mtl4_signal),
    HD(H_4_WAIT, "waitForEvent:value:", h_mtl4_wait),
    HD(H_4_DRAWABLE_SIGNAL, "signalDrawable:", h_mtl4_drawable_signal),
    HD(H_4_DRAWABLE_WAIT, "waitForDrawable:", h_mtl4_drawable_wait),
    HD(H_4_RES_ADD, "addResidencySet:", h_mtl4_res_add),
    HD(H_4_RES_ADDS, "addResidencySets:count:", h_mtl4_res_adds),
    HD(H_4_RES_REMOVE, "removeResidencySet:", h_mtl4_res_remove),
    HD(H_4_RES_REMOVES, "removeResidencySets:count:", h_mtl4_res_removes),
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
    if (@available(macOS 26.0, *)) {
        install(cls, 104, mtl4_device_hooks, N(mtl4_device_hooks));
        if (g_capture_available) {
            Method factory = class_getInstanceMethod(cls, sel_registerName("newCommandBuffer"));
            if (factory && method_getImplementation(factory) != (IMP)h_mtl4_dev_cb &&
                capture_mtl4_tracking_lost("Metal 4 command buffer factory hook unavailable"))
                capture_interrupt_start("Metal 4 command buffer tracking lost during capture start");
        }
    }
}
static void hook_queue(Class cls) { install(cls, 101, queue_hooks, N(queue_hooks)); }
static void hook_cb(Class cls) { install(cls, 102, cb_hooks, N(cb_hooks)); }
static void hook_event(Class cls) { install(cls, 103, event_hooks, N(event_hooks)); }
static void hook_mtl4_cb(Class cls)
{
    if (@available(macOS 26.0, *)) install(cls, 105, mtl4_cb_hooks, N(mtl4_cb_hooks));
}
static void hook_mtl4_encoder(Class cls, int kind)
{
    if (kind == FPK_RENDER) install(cls, 106, mtl4_render_hooks, N(mtl4_render_hooks));
    else if (kind == FPK_ML) {
        static const hook_def ml_hooks[] = { HD(H_END, "endEncoding", h_end) };
        install(cls, 114, ml_hooks, N(ml_hooks));
    }
    else if (@available(macOS 26.0, *)) install(cls, 112, mtl4_compute_hooks, N(mtl4_compute_hooks));
}
static void hook_mtl4_argument(Class cls) { install(cls, 111, mtl4_argument_hooks, N(mtl4_argument_hooks)); }
static void hook_mtl4_compiler(Class cls)
{
    if (@available(macOS 26.0, *)) install(cls, 113, mtl4_compiler_hooks, N(mtl4_compiler_hooks));
}
static void hook_mtl4_queue(Class cls)
{
    if (@available(macOS 26.0, *)) install(cls, 107, mtl4_queue_hooks, N(mtl4_queue_hooks));
}
static void hook_mtl4_heap(Class cls) { install(cls, 108, mtl4_heap_hooks, N(mtl4_heap_hooks)); }
static void hook_mtl4_buffer(Class cls) { install(cls, 109, mtl4_buffer_hooks, N(mtl4_buffer_hooks)); }
static void hook_mtl4_texture(Class cls) { install(cls, 110, mtl4_texture_hooks, N(mtl4_texture_hooks)); }
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
    if (g_capture_reporter) dispatch_sync(g_capture_reporter, ^{});
    size_t length = 0;
    tl_log_final = open_memstream(&g_log_final, &length);
    if (g_st[ST_NODES] || g_mtl4_gpu_errors ||
        g_st[ST_ABANDONED_ENCODERS] || g_st[ST_NODE_POOL_EXHAUSTED]) summary(1);
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
    if (tl_log_final) {
        fclose(tl_log_final);
        tl_log_final = NULL;
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
        g_capture_reporter = dispatch_queue_create("fencepost.capture-reporter", DISPATCH_QUEUE_SERIAL);
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
    g_trace_history = calloc(FP_POOL, sizeof *g_trace_history);
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
    const char *p = getenv("FENCEPOST_LOG");
    if (p && *p) g_out = fp_open_path(p, "a");
    if (g_out) g_log_fd = fileno(g_out);
    g_log_queue = dispatch_queue_create("fencepost.log", DISPATCH_QUEUE_SERIAL);
    g_log_group = dispatch_group_create();
    atexit(fp_log_finish);

#ifdef FENCEPOST_TEST_HOOKS
    g_mtl4_test_error_commit = trace_unsigned("FENCEPOST_MTL4_TEST_ERROR_COMMIT", 0);
    g_mtl4_test_reflection_fail = trace_unsigned("FENCEPOST_MTL4_TEST_REFLECTION_FAIL", 0);
#endif
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
    if ((p = getenv("FENCEPOST_CAPTURE")) && *p) capture_init(p);
    if ((p = getenv("FENCEPOST_TRACE")) && *p) trace_init(p);
    if ((p = getenv("FENCEPOST_MAX_REPORTS"))) g_max_reports = atol(p);
    if ((p = getenv("FENCEPOST_POSSIBLE"))) g_show_possible = atoi(p);
    if ((p = getenv("FENCEPOST_SUMMARY_SECS"))) g_summary_secs = atoi(p);
    if ((p = getenv("FENCEPOST_BIND_MAYWRITE"))) g_bind_maywrite = atoi(p);
    if ((p = getenv("FENCEPOST_MTL4_RW"))) g_mtl4_rw_possible = !strcmp(p, "possible");
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
/* A detector-only fixture lets the later signal settle first without
 * encoding a GPU wait that depends on an unsubmitted signal. */
__attribute__((visibility("default"))) int fencepost_test_wait_source(void)
{
    NSObject *object = [[NSObject alloc] init];
    void *key = event_key(object, 1);
    pthread_mutex_lock(&g_lock);
    static fp_queue producer, consumer;
    fp_node *first = node_new(FPK_SIGNAL), *second = node_new(FPK_SIGNAL), *waiter = node_new(FPK_COMPUTE);
    int passed = 0;
    if (!first || !second || !waiter) goto done;
    first->req_q = second->req_q = &producer;
    second->aseq = 1;
    waiter->req_q = &consumer;
    fp_ev *event = ev_get(key);
    record_event_signal(event, 1, first->uid);
    record_event_signal(event, 2, second->uid);
    fp_evwait wait = { key, 1, 0, 0, 0 };
    PUSH(waiter->ew, waiter->nev, waiter->cev, wait);
    int pending = !ready(waiter, 0) && !waiter->ned;
    first->aseq = 2;
    int selected = ready(waiter, 0) && waiter->ned == 1 && waiter->ed[0].src == first->uid;
    passed = pending && selected;
done:
    if (first) first->state = 0;
    if (second) second->state = 0;
    if (waiter) waiter->state = 0;
    pthread_mutex_unlock(&g_lock);
    [object release];
    return passed;
}

/* The earliest satisfying signal's node was recycled: the wait settles with no edge,
 * not with an edge to the later, analysed signal. */
__attribute__((visibility("default"))) int fencepost_test_recycled_signal(void)
{
    NSObject *object = [[NSObject alloc] init];
    void *key = event_key(object, 1);
    pthread_mutex_lock(&g_lock);
    static fp_queue producer, consumer;
    fp_node *later = node_new(FPK_SIGNAL), *waiter = node_new(FPK_COMPUTE);
    int settled = 0;
    if (!later || !waiter) goto done;
    later->req_q = &producer;
    later->aseq = 1;
    waiter->req_q = &consumer;
    uint64_t recycled = later->uid > FP_POOL ? later->uid - FP_POOL : later->uid + FP_POOL;
    fp_ev *event = ev_get(key);
    record_event_signal(event, 1, recycled);
    record_event_signal(event, 2, later->uid);
    fp_evwait wait = { key, 1, 0, 0, 0 };
    PUSH(waiter->ew, waiter->nev, waiter->cev, wait);
    long before = g_st[ST_EVENT_UNKNOWN_SIGNALS];
    settled = !node_get(recycled) && ready(waiter, 0) && !waiter->ned && !waiter->nev &&
        g_st[ST_EVENT_UNKNOWN_SIGNALS] == before + 1;
done:
    if (later) later->state = 0;
    if (waiter) waiter->state = 0;
    pthread_mutex_unlock(&g_lock);
    [object release];
    return settled;
}

__attribute__((visibility("default"))) int fencepost_test_rotated_signal(void)
{
    NSObject *object = [[NSObject alloc] init];
    void *key = event_key(object, 1);
    pthread_mutex_lock(&g_lock);
    static fp_queue producer, consumer;
    fp_node *first = node_new(FPK_SIGNAL), *later = node_new(FPK_SIGNAL), *waiter = node_new(FPK_COMPUTE);
    int passed = 0;
    if (!first || !later || !waiter) goto done;
    first->req_q = later->req_q = &producer;
    first->aseq = 1;
    later->aseq = 2;
    waiter->req_q = &consumer;
    fp_ev *event = ev_get(key);
    record_event_signal(event, 2, first->uid);
    record_event_signal(event, 1, first->uid);
    for (unsigned i = 0; i < FP_EVHIST; i++) record_event_signal(event, 3, later->uid);
    long before = g_st[ST_EVENT_UNKNOWN_SIGNALS];
    fp_evwait wait = { key, 2, 0, 0, 0 };
    PUSH(waiter->ew, waiter->nev, waiter->cev, wait);
    int unknown = ready(waiter, 0) && !waiter->ned && !waiter->nev &&
        g_st[ST_EVENT_UNKNOWN_SIGNALS] == before + 1;
    wait.val = 3;
    PUSH(waiter->ew, waiter->nev, waiter->cev, wait);
    int selected = ready(waiter, 0) && waiter->ned == 1 && waiter->ed[0].src == later->uid;
    passed = unknown && selected;
done:
    if (first) first->state = 0;
    if (later) later->state = 0;
    if (waiter) waiter->state = 0;
    pthread_mutex_unlock(&g_lock);
    [object release];
    return passed;
}

__attribute__((visibility("default"))) int fencepost_test_event_lifetime(void)
{
    NSObject *old = [[NSObject alloc] init];
    void *expired = event_key(old, 1);
    [old release];
    NSObject *replacement = [[NSObject alloc] init];
    void *fresh = event_key(replacement, 1);

    pthread_mutex_lock(&g_lock);
    long before = g_st[ST_EVENT_LOOKUP_MISSES];
    fp_node *waiter = node_new(FPK_COMPUTE);
    int settled = 0;
    if (waiter) {
        fp_evwait wait = { expired, 1, 0, 0, 0 };
        PUSH(waiter->ew, waiter->nev, waiter->cev, wait);
        settled = expired && fresh && fresh != expired && ev_get(fresh) && !ev_get(expired) &&
            ready(waiter, 0) && !waiter->ned && !waiter->nev && g_st[ST_EVENT_LOOKUP_MISSES] == before + 1;
        waiter->state = 0;
    }
    pthread_mutex_unlock(&g_lock);
    [replacement release];
    return settled;
}

__attribute__((visibility("default"))) long fencepost_test_trace_cbs(long *peak)
{
    pthread_mutex_lock(&g_lock);
    long live = g_mtl4_trace_live;
    *peak = g_mtl4_trace_peak;
    pthread_mutex_unlock(&g_lock);
    return live;
}

__attribute__((visibility("default"))) int fencepost_test_reflection_failures(void)
{
    return g_mtl4_test_reflection_failed;
}

__attribute__((visibility("default"))) long fencepost_test_retired_signals(void)
{
    pthread_mutex_lock(&g_lock);
    long retired = g_mtl4_signals_retired;
    pthread_mutex_unlock(&g_lock);
    return retired;
}

__attribute__((visibility("default"))) int fencepost_test_signal_window(void)
{
    pthread_mutex_lock(&g_lock);
    fp_node *signal = node_new(FPK_SIGNAL);
    if (!signal) {
        pthread_mutex_unlock(&g_lock);
        return 0;
    }
    fp_queue_signal entry = { signal->uid, 0 };
    fp_queue queue = { .nrecent = FP_QRECENT + 1, .mtl4_signals = &entry, .mtl4_nsignals = 1 };
    uint64_t saved_tick = g_tick;
    g_tick = 1;
    mtl4_retire_signals(&queue, 1);
    g_tick = saved_tick;
    int retired = signal->done_tick == 1 && !queue.mtl4_nsignals;
    signal->state = 0;
    g_mtl4_signals_retired -= retired;
    pthread_mutex_unlock(&g_lock);
    return retired;
}

__attribute__((visibility("default"))) void fencepost_test_safety_log(void)
{
    atexit(final_summary);
    pthread_mutex_lock(&g_lock);
    g_st[ST_NODES] = 1;
    for (int i = 0; i < 16000; i++)
        fp_log("fencepost probe: %05d --------------------------------------------------\n", i);
    pthread_mutex_unlock(&g_lock);
}

__attribute__((visibility("default"))) int fencepost_test_safety_pool(void)
{
    atexit(final_summary);
    pthread_mutex_lock(&g_lock);
    uint64_t first_uid = g_uid + 1;
    for (int i = 0; i < FP_POOL; i++) node_new(FPK_COMPUTE);
    fp_node *first = node_get(first_uid);
    long before = g_st[ST_NODE_POOL_EXHAUSTED];
    int full = !node_new(FPK_COMPUTE) && first && g_st[ST_NODE_POOL_EXHAUSTED] == before + 1;
    pthread_mutex_unlock(&g_lock);
    /* Each fixture must return failure with no available slots. */
    int hooks_safe = !fencepost_test_wait_source() && !fencepost_test_recycled_signal() &&
        !fencepost_test_rotated_signal() && !fencepost_test_event_lifetime() && !fencepost_test_signal_window();
    pthread_mutex_lock(&g_lock);
    fp_node *last = node_get(first_uid + FP_POOL - 1);
    if (last) last->state = 2;
    fp_node *next = node_new(FPK_COMPUTE);
    int passed = full && hooks_safe && next && next == last && node_get(first_uid) == first;
    for (int i = 0; i < FP_POOL; i++) g_pool[i].state = 0;
    pthread_mutex_unlock(&g_lock);
    return passed;
}

__attribute__((visibility("default"))) int fencepost_test_safety_abandoned(void)
{
    atexit(final_summary);
    NSObject *cb = [[NSObject alloc] init], *encoder = [[NSObject alloc] init];
    pthread_mutex_lock(&g_lock);
    cb_record_new(cb, NULL, 0, NULL);
    pthread_mutex_unlock(&g_lock);
    fp_node *first = on_encoder(cb, encoder, FPK_COMPUTE, NULL);
    if (!first) {
        [encoder release];
        [cb release];
        return 0;
    }
    uint64_t uid = first->uid;
    pthread_mutex_lock(&g_lock);
    long before = g_st[ST_ABANDONED_ENCODERS];
    for (int i = 1; i < FP_POOL; i++) node_new(FPK_COMPUTE);
    pthread_mutex_unlock(&g_lock);
    [encoder release];
    pthread_mutex_lock(&g_lock);
    int reclaimed = !node_get(uid) && g_st[ST_ABANDONED_ENCODERS] == before + 1;
    fp_node *replacement = node_new(FPK_COMPUTE);
    for (int i = 0; i < FP_POOL; i++) g_pool[i].state = 0;
    cb_close(cb);
    pthread_mutex_unlock(&g_lock);
    [cb release];
    return reclaimed && replacement == first;
}

@interface FPSafetyResource : NSObject
@end
@implementation FPSafetyResource
- (NSString *)label { return @"released safety buffer"; }
- (NSUInteger)length { return 16; }
@end

__attribute__((visibility("default"))) int fencepost_test_safety_resource(void)
{
    FPSafetyResource *resource = [[FPSafetyResource alloc] init];
    fp_node first = { .kind = FPK_BLIT }, second = { .kind = FPK_BLIT };
    fp_acc access = { .store = resource, .a0 = 4, .a1 = 12, .rw = FP_WR, .via = VIA_FILL };
    record_store(&first, &access, (id<MTLResource>)resource, 16);
    record_store(&second, &access, (id<MTLResource>)resource, 16);
    [resource release];
    pthread_mutex_lock(&g_lock);
    report(&first, &first.acc[0], &second, &second.acc[0]);
    pthread_mutex_unlock(&g_lock);
    free(first.acc); free(first.descriptions);
    free(second.acc); free(second.descriptions);
    return 1;
}

__attribute__((visibility("default"))) long fencepost_test_safety_trace_live(long *endpoints)
{
    pthread_mutex_lock(&g_lock);
    long buffers = g_mtl3_trace_live;
    *endpoints = g_trace_nodes ? CFDictionaryGetCount(g_trace_nodes) : -1;
    pthread_mutex_unlock(&g_lock);
    return buffers;
}

__attribute__((visibility("default"))) int fencepost_test_safety_trace_recycled(void)
{
    if (!g_trace_enabled) return 0;
    NSObject *cb = [[NSObject alloc] init];
    static fp_queue queue;
    pthread_mutex_lock(&g_lock);
    fp_cb *buffer = cb_record_new(cb, &queue, 0, NULL);
    pthread_mutex_unlock(&g_lock);
    fp_trace_encoder *encoder = trace_encoder_begin(cb);
    pthread_mutex_lock(&g_lock);
    fp_node *n = node_new(FPK_COMPUTE);
    if (!n || !encoder) {
        if (n) n->state = 0;
        cb_close(cb);
        pthread_mutex_unlock(&g_lock);
        [cb release];
        return 0;
    }
    n->cb_uid = buffer->uid;
    n->state = 4;
    snprintf(n->label, sizeof n->label, "recycled safety encoder");
    trace_encoder_register(encoder, n);
    trace_encoder_end(n);
    fp_trace_cb *trace = buffer->trace;
    trace_phase_node(trace, n, NULL, 0);
    trace_analyse(n);
    trace->phased = 1;
    trace->owner->record = NULL;
    trace->owner = nil;
    uint64_t uid = n->uid;
    for (int i = 0; i < FP_POOL; i++) {
        fp_node *replacement = node_new(FPK_SIGNAL);
        if (!replacement) break;
        replacement->state = 4;
    }
    int recycled = !node_get(uid);
    uint64_t before = g_trace_encoders;
    trace_cb_completed(trace, 0, 0, NULL, (fp_trace_calibration){0});
    int written = g_trace_encoders == before + 1;
    cb_close(cb);
    pthread_mutex_unlock(&g_lock);
    [cb release];
    return recycled && written;
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
    if (c) { fp_item it = { 3, 0, NULL, (uint64_t)c->nenc }; PUSH(c->it, c->nit, c->cit, it); }
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
