/* fencepost self-test: small Metal programs with and without the synchronisation they
 * need, each with the number of races fencepost must report. No shaders: render passes
 * that clear, compute encoders that declare resources with useResource, blits.
 *
 *   DYLD_INSERT_LIBRARIES=out/libfencepost.dylib out/fptest
 *
 * Command buffers that must stay in flight wait on a "gate" event the CPU opens at the end. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

static id<MTLDevice> dev;
static id<MTLCommandQueue> q1, q2;
static NSMutableArray *gates, *cbs;
static int (*fp_stats)(long *, int);
static void (*fp_flush)(void);
static void (*fp_barrier)(id);
static long base[4];
enum { S_RACE = 0, S_POSS = 1, S_ORDER = 15, S_PARTIAL = 17 };   /* fencepost's counter indices */
static int fails, total;
static id<MTLTexture> scratch;   /* tracked: a render target for passes that only read */

static id<MTLTexture> tex(int untracked, int levels)
{
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:64 height:64 mipmapped:levels > 1];
    d.mipmapLevelCount = levels;
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    d.storageMode = MTLStorageModePrivate;
    d.hazardTrackingMode = untracked ? MTLHazardTrackingModeUntracked : MTLHazardTrackingModeTracked;
    return [dev newTextureWithDescriptor:d];
}
static id<MTLTexture> level_view(id<MTLTexture> t, NSUInteger level)
{
    return [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D
                                     levels:NSMakeRange(level, 1) slices:NSMakeRange(0, 1)];
}
static id<MTLBuffer> buf(void)
{
    return [dev newBufferWithLength:4096 options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
}

static id<MTLCommandBuffer> newcb(id<MTLCommandQueue> q, int gated)
{
    id<MTLCommandBuffer> cb = [q commandBuffer];
    if (gated) {
        id<MTLSharedEvent> g = [dev newSharedEvent];
        [gates addObject:g];
        [cb encodeWaitForEvent:g value:1];
    }
    [cbs addObject:cb];
    return cb;
}

/* render pass that clears t and stores it: a write in the fragment stage */
static void rclear(id<MTLCommandBuffer> cb, id<MTLTexture> t, id<MTLFence> upd, MTLRenderStages after,
                   id<MTLFence> wait, MTLRenderStages before)
{
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = t;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
    if (wait) [e waitForFence:wait beforeStages:before];
    if (upd) [e updateFence:upd afterStages:after];
    [e endEncoding];
}
/* render pass that reads r in the given stages (target: a tracked scratch texture) */
static void ruse(id<MTLCommandBuffer> cb, id r, MTLRenderStages stages, id<MTLFence> wait, MTLRenderStages before)
{
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = scratch;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
    if (wait) [e waitForFence:wait beforeStages:before];
    [e useResource:r usage:MTLResourceUsageRead stages:stages];
    [e endEncoding];
}
static void cuse(id<MTLCommandBuffer> cb, id r, MTLResourceUsage u, id<MTLFence> wait, id<MTLFence> upd)
{
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (wait) [e waitForFence:wait];
    [e useResource:r usage:u];
    if (upd) [e updateFence:upd];
    [e endEncoding];
}
static void bfill(id<MTLCommandBuffer> cb, id<MTLBuffer> b, NSUInteger off, NSUInteger len, id<MTLFence> wait, id<MTLFence> upd)
{
    id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
    if (wait) [e waitForFence:wait];
    if (b) [e fillBuffer:b range:NSMakeRange(off, len) value:0];
    if (upd) [e updateFence:upd];
    [e endEncoding];
}

static void check4(const char *name, long want_races, long want_possible, long want_order, long want_partial)
{
    long s[32];
    const int idx[4] = { S_RACE, S_POSS, S_ORDER, S_PARTIAL };
    const long want[4] = { want_races, want_possible, want_order, want_partial };
    long got[4];
    int ok = 1;
    fp_flush();
    fp_stats(s, 32);
    for (int i = 0; i < 4; i++) { got[i] = s[idx[i]] - base[i]; base[i] = s[idx[i]]; if (got[i] != want[i]) ok = 0; }
    total++;
    if (!ok) fails++;
    fprintf(stderr, "%s %-58s races %ld/%ld, possible %ld/%ld, order %ld/%ld, partial %ld/%ld (got/want)\n\n",
            ok ? "[ok]  " : "[FAIL]", name, got[0], want[0], got[1], want[1], got[2], want[2], got[3], want[3]);
}
#define check(name, r, p) check4(name, r, p, 0, 0)

#define MAIN_VIS 0
#define FRAG MTLRenderStageFragment
#define VERT MTLRenderStageVertex
#define RD MTLResourceUsageRead

int main(void)
{
    @autoreleasepool {
        fp_stats = (int (*)(long *, int))dlsym(RTLD_DEFAULT, "fencepost_stats");
        fp_flush = (void (*)(void))dlsym(RTLD_DEFAULT, "fencepost_flush");
        fp_barrier = (void (*)(id))dlsym(RTLD_DEFAULT, "fencepost_barrier");
        if (!fp_stats || !fp_flush || !fp_barrier) {
            fprintf(stderr, "fencepost is not loaded: run with DYLD_INSERT_LIBRARIES=out/libfencepost.dylib\n");
            return 2;
        }
        dev = MTLCreateSystemDefaultDevice();
        q1 = [dev newCommandQueue];
        q2 = [dev newCommandQueue];
        gates = [NSMutableArray array];
        cbs = [NSMutableArray array];
        scratch = tex(0, 1);
        id<MTLCommandBuffer> a, b;
        id<MTLTexture> t;
        id<MTLFence> f, f2;
        id<MTLSharedEvent> ev;
        id<MTLBuffer> bb;

        /* a command buffer that completed before the next was committed orders them */
        t = tex(1, 1);
        a = newcb(q1, 0); rclear(a, t, nil, 0, nil, 0); [a commit]; [a waitUntilCompleted];
        usleep(50000);   /* let the completed handlers run */
        b = newcb(q1, 0); cuse(b, t, RD, nil, nil); [b commit]; [b waitUntilCompleted];
        check("completed before commit: ordered", 0, 0);

        t = tex(1, 1);
        a = newcb(q1, 0); rclear(a, t, nil, 0, nil, 0); cuse(a, t, RD, nil, nil); [a commit];
        check("render writes, compute reads, no fence", 1, 0);

        t = tex(1, 1); f = [dev newFence];
        a = newcb(q1, 0); rclear(a, t, f, FRAG, nil, 0); cuse(a, t, RD, f, nil); [a commit];
        check("render writes, fence, compute reads", 0, 0);

        t = tex(1, 1); f = [dev newFence]; f2 = [dev newFence];
        a = newcb(q1, 0); bfill(a, nil, 0, 0, nil, f2); rclear(a, t, f, FRAG, nil, 0); cuse(a, t, RD, f2, nil); [a commit];
        check("reader waits on the wrong fence", 1, 0);

        t = tex(1, 1); f = [dev newFence];
        a = newcb(q1, 0); rclear(a, t, f, VERT, nil, 0); cuse(a, t, RD, f, nil); [a commit];
        check("fence updated after vertex only; write is in fragment", 1, 0);

        t = tex(1, 1); f = [dev newFence];
        a = newcb(q1, 0); rclear(a, t, f, FRAG, nil, 0); ruse(a, t, VERT, f, FRAG); [a commit];
        check("vertex-stage read, wait only before fragment", 1, 0);

        t = tex(1, 1); f = [dev newFence];
        a = newcb(q1, 0); rclear(a, t, f, FRAG, nil, 0); ruse(a, t, FRAG, f, FRAG); [a commit];
        check("fragment-stage read, wait before fragment", 0, 0);

        t = tex(1, 1);
        a = newcb(q1, 0); cuse(a, t, RD, nil, nil); rclear(a, t, nil, 0, nil, 0); [a commit];
        check("compute reads, render overwrites, no fence (WAR)", 1, 0);

        bb = buf();
        a = newcb(q1, 0); bfill(a, bb, 0, 256, nil, nil); bfill(a, bb, 128, 256, nil, nil); [a commit];
        check("two fills overlap, no fence (WAW)", 1, 0);

        bb = buf();
        a = newcb(q1, 0); bfill(a, bb, 0, 256, nil, nil); bfill(a, bb, 256, 256, nil, nil); [a commit];
        check("two fills, disjoint ranges", 0, 0);

        t = tex(1, 2);
        a = newcb(q1, 0); rclear(a, level_view(t, 0), nil, 0, nil, 0); cuse(a, level_view(t, 1), RD, nil, nil); [a commit];
        check("write mip 0 through a view, read mip 1 through a view", 0, 0);

        t = tex(1, 2);
        a = newcb(q1, 0); rclear(a, level_view(t, 0), nil, 0, nil, 0); cuse(a, level_view(t, 0), RD, nil, nil); [a commit];
        check("write mip 0 through a view, read mip 0 through another", 1, 0);

        t = tex(1, 1); f = [dev newFence]; f2 = [dev newFence];
        a = newcb(q1, 0); rclear(a, t, f, FRAG, nil, 0); bfill(a, nil, 0, 0, f, f2); cuse(a, t, RD, f2, nil); [a commit];
        check("ordered through a chain of two fences", 0, 0);

        t = tex(1, 1); f = [dev newFence];
        a = newcb(q1, 0); rclear(a, t, f, FRAG, nil, 0); bfill(a, nil, 0, 0, nil, f); cuse(a, t, RD, f, nil); [a commit];
        check("fence re-updated by an unrelated encoder before the wait", 1, 0);

        t = tex(0, 1);
        a = newcb(q1, 0); rclear(a, t, nil, 0, nil, 0); cuse(a, t, RD, nil, nil); [a commit];
        check("tracked texture: Metal orders it, not reported", 0, 0);

        bb = buf();
        a = newcb(q1, 0); cuse(a, bb, RD | MTLResourceUsageWrite, nil, nil); cuse(a, bb, RD | MTLResourceUsageWrite, nil, nil); [a commit];
        check("two read+write declarations, no fence: possible only", 0, 1);

        /* from here q1's command buffers stay in flight */
        t = tex(1, 1);
        a = newcb(q1, 1); rclear(a, t, nil, 0, nil, 0); [a commit];
        b = newcb(q1, 0); cuse(b, t, RD, nil, nil); [b commit];
        check("across command buffers, first still running, no fence", 1, 0);

        t = tex(1, 1); f = [dev newFence];
        a = newcb(q1, 1); rclear(a, t, f, FRAG, nil, 0); [a commit];
        b = newcb(q1, 0); cuse(b, t, RD, f, nil); [b commit];
        check("across command buffers with a fence", 0, 0);

        t = tex(1, 1); ev = [dev newSharedEvent];
        a = newcb(q1, 1); rclear(a, t, nil, 0, nil, 0); [a encodeSignalEvent:ev value:1]; [a commit];
        b = newcb(q2, 0); [b encodeWaitForEvent:ev value:1]; cuse(b, t, RD, nil, nil); [b commit];
        check("across queues, event signal then wait", 0, 0);

        t = tex(1, 1); ev = [dev newSharedEvent];
        a = newcb(q1, 1); rclear(a, t, nil, 0, nil, 0); [a encodeSignalEvent:ev value:1]; [a commit];
        b = newcb(q2, 0); cuse(b, t, RD, nil, nil); [b commit];
        check("across queues, signal but no wait", 1, 0);

        t = tex(1, 1); ev = [dev newSharedEvent];
        b = newcb(q2, 0); [b encodeWaitForEvent:ev value:1]; cuse(b, t, RD, nil, nil); [b commit];
        a = newcb(q1, 0); rclear(a, t, nil, 0, nil, 0); [a encodeSignalEvent:ev value:1]; [a commit];
        check("across queues, wait committed before the signal", 0, 0);

        t = tex(1, 1); f = [dev newFence];
        a = newcb(q1, 0); b = newcb(q1, 0);
        [a enqueue]; [b enqueue];
        cuse(b, t, RD, f, nil); [b commit];   /* committed first, but enqueued second */
        rclear(a, t, f, FRAG, nil, 0); [a commit];
        check("fence resolved in enqueue order, not commit order", 0, 0);

        /* declared barriers (fencepost_barrier); each on its own queue, the reads hit a
         * tracked texture so only the ordering is judged */
        id<MTLFence> f0;
        t = tex(1, 1); f = [dev newFence];
        a = newcb([dev newCommandQueue], 0); rclear(a, t, f, FRAG, nil, 0); fp_barrier(a); cuse(a, scratch, RD, f, nil); [a commit];
        check4("barrier, then a wait on the fence the writer updated", 0, 0, 0, 0);

        t = tex(1, 1);
        a = newcb([dev newCommandQueue], 0); rclear(a, t, nil, 0, nil, 0); fp_barrier(a); cuse(a, scratch, RD, nil, nil); [a commit];
        check4("barrier, encoder after it waits for nothing", 0, 0, 1, 0);

        t = tex(1, 1); f0 = [dev newFence]; f = [dev newFence];
        a = newcb([dev newCommandQueue], 0); bfill(a, nil, 0, 0, nil, f0); rclear(a, t, f, FRAG, nil, 0); fp_barrier(a);
        cuse(a, scratch, RD, f, nil); [a commit];
        check4("barrier, after-encoder follows the last one but not the one before", 0, 0, 1, 0);

        t = tex(1, 1); f0 = [dev newFence]; f = [dev newFence];
        a = newcb([dev newCommandQueue], 0); bfill(a, nil, 0, 0, nil, f0); rclear(a, t, f, FRAG, f0, VERT); fp_barrier(a);
        cuse(a, scratch, RD, f, nil); [a commit];
        check4("barrier, everything before chained into the waited fence", 0, 0, 0, 0);

        f = [dev newFence];
        a = newcb([dev newCommandQueue], 0); cuse(a, scratch, RD, nil, f); fp_barrier(a); ruse(a, scratch, VERT, f, FRAG); [a commit];
        check4("barrier, render after it waits only before fragment", 0, 0, 0, 1);

        t = tex(1, 1); f = [dev newFence];
        { id<MTLCommandQueue> qb = [dev newCommandQueue];
          a = newcb(qb, 1); rclear(a, t, f, FRAG, nil, 0); [a commit];
          b = newcb(qb, 0); fp_barrier(b); cuse(b, scratch, RD, f, nil); [b commit]; }
        check4("barrier at the start of the next command buffer, fenced", 0, 0, 0, 0);

        t = tex(1, 1);
        { id<MTLCommandQueue> qb = [dev newCommandQueue];
          a = newcb(qb, 1); rclear(a, t, nil, 0, nil, 0); [a commit];
          b = newcb(qb, 0); fp_barrier(b); cuse(b, scratch, RD, nil, nil); [b commit]; }
        check4("barrier at the start of the next command buffer, not fenced", 0, 0, 1, 0);

        for (id<MTLSharedEvent> g in gates) g.signaledValue = 1;
        for (id<MTLCommandBuffer> c in cbs) [c waitUntilCompleted];
        fprintf(stderr, "fptest: %d/%d passed\n", total - fails, total);
    }
    return fails ? 1 : 0;
}
