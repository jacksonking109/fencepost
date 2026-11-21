/* Safe, serial blits with optional explicit frame boundaries. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int submit(id<MTLCommandBuffer> cb, id<MTLBuffer> buffer, void (*frame)(id))
{
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    if (!enc) return 0;
    [enc fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:42];
    [enc endEncoding];
    if (frame) frame(cb);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusCompleted) return 1;
    fprintf(stderr, "capturetest: %s\n", cb.error.localizedDescription.UTF8String ?: "command buffer failed");
    return 0;
}

struct API_AVAILABLE(macos(26.0)) mtl4_fixture {
    id<MTL4CommandQueue> queue;
    id<MTL4CommandBuffer> cb;
    id<MTL4CommandAllocator> allocator;
    id<MTLBuffer> buffer;
};

API_AVAILABLE(macos(26.0))
static BOOL mtl4_setup(id<MTLDevice> device, struct mtl4_fixture *f)
{
    f->queue = [[device newMTL4CommandQueue] autorelease];
    f->cb = [[device newCommandBuffer] autorelease];
    f->allocator = [[device newCommandAllocator] autorelease];
    f->buffer = [[device newBufferWithLength:8192 options:MTLResourceStorageModeShared] autorelease];
    MTLResidencySetDescriptor *desc = [[[MTLResidencySetDescriptor alloc] init] autorelease];
    id<MTLResidencySet> residency = [[device newResidencySetWithDescriptor:desc error:NULL] autorelease];
    if (!f->queue || !f->cb || !f->allocator || !f->buffer || !residency) return NO;
    [residency addAllocation:f->buffer];
    [residency commit];
    [f->queue addResidencySet:residency];
    return YES;
}

API_AVAILABLE(macos(26.0))
static BOOL mtl4_encode(struct mtl4_fixture f, NSUInteger index, void (*frame)(id))
{
    [f.allocator reset];
    if (index % 2) [f.cb beginCommandBufferWithAllocator:f.allocator];
    else {
        MTL4CommandBufferOptions *options = [[[MTL4CommandBufferOptions alloc] init] autorelease];
        [f.cb beginCommandBufferWithAllocator:f.allocator options:options];
    }
    // Each encoder fills its own half of a small resident buffer.
    for (NSUInteger part = 0; part < 2; part++) {
        id<MTL4ComputeCommandEncoder> encoder = [f.cb computeCommandEncoder];
        if (!encoder) return NO;
        encoder.label = [NSString stringWithFormat:@"fp4-capture-frame-%lu-encoder-%lu", index, part];
        [encoder fillBuffer:f.buffer range:NSMakeRange(part * 4096, 4096) value:(uint8_t)index];
        [encoder endEncoding];
    }
    if (frame) frame(f.cb);
    [f.cb endCommandBuffer];
    return YES;
}

API_AVAILABLE(macos(26.0))
static BOOL mtl4_submit(struct mtl4_fixture f)
{
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block BOOL success = NO;
    MTL4CommitOptions *options = [[[MTL4CommitOptions alloc] init] autorelease];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        success = !feedback.error;
        dispatch_semaphore_signal(done);
    }];
    [f.queue commit:&f.cb count:1 options:options];
    BOOL ready = !dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
    if (ready) dispatch_release(done);
    return ready && success;
}

static BOOL wait_capture_stop(MTLCaptureManager *manager)
{
    for (int i = 0; i < 800 && manager.isCapturing; i++) usleep(10000);
    return !manager.isCapturing;
}

API_AVAILABLE(macos(26.0))
static int run_mtl4(const char *mode, id<MTLDevice> device, void (*frame)(id), void (*capture)(const char *))
{
    BOOL discard = !strcmp(mode, "mtl4-discard");
    BOOL defer = !strcmp(mode, "mtl4-defer") || discard, blocked = !strcmp(mode, "mtl4-blocked");
    BOOL mixed = !strcmp(mode, "mtl4-mixed"), quota = !strcmp(mode, "mtl4-quota");
    BOOL cooldown = !strcmp(mode, "mtl4-cooldown"), space = !strcmp(mode, "mtl4-space");
    BOOL multi = !strcmp(mode, "mtl4-multi"), idle = !strcmp(mode, "mtl4-idle");
    BOOL hard = !strcmp(mode, "mtl4-hard-limit");
    struct mtl4_fixture main = {0};
    if (!mtl4_setup(device, &main)) return 2;
    MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
    dispatch_semaphore_t ended = NULL, release_held = NULL;
    dispatch_group_t worker = NULL;
    __block BOOL ended_ok = NO, held_ok = NO;
    if (defer || blocked) {
        ended = dispatch_semaphore_create(0);
        release_held = dispatch_semaphore_create(0);
        worker = dispatch_group_create();
        dispatch_group_async(worker, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            @autoreleasepool {
                struct mtl4_fixture held = {0};
                ended_ok = mtl4_setup(device, &held) && mtl4_encode(held, 99, NULL);
                dispatch_semaphore_signal(ended);
                if (!ended_ok || dispatch_semaphore_wait(release_held, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)))
                    return;
                if (discard) {
                    // Drain the worker pool and release the ended CB without submitting it.
                    held_ok = YES;
                    return;
                }
                id<MTLSharedEvent> finished = [[device newSharedEvent] autorelease];
                if (!finished) return;
                // Exercise the plain commit hook on a different queue.
                [held.queue commit:&held.cb count:1];
                [held.queue signalEvent:finished value:1];
                held_ok = [finished waitUntilSignaledValue:1 timeoutMS:5000];
            }
        });
        if (dispatch_semaphore_wait(ended, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) || !ended_ok)
            return 2;
    }

    if (!mtl4_encode(main, 1, frame) || !mtl4_submit(main)) return 1;
    capture("API Request");
    if (mixed) {
        id<MTLCommandQueue> queue = [[device newCommandQueue] autorelease];
        id<MTLBuffer> buffer = [[device newBufferWithLength:4096 options:MTLResourceStorageModeShared] autorelease];
        id<MTLCommandBuffer> first = [queue commandBuffer];
        if (!first || !buffer || manager.isCapturing || !submit(first, buffer, NULL)) return 3;
    }
    if (defer || blocked) {
        // The first begin after a frame boundary must see the worker's held buffer.
        for (NSUInteger frame_no = 2; frame_no <= (blocked ? 4 : 2); frame_no++) {
            if (!mtl4_encode(main, frame_no, defer ? NULL : frame) || manager.isCapturing || !mtl4_submit(main)) return 3;
        }
        dispatch_semaphore_signal(release_held);
        if (dispatch_group_wait(worker, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) || !held_ok) return 2;
        dispatch_release(worker);
        dispatch_release(release_held);
        dispatch_release(ended);
        if (defer) {
            // The blocker cleared, but frame 2 already has work outside the capture.
            if (!mtl4_encode(main, 2, frame) || manager.isCapturing || !mtl4_submit(main)) return 3;
        }
    }

    NSUInteger first_frame = defer ? 3 : blocked ? 5 : 2;
    if (!mtl4_encode(main, first_frame, frame) || manager.isCapturing != !(blocked || space) || !mtl4_submit(main))
        return 3;
    if (multi) {
        if (!manager.isCapturing || !mtl4_encode(main, 3, frame) || !mtl4_submit(main)) return 3;
    }
    if (hard) {
        for (NSUInteger i = 3; i < 60 && manager.isCapturing; i++) @autoreleasepool {
            usleep(100000);
            if (!mtl4_encode(main, i, frame) || !mtl4_submit(main)) return 1;
        }
    }
    if (!wait_capture_stop(manager)) return 3;
    if (quota || cooldown) capture("Second Request");
    if (!idle && !hard) {
        NSUInteger next = first_frame + (multi ? 2 : 1);
        if (!mtl4_encode(main, next, frame) || manager.isCapturing || !mtl4_submit(main)) return 3;
    }
    return 0;
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        const char *mode = argc == 2 ? argv[1] : "";
        int file = !strcmp(mode, "file"), noFrame = !strcmp(mode, "no-frame");
        int emptyFile = !strcmp(mode, "empty-file"), wrongPid = !strcmp(mode, "wrong-pid");
        int api = !strcmp(mode, "api"), exitActive = !strcmp(mode, "exit-active");
        int appCapture = !strcmp(mode, "app-capture"), multi = !strcmp(mode, "multi-frame");
        int idle = !strcmp(mode, "idle"), slow = !strcmp(mode, "slow-frame");
        int hardLimit = !strcmp(mode, "hard-limit");
        int mtl4 = !strncmp(mode, "mtl4-", 5);
        if (!mtl4 && !file && !emptyFile && !wrongPid && !noFrame && !api && !exitActive && !appCapture &&
            !multi && !idle && !slow && !hardLimit) return 2;

        void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
        void (*capture)(const char *) = dlsym(RTLD_DEFAULT, "fencepost_capture");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !frame || !capture) return 2;
        if (mtl4) {
            if (@available(macOS 26.0, *)) return run_mtl4(mode, device, frame, capture);
            return 2;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        // The capture stores the larger buffer in a named snapshot file.
        NSUInteger length = api ? 8 * 1024 * 1024 : 4096;
        id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> first = [queue commandBuffer];
        if (!first || !buffer) return 2;

        if (file || emptyFile || wrongPid || noFrame) {
            puts("ready");
            fflush(stdout);
            if (getchar() == EOF) return 2;
        }
        if (!file && !emptyFile && !wrongPid) capture("API Request");
        MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
        if (manager.isCapturing) return 3;
        if (!submit(first, buffer, noFrame ? NULL : frame)) return 1;

        if (appCapture) {
            NSString *path = [NSString stringWithUTF8String:getenv("CAPTURETEST_APP_TRACE") ?: ""];
            if (!path.length) return 2;
            MTLCaptureDescriptor *captureDesc = [[[MTLCaptureDescriptor alloc] init] autorelease];
            captureDesc.captureObject = device;
            captureDesc.destination = MTLCaptureDestinationGPUTraceDocument;
            captureDesc.outputURL = [NSURL fileURLWithPath:path];
            NSError *error;
            if (![manager startCaptureWithDescriptor:captureDesc error:&error]) {
                fprintf(stderr, "capturetest: %s\n", error.localizedDescription.UTF8String);
                return 3;
            }
        }

        MTLCommandBufferDescriptor *descriptor = [[[MTLCommandBufferDescriptor alloc] init] autorelease];
        id<MTLCommandBuffer> second = [queue commandBufferWithDescriptor:descriptor];
        if ((emptyFile || wrongPid) && manager.isCapturing) return 3;
        if (api || exitActive || multi || idle || slow || hardLimit) {
            if (!manager.isCapturing) return 3;
            capture("Ignore While Active");
        }
        if (!submit(second, buffer, noFrame || slow ? NULL : frame)) return 1;
        if (slow) {
            /* Frame 2 takes more than 2 s, but commits arrive every 750 ms. */
            for (int i = 0; i < 4; i++) @autoreleasepool {
                usleep(750000);
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                if (!manager.isCapturing) return 3;
                if (!submit(cb, buffer, i == 3 ? frame : NULL)) return 1;
            }
            return manager.isCapturing ? 3 : 0;
        }
        if (hardLimit) {
            for (int i = 0; i < 60 && manager.isCapturing; i++) @autoreleasepool {
                usleep(100000);
                if (!submit([queue commandBuffer], buffer, frame)) return 1;
            }
            return manager.isCapturing ? 3 : 0;
        }
        if (emptyFile || wrongPid) {
            puts("request polled");
            fflush(stdout);
            if (getchar() == EOF) return 2;
        }
        if (wrongPid) {
            id<MTLCommandBuffer> sameStamp = [queue commandBuffer];
            if (manager.isCapturing) return 3;
            if (!submit(sameStamp, buffer, frame)) return 1;
            puts("same mtime polled");
            fflush(stdout);
            if (getchar() == EOF) return 2;
        }
        if (idle) {
            if (getchar() == EOF) return 2;
            return manager.isCapturing ? 3 : 0;
        }
        if (exitActive) return manager.isCapturing ? 0 : 3;
        if (appCapture) {
            if (!manager.isCapturing) return 3;
            [manager stopCapture];
        }
        if ((api || multi) && manager.isCapturing != multi) return 3;

        for (int i = 0; i < 3; i++) @autoreleasepool {
            if (!submit([queue commandBufferWithUnretainedReferences], buffer, noFrame ? NULL : frame)) return 1;
            if ((api || multi) && manager.isCapturing) return 3;
        }
        [buffer release];
        [queue release];
    }
    return 0;
}
