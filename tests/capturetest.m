/* Safe, serial blits with optional explicit frame boundaries. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

enum gate_case {
    GATE_NONE, GATE_RACE, GATE_REENTRANT, GATE_REENTRANT_RECORDING, GATE_TIMEOUT, GATE_FAILED,
    GATE_ALLOC_FAILED, GATE_TEARDOWN, GATE_TEARDOWN_FILE, GATE_FEEDBACK
};
static struct {
    enum gate_case mode;
    dispatch_semaphore_t enter_end, end_tracked, release_end;
    id nested_end;
    BOOL replay_pending, replayed, throw_commit, failed_end, fail_alloc, requested;
    void (*capture)(const char *);
} gate;

static void phase_wait(dispatch_semaphore_t sem)
{
    if (!dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC))) return;
    fputs("capturetest: gate handshake timed out\n", stderr);
    _exit(4);
}

static BOOL reentrant_end(void)
{
    return gate.mode == GATE_REENTRANT || gate.mode == GATE_REENTRANT_RECORDING;
}

static int gate_hook(const char *phase)
{
    if (gate.mode == GATE_ALLOC_FAILED && gate.fail_alloc && !strcmp(phase, "mtl4-capture-owners-allocation")) {
        gate.fail_alloc = NO;
        return 1;
    }
    if (gate.mode == GATE_TIMEOUT && !strcmp(phase, "start-reserved")) {
        dispatch_semaphore_signal(gate.enter_end);
        phase_wait(gate.end_tracked);
    }
    if (gate.mode == GATE_RACE) {
        if (!strcmp(phase, "before-start-gate") && !gate.requested) {
            gate.requested = YES;
            dispatch_semaphore_signal(gate.enter_end);
            phase_wait(gate.end_tracked);
        } else if (!strcmp(phase, "before-native-end") && !pthread_main_np()) {
            dispatch_semaphore_signal(gate.end_tracked);
            phase_wait(gate.release_end);
        }
    }
    if (reentrant_end()) {
        BOOL recording = gate.mode == GATE_REENTRANT_RECORDING;
        if (!strcmp(phase, recording ? "start-recording" : "start-reserved") && gate.nested_end) {
            if (MTLCaptureManager.sharedCaptureManager.isCapturing != recording) _exit(4);
            id cb = gate.nested_end;
            gate.nested_end = nil;
            gate.replay_pending = YES;
            if (@available(macOS 26.0, *)) [(id<MTL4CommandBuffer>)cb endCommandBuffer];
            // A nested end must unwind the transition before it reaches Metal.
            _exit(4);
        }
        if (!strcmp(phase, "before-native-end") && gate.replay_pending) {
            if (MTLCaptureManager.sharedCaptureManager.isCapturing) _exit(4);
            gate.replay_pending = NO;
            gate.replayed = YES;
        }
    }
    if (gate.mode == GATE_FAILED) {
        if (!strcmp(phase, "mtl4-commit") && gate.throw_commit) {
            gate.throw_commit = NO;
            gate.failed_end = YES;
            @throw [NSException exceptionWithName:@"CaptureTestCommit" reason:@"injected before native commit" userInfo:nil];
        }
        if (!strcmp(phase, "start-reserved") && gate.failed_end) _exit(4);
    }
    if ((gate.mode == GATE_TEARDOWN || gate.mode == GATE_TEARDOWN_FILE) &&
        !strcmp(phase, "stop-armed") && !gate.requested) {
        gate.requested = YES;
        if (gate.mode == GATE_TEARDOWN) {
            gate.capture("Teardown Request");
        } else {
            FILE *request = fopen(getenv("CAPTURETEST_REQUEST_FILE") ?: "", "w");
            if (!request) _exit(4);
            fputs("Teardown Request\n", request);
            if (fclose(request)) _exit(4);
        }
    }
    return gate.mode == GATE_FEEDBACK && !strcmp(phase, "mtl4-feedback");
}

static int submit(id<MTLCommandBuffer> cb, id<MTLBuffer> buffer, void (*frame)(id))
{
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    if (!enc) return 0;
    [enc fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:42];
    [enc endEncoding];
    if (frame) frame(cb);
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) { dispatch_semaphore_signal(done); }];
    [cb commit];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC))) return 0;
    dispatch_release(done);
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
static BOOL mtl4_record(struct mtl4_fixture f, NSUInteger index, void (*frame)(id))
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
    return YES;
}

API_AVAILABLE(macos(26.0))
static BOOL mtl4_encode(struct mtl4_fixture f, NSUInteger index, void (*frame)(id))
{
    if (!mtl4_record(f, index, frame)) return NO;
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
static int run_gate(id<MTLDevice> device, void (*frame)(id), void (*capture)(const char *))
{
    struct mtl4_fixture main = {0}, held = {0};
    if (!mtl4_setup(device, &main) || !mtl4_setup(device, &held) ||
        !mtl4_record(held, 99, NULL) || !mtl4_encode(main, 1, frame) || !mtl4_submit(main)) return 2;
    MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
    dispatch_group_t worker = NULL;
    if (gate.mode == GATE_RACE || gate.mode == GATE_TIMEOUT) {
        gate.enter_end = dispatch_semaphore_create(0);
        gate.end_tracked = dispatch_semaphore_create(0);
        if (gate.mode == GATE_RACE) gate.release_end = dispatch_semaphore_create(0);
        worker = dispatch_group_create();
        dispatch_group_async(worker, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            @autoreleasepool {
                phase_wait(gate.enter_end);
                [held.cb endCommandBuffer];
                if (gate.mode == GATE_TIMEOUT) dispatch_semaphore_signal(gate.end_tracked);
            }
        });
    } else if (reentrant_end()) {
        gate.nested_end = held.cb;
    } else if (gate.mode == GATE_ALLOC_FAILED) {
        [held.cb endCommandBuffer];
        gate.fail_alloc = YES;
        if (!mtl4_submit(held) || gate.fail_alloc) return 3;
    } else {
        [held.cb endCommandBuffer];
        gate.throw_commit = YES;
        @try {
            [held.queue commit:&held.cb count:1];
            return 3;
        } @catch (NSException *exception) {
            if (![exception.name isEqualToString:@"CaptureTestCommit"]) @throw;
        }
    }
    capture("API Request");
    if (!mtl4_encode(main, 2, frame) || manager.isCapturing || !mtl4_submit(main)) return 3;
    if (gate.mode == GATE_ALLOC_FAILED) return 0;
    if (worker) {
        if (gate.release_end) dispatch_semaphore_signal(gate.release_end);
        if (dispatch_group_wait(worker, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC))) return 3;
        dispatch_release(worker);
        if (gate.release_end) dispatch_release(gate.release_end);
        dispatch_release(gate.end_tracked);
        dispatch_release(gate.enter_end);
    }
    if (reentrant_end() && !gate.replayed) return 3;
    if (!mtl4_submit(held)) return 3;
    if (gate.mode == GATE_TIMEOUT) {
        capture("API Request");
        return mtl4_encode(main, 3, frame) && !manager.isCapturing && mtl4_submit(main) ? 0 : 3;
    }
    gate.failed_end = NO;
    if (reentrant_end()) capture("API Request");
    if (!mtl4_encode(main, 3, frame) || !manager.isCapturing || !mtl4_submit(main)) return 3;
    return wait_capture_stop(manager) ? 0 : 3;
}

API_AVAILABLE(macos(26.0))
static int run_mixed_metal3(id<MTLDevice> device, void (*frame)(id), void (*capture)(const char *))
{
    id<MTL4CommandQueue> metal4 = [[device newMTL4CommandQueue] autorelease];
    id<MTLCommandQueue> metal3 = [[device newCommandQueue] autorelease];
    id<MTLBuffer> buffer = [[device newBufferWithLength:4096 options:MTLResourceStorageModeShared] autorelease];
    id<MTLCommandBuffer> first = [metal3 commandBuffer];
    if (!metal4 || !first || !buffer || !submit(first, buffer, frame)) return 2;
    capture("API Request");
    id<MTLCommandBuffer> second = [metal3 commandBuffer];
    MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
    if (!second || !manager.isCapturing || !submit(second, buffer, frame)) return 3;
    return wait_capture_stop(manager) ? 0 : 3;
}

API_AVAILABLE(macos(26.0))
static int run_mtl4(const char *mode, id<MTLDevice> device, void (*frame)(id), void (*capture)(const char *))
{
    if (gate.mode == GATE_RACE || reentrant_end() || gate.mode == GATE_TIMEOUT ||
        gate.mode == GATE_FAILED || gate.mode == GATE_ALLOC_FAILED)
        return run_gate(device, frame, capture);
    if (!strcmp(mode, "mtl4-mixed-metal3")) return run_mixed_metal3(device, frame, capture);
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

    if (mixed) {
        id<MTLCommandQueue> queue = [[device newCommandQueue] autorelease];
        id<MTLBuffer> buffer = [[device newBufferWithLength:4096 options:MTLResourceStorageModeShared] autorelease];
        id<MTLCommandBuffer> first = [queue commandBuffer];
        if (!first || !buffer || manager.isCapturing || !submit(first, buffer, NULL)) return 3;
    }
    if (!mtl4_encode(main, 1, frame) || !mtl4_submit(main)) return 1;
    capture("API Request");
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
    if (!mtl4_encode(main, first_frame, frame) || manager.isCapturing != !(blocked || space))
        return 3;
    if (gate.mode == GATE_FEEDBACK) {
        id<MTLSharedEvent> done = [[device newSharedEvent] autorelease];
        if (!done) return 2;
        [main.queue commit:&main.cb count:1];
        [main.queue signalEvent:done value:1];
        if (![done waitUntilSignaledValue:1 timeoutMS:5000]) return 3;
    } else if (!mtl4_submit(main)) return 3;
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
        if (!strcmp(mode, "mtl4-end-race")) gate.mode = GATE_RACE;
        if (!strcmp(mode, "mtl4-reentrant-end")) gate.mode = GATE_REENTRANT;
        if (!strcmp(mode, "mtl4-reentrant-recording-end")) gate.mode = GATE_REENTRANT_RECORDING;
        if (!strcmp(mode, "mtl4-end-start-timeout")) gate.mode = GATE_TIMEOUT;
        if (!strcmp(mode, "mtl4-failed-commit")) gate.mode = GATE_FAILED;
        if (!strcmp(mode, "mtl4-failed-owner-allocation")) gate.mode = GATE_ALLOC_FAILED;
        if (!strcmp(mode, "mtl4-feedback-fallback")) gate.mode = GATE_FEEDBACK;
        if (!strcmp(mode, "teardown-api")) gate.mode = GATE_TEARDOWN;
        if (!strcmp(mode, "teardown-file")) gate.mode = GATE_TEARDOWN_FILE;
        BOOL teardown = gate.mode == GATE_TEARDOWN || gate.mode == GATE_TEARDOWN_FILE;
        if (!mtl4 && !file && !emptyFile && !wrongPid && !noFrame && !api && !exitActive && !appCapture &&
            !multi && !idle && !slow && !hardLimit && !teardown) return 2;

        void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
        void (*capture)(const char *) = dlsym(RTLD_DEFAULT, "fencepost_capture");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !frame || !capture) return 2;
        if (gate.mode != GATE_NONE) {
            void (*set_hook)(int (*)(const char *)) = dlsym(RTLD_DEFAULT, "fencepost_test_capture_hook");
            if (!set_hook) return 2;
            gate.capture = capture;
            set_hook(gate_hook);
        }
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
        if (teardown) {
            if (!gate.requested || manager.isCapturing) return 3;
            id<MTLCommandBuffer> third = [queue commandBuffer];
            if (!third || !manager.isCapturing || !submit(third, buffer, frame) || manager.isCapturing) return 3;
            return 0;
        }
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
