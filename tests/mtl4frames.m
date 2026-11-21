#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <time.h>

enum { FrameCount = 6, BufferSize = 16, ThreadCommits = 256, ThreadSlots = 8 };
enum { WaitDefault, WaitSelf, WaitMany };

static int fail(const char *message)
{
    fprintf(stderr, "mtl4frames: %s\n", message);
    return 1;
}

static bool wait_cpu(dispatch_semaphore_t ready)
{
    return dispatch_semaphore_wait(ready, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0;
}

typedef struct {
    atomic_int inside;
    atomic_bool overlapped;
} SubmitOverlap;

API_AVAILABLE(macos(26.0))
static bool submit_thread(id<MTLDevice> device, id<MTL4CommandQueue> queue, id<MTLSharedEvent> event,
                          id<MTLBuffer> source, id<MTLBuffer> dest, dispatch_semaphore_t signal_scheduled,
                          dispatch_group_t first_submit, SubmitOverlap *overlap,
                          bool producer, atomic_int *callbacks, atomic_int *errors)
{
    id<MTL4CommandBuffer> cb = [device newCommandBuffer];
    MTL4CommitOptions *opts = [MTL4CommitOptions new];
    id<MTL4CommandAllocator> allocators[ThreadSlots];
    dispatch_semaphore_t retired[ThreadSlots];
    if (!cb || !opts) return false;
    for (int slot = 0; slot < ThreadSlots; slot++) {
        allocators[slot] = [device newCommandAllocator];
        retired[slot] = dispatch_semaphore_create(0);
        if (!allocators[slot]) return false;
    }
    for (NSUInteger i = 0; i < ThreadCommits; i++) @autoreleasepool {
        NSUInteger slot = i % ThreadSlots, offset = i * BufferSize;
        id<MTL4CommandAllocator> allocator = allocators[slot];
        dispatch_semaphore_t done = retired[slot];
        if (i >= ThreadSlots) {
            if (!wait_cpu(done)) return false;
            [allocator reset];
        }
        if (!producer && i) {
            // The producer publishes only after submitting this signal to Metal.
            if (!wait_cpu(signal_scheduled)) return false;
            [queue waitForEvent:event value:i + 1];
        }
        [cb beginCommandBufferWithAllocator:allocator];
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return false;
        if (producer || !i)
            [enc fillBuffer:producer ? source : dest range:NSMakeRange(offset, BufferSize) value:(uint8_t)i];
        else
            [enc copyFromBuffer:source sourceOffset:offset toBuffer:dest destinationOffset:offset size:BufferSize];
        [enc endEncoding];
        [cb endCommandBuffer];
        [opts addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            if (feedback.error) atomic_fetch_add(errors, 1);
            atomic_fetch_add(callbacks, 1);
            dispatch_semaphore_signal(done);
        }];
        if (atomic_fetch_add(&overlap->inside, 1)) atomic_store(&overlap->overlapped, true);
        if (!i) {
            // First commits use independent buffers. Bring both submitters to
            // the call together without introducing a GPU dependency.
            dispatch_group_leave(first_submit);
            if (dispatch_group_wait(first_submit, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)))
                return false;
        }
        [queue commit:&cb count:1 options:opts];
        atomic_fetch_sub(&overlap->inside, 1);
        if (producer && i) {
            [queue signalEvent:event value:i + 1];
            dispatch_semaphore_signal(signal_scheduled);
        }
    }
    for (int slot = 0; slot < ThreadSlots; slot++)
        if (!wait_cpu(retired[slot])) return false;
    return true;
}

API_AVAILABLE(macos(26.0))
static int run_threads(id<MTLDevice> device)
{
    id<MTL4CommandQueue> producer = [device newMTL4CommandQueue], consumer = [device newMTL4CommandQueue];
    id<MTLSharedEvent> event = [device newSharedEvent];
    id<MTLBuffer> source = [device newBufferWithLength:ThreadCommits * BufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> dest = [device newBufferWithLength:ThreadCommits * BufferSize options:MTLResourceStorageModeShared];
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:NULL];
    if (!producer || !consumer || !event || !source || !dest || !residency) return fail("thread setup failed");
    [residency addAllocation:source];
    [residency addAllocation:dest];
    [residency commit];
    [producer addResidencySet:residency];
    [consumer addResidencySet:residency];
    memset(dest.contents, 0xFE, dest.length);
    dispatch_group_t threads = dispatch_group_create();
    dispatch_group_t first_submit = dispatch_group_create();
    dispatch_group_enter(first_submit);
    dispatch_group_enter(first_submit);
    dispatch_semaphore_t signal_scheduled = dispatch_semaphore_create(0);
    __block atomic_int writes = 0, reads = 0, errors = 0;
    __block bool producer_ok = false, consumer_ok = false;
    __block SubmitOverlap overlap = { 0 };
    uint64_t started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    dispatch_group_async(threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        producer_ok = submit_thread(device, producer, event, source, dest, signal_scheduled,
                                    first_submit, &overlap, true, &writes, &errors);
    });
    dispatch_group_async(threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        consumer_ok = submit_thread(device, consumer, event, source, dest, signal_scheduled,
                                    first_submit, &overlap, false, &reads, &errors);
    });
    if (dispatch_group_wait(threads, dispatch_time(DISPATCH_TIME_NOW, 15 * NSEC_PER_SEC)) || !producer_ok || !consumer_ok)
        return fail("threaded submissions did not complete");
    if (!atomic_load(&overlap.overlapped)) return fail("submit paths did not overlap");
    double wall_ms = (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started) / 1e6;
    unsigned checksum = 0;
    const uint8_t *bytes = dest.contents;
    for (unsigned i = 0; i < ThreadCommits * BufferSize; i++) {
        if (bytes[i] != i / BufferSize) return fail("threaded copy mismatch");
        checksum += bytes[i];
    }
    int nwrite = atomic_load(&writes), nread = atomic_load(&reads), nerror = atomic_load(&errors);
    printf("threaded commits=%d checksum=%u producer_feedback=%d consumer_feedback=%d errors=%d overlap=yes\n",
           ThreadCommits * 2, checksum, nwrite, nread, nerror);
    printf("threaded wall time %.3f ms\n", wall_ms);
    return nwrite != ThreadCommits || nread != ThreadCommits || nerror != 0;
}

API_AVAILABLE(macos(26.0))
static bool encode_render(id<MTL4CommandBuffer> cb, id<MTL4CommandAllocator> allocator,
                          id<MTLTexture> texture, bool with_options, void (*frame)(id))
{
    if (with_options)
        [cb beginCommandBufferWithAllocator:allocator options:[MTL4CommandBufferOptions new]];
    else
        [cb beginCommandBufferWithAllocator:allocator];
    MTL4RenderPassDescriptor *pass = [MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0, 0.5, 1);
    id<MTL4RenderCommandEncoder> render = with_options
        ? [cb renderCommandEncoderWithDescriptor:pass options:0]
        : [cb renderCommandEncoderWithDescriptor:pass];
    if (!render) return false;
    [render endEncoding];
    if (frame) frame(cb);
    [cb endCommandBuffer];
    return true;
}

API_AVAILABLE(macos(26.0))
static int run(id<MTLDevice> device, bool drawable_mode, bool pause_for_trigger, unsigned frames, int wait_mode)
{
    void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
    long (*trace_cbs)(long *) = dlsym(RTLD_DEFAULT, "fencepost_test_trace_cbs");
    long (*retired_signals)(void) = dlsym(RTLD_DEFAULT, "fencepost_test_retired_signals");
    int (*stats)(long *, int) = dlsym(RTLD_DEFAULT, "fencepost_stats");
    id<MTL4CommandQueue> producer = [device newMTL4CommandQueue];
    MTL4CommandQueueDescriptor *description = [MTL4CommandQueueDescriptor new];
    description.label = @"frames render queue";
    id<MTL4CommandQueue> consumer = [device newMTL4CommandQueueWithDescriptor:description error:NULL];
    id<MTLSharedEvent> produced = [device newSharedEvent];
    id<MTLSharedEvent> done = [device newSharedEvent];
    id<MTL4CommandBuffer> compute = [device newCommandBuffer];
    id<MTL4CommandBuffer> render[2] = { [device newCommandBuffer], [device newCommandBuffer] };
    id<MTL4CommandAllocator> compute_allocator = [device newCommandAllocator];
    id<MTL4CommandAllocator> render_allocators[2] = { [device newCommandAllocator], [device newCommandAllocator] };
    if (!producer || !consumer || !produced || !done || !compute || !render[0] || !render[1] ||
        !compute_allocator || !render_allocators[0] || !render_allocators[1])
        return fail("command setup failed");
    compute.label = @"producer";
    render[0].label = @"batch first";
    render[1].label = @"batch second";

    id<MTLBuffer> bytes = [device newBufferWithLength:BufferSize options:MTLResourceStorageModeShared];
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:32 height:32 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> targets[2] = { [device newTextureWithDescriptor:td], [device newTextureWithDescriptor:td] };
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:NULL];
    if (!bytes || !targets[0] || !targets[1] || !residency) return fail("resource setup failed");
    [residency addAllocation:bytes];
    [residency addAllocation:targets[0]];
    [residency addAllocation:targets[1]];
    [residency commit];
    [producer addResidencySet:residency];
    [consumer addResidencySet:residency];

    CAMetalLayer *layer = nil;
    if (drawable_mode) {
        layer = [CAMetalLayer layer];
        layer.device = device;
        layer.pixelFormat = td.pixelFormat;
        layer.bounds = CGRectMake(0, 0, 32, 32);
        layer.drawableSize = CGSizeMake(32, 32);
        layer.allowsNextDrawableTimeout = YES;
        if (!layer.residencySet) return fail("no CAMetalLayer residency set");
        [consumer addResidencySet:layer.residencySet];
    }

    __block atomic_int errors = 0, fresh_count = 0, shared_count = 0, added_count = 0, batch_count = 0;
    dispatch_semaphore_t app_feedback = dispatch_semaphore_create(0);
    dispatch_semaphore_t batch_feedback = dispatch_semaphore_create(0);
    MTL4CommitOptions *fresh = [MTL4CommitOptions new];
    MTL4CommitOptions *shared = [MTL4CommitOptions new];
    [fresh addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        if (feedback.error) atomic_fetch_add(&errors, 1);
        atomic_fetch_add(&fresh_count, 1);
        dispatch_semaphore_signal(app_feedback);
    }];
    [shared addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        if (feedback.error) atomic_fetch_add(&errors, 1);
        atomic_fetch_add(&shared_count, 1);
        dispatch_semaphore_signal(app_feedback);
    }];

    enum { S_FRAME_CHAIN = 24 };
    long initial_live = -1, peak = 0, first_chain = 0, last_chain = 0, max_chain = 0;
    for (unsigned i = 1; i <= frames; i++) @autoreleasepool {
        if (pause_for_trigger && i == 3) {
            puts("ready for trigger");
            if (getchar() != '\n') return fail("trigger handshake closed");
        }

        if (i == 5) {
            [shared addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
                if (feedback.error) atomic_fetch_add(&errors, 1);
                atomic_fetch_add(&added_count, 1);
                dispatch_semaphore_signal(app_feedback);
            }];
        }

        [compute beginCommandBufferWithAllocator:compute_allocator];
        id<MTL4ComputeCommandEncoder> fill = [compute computeCommandEncoder];
        if (!fill) return fail("compute encoder creation failed");
        [fill fillBuffer:bytes range:NSMakeRange(0, BufferSize) value:(uint8_t)i];
        [fill endEncoding];
        [compute endCommandBuffer];

        id<CAMetalDrawable> drawable = drawable_mode ? [layer nextDrawable] : nil;
        if (drawable_mode && !drawable) {
            fprintf(stderr, "mtl4frames: headless nextDrawable unavailable in frame %u\n", i);
            return 2;
        }
        if (!encode_render(render[0], render_allocators[0], targets[0], false, NULL) ||
            !encode_render(render[1], render_allocators[1], targets[1], true,
                           !drawable_mode || i == frames ? frame : NULL))
            return fail("render encoder creation failed");

        if (i == 1) [producer commit:&compute count:1];
        else [producer commit:&compute count:1 options:i == 2 ? fresh : shared];
        // Re-register on the same object while the producer can still be in flight.
        MTL4CommitOptions *batch = i >= 3 ? shared : [MTL4CommitOptions new];
        [batch addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            if (feedback.error) atomic_fetch_add(&errors, 1);
            atomic_fetch_add(&batch_count, 1);
            dispatch_semaphore_signal(batch_feedback);
        }];
        if (wait_mode == WaitSelf) {
            // The producer is already committed and independent. Submit its
            // signal immediately after the consumer's wait and dependent signal.
            [consumer waitForEvent:produced value:1];
            [consumer signalEvent:produced value:2];
            [producer signalEvent:produced value:1];
            [consumer waitForEvent:produced value:2];
        } else if (wait_mode == WaitMany) {
            [producer signalEvent:produced value:40];
            for (uint64_t value = 1; value <= 40; value++) [consumer waitForEvent:produced value:value];
        } else {
            [producer signalEvent:produced value:i];
            [consumer waitForEvent:produced value:i];
        }
        [consumer commit:render count:2 options:batch];
        if (drawable) {
            [consumer signalDrawable:drawable];
            [drawable present];
        }
        [consumer signalEvent:done value:i];
        if (![done waitUntilSignaledValue:i timeoutMS:3000]) return fail("GPU completion timeout");
        if (dispatch_semaphore_wait(batch_feedback, dispatch_time(DISPATCH_TIME_NOW, 3 * NSEC_PER_SEC)))
            return fail("batch feedback timeout");
        if ((i == 2 || i == 3 || i == 5) &&
            dispatch_semaphore_wait(app_feedback, dispatch_time(DISPATCH_TIME_NOW, 3 * NSEC_PER_SEC)))
            return fail("producer feedback timeout");
        for (unsigned k = 0; k < BufferSize; k++)
            if (((const uint8_t *)bytes.contents)[k] != (uint8_t)i) return fail("GPU fill mismatch");
        [compute_allocator reset];
        [render_allocators[0] reset];
        [render_allocators[1] reset];
        if (trace_cbs && i == 64) initial_live = trace_cbs(&peak);
        if (stats && frames > FrameCount && frame) {
            long counters[S_FRAME_CHAIN + 1];
            stats(counters, S_FRAME_CHAIN + 1);
            last_chain = counters[S_FRAME_CHAIN];
            if (i == 1) first_chain = last_chain;
            max_chain = MAX(max_chain, last_chain);
        }
    }
    // Feedback runs on Metal's dispatch queues. Let the last callback drain before exit.
    [NSThread sleepForTimeInterval:0.05];
    int fresh_total = atomic_load(&fresh_count), shared_total = atomic_load(&shared_count);
    int added_total = atomic_load(&added_count), batch_total = atomic_load(&batch_count);
    int error_total = atomic_load(&errors);
    printf("frames=%u boundary=%s app_feedback fresh=%d reused=%d added=%d batch=%d errors=%d\n",
           frames, drawable_mode ? "drawable" : "api",
           fresh_total, shared_total, added_total, batch_total, error_total);
    if (initial_live >= 0) {
        long final_live = trace_cbs(&peak);
        printf("live Metal 4 trace records at frame 64 and end: %ld, %ld (peak %ld)\n",
               initial_live, final_live, peak);
        if (initial_live > 8 || final_live > 8 || peak > 16) return fail("unbounded live trace records");
    }
    if (max_chain) {
        printf("longest frame chain at start and end: %ld, %ld (max %ld)\n", first_chain, last_chain, max_chain);
        if (first_chain != 2 || last_chain != 2 || max_chain > 2) return fail("unbounded frame chain");
    }
    if (retired_signals && frames > FrameCount) {
        long retired = retired_signals();
        printf("retired Metal 4 queue signals: %ld\n", retired);
        if (retired != 2L * (frames - 1)) return fail("queue signals did not retire");
    }
    int expected_producer_feedback = wait_mode == WaitDefault ? 1 : 0;
    return fresh_total != expected_producer_feedback || shared_total != expected_producer_feedback ||
           added_total != expected_producer_feedback || batch_total != (int)frames || error_total != 0;
}

int main(int argc, const char **argv)
{
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        bool drawable = false, pause_for_trigger = false, threaded = false;
        int wait_mode = WaitDefault;
        unsigned frames = FrameCount;
        if (argc == 2 && !strcmp(argv[1], "--wait-source")) {
            int (*test_source)(void) = dlsym(RTLD_DEFAULT, "fencepost_test_wait_source");
            if (!test_source || !test_source()) return fail("first pending signal was bypassed");
            puts("cross-queue wait selected the first signal");
            return 0;
        }
        if (argc == 2 && !strcmp(argv[1], "--recycled-signal")) {
            int (*test_recycled)(void) = dlsym(RTLD_DEFAULT, "fencepost_test_recycled_signal");
            if (!test_recycled || !test_recycled()) return fail("a recycled first signal fell through to a later one");
            puts("a recycled first signal settled the wait without an edge");
            return 0;
        }
        if (argc == 2 && !strcmp(argv[1], "--signal-window")) {
            int (*test_window)(void) = dlsym(RTLD_DEFAULT, "fencepost_test_signal_window");
            if (!test_window || !test_window()) return fail("signal outside recent queue entries did not retire");
            puts("signal outside the queue window retired");
            return 0;
        }
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--drawable")) drawable = true;
            else if (!strcmp(argv[i], "--trigger")) pause_for_trigger = true;
            else if (!strcmp(argv[i], "--threads")) threaded = true;
            else if (!strcmp(argv[i], "--long")) frames = 512;
            else if (!strcmp(argv[i], "--self-wait")) wait_mode = WaitSelf;
            else if (!strcmp(argv[i], "--many-waits")) wait_mode = WaitMany;
            else return fail("usage: mtl4frames [--drawable] [--trigger] [--long] | --threads | --self-wait | --many-waits");
        }
        if (@available(macOS 26.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device && [device supportsFamily:MTLGPUFamilyMetal4])
                return threaded ? run_threads(device)
                                : run(device, drawable, pause_for_trigger, wait_mode == WaitDefault ? frames : 1, wait_mode);
        }
        puts("skip: Metal 4 unavailable");
        return 77;
    }
}
