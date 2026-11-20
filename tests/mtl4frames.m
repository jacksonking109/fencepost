#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <dlfcn.h>
#include <stdatomic.h>

enum { FrameCount = 6, BufferSize = 16 };

static int fail(const char *message)
{
    fprintf(stderr, "mtl4frames: %s\n", message);
    return 1;
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
static int run(id<MTLDevice> device, bool drawable_mode, bool pause_for_trigger)
{
    void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
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

    for (unsigned i = 1; i <= FrameCount; i++) @autoreleasepool {
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
                           !drawable_mode || i == FrameCount ? frame : NULL))
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
        [producer signalEvent:produced value:i];   // Submit the signal before any GPU wait.
        [consumer waitForEvent:produced value:i];
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
            if (((const uint8_t *)bytes.contents)[k] != i) return fail("GPU fill mismatch");
        [compute_allocator reset];
        [render_allocators[0] reset];
        [render_allocators[1] reset];
    }
    // Feedback runs on Metal's dispatch queues. Let the last callback drain before exit.
    [NSThread sleepForTimeInterval:0.05];
    int fresh_total = atomic_load(&fresh_count), shared_total = atomic_load(&shared_count);
    int added_total = atomic_load(&added_count), batch_total = atomic_load(&batch_count);
    int error_total = atomic_load(&errors);
    printf("frames=%d boundary=%s app_feedback fresh=%d reused=%d added=%d batch=%d errors=%d\n",
           FrameCount, drawable_mode ? "drawable" : "api",
           fresh_total, shared_total, added_total, batch_total, error_total);
    return fresh_total != 1 || shared_total != 1 || added_total != 1 || batch_total != FrameCount || error_total != 0;
}

int main(int argc, const char **argv)
{
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        bool drawable = false, pause_for_trigger = false;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--drawable")) drawable = true;
            else if (!strcmp(argv[i], "--trigger")) pause_for_trigger = true;
            else return fail("usage: mtl4frames [--drawable] [--trigger]");
        }
        if (@available(macOS 26.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device && [device supportsFamily:MTLGPUFamilyMetal4]) return run(device, drawable, pause_for_trigger);
        }
        puts("skip: Metal 4 unavailable");
        return 77;
    }
}
