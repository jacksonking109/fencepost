// Small resident workloads. Intentionally unordered accesses never control
// addresses, iteration counts, dispatch sizes, or GPU synchronization.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdatomic.h>

static int fail(const char *message)
{
    fprintf(stderr, "mtl4races: %s\n", message);
    return 1;
}

static BOOL wait_cpu(dispatch_semaphore_t sem)
{
    return !dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
}

API_AVAILABLE(macos(26.0))
static void submit(id<MTL4CommandQueue> queue, id<MTL4CommandBuffer> cb,
                   dispatch_semaphore_t complete, atomic_int *errors)
{
    MTL4CommitOptions *options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        if (feedback.error) atomic_fetch_add(errors, 1);
        dispatch_semaphore_signal(complete);
    }];
    [queue commit:&cb count:1 options:options];
}

static NSString *const kernels =
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void word(const device uint *src [[buffer(0)]], device uint *dst [[buffer(1)]]) { *dst = *src; }\n"
    "kernel void texel(texture2d<float, access::read> src [[texture(0)]], device uint *dst [[buffer(0)]]) {\n"
    "    *dst = uint(src.read(uint2(0)).r * 255.0f);\n"
    "}\n";

API_AVAILABLE(macos(26.0))
static int run(id<MTLDevice> device, NSString *mode)
{
    BOOL split = [mode hasPrefix:@"commits"];
    BOOL color = [mode hasPrefix:@"color"];
    BOOL mutate = [mode isEqualToString:@"mutate"];
    BOOL placement = [mode isEqualToString:@"placement"];
    BOOL order = [mode isEqualToString:@"order"], partial = [mode isEqualToString:@"partial"];
    BOOL shader = color || mutate || placement || partial || [mode isEqualToString:@"disjoint"] ||
                  [mode isEqualToString:@"overlap"];
    BOOL use_fence = [mode hasPrefix:@"fence"];

    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandAllocator> nextAllocator = split ? [device newCommandAllocator] : allocator;
    id<MTL4CommandBuffer> a = [device newCommandBuffer], b = split ? [device newCommandBuffer] : a;
    if (!queue || !allocator || !nextAllocator || !a || !b) return fail("command setup failed");
    a.label = @"producer";
    if (split) b.label = @"consumer";

    __attribute__((objc_precise_lifetime)) id<MTLHeap> heap = nil;
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> source = nil;
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> secondSource = nil;
    id<MTLBuffer> result = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (placement) {
        MTLResourceOptions options = MTLResourceStorageModePrivate;
        MTLSizeAndAlign align = [device heapBufferSizeAndAlignWithLength:64 options:options];
        MTLHeapDescriptor *desc = [MTLHeapDescriptor new];
        desc.type = MTLHeapTypePlacement; desc.storageMode = MTLStorageModePrivate; desc.size = align.size;
        heap = [device newHeapWithDescriptor:desc];
        source = [heap newBufferWithLength:64 options:options offset:0];
    } else source = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (!source || !result) return fail("buffer allocation failed");
    if (!placement) memset(source.contents, 0, source.length);
    memset(result.contents, 0, result.length);
    if (mutate) {
        secondSource = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        if (!secondSource) return fail("second buffer allocation failed");
        *(uint32_t *)source.contents = 11;
        *(uint32_t *)secondSource.contents = 22;
    }

    id<MTLTexture> texture = nil;
    __attribute__((objc_precise_lifetime)) id<MTLTexture> view = nil;
    MTL4RenderPassDescriptor *pass = nil;
    if (color) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
            width:16 height:16 mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
        texture = [device newTextureWithDescriptor:td];
        view = [texture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
        if (!texture || !view) return fail("texture or view allocation failed");
        pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
    }

    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:NULL];
    if (!residency) return fail("residency allocation failed");
    [residency addAllocation:heap ?: (id<MTLAllocation>)source];
    [residency addAllocation:result];
    if (secondSource) [residency addAllocation:secondSource];
    if (texture) [residency addAllocation:texture];
    [residency commit];
    [queue addResidencySet:residency];

    __attribute__((objc_precise_lifetime)) id<MTLComputePipelineState> pipeline = nil;
    id<MTL4ArgumentTable> args = nil;
    if (shader) {
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kernels options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:color ? @"texel" : @"word"];
        if (!function) return fail(error.localizedDescription.UTF8String ?: "shader compilation failed");
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) return fail(error.localizedDescription.UTF8String ?: "pipeline creation failed");
        MTL4ArgumentTableDescriptor *layout = [MTL4ArgumentTableDescriptor new];
        layout.maxBufferBindCount = color ? 1 : 2;
        layout.maxTextureBindCount = color ? 1 : 0;
        args = [device newArgumentTableWithDescriptor:layout error:&error];
        if (!args) return fail("argument table creation failed");
        if (color) [args setTexture:view.gpuResourceID atIndex:0];
        else {
            NSUInteger offset = [mode isEqualToString:@"disjoint"] || [mode isEqualToString:@"overlap"] ? 16 : 0;
            [args setAddress:source.gpuAddress + offset atIndex:0];
        }
        [args setAddress:result.gpuAddress atIndex:color ? 0 : 1];
    }

    __attribute__((objc_precise_lifetime)) id<MTLFence> fence = use_fence ? [device newFence] : nil;
    if (use_fence && !fence) return fail("fence creation failed");
    [a beginCommandBufferWithAllocator:allocator];
    if (color) {
        id<MTL4RenderCommandEncoder> render = [a renderCommandEncoderWithDescriptor:pass];
        if (!render) return fail("render encoder creation failed");
        render.label = @"clear";
        [render endEncoding];
    } else if (!mutate) {
        id<MTL4ComputeCommandEncoder> fill = [a computeCommandEncoder];
        if (!fill) return fail("producer encoder creation failed");
        fill.label = @"write";
        NSUInteger start = [mode isEqualToString:@"overlap"] ? 16 : 0;
        [fill fillBuffer:source range:NSMakeRange(start, 16) value:0x5A];
        if ([mode isEqualToString:@"producer"])
            [fill barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        if (fence) [fill updateFence:fence afterEncoderStages:MTLStageBlit];
        [fill endEncoding];
    }
    if (order || partial) {
        void (*declare_barrier)(id) = dlsym(RTLD_DEFAULT, "fencepost_barrier");
        if (!declare_barrier) return fail("fencepost_barrier is missing");
        declare_barrier(a);
    }
    if (split) {
        [a endCommandBuffer];
        [b beginCommandBufferWithAllocator:nextAllocator];
    }

    id<MTL4ComputeCommandEncoder> read = [b computeCommandEncoder];
    if (!read) return fail("consumer encoder creation failed");
    read.label = mutate ? @"mutate" : @"read";
    if ([mode isEqualToString:@"fence"]) [read waitForFence:fence beforeEncoderStages:MTLStageBlit];
    if ([mode isEqualToString:@"commits-barrier"])
        [read barrierAfterQueueStages:MTLStageBlit beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    if (color || placement || partial) {
        MTLStages after = placement || partial ? MTLStageBlit
            : [mode isEqualToString:@"color-fragment"] ? MTLStageFragment : MTLStageVertex;
        [read barrierAfterQueueStages:after beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    }
    if (shader) {
        [read setComputePipelineState:pipeline];
        [read setArgumentTable:args];
        [read dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        if (mutate) {
            [args setAddress:secondSource.gpuAddress atIndex:0];
            [args setAddress:result.gpuAddress + 16 atIndex:1];
            [read dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
    } else if (!order) [read copyFromBuffer:source sourceOffset:0 toBuffer:result destinationOffset:0 size:16];
    if (order || partial) [read fillBuffer:result range:NSMakeRange(32, 16) value:0];
    [read endEncoding];
    [b endCommandBuffer];

    dispatch_semaphore_t releaseFeedback = nil;
    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    __block atomic_int errors = 0;
    if (split) {
        // Hold the serial feedback queue, never GPU execution, until both commits are issued.
        releaseFeedback = dispatch_semaphore_create(0);
        dispatch_semaphore_t blocked = dispatch_semaphore_create(0);
        id<MTL4CommandBuffer> warm = [device newCommandBuffer];
        __attribute__((objc_precise_lifetime)) id<MTL4CommandAllocator> warmAllocator = [device newCommandAllocator];
        if (!warm || !warmAllocator) return fail("warm-up setup failed");
        [warm beginCommandBufferWithAllocator:warmAllocator];
        id<MTL4ComputeCommandEncoder> fill = [warm computeCommandEncoder];
        if (!fill) return fail("warm-up encoder creation failed");
        [fill fillBuffer:result range:NSMakeRange(48, 16) value:0];
        [fill endEncoding];
        [warm endCommandBuffer];
        MTL4CommitOptions *options = [MTL4CommitOptions new];
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            if (feedback.error) atomic_fetch_add(&errors, 1);
            dispatch_semaphore_signal(blocked);
            wait_cpu(releaseFeedback);
        }];
        [queue commit:&warm count:1 options:options];
        if (!wait_cpu(blocked)) return fail("feedback gate timeout");
        submit(queue, a, finished, &errors);
    }
    submit(queue, b, finished, &errors);
    if (releaseFeedback) dispatch_semaphore_signal(releaseFeedback);
    if (!wait_cpu(finished) || (split && !wait_cpu(finished))) return fail("commit feedback timeout");
    if (atomic_load(&errors)) return fail("GPU feedback error");

    uint32_t *words = result.contents;
    if (mutate && (words[0] != 11 || words[4] != 22)) return fail("argument-table snapshot mismatch");
    if ([mode isEqualToString:@"color-fragment"] && words[0] != 255) return fail("texture result mismatch");
    if ([mode isEqualToString:@"disjoint"] && words[0]) return fail("disjoint range result mismatch");
    if ((placement || partial || [mode isEqualToString:@"producer"] || [mode isEqualToString:@"fence"] ||
         [mode isEqualToString:@"commits-barrier"]) && words[0] != 0x5A5A5A5A)
        return fail("synchronized copy mismatch");
    printf("case %s: submitted successfully\n", mode.UTF8String);
    return 0;
}

int main(int argc, const char **argv)
{
    @autoreleasepool {
        NSString *mode = argc == 2 ? [NSString stringWithUTF8String:argv[1]] : nil;
        if (!mode || ![@[@"producer", @"fence", @"fence-missing", @"commits", @"commits-barrier", @"color-fragment",
                        @"color-vertex", @"mutate", @"disjoint", @"overlap", @"placement", @"order", @"partial"]
                       containsObject:mode])
            return fail("expected a named Metal 4 race case");
        if (@available(macOS 26.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device && [device supportsFamily:MTLGPUFamilyMetal4]) return run(device, mode);
        }
        return fail("Metal 4 unavailable");
    }
}
