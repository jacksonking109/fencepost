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

/* Hold the feedback queue until both commits are issued. The GPU never waits on the gate. */
API_AVAILABLE(macos(26.0))
static int submit_split(id<MTLDevice> device, id<MTL4CommandQueue> queue, id<MTL4CommandBuffer> a,
                        id<MTL4CommandBuffer> b)
{
    dispatch_semaphore_t releaseFeedback = dispatch_semaphore_create(0);
    dispatch_semaphore_t blocked = dispatch_semaphore_create(0);
    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    __block atomic_int errors = 0;
    id<MTL4CommandBuffer> warm = [device newCommandBuffer];
    __attribute__((objc_precise_lifetime)) id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    if (!warm || !allocator) return fail("warm-up setup failed");
    [warm beginCommandBufferWithAllocator:allocator];
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
    submit(queue, b, finished, &errors);
    dispatch_semaphore_signal(releaseFeedback);
    if (!wait_cpu(finished) || !wait_cpu(finished)) return fail("commit feedback timeout");
    return atomic_load(&errors) ? fail("GPU feedback error") : 0;
}

static NSString *const kernels =
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void word(const device uint *src [[buffer(0)]], device uint *dst [[buffer(1)]]) { *dst = *src; }\n"
    "kernel void texel(texture2d<float, access::read> src [[texture(0)]], device uint *dst [[buffer(0)]]) {\n"
    "    *dst = uint(src.read(uint2(0)).r * 255.0f);\n"
    "}\n"
    "struct VOut { float4 position [[position]]; };\n"
    "vertex VOut vtx(uint vid [[vertex_id]]) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VOut out; out.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f); return out;\n"
    "}\n"
    "fragment float4 frag(VOut in [[stage_in]], const device uint *data [[buffer(0)]]) {\n"
    "    return float4(float(data[0] & 255u) / 255.0f, 0.0f, 0.0f, 1.0f);\n"
    "}\n"
    "fragment float4 solid() { return float4(0.0f, 1.0f, 0.0f, 1.0f); }\n";

API_AVAILABLE(macos(26.0))
static int commit_and_wait(id<MTL4CommandQueue> queue, id<MTL4CommandBuffer> cb)
{
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block atomic_int errors = 0;
    submit(queue, cb, done, &errors);
    if (!wait_cpu(done)) return fail("commit feedback timeout");
    return atomic_load(&errors) ? fail("GPU feedback error") : 0;
}

API_AVAILABLE(macos(26.0))
static BOOL encode_word(id<MTL4CommandBuffer> cb, id<MTLComputePipelineState> pipeline,
                        id<MTL4ArgumentTable> table, id<MTLBuffer> source, id<MTLBuffer> dest, NSString *label)
{
    id<MTL4ComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (!e) return NO;
    e.label = label;
    [table setAddress:source.gpuAddress atIndex:0];
    [table setAddress:dest.gpuAddress atIndex:1];
    [e setComputePipelineState:pipeline];
    [e setArgumentTable:table];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    return YES;
}

API_AVAILABLE(macos(26.0))
static int run_consumer_barriers(id<MTLDevice> device, NSString *mode)
{
    BOOL separate = [mode isEqualToString:@"consumer-buffers"] || [mode isEqualToString:@"consumer-commits"];
    BOOL intervening = [mode isEqualToString:@"consumer-intervening"];
    BOOL chain = [mode hasPrefix:@"chain-"];
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> a1 = [device newCommandAllocator];
    id<MTL4CommandAllocator> a2 = separate ? [device newCommandAllocator] : a1;
    id<MTL4CommandBuffer> a = [device newCommandBuffer], b = separate ? [device newCommandBuffer] : a;
    if (!queue || !a1 || !a2 || !a || !b) return fail("command setup failed");
    id<MTLLibrary> lib = [device newLibraryWithSource:kernels options:nil error:NULL];
    id<MTLFunction> function = [lib newFunctionWithName:@"word"];
    id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:NULL] : nil;
    MTL4ArgumentTableDescriptor *layout = [MTL4ArgumentTableDescriptor new];
    layout.maxBufferBindCount = 2;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:layout error:NULL];
    id<MTLBuffer> input = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> data = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scratch = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:NULL];
    BOOL fenceBridge = [mode isEqualToString:@"chain-fence"];
    __attribute__((objc_precise_lifetime)) id<MTLFence> fence = fenceBridge ? [device newFence] : nil;
    if (!pipeline || !table || !input || !data || !output || !scratch || !residency) return fail("work setup failed");
    if (fenceBridge && !fence) return fail("fence creation failed");
    *(uint32_t *)input.contents = 42;
    *(uint32_t *)data.contents = 0;
    for (id<MTLBuffer> buffer in @[input, data, output, scratch]) [residency addAllocation:buffer];
    [residency commit];
    [queue addResidencySet:residency];

    [a beginCommandBufferWithAllocator:a1];
    if (!encode_word(a, pipeline, table, input, intervening ? scratch : data, @"first")) return fail("first encoder failed");
    id<MTL4ComputeCommandEncoder> barrier = [a computeCommandEncoder];
    if (!barrier) return fail("barrier encoder failed");
    if ([mode isEqualToString:@"chain-before-blit"])
        [barrier fillBuffer:scratch range:NSMakeRange(0, 16) value:0];
    [barrier barrierAfterQueueStages:MTLStageDispatch beforeStages:chain ? MTLStageBlit : MTLStageDispatch
                   visibilityOptions:MTL4VisibilityOptionDevice];
    if ([mode isEqualToString:@"chain-blit"])
        [barrier fillBuffer:scratch range:NSMakeRange(0, 16) value:0];
    if (fence) [barrier updateFence:fence afterEncoderStages:MTLStageBlit];
    [barrier endEncoding];
    if (chain) {
        id<MTL4ComputeCommandEncoder> bridge = [a computeCommandEncoder];
        if (!bridge) return fail("second barrier encoder failed");
        [bridge barrierAfterQueueStages:MTLStageBlit beforeStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
        [bridge endEncoding];
    }
    if (intervening && !encode_word(a, pipeline, table, input, data, @"intervening"))
        return fail("intervening encoder failed");
    if (separate) {
        [a endCommandBuffer];
        [b beginCommandBufferWithAllocator:a2];
    }
    if (!encode_word(b, pipeline, table, data, output, @"last")) return fail("last encoder failed");
    [b endCommandBuffer];

    if ([mode isEqualToString:@"consumer-commits"]) return submit_split(device, queue, a, b);
    if (!separate) return commit_and_wait(queue, a);
    id<MTL4CommandBuffer> const both[] = {a, b};
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block BOOL error = NO;
    MTL4CommitOptions *options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        error = feedback.error != nil;
        dispatch_semaphore_signal(done);
    }];
    [queue commit:both count:2 options:options];
    if (!wait_cpu(done)) return fail("commit feedback timeout");
    return error ? fail("GPU feedback error") : 0;
}

API_AVAILABLE(macos(26.0))
static BOOL binding_access(NSArray<id<MTLBinding>> *bindings, NSUInteger index, MTLBindingAccess access)
{
    for (id<MTLBinding> binding in bindings)
        if (binding.type == MTLBindingTypeBuffer && binding.index == index) return binding.used && binding.access == access;
    return NO;
}

/* Attachment loads against queue barriers: a clear-and-store pass, then a load-and-store pass
 * whose Fragment -> Fragment consumer barrier comes before its first draw or between draws. */
API_AVAILABLE(macos(26.0))
static int run_attachments(id<MTLDevice> device, NSString *mode)
{
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> cb = [device newCommandBuffer];
    if (!queue || !allocator || !cb) return fail("command setup failed");
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kernels options:nil error:&error];
    MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
    rd.vertexFunction = [library newFunctionWithName:@"vtx"];
    rd.fragmentFunction = [library newFunctionWithName:@"solid"];
    rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline = library ? [device newRenderPipelineStateWithDescriptor:rd error:&error] : nil;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:NULL];
    if (!pipeline || !target || !residency) return fail("render setup failed");
    [residency addAllocation:target];
    [residency commit];
    [queue addResidencySet:residency];

    [cb beginCommandBufferWithAllocator:allocator];
    BOOL early = [mode isEqualToString:@"load-early"];
    for (int pass = 0; pass < 2; pass++) {
        MTL4RenderPassDescriptor *desc = [MTL4RenderPassDescriptor new];
        desc.colorAttachments[0].texture = target;
        desc.colorAttachments[0].loadAction = pass ? MTLLoadActionLoad : MTLLoadActionClear;
        desc.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTL4RenderCommandEncoder> draw = [cb renderCommandEncoderWithDescriptor:desc];
        if (!draw) return fail("render encoder creation failed");
        draw.label = pass ? @"load" : @"clear";
        [draw setRenderPipelineState:pipeline];
        if (pass && early)
            [draw barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageFragment
                        visibilityOptions:MTL4VisibilityOptionDevice];
        [draw drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        if (pass && !early) {
            [draw barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageFragment
                        visibilityOptions:MTL4VisibilityOptionDevice];
            [draw drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [draw endEncoding];
    }
    [cb endCommandBuffer];
    int failed = commit_and_wait(queue, cb);
    if (!failed) printf("case %s: submitted successfully\n", mode.UTF8String);
    return failed;
}

/* Binding access from pipeline reflection: each mode is one queue and one command buffer. */
API_AVAILABLE(macos(26.0))
static int run_reflection(id<MTLDevice> device, NSString *mode)
{
    BOOL render = [mode hasPrefix:@"fragment"], compiler = [mode isEqualToString:@"compiler-race"];
    BOOL app = [mode isEqualToString:@"reflect-app"];
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> cb = [device newCommandBuffer];
    if (!queue || !allocator || !cb) return fail("command setup failed");
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kernels options:nil error:&error];
    if (!library) return fail(error.localizedDescription.UTF8String ?: "shader compilation failed");

    id<MTLBuffer> source = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> first = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> second = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (!source || !first || !second) return fail("buffer allocation failed");
    memset(source.contents, 0, 64);
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:NULL];
    if (!residency) return fail("residency allocation failed");
    [residency addAllocation:source];
    [residency addAllocation:first];
    [residency addAllocation:second];

    MTL4ArgumentTableDescriptor *layout = [MTL4ArgumentTableDescriptor new];
    layout.maxBufferBindCount = 2;
    id<MTL4ArgumentTable> one = [device newArgumentTableWithDescriptor:layout error:&error];
    id<MTL4ArgumentTable> two = [device newArgumentTableWithDescriptor:layout error:&error];
    if (!one || !two) return fail("argument table creation failed");
    [one setAddress:source.gpuAddress atIndex:0];
    [one setAddress:first.gpuAddress atIndex:1];
    [two setAddress:source.gpuAddress atIndex:0];
    [two setAddress:(compiler ? first : second).gpuAddress atIndex:1];

    [cb beginCommandBufferWithAllocator:allocator];
    if (render) {
        MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
        rd.vertexFunction = [library newFunctionWithName:@"vtx"];
        rd.fragmentFunction = [library newFunctionWithName:@"frag"];
        rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:rd error:&error];
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
            width:4 height:4 mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> target = [device newTextureWithDescriptor:td];
        if (!pipeline || !target) return fail("render setup failed");
        [residency addAllocation:target];
        [residency commit];
        [queue addResidencySet:residency];

        id<MTL4ComputeCommandEncoder> fill = [cb computeCommandEncoder];
        fill.label = @"write";
        [fill fillBuffer:source range:NSMakeRange(0, 64) value:0x40];
        [fill endEncoding];
        MTL4RenderPassDescriptor *pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTL4RenderCommandEncoder> draw = [cb renderCommandEncoderWithDescriptor:pass];
        if (!draw) return fail("render encoder creation failed");
        draw.label = @"draw";
        MTLStages before = [mode isEqualToString:@"fragment-only"] ? MTLStageFragment : MTLStageVertex;
        [draw barrierAfterQueueStages:MTLStageBlit beforeStages:before visibilityOptions:MTL4VisibilityOptionDevice];
        [draw setRenderPipelineState:pipeline];
        [draw setArgumentTable:one atStages:MTLRenderStageVertex | MTLRenderStageFragment];
        [draw drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [draw endEncoding];
    } else {
        [residency commit];
        [queue addResidencySet:residency];
        id<MTLFunction> function = [library newFunctionWithName:@"word"];
        id<MTLComputePipelineState> pipeline = nil;
        if (compiler || app) {
            id<MTL4Compiler> c = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
            MTL4LibraryFunctionDescriptor *fd = [MTL4LibraryFunctionDescriptor new];
            fd.library = library;
            fd.name = @"word";
            MTL4ComputePipelineDescriptor *cd = [MTL4ComputePipelineDescriptor new];
            cd.computeFunctionDescriptor = fd;
            MTL4PipelineOptions *options = app ? [MTL4PipelineOptions new] : nil;
            options.shaderReflection = MTL4ShaderReflectionBindingInfo;
            cd.options = options;
            pipeline = [c newComputePipelineStateWithDescriptor:cd compilerTaskOptions:nil error:&error];
            if (!pipeline) return fail("compiler pipeline creation failed");
            // fencepost compiles a copy: the app's descriptor must be as it left it.
            if (cd.options != options || (app && options.shaderReflection != MTL4ShaderReflectionBindingInfo))
                return fail("the app's pipeline descriptor changed");
            if (app && (!binding_access(pipeline.reflection.bindings, 0, MTLBindingAccessReadOnly) ||
                        !binding_access(pipeline.reflection.bindings, 1, MTLBindingAccessReadWrite)))
                return fail("compiler reflection the app asked for is missing");
        }
        if (app) {
            MTLComputePipelineReflection *reflection = nil;
            id<MTLComputePipelineState> own = [device newComputePipelineStateWithFunction:function
                options:MTLPipelineOptionBindingInfo reflection:&reflection error:&error];
            if (!own || !binding_access(reflection.bindings, 0, MTLBindingAccessReadOnly) ||
                !binding_access(reflection.bindings, 1, MTLBindingAccessReadWrite))
                return fail("device reflection the app asked for is missing");
            reflection = nil;
            own = [device newComputePipelineStateWithFunction:function options:MTLPipelineOptionNone
                                                   reflection:&reflection error:&error];
            if (!own || !reflection) return fail("device reflection out-parameter is missing");
        }
        if (!pipeline) pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) return fail("pipeline creation failed");
        for (int pass = 0; pass < (app ? 1 : 2); pass++) {
            id<MTL4ComputeCommandEncoder> encoder = [cb computeCommandEncoder];
            if (!encoder) return fail("encoder creation failed");
            encoder.label = pass ? @"second" : @"first";
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:pass ? two : one];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
        }
    }
    [cb endCommandBuffer];
    int failed = commit_and_wait(queue, cb);
    if (!failed) printf("case %s: submitted successfully\n", mode.UTF8String);
    return failed;
}

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
    if ([@[@"const-shared", @"fragment-only", @"fragment-vertex", @"reflect-app", @"compiler-race"] containsObject:mode])
        return run_reflection(device, mode);
    if ([mode hasPrefix:@"load-"]) return run_attachments(device, mode);
    if ([mode hasPrefix:@"consumer-"] || [mode hasPrefix:@"chain-"]) return run_consumer_barriers(device, mode);

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

    int failed = split ? submit_split(device, queue, a, b) : commit_and_wait(queue, b);
    if (failed) return failed;

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
                        @"color-vertex", @"mutate", @"disjoint", @"overlap", @"placement", @"order", @"partial",
                        @"const-shared", @"fragment-only", @"fragment-vertex", @"reflect-app", @"compiler-race",
                        @"load-early", @"load-late", @"consumer-buffer", @"consumer-buffers", @"consumer-commits",
                        @"consumer-intervening", @"chain-empty", @"chain-before-blit", @"chain-blit", @"chain-fence"]
                       containsObject:mode])
            return fail("expected a named Metal 4 race case");
        if (@available(macOS 26.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device && [device supportsFamily:MTLGPUFamilyMetal4]) return run(device, mode);
        }
        return fail("Metal 4 unavailable");
    }
}
