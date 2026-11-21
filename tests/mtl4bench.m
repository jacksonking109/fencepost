#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <mach/mach_time.h>
#include <stdatomic.h>

enum { FrameCount = 60, WarmupCount = 5, SampleCount = FrameCount - WarmupCount,
       CommitCount = 1024, CommitWarmup = 64, CommitSlots = 8 };

static NSString *const shader = @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "vertex float4 benchVertex(const device float4 *vertices [[buffer(0)]], uint i [[vertex_id]]) {\n"
    "    return vertices[i];\n"
    "}\n"
    "fragment half4 benchFragment(texture2d<half, access::read> tex [[texture(0)]]) {\n"
    "    return tex.read(uint2(0));\n"
    "}\n";

// An oversized triangle through a 1 x 1 viewport shades exactly one pixel per draw.
static const float vertices[][4] = {{-1, -1, 0, 1}, {3, -1, 0, 1}, {-1, 3, 0, 1}};
static const MTLViewport viewport = {0, 0, 1, 1, 0, 1};
static const uint8_t colors[][4] = {{255, 0, 0, 255}, {0, 255, 0, 255}};

static BOOL failure(NSString *operation, NSError *error)
{
    fprintf(stderr, "mtl4bench: %s%s%s\n", operation.UTF8String, error ? ": " : "",
            error ? error.description.UTF8String : "");
    return NO;
}

static void configureAttachment(MTLRenderPassColorAttachmentDescriptor *color, id<MTLTexture> target)
{
    color.texture = target;
    color.loadAction = MTLLoadActionClear;
    color.storeAction = MTLStoreActionStore;
    color.clearColor = MTLClearColorMake(0, 0, 1, 1);
}

static BOOL runMetal3(id<MTLDevice> device, id<MTLRenderPipelineState> pipeline,
                      id<MTLBuffer> vertexBuffer, NSArray<id<MTLTexture>> *textures,
                      id<MTLTexture> target, NSUInteger draws, uint64_t ticks[SampleCount])
{
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return failure(@"Metal 3 queue creation failed", nil);
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    configureAttachment(pass.colorAttachments[0], target);
    id<MTLTexture> bindings[] = {textures[0], textures[1]};

    for (int frame = 0; frame < FrameCount; ++frame) @autoreleasepool {
        uint64_t start = mach_absolute_time();
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLRenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
        if (!render) return failure(@"Metal 3 encoder creation failed", nil);
        [render setRenderPipelineState:pipeline];
        [render setViewport:viewport];
        for (NSUInteger draw = 0; draw < draws; ++draw) {
            NSUInteger binding = draw % 2;
            [render setVertexBuffer:vertexBuffer offset:binding * sizeof(vertices) atIndex:0];
            [render setFragmentTexture:bindings[binding] atIndex:0];
            [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [render endEncoding];
        [commands commit];
        uint64_t elapsed = mach_absolute_time() - start;
        [commands waitUntilCompleted];
        if (commands.status != MTLCommandBufferStatusCompleted)
            return failure(@"Metal 3 render failed", commands.error);
        if (frame >= WarmupCount) ticks[frame - WarmupCount] = elapsed;
    }
    return YES;
}

API_AVAILABLE(macos(26.0))
static BOOL runMetal4(id<MTLDevice> device, id<MTL4CommandQueue> queue, id<MTLRenderPipelineState> pipeline,
                      id<MTLBuffer> vertexBuffer, NSArray<id<MTLTexture>> *textures,
                      NSArray<id<MTLTexture>> *targets, NSUInteger draws, BOOL commit_heavy,
                      uint64_t *ticks)
{
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    if (!queue || !commands) return failure(@"Metal 4 command setup failed", nil);

    NSError *error = nil;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    if (!residency) return failure(@"residency set creation failed", error);
    [residency addAllocation:vertexBuffer];
    for (id<MTLTexture> target in targets) [residency addAllocation:target];
    for (id<MTLTexture> texture in textures) [residency addAllocation:texture];
    [residency commit];
    [queue addResidencySet:residency];

    MTL4ArgumentTableDescriptor *layout = [MTL4ArgumentTableDescriptor new];
    layout.maxBufferBindCount = 1;
    layout.maxTextureBindCount = 1;
    id<MTL4ArgumentTable> args = [device newArgumentTableWithDescriptor:layout error:&error];
    if (!args) return failure(@"argument table creation failed", error);
    MTLGPUAddress address = vertexBuffer.gpuAddress;
    MTLResourceID bindings[] = {textures[0].gpuResourceID, textures[1].gpuResourceID};

    NSUInteger slots = targets.count;
    id<MTL4CommandAllocator> allocators[CommitSlots];
    MTL4RenderPassDescriptor *passes[CommitSlots];
    dispatch_semaphore_t retired[CommitSlots];
    for (NSUInteger slot = 0; slot < slots; slot++) {
        allocators[slot] = [device newCommandAllocator];
        if (!allocators[slot]) return failure(@"Metal 4 command allocator creation failed", nil);
        passes[slot] = [MTL4RenderPassDescriptor new];
        configureAttachment(passes[slot].colorAttachments[0], targets[slot]);
        retired[slot] = dispatch_semaphore_create(0);
    }

    __block atomic_int gpu_errors = 0;
    NSUInteger commits = commit_heavy ? CommitCount : FrameCount;
    NSUInteger warmup = commit_heavy ? CommitWarmup : WarmupCount;
    for (NSUInteger i = 0; i < commits; i++) @autoreleasepool {
        NSUInteger slot = i % slots;
        id<MTL4CommandAllocator> allocator = allocators[slot];
        dispatch_semaphore_t completion = retired[slot];
        // Every in-flight slot has its own allocator and render target.
        if (i >= slots) {
            if (dispatch_semaphore_wait(completion, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)))
                return failure(@"timed out waiting for Metal 4 commit feedback", nil);
            if (atomic_load(&gpu_errors)) return NO;
            [allocator reset];
        }
        uint64_t start = mach_absolute_time();
        MTL4CommitOptions *commit = [MTL4CommitOptions new];
        [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            if (feedback.error) {
                failure(@"Metal 4 render failed", feedback.error);
                atomic_fetch_add(&gpu_errors, 1);
            }
            dispatch_semaphore_signal(completion);
        }];
        if (!commit_heavy) start = mach_absolute_time();  // Keep the existing per-draw measurement.
        [commands beginCommandBufferWithAllocator:allocator];
        id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:passes[slot]];
        if (!render) return failure(@"Metal 4 encoder creation failed", nil);
        [render setRenderPipelineState:pipeline];
        [render setViewport:viewport];
        [render setArgumentTable:args atStages:MTLRenderStageVertex | MTLRenderStageFragment];
        for (NSUInteger draw = 0; draw < draws; ++draw) {
            NSUInteger binding = draw % 2;
            [args setAddress:address + binding * sizeof(vertices) atIndex:0];
            [args setTexture:bindings[binding] atIndex:0];
            [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [render endEncoding];
        [commands endCommandBuffer];
        [queue commit:&commands count:1 options:commit];
        uint64_t elapsed = mach_absolute_time() - start;
        if (i >= warmup) ticks[i - warmup] = elapsed;
    }
    for (NSUInteger slot = 0; slot < slots; slot++)
        if (dispatch_semaphore_wait(retired[slot], dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)))
            return failure(@"timed out draining Metal 4 commit feedback", nil);
    return atomic_load(&gpu_errors) == 0;
}

static int compareTicks(const void *a, const void *b)
{
    uint64_t first = *(const uint64_t *)a, second = *(const uint64_t *)b;
    return (first > second) - (first < second);
}

/* Pipeline creation time, the cost of the binding reflection fencepost requests. A distinct
 * constant per library keeps the pipelines distinct within a run; repeated runs reuse
 * Metal's shader cache, so the medians measure pipeline building rather than compiling. */
API_AVAILABLE(macos(26.0))
static int pipelines(void)
{
    enum { Count = 20 };
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) { puts("skip: Metal 4 family unsupported"); return 0; }
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    NSError *error = nil;
    id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    if (!queue || !compiler) { failure(@"Metal 4 compiler setup failed", error); return 1; }
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double medians[2];
    for (int path = 0; path < 2; path++) {
        uint64_t ticks[Count];
        for (int i = 0; i < Count; i++) @autoreleasepool {
            NSString *source = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n"
                "constant uint salt = %d;\n"
                "kernel void k(const device uint *in [[buffer(0)]], device uint *out [[buffer(1)]],"
                " uint i [[thread_position_in_grid]]) { out[i] = in[i] + salt; }\n", path * Count + i];
            id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
            if (!library) { failure(@"shader compilation failed", error); return 1; }
            id pipeline;
            uint64_t start;
            if (path == 0) {
                MTL4LibraryFunctionDescriptor *function = [MTL4LibraryFunctionDescriptor new];
                function.library = library;
                function.name = @"k";
                MTL4ComputePipelineDescriptor *desc = [MTL4ComputePipelineDescriptor new];
                desc.computeFunctionDescriptor = function;
                start = mach_absolute_time();
                pipeline = [compiler newComputePipelineStateWithDescriptor:desc compilerTaskOptions:nil error:&error];
            } else {
                id<MTLFunction> function = [library newFunctionWithName:@"k"];
                start = mach_absolute_time();
                pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            }
            ticks[i] = mach_absolute_time() - start;
            if (!pipeline) { failure(@"pipeline creation failed", error); return 1; }
        }
        qsort(ticks, Count, sizeof ticks[0], compareTicks);
        medians[path] = (ticks[Count / 2 - 1] + ticks[Count / 2]) / 2.0 * timebase.numer / timebase.denom / 1e6;
    }
    printf("pipelines: %d fresh compute pipelines each, median %.4f ms MTL4Compiler, %.4f ms device\n",
           Count, medians[0], medians[1]);
    int (*stats)(long *, int) = (void *)dlsym(RTLD_DEFAULT, "fencepost_stats");
    if (stats) {
        long reports[2];   /* ST_RACE_UNIQ, ST_POSS_UNIQ */
        if (stats(reports, 2) < 2) return 1;
        printf("fencepost stats: %ld races, %ld possible\n", reports[0], reports[1]);
    }
    return 0;
}

static int benchmark(BOOL metal4, BOOL commit_heavy, NSUInteger draws)
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        puts("skip: Metal returned no default device");
        return 0;
    }
    if (metal4) {
        if (@available(macOS 26.0, *)) {
            if (![device supportsFamily:MTLGPUFamilyMetal4]) {
                puts("skip: Metal 4 family unsupported");
                return 0;
            }
        } else {
            puts("skip: Metal 4 requires macOS 26 or later");
            return 0;
        }
    }

    // A Metal 4 program makes its queue before its pipelines, so they carry binding reflection.
    id queue = nil;
    if (metal4) {
        if (@available(macOS 26.0, *)) queue = [device newMTL4CommandQueue];
    }
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shader options:nil error:&error];
    if (!library) { failure(@"shader compilation failed", error); return 1; }
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"benchVertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@"benchFragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!pipeline) { failure(@"render pipeline creation failed", error); return 1; }

    float vertexData[2][3][4];
    memcpy(vertexData[0], vertices, sizeof(vertices));
    memcpy(vertexData[1], vertices, sizeof(vertices));
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> vertexBuffer =
        [device newBufferWithBytes:vertexData length:sizeof(vertexData) options:MTLResourceStorageModeShared];
    NSUInteger size = commit_heavy ? 16 : 256;
    MTLTextureDescriptor *targetDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:size height:size mipmapped:NO];
    targetDesc.storageMode = MTLStorageModeShared;
    targetDesc.usage = MTLTextureUsageRenderTarget;
    NSMutableArray<id<MTLTexture>> *targets = [NSMutableArray new];
    NSUInteger slots = commit_heavy ? CommitSlots : 1;
    for (NSUInteger slot = 0; slot < slots; slot++) {
        id<MTLTexture> target = [device newTextureWithDescriptor:targetDesc];
        if (!target) { failure(@"render target allocation failed", nil); return 1; }
        [targets addObject:target];
    }

    MTLTextureDescriptor *sourceDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:1 height:1 mipmapped:NO];
    sourceDesc.storageMode = MTLStorageModeShared;
    sourceDesc.usage = MTLTextureUsageShaderRead;
    __attribute__((objc_precise_lifetime)) id<MTLTexture> first = [device newTextureWithDescriptor:sourceDesc];
    __attribute__((objc_precise_lifetime)) id<MTLTexture> second = [device newTextureWithDescriptor:sourceDesc];
    if (!vertexBuffer || !first || !second) { failure(@"resource allocation failed", nil); return 1; }
    MTLRegion pixelRegion = MTLRegionMake2D(0, 0, 1, 1);
    [first replaceRegion:pixelRegion mipmapLevel:0 withBytes:colors[0] bytesPerRow:sizeof(colors[0])];
    [second replaceRegion:pixelRegion mipmapLevel:0 withBytes:colors[1] bytesPerRow:sizeof(colors[1])];
    NSArray<id<MTLTexture>> *textures = @[first, second];

    uint64_t ticks[CommitCount - CommitWarmup];
    BOOL complete;
    if (metal4) {
        if (@available(macOS 26.0, *))
            complete = runMetal4(device, queue, pipeline, vertexBuffer, textures, targets, draws, commit_heavy, ticks);
        else return 1;
    } else {
        complete = runMetal3(device, pipeline, vertexBuffer, textures, targets[0], draws, ticks);
    }
    if (!complete) return 1;

    uint8_t pixel[4];
    [targets[0] getBytes:pixel bytesPerRow:sizeof(pixel) fromRegion:pixelRegion mipmapLevel:0];
    printf("pixel: %u,%u,%u,%u\n", pixel[0], pixel[1], pixel[2], pixel[3]);
    if (memcmp(pixel, colors[(draws - 1) % 2], sizeof(pixel)) != 0) {
        failure(@"rendered pixel differs from the final draw color", nil);
        return 1;
    }

    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    int samples = commit_heavy ? CommitCount - CommitWarmup : SampleCount;
    double microsecondsPerTick = (double)timebase.numer / timebase.denom / 1000;
    qsort(ticks, (size_t)samples, sizeof(ticks[0]), compareTicks);
    double median = ticks[samples / 2] * microsecondsPerTick;
    double p90 = ticks[(samples * 9 + 9) / 10 - 1] * microsecondsPerTick;
    if (commit_heavy)
        printf("metal4-commit: %lu draws/commit, %d commits (%d warmup), %d slots, median %.6f p90 %.6f CPU us/commit\n",
               (unsigned long)draws, CommitCount, CommitWarmup, CommitSlots, median, p90);
    else
        printf("%s: %lu draws/frame, %d frames (%d warmup), median %.6f p90 %.6f CPU us/draw\n",
               metal4 ? "metal4" : "metal3", (unsigned long)draws, FrameCount, WarmupCount, median / draws, p90 / draws);
    return 0;
}

int main(int argc, const char *argv[])
{
    @autoreleasepool {
        BOOL commit_heavy = argc >= 2 && strcmp(argv[1], "metal4-commit") == 0;
        if (argc == 2 && !strcmp(argv[1], "pipelines")) {
            if (@available(macOS 26.0, *)) return pipelines();
            puts("skip: Metal 4 requires macOS 26 or later");
            return 0;
        }
        if (argc < 2 || argc > 3 || (!commit_heavy && strcmp(argv[1], "metal3") && strcmp(argv[1], "metal4"))) {
            fprintf(stderr, "usage: mtl4bench {metal3|metal4} [draws-per-frame] | metal4-commit [draws-per-commit] | pipelines\n");
            return 2;
        }
        unsigned long draws = commit_heavy ? 4 : 10000;
        if (argc == 3) {
            char *end;
            draws = strtoul(argv[2], &end, 10);
            unsigned long max_draws = commit_heavy ? 10 : 100000;
            if (*end || draws == 0 || draws > max_draws) {
                fprintf(stderr, "draws must be an integer from 1 to %lu\n", max_draws);
                return 2;
            }
        }
        return benchmark(commit_heavy || strcmp(argv[1], "metal4") == 0, commit_heavy, draws);
    }
}
