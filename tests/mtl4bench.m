#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>

enum { FrameCount = 60, WarmupCount = 5, SampleCount = FrameCount - WarmupCount };

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
static BOOL runMetal4(id<MTLDevice> device, id<MTLRenderPipelineState> pipeline,
                      id<MTLBuffer> vertexBuffer, NSArray<id<MTLTexture>> *textures,
                      id<MTLTexture> target, NSUInteger draws, uint64_t ticks[SampleCount])
{
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    if (!queue || !allocator || !commands) return failure(@"Metal 4 command setup failed", nil);

    NSError *error = nil;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    if (!residency) return failure(@"residency set creation failed", error);
    [residency addAllocation:vertexBuffer];
    [residency addAllocation:target];
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

    MTL4RenderPassDescriptor *pass = [MTL4RenderPassDescriptor new];
    configureAttachment(pass.colorAttachments[0], target);
    dispatch_semaphore_t completion = dispatch_semaphore_create(0);
    __block NSError *gpuError = nil;

    for (int frame = 0; frame < FrameCount; ++frame) @autoreleasepool {
        // Register fresh feedback outside the timed section. Reusing options can lose a callback.
        MTL4CommitOptions *commit = [MTL4CommitOptions new];
        [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            gpuError = feedback.error;
            dispatch_semaphore_signal(completion);
        }];
        uint64_t start = mach_absolute_time();
        [commands beginCommandBufferWithAllocator:allocator];
        id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
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
        if (dispatch_semaphore_wait(completion, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)))
            return failure(@"timed out waiting for Metal 4 commit feedback", nil);
        if (gpuError) return failure(@"Metal 4 render failed", gpuError);
        [allocator reset];
        if (frame >= WarmupCount) ticks[frame - WarmupCount] = elapsed;
    }
    return YES;
}

static int compareTicks(const void *a, const void *b)
{
    uint64_t first = *(const uint64_t *)a, second = *(const uint64_t *)b;
    return (first > second) - (first < second);
}

static int benchmark(BOOL metal4, NSUInteger draws)
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
    MTLTextureDescriptor *targetDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:256 height:256 mipmapped:NO];
    targetDesc.storageMode = MTLStorageModeShared;
    targetDesc.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [device newTextureWithDescriptor:targetDesc];

    MTLTextureDescriptor *sourceDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:1 height:1 mipmapped:NO];
    sourceDesc.storageMode = MTLStorageModeShared;
    sourceDesc.usage = MTLTextureUsageShaderRead;
    __attribute__((objc_precise_lifetime)) id<MTLTexture> first = [device newTextureWithDescriptor:sourceDesc];
    __attribute__((objc_precise_lifetime)) id<MTLTexture> second = [device newTextureWithDescriptor:sourceDesc];
    if (!vertexBuffer || !target || !first || !second) { failure(@"resource allocation failed", nil); return 1; }
    MTLRegion pixelRegion = MTLRegionMake2D(0, 0, 1, 1);
    [first replaceRegion:pixelRegion mipmapLevel:0 withBytes:colors[0] bytesPerRow:sizeof(colors[0])];
    [second replaceRegion:pixelRegion mipmapLevel:0 withBytes:colors[1] bytesPerRow:sizeof(colors[1])];
    NSArray<id<MTLTexture>> *textures = @[first, second];

    uint64_t ticks[SampleCount];
    BOOL complete;
    if (metal4) {
        if (@available(macOS 26.0, *)) complete = runMetal4(device, pipeline, vertexBuffer, textures, target, draws, ticks);
        else return 1;
    } else {
        complete = runMetal3(device, pipeline, vertexBuffer, textures, target, draws, ticks);
    }
    if (!complete) return 1;

    uint8_t pixel[4];
    [target getBytes:pixel bytesPerRow:sizeof(pixel) fromRegion:pixelRegion mipmapLevel:0];
    printf("pixel: %u,%u,%u,%u\n", pixel[0], pixel[1], pixel[2], pixel[3]);
    if (memcmp(pixel, colors[(draws - 1) % 2], sizeof(pixel)) != 0) {
        failure(@"rendered pixel differs from the final draw color", nil);
        return 1;
    }

    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double microsecondsPerTickPerDraw = (double)timebase.numer / timebase.denom / 1000 / draws;
    qsort(ticks, SampleCount, sizeof(ticks[0]), compareTicks);
    printf("%s: %lu draws/frame, %d frames (%d warmup), median %.6f p90 %.6f CPU us/draw\n",
           metal4 ? "metal4" : "metal3", (unsigned long)draws, FrameCount, WarmupCount,
           ticks[SampleCount / 2] * microsecondsPerTickPerDraw,
           ticks[(SampleCount * 9 + 9) / 10 - 1] * microsecondsPerTickPerDraw);
    return 0;
}

int main(int argc, const char *argv[])
{
    @autoreleasepool {
        if (argc < 2 || argc > 3 || (strcmp(argv[1], "metal3") && strcmp(argv[1], "metal4"))) {
            fprintf(stderr, "usage: mtl4bench {metal3|metal4} [draws-per-frame]\n");
            return 2;
        }
        unsigned long draws = 10000;
        if (argc == 3) {
            char *end;
            draws = strtoul(argv[2], &end, 10);
            if (*end || draws == 0 || draws > 100000) {
                fprintf(stderr, "draws-per-frame must be an integer from 1 to 100000\n");
                return 2;
            }
        }
        return benchmark(strcmp(argv[1], "metal4") == 0, draws);
    }
}
