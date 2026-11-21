#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

enum { WordCount = 65536, GroupSize = 64 };

static NSString *const shader = @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void produce(device uint *data [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    "    data[i] = i + 1;\n"
    "}\n"
    "kernel void consume(const device uint *data [[buffer(0)]], device uint *sum [[buffer(1)]]) {\n"
    "    uint total = 0;\n"
    "    for (uint i = 0; i < wordCount; ++i) total += data[i];\n"
    "    *sum = total;\n"
    "}\n";

static int failure(NSString *operation, NSError *error)
{
    fprintf(stderr, "mtl4test: %s%s%s\n", operation.UTF8String, error ? ": " : "",
            error ? error.description.UTF8String : "");
    return 1;
}

static BOOL waitForFeedback(dispatch_semaphore_t completion, NSString *label, id<MTLSharedEvent> event)
{
    if (!dispatch_semaphore_wait(completion, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)))
        return YES;

    failure([NSString stringWithFormat:@"timed out waiting for %@ commit feedback", label], nil);
    if (event) {
        fprintf(stderr, "event.signaledValue at timeout: %llu\n", (unsigned long long)event.signaledValue);
        [NSThread sleepForTimeInterval:5];
        fprintf(stderr, "event.signaledValue 5 s later: %llu\n", (unsigned long long)event.signaledValue);
    }
    return NO;
}

API_AVAILABLE(macos(26.0))
static BOOL timing_commit(id<MTL4CommandQueue> queue, const id<MTL4CommandBuffer> *buffers, NSUInteger count)
{
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block NSError *error = nil;
    MTL4CommitOptions *options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    [queue commit:buffers count:count options:options];
    if (!waitForFeedback(completed, @"timing", nil)) return NO;
    if (error) failure(@"timing workload failed", error);
    return error == nil;
}

API_AVAILABLE(macos(26.0))
static int timing_dispatches(id<MTLDevice> device, BOOL ordered)
{
    static const unsigned iterations[] = { 128, 512, 2048 };
    NSError *error = nil;
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandBuffer> cb = [device newCommandBuffer];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    if (!queue || !cb || !allocator || !residency) return failure(@"timing command setup failed", error);

    MTL4ArgumentTableDescriptor *layout = [MTL4ArgumentTableDescriptor new];
    layout.maxBufferBindCount = 1;
    id<MTLComputePipelineState> pipelines[3];
    id<MTL4ArgumentTable> tables[3];
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> outputs[3];
    for (NSUInteger i = 0; i < 3; i++) {
        NSString *source = [NSString stringWithFormat:
            @"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void busy(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
            "  float4 a = float4(float(gid & 255u), 1.0, 2.0, 3.0);\n"
            "  float4 b = float4(1.0000001, 0.9999999, 1.0000002, 0.9999998);\n"
            "  for (uint k = 0; k < %uu; ++k) a = fma(a, b, float4(0.5, 0.25, 0.125, 0.0625));\n"
            "  out[gid] = uint(a.x + a.y + a.z + a.w);\n"
            "}\n", iterations[i]];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) return failure(@"timing shader compile failed", error);
        pipelines[i] = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"busy"] error:&error];
        tables[i] = [device newArgumentTableWithDescriptor:layout error:&error];
        outputs[i] = [device newBufferWithLength:WordCount * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!pipelines[i] || !tables[i] || !outputs[i]) return failure(@"timing dispatch setup failed", error);
        [tables[i] setAddress:outputs[i].gpuAddress atIndex:0];
        [residency addAllocation:outputs[i]];
    }
    [residency commit];
    [queue addResidencySet:residency];

    /* Warm the pipelines and wait before reusing their independent outputs. */
    for (int run = 0; run < 3; run++) {
        [allocator reset];
        [cb beginCommandBufferWithAllocator:allocator];
        cb.label = run < 2 ? @"timing warmup" : @"timing dispatches";
        for (NSUInteger i = 0; i < 3; i++) {
            id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!enc) return failure(@"timing compute encoder creation failed", nil);
            enc.label = [NSString stringWithFormat:@"timing %lux", 1ul << (2 * i)];
            if (ordered && i)
                [enc barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch
                           visibilityOptions:MTL4VisibilityOptionNone];
            [enc setComputePipelineState:pipelines[i]];
            [enc setArgumentTable:tables[i]];
            [enc dispatchThreads:MTLSizeMake(WordCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        [cb endCommandBuffer];
        if (!timing_commit(queue, &cb, 1)) return 1;
    }
    return 0;
}

API_AVAILABLE(macos(26.0))
static int timing_overflow(id<MTLDevice> device)
{
    enum { BufferCount = 4, EncodersPerBuffer = 513, Outputs = BufferCount * EncodersPerBuffer };
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandBuffer> buffers[BufferCount];
    id<MTL4CommandAllocator> allocators[BufferCount];
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> outputs[Outputs];
    NSError *error = nil;
    MTLResidencySetDescriptor *residencyDesc = [MTLResidencySetDescriptor new];
    residencyDesc.initialCapacity = Outputs;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residencyDesc error:&error];
    if (!queue || !residency) return failure(@"timing overflow setup failed", error);
    for (NSUInteger i = 0; i < Outputs; i++) {
        outputs[i] = [device newBufferWithLength:GroupSize * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!outputs[i]) return failure(@"timing overflow allocation failed", nil);
        [residency addAllocation:outputs[i]];
    }
    [residency commit];
    [queue addResidencySet:residency];
    MTLCompileOptions *compile = [MTLCompileOptions new];
    compile.preprocessorMacros = @{@"wordCount": @(GroupSize)};
    id<MTLLibrary> library = [device newLibraryWithSource:shader options:compile error:&error];
    if (!library) return failure(@"timing overflow shader compile failed", error);
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"produce"] error:&error];
    MTL4ArgumentTableDescriptor *layout = [MTL4ArgumentTableDescriptor new];
    layout.maxBufferBindCount = 1;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:layout error:&error];
    if (!pipeline || !table) return failure(@"timing overflow dispatch setup failed", error);

    MTLSize group = MTLSizeMake(GroupSize, 1, 1);
    for (NSUInteger i = 0; i < BufferCount; i++) {
        buffers[i] = [device newCommandBuffer];
        allocators[i] = [device newCommandAllocator];
        if (!buffers[i] || !allocators[i]) return failure(@"timing overflow command setup failed", nil);
        [buffers[i] beginCommandBufferWithAllocator:allocators[i]];
        buffers[i].label = @"timing overflow";
        for (NSUInteger j = 0; j < EncodersPerBuffer; j++) @autoreleasepool {
            id<MTL4ComputeCommandEncoder> enc = [buffers[i] computeCommandEncoder];
            if (!enc) return failure(@"timing overflow encoder creation failed", nil);
            NSUInteger index = i * EncodersPerBuffer + j;
            [table setAddress:outputs[index].gpuAddress atIndex:0];
            [enc setComputePipelineState:pipeline];
            [enc setArgumentTable:table];
            [enc dispatchThreads:group threadsPerThreadgroup:group];
            [enc endEncoding];
        }
        [buffers[i] endCommandBuffer];
    }
    if (!timing_commit(queue, buffers, BufferCount)) return 1;

    [allocators[0] reset];
    [buffers[0] beginCommandBufferWithAllocator:allocators[0]];
    buffers[0].label = @"timing reuse";
    id<MTL4ComputeCommandEncoder> empty = [buffers[0] computeCommandEncoder];
    if (!empty) return failure(@"timing empty encoder creation failed", nil);
    empty.label = @"timing empty";
    [empty endEncoding];
    id<MTL4ComputeCommandEncoder> enc = [buffers[0] computeCommandEncoder];
    if (!enc) return failure(@"timing reuse encoder creation failed", nil);
    enc.label = @"timing reused pair";
    [table setAddress:outputs[0].gpuAddress atIndex:0];
    [enc setComputePipelineState:pipeline];
    [enc setArgumentTable:table];
    [enc dispatchThreads:group threadsPerThreadgroup:group];
    [enc endEncoding];
    [buffers[0] endCommandBuffer];
    return !timing_commit(queue, buffers, 1);
}

API_AVAILABLE(macos(26.0))
static int probe(id<MTLDevice> device, NSString *test)
{
    BOOL supported = [device supportsFamily:MTLGPUFamilyMetal4];
    printf("device: %s\nsupportsFamily:MTLGPUFamilyMetal4: %s\n", device.name.UTF8String,
           supported ? "yes" : "no");
    BOOL hasQueue = [device respondsToSelector:@selector(newMTL4CommandQueue)];
    BOOL hasAllocator = [device respondsToSelector:@selector(newCommandAllocator)];
    BOOL hasBuffer = [device respondsToSelector:@selector(newCommandBuffer)];
    id<MTL4CommandQueue> producer = hasQueue ? [device newMTL4CommandQueue] : nil;
    __attribute__((objc_precise_lifetime)) id<MTL4CommandAllocator> allocator =
        hasAllocator ? [device newCommandAllocator] : nil;
    id<MTL4CommandBuffer> commands = hasBuffer ? [device newCommandBuffer] : nil;
    printf("newMTL4CommandQueue: %s\n", producer ? "ok" : hasQueue ? "nil" : "unavailable");
    printf("newCommandAllocator: %s\n", allocator ? "ok" : hasAllocator ? "nil" : "unavailable");
    printf("newCommandBuffer: %s\n", commands ? "ok" : hasBuffer ? "nil" : "unavailable");
    if (!supported || !producer || !allocator || !commands) {
        puts(supported ? "skip: Metal 4 object creation failed" : "skip: Metal 4 family unsupported");
        return 0;
    }

    BOOL late = [test isEqualToString:@"eventlate"];
    BOOL crossQueue = late || [test isEqualToString:@"event"];
    id<MTL4CommandQueue> consumer = crossQueue ? [device newMTL4CommandQueue] : producer;
    id<MTL4CommandBuffer> readCommands = crossQueue ? [device newCommandBuffer] : commands;
    __attribute__((objc_precise_lifetime)) id<MTL4CommandAllocator> readAllocator =
        crossQueue ? [device newCommandAllocator] : allocator;
    id<MTLSharedEvent> event = crossQueue ? [device newSharedEvent] : nil;
    if (!consumer || !readCommands || !readAllocator || (crossQueue && !event))
        return failure(@"cross-queue setup failed", nil);

    NSError *error = nil;
    MTLCompileOptions *compile = [MTLCompileOptions new];
    compile.preprocessorMacros = @{@"wordCount": @(WordCount)};
    id<MTLLibrary> library = [device newLibraryWithSource:shader options:compile error:&error];
    if (!library) return failure(@"shader compilation failed", error);
    __attribute__((objc_precise_lifetime)) id<MTLComputePipelineState> writePipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"produce"] error:&error];
    if (!writePipeline) return failure(@"producer pipeline creation failed", error);
    __attribute__((objc_precise_lifetime)) id<MTLComputePipelineState> readPipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"consume"] error:&error];
    if (!readPipeline) return failure(@"consumer pipeline creation failed", error);

    size_t dataSize = WordCount * sizeof(uint32_t);
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> data =
        [device newBufferWithLength:dataSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> checksum = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (!data || !checksum) return failure(@"buffer allocation failed", nil);
    memset(data.contents, 0, dataSize);
    memset(checksum.contents, 0, sizeof(uint32_t));

    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    if (!residency) return failure(@"residency set creation failed", error);
    [residency addAllocation:data];
    [residency addAllocation:checksum];
    [residency commit];
    [producer addResidencySet:residency];
    if (crossQueue) [consumer addResidencySet:residency];

    MTL4ArgumentTableDescriptor *layout = [MTL4ArgumentTableDescriptor new];
    layout.maxBufferBindCount = 1;
    id<MTL4ArgumentTable> writeArgs = [device newArgumentTableWithDescriptor:layout error:&error];
    if (!writeArgs) return failure(@"producer argument table creation failed", error);
    [writeArgs setAddress:data.gpuAddress atIndex:0];
    layout.maxBufferBindCount = 2;
    id<MTL4ArgumentTable> args = [device newArgumentTableWithDescriptor:layout error:&error];
    if (!args) return failure(@"argument table creation failed", error);
    [args setAddress:data.gpuAddress atIndex:0];
    [args setAddress:checksum.gpuAddress atIndex:1];

    [commands beginCommandBufferWithAllocator:allocator];
    id<MTL4ComputeCommandEncoder> write = [commands computeCommandEncoder];
    if (!write) return failure(@"producer encoder creation failed", nil);
    [write setComputePipelineState:writePipeline];
    [write setArgumentTable:writeArgs];
    [write dispatchThreadgroups:MTLSizeMake(WordCount / GroupSize, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(GroupSize, 1, 1)];
    [write endEncoding];
    if (crossQueue) {
        [commands endCommandBuffer];
        [readCommands beginCommandBufferWithAllocator:readAllocator];
    }

    id<MTL4ComputeCommandEncoder> read = [readCommands computeCommandEncoder];
    if (!read) return failure(@"consumer encoder creation failed", nil);
    [read setComputePipelineState:readPipeline];
    [read setArgumentTable:args];
    if ([test isEqualToString:@"ok"] || [test isEqualToString:@"wrongstage"]) {
        MTLStages after = [test isEqualToString:@"ok"] ? MTLStageDispatch : MTLStageBlit;
        [read barrierAfterQueueStages:after beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    }
    [read dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [read endEncoding];
    [readCommands endCommandBuffer];

    dispatch_semaphore_t writeDone = dispatch_semaphore_create(0);
    __block NSError *writeError = nil;
    MTL4CommitOptions *writeCommit = [MTL4CommitOptions new];
    [writeCommit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        writeError = feedback.error;
        if (crossQueue) puts("producer commit feedback received");
        dispatch_semaphore_signal(writeDone);
    }];

    dispatch_semaphore_t readDone = crossQueue ? dispatch_semaphore_create(0) : nil;
    __block NSError *readError = nil;
    MTL4CommitOptions *readCommit = crossQueue ? [MTL4CommitOptions new] : nil;
    if (crossQueue) {
        [readCommit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            readError = feedback.error;
            puts("consumer commit feedback received");
            dispatch_semaphore_signal(readDone);
        }];
    }

    id<MTLSharedEvent> gate = nil;
    if (late) {
        /* The producer queue waits for a gate the CPU opens only after the consumer is
         * committed, so the producer cannot complete, from the CPU's point of view, before
         * the deliberately early wait is submitted. Every wait has a committed or CPU signal. */
        gate = [device newSharedEvent];
        if (!gate) return failure(@"gate event creation failed", nil);
        [producer waitForEvent:gate value:1];
        [producer signalEvent:event value:1];
    }
    [producer commit:&commands count:1 options:writeCommit];
    if (crossQueue) {
        // Queue signaling covers prior work, and queue waiting gates future work.
        [producer signalEvent:event value:2];
        [consumer waitForEvent:event value:late ? 1 : 2];
        [consumer commit:&readCommands count:1 options:readCommit];
    }
    if (gate) gate.signaledValue = 1;
    if (!waitForFeedback(writeDone, crossQueue ? @"producer" : @"workload", event) ||
        (crossQueue && !waitForFeedback(readDone, @"consumer", event))) return 1;
    if (writeError) return failure(@"producer workload failed", writeError);
    if (readError) return failure(@"consumer workload failed", readError);

    uint32_t expected = (uint32_t)((uint64_t)WordCount * (WordCount + 1) / 2);
    uint32_t got = *(const uint32_t *)checksum.contents;
    printf("case %s: expected %u got %u\n", test.UTF8String, expected, got);
    return ([test isEqualToString:@"ok"] || [test isEqualToString:@"event"]) && got != expected;
}

int main(int argc, const char *argv[])
{
    @autoreleasepool {
        NSString *test = argc == 2 ? [NSString stringWithUTF8String:argv[1]] : nil;
        if (!test || ![@[@"ok", @"nobarrier", @"wrongstage", @"event", @"eventlate",
                        @"timing-ordered", @"timing-overlap", @"timing-overflow"] containsObject:test]) {
            fprintf(stderr, "usage: mtl4test {ok|nobarrier|wrongstage|event|eventlate|"
                            "timing-ordered|timing-overlap|timing-overflow}\n");
            return 2;
        }
        setvbuf(stdout, NULL, _IOLBF, 0);
        if (@available(macOS 26.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device) {
                if ([test hasPrefix:@"timing-"]) {
                    if (![device supportsFamily:MTLGPUFamilyMetal4]) return failure(@"Metal 4 unavailable", nil);
                    if ([test isEqualToString:@"timing-overflow"]) return timing_overflow(device);
                    return timing_dispatches(device, [test isEqualToString:@"timing-ordered"]);
                }
                return probe(device, test);
            }
            puts("skip: Metal returned no default device");
        } else {
            puts("skip: Metal 4 requires macOS 26 or later");
        }
        return 0;
    }
}
