/* A fence, a planted race and a cross-queue event in two frames. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static void fill(id<MTLBlitCommandEncoder> enc, NSString *label, id<MTLBuffer> buffer,
                 id<MTLBuffer> tracked, id<MTLFence> wait, id<MTLFence> update)
{
    enc.label = label;
    if (wait) [enc waitForFence:wait];
    [enc fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:42];
    if (tracked) [enc fillBuffer:tracked range:NSMakeRange(0, tracked.length) value:42];
    if (update) [enc updateFence:update];
    [enc endEncoding];
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        int app_counters = argc == 2 && !strcmp(argv[1], "--app-counters");
        void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
        if (!frame) {
            fprintf(stderr, "tracetest: inject libfencepost.dylib\n");
            return 2;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q0 = [dev newCommandQueue], q1 = [dev newCommandQueue];
        q0.label = @"producer \"GPU\"";
        q1.label = @"consumer";
        id<MTLCommandBuffer> produce = [q0 commandBuffer], consume = [q1 commandBuffer];
        produce.label = @"produce";
        consume.label = @"consume";
        MTLResourceOptions options = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> independent = [dev newBufferWithLength:65536 options:options];
        id<MTLBuffer> raced = [dev newBufferWithLength:65536 options:options];
        /* Give the viewer a forward-in-time race arrow while still planting a
         * missing fence on the untracked resource fencepost actually checks. */
        id<MTLBuffer> tracked = [dev newBufferWithLength:65536
                                               options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked];
        raced.label = @"raced \\ buffer";
        id<MTLFence> fence = [dev newFence];
        id<MTLSharedEvent> signal = [dev newSharedEvent];
        id<MTLSharedEvent> producer_gate = [dev newSharedEvent], consumer_gate = [dev newSharedEvent];

        [produce encodeWaitForEvent:producer_gate value:1];
        fill([produce blitCommandEncoder], @"fence update", independent, nil, nil, fence);
        id<MTLBlitCommandEncoder> empty = [produce blitCommandEncoder];
        empty.label = @"empty blit";
        [empty endEncoding];
        fill([produce blitCommandEncoder], @"fence wait", raced, tracked, fence, nil);

        MTLBlitPassDescriptor *pass = [MTLBlitPassDescriptor blitPassDescriptor];
        id<MTLCounterSampleBuffer> app_buffer = nil;
        if (app_counters && [dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            for (id<MTLCounterSet> set in dev.counterSets) {
                if (![set.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
                MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
                d.counterSet = set; d.sampleCount = 2; d.storageMode = MTLStorageModeShared;
                app_buffer = [dev newCounterSampleBufferWithDescriptor:d error:NULL];
                [d release];
                break;
            }
        }
        if (app_buffer) {
            pass.sampleBufferAttachments[0].sampleBuffer = app_buffer;
            pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
            pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
        }
        fill([produce blitCommandEncoderWithDescriptor:pass], @"racing write", raced, tracked, nil, nil);
        if (pass.sampleBufferAttachments[0].sampleBuffer != app_buffer || pass.sampleBufferAttachments[1].sampleBuffer) {
            fprintf(stderr, "tracetest: app's sample buffer attachments changed\n");
            return 1;
        }
        [produce encodeSignalEvent:signal value:1];
        frame(produce);
        [produce enqueue];
        [produce commit];

        [consume encodeWaitForEvent:signal value:1];
        /* Commit both before either completes, and give the wait pass a later GPU span. */
        [consume encodeWaitForEvent:consumer_gate value:1];
        id<MTLBuffer> other_queue = [dev newBufferWithLength:65536 options:options];
        fill([consume blitCommandEncoder], @"event wait", other_queue, nil, nil, nil);
        frame(consume);
        [consume commit];
        producer_gate.signaledValue = 1;
        [produce waitUntilCompleted];
        consumer_gate.signaledValue = 1;
        [consume waitUntilCompleted];
        if (produce.status != MTLCommandBufferStatusCompleted || consume.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"tracetest: Metal command failed: %@ / %@", produce.error, consume.error);
            return 1;
        }
        if (app_buffer) {
            NSData *samples = [app_buffer resolveCounterRange:NSMakeRange(0, 2)];
            const MTLCounterResultTimestamp *ts = samples.bytes;
            if (samples.length != 2 * sizeof *ts || !ts[0].timestamp ||
                    ts[1].timestamp <= ts[0].timestamp || ts[1].timestamp == MTLCounterErrorValue) {
                fprintf(stderr, "tracetest: app's counter samples are invalid\n");
                return 1;
            }
        }
        [app_buffer release];
        [independent release]; [raced release]; [tracked release]; [other_queue release];
        [fence release]; [signal release]; [producer_gate release]; [consumer_gate release];
        [q0 release]; [q1 release];
        fprintf(stderr, "tracetest: both command buffers completed\n");
    }
    return 0;
}
