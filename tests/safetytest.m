#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static int stream_trace(void)
{
    long (*counts)(long *) = dlsym(RTLD_DEFAULT, "fencepost_test_safety_trace_live");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [[device newCommandQueue] autorelease];
    id<MTLSharedEvent> event = [[device newSharedEvent] autorelease];
    id<MTLBuffer> buffer = [[device newBufferWithLength:4096 options:MTLResourceStorageModeShared] autorelease];
    if (!counts || !queue || !event || !buffer) return 2;
    for (unsigned i = 1; i <= 20000; i++) @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
        if (!encoder) return 2;
        [encoder fillBuffer:buffer range:NSMakeRange(0, 4096) value:0];
        [encoder endEncoding];
        [cb encodeSignalEvent:event value:i];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return 2;
    }
    long endpoints;
    long buffers = counts(&endpoints);
    printf("20000 completed, %ld live CB records, %ld live endpoints\n", buffers, endpoints);
    return buffers || endpoints;
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        if (argc != 2) return 2;
        const char *mode = argv[1];
        if (!strcmp(mode, "stream")) return stream_trace();
        if (!strcmp(mode, "log")) {
            void (*probe)(void) = dlsym(RTLD_DEFAULT, "fencepost_test_safety_log");
            if (!probe) return 2;
            probe();
            puts("detector returned with blocked pipe");
            fflush(stdout);
            return 0;
        }
        char symbol[96];
        snprintf(symbol, sizeof symbol, "fencepost_test_safety_%s", mode);
        int (*probe)(void) = dlsym(RTLD_DEFAULT, symbol);
        return probe ? !probe() : 2;
    }
}
