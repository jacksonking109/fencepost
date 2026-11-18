/* One command buffer and an explicit frame boundary per iteration. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>

int main(void)
{
    @autoreleasepool {
        void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
        void (*trigger)(const char *) = dlsym(RTLD_DEFAULT, "fencepost_trigger");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !frame || !trigger) return 2;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> buffer = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        trigger("idle");
        for (int i = 1; i <= 40; i++) @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
            [enc fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:42];
            [enc endEncoding];
            if (i == 30) trigger("Bad Frame");
            frame(cb);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) return 1;
        }
        [buffer release];
        [queue release];
    }
    return 0;
}
