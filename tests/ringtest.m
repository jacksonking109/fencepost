/* One command buffer and an explicit frame boundary per iteration. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void fill(id<MTLCommandBuffer> cb, id<MTLBuffer> buffer)
{
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:42];
    [enc endEncoding];
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
        void (*trigger)(const char *) = dlsym(RTLD_DEFAULT, "fencepost_trigger");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !frame || !trigger) return 2;
        const char *mode = argc == 2 ? argv[1] : "api";
        int race = !strcmp(mode, "race"), spike = !strcmp(mode, "spike"), file = !strcmp(mode, "file");
        if (!race && !spike && !file && strcmp(mode, "api")) return 2;
        const char *path = getenv("FENCEPOST_TRIGGER_FILE");
        if (file && (!path || !*path)) return 2;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> buffer = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        trigger("idle");
        for (int i = 1; i <= (spike ? 140 : 40); i++) @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            fill(cb, buffer);
            if (race && (i == 30 || i == 35)) fill(cb, buffer);
            if (!strcmp(mode, "api") && i == 30) trigger("Bad Frame");
            if (spike) usleep(i == 131 ? 60000 : 5000);
            if (file && (i == 30 || i == 33)) {
                usleep(300000); /* cross the stat throttle */
                FILE *f = fopen(path, "w");
                if (!f) return 1;
                fputs("flicker\n", f);
                if (fclose(f)) return 1;
            }
            if (file && i == 36) {
                /* A later dump exposes suppression after the first dump closed. */
                usleep(1100000);
                trigger("checkpoint");
            }
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
