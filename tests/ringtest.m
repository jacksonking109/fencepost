/* One command buffer and an explicit frame boundary per iteration. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
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
        int external = !strcmp(mode, "external");
        if (!race && !spike && !file && !external && strcmp(mode, "api")) return 2;
        const char *path = getenv("FENCEPOST_TRIGGER_FILE");
        if ((file || external) && (!path || !*path)) return 2;
        struct stat initial = {0};
        if (external && stat(path, &initial) == -1 && errno != ENOENT) return 1;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> buffer = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (!external) trigger("idle");
        uint64_t deadline = external ? clock_gettime_nsec_np(CLOCK_MONOTONIC) + 10000000000ULL : 0;
        int changed = 0;
        for (int i = 1; external ? clock_gettime_nsec_np(CLOCK_MONOTONIC) < deadline
                                 : i <= (spike ? 140 : 40); i++) @autoreleasepool {
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
                usleep(3100000);
                trigger("checkpoint");
            }
            frame(cb);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) return 1;
            if (external) {
                if (i == 10) {
                    puts("ready");
                    fflush(stdout);
                }
                struct stat current;
                if (!changed && stat(path, &current) == 0 &&
                    (current.st_ino != initial.st_ino ||
                     current.st_mtimespec.tv_sec != initial.st_mtimespec.tv_sec ||
                     current.st_mtimespec.tv_nsec != initial.st_mtimespec.tv_nsec)) {
                    changed = 1;
                    /* Give fencepost's 250 ms poll and post-trigger frames time to finish. */
                    deadline = MIN(deadline, clock_gettime_nsec_np(CLOCK_MONOTONIC) + 400000000ULL);
                }
                usleep(10000);
            }
        }
        [buffer release];
        [queue release];
    }
    return 0;
}
