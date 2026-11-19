/* Safe, serial blits with optional explicit frame boundaries. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int submit(id<MTLCommandBuffer> cb, id<MTLBuffer> buffer, void (*frame)(id))
{
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    if (!enc) return 0;
    [enc fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:42];
    [enc endEncoding];
    if (frame) frame(cb);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusCompleted) return 1;
    fprintf(stderr, "capturetest: %s\n", cb.error.localizedDescription.UTF8String ?: "command buffer failed");
    return 0;
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        const char *mode = argc == 2 ? argv[1] : "";
        int file = !strcmp(mode, "file"), noFrame = !strcmp(mode, "no-frame");
        int emptyFile = !strcmp(mode, "empty-file"), wrongPid = !strcmp(mode, "wrong-pid");
        int api = !strcmp(mode, "api"), exitActive = !strcmp(mode, "exit-active");
        int appCapture = !strcmp(mode, "app-capture"), multi = !strcmp(mode, "multi-frame");
        int idle = !strcmp(mode, "idle"), slow = !strcmp(mode, "slow-frame");
        int hardLimit = !strcmp(mode, "hard-limit");
        if (!file && !emptyFile && !wrongPid && !noFrame && !api && !exitActive && !appCapture && !multi &&
            !idle && !slow && !hardLimit) return 2;

        void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
        void (*capture)(const char *) = dlsym(RTLD_DEFAULT, "fencepost_capture");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !frame || !capture) return 2;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        // The capture stores the larger buffer in a named snapshot file.
        NSUInteger length = api ? 8 * 1024 * 1024 : 4096;
        id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> first = [queue commandBuffer];
        if (!first || !buffer) return 2;

        if (file || emptyFile || wrongPid || noFrame) {
            puts("ready");
            fflush(stdout);
            if (getchar() == EOF) return 2;
        }
        if (!file && !emptyFile && !wrongPid) capture("API Request");
        MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
        if (manager.isCapturing) return 3;
        if (!submit(first, buffer, noFrame ? NULL : frame)) return 1;

        if (appCapture) {
            NSString *path = [NSString stringWithUTF8String:getenv("CAPTURETEST_APP_TRACE") ?: ""];
            if (!path.length) return 2;
            MTLCaptureDescriptor *captureDesc = [[[MTLCaptureDescriptor alloc] init] autorelease];
            captureDesc.captureObject = device;
            captureDesc.destination = MTLCaptureDestinationGPUTraceDocument;
            captureDesc.outputURL = [NSURL fileURLWithPath:path];
            NSError *error;
            if (![manager startCaptureWithDescriptor:captureDesc error:&error]) {
                fprintf(stderr, "capturetest: %s\n", error.localizedDescription.UTF8String);
                return 3;
            }
        }

        MTLCommandBufferDescriptor *descriptor = [[[MTLCommandBufferDescriptor alloc] init] autorelease];
        id<MTLCommandBuffer> second = [queue commandBufferWithDescriptor:descriptor];
        if ((emptyFile || wrongPid) && manager.isCapturing) return 3;
        if (api || exitActive || multi || idle || slow || hardLimit) {
            if (!manager.isCapturing) return 3;
            capture("Ignore While Active");
        }
        if (!submit(second, buffer, noFrame || slow ? NULL : frame)) return 1;
        if (slow) {
            /* Frame 2 takes more than 2 s, but commits arrive every 750 ms. */
            for (int i = 0; i < 4; i++) @autoreleasepool {
                usleep(750000);
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                if (!manager.isCapturing) return 3;
                if (!submit(cb, buffer, i == 3 ? frame : NULL)) return 1;
            }
            return manager.isCapturing ? 3 : 0;
        }
        if (hardLimit) {
            for (int i = 0; i < 60 && manager.isCapturing; i++) @autoreleasepool {
                usleep(100000);
                if (!submit([queue commandBuffer], buffer, frame)) return 1;
            }
            return manager.isCapturing ? 3 : 0;
        }
        if (emptyFile || wrongPid) {
            puts("request polled");
            fflush(stdout);
            if (getchar() == EOF) return 2;
        }
        if (wrongPid) {
            id<MTLCommandBuffer> sameStamp = [queue commandBuffer];
            if (manager.isCapturing) return 3;
            if (!submit(sameStamp, buffer, frame)) return 1;
            puts("same mtime polled");
            fflush(stdout);
            if (getchar() == EOF) return 2;
        }
        if (idle) {
            if (getchar() == EOF) return 2;
            return manager.isCapturing ? 3 : 0;
        }
        if (exitActive) return manager.isCapturing ? 0 : 3;
        if (appCapture) {
            if (!manager.isCapturing) return 3;
            [manager stopCapture];
        }
        if ((api || multi) && manager.isCapturing != multi) return 3;

        for (int i = 0; i < 3; i++) @autoreleasepool {
            if (!submit([queue commandBufferWithUnretainedReferences], buffer, noFrame ? NULL : frame)) return 1;
            if ((api || multi) && manager.isCapturing) return 3;
        }
        [buffer release];
        [queue release];
    }
    return 0;
}
