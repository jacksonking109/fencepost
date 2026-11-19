/* Safe, serial blits with optional explicit frame boundaries. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

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
        int api = !strcmp(mode, "api"), exitActive = !strcmp(mode, "exit-active");
        int appCapture = !strcmp(mode, "app-capture"), multi = !strcmp(mode, "multi-frame");
        int idle = !strcmp(mode, "idle");
        if (!file && !noFrame && !api && !exitActive && !appCapture && !multi && !idle) return 2;

        void (*frame)(id) = dlsym(RTLD_DEFAULT, "fencepost_frame");
        void (*capture)(const char *) = dlsym(RTLD_DEFAULT, "fencepost_capture");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !frame || !capture) return 2;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> buffer = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> first = [queue commandBuffer];
        if (!first || !buffer) return 2;

        if (file || noFrame) {
            puts("ready");
            fflush(stdout);
            if (getchar() == EOF) return 2;
        }
        if (!file) capture("API Request");
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
        if (api || exitActive || multi || idle) {
            if (!manager.isCapturing) return 3;
            capture("Ignore While Active");
        }
        if (!submit(second, buffer, noFrame ? NULL : frame)) return 1;
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
