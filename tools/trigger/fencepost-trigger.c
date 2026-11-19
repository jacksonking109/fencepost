#include "trigger_write.h"
#include <limits.h>

int main(int argc, char **argv)
{
    const char *path = NULL, *reason = NULL;
    int options = 1, capture = 0;
    pid_t pid = 0;
    for (int i = 1; i < argc; i++) {
        if (options && !strcmp(argv[i], "--")) {
            options = 0;
        } else if (options && !strcmp(argv[i], "--file")) {
            if (path || ++i == argc) goto usage;
            path = argv[i];
        } else if (options && !strcmp(argv[i], "--capture")) {
            if (capture) goto usage;
            capture = 1;
        } else if (options && !strcmp(argv[i], "--pid")) {
            if (pid || ++i == argc) goto usage;
            char *end;
            long value = strtol(argv[i], &end, 10);
            if (*end || value <= 0 || value > INT_MAX) goto usage;
            pid = (pid_t)value;
        } else {
            if (reason || (options && argv[i][0] == '-')) goto usage;
            reason = argv[i];
        }
    }

    char default_path[64];
    if (!path) path = capture ? capture_default_path(default_path) : trigger_default_path(default_path);
    if (trigger_write(path, reason ? reason : "manual", pid) == -1) {
        fprintf(stderr, "fencepost-trigger: %s: %s\n", path, strerror(errno));
        return 1;
    }
    return 0;

usage:
    fputs("usage: fencepost-trigger [reason] [--capture] [--file PATH] [--pid N]\n", stderr);
    return 2;
}
