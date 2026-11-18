#include "trigger_write.h"

int main(int argc, char **argv)
{
    const char *path = NULL, *reason = NULL;
    int options = 1;
    for (int i = 1; i < argc; i++) {
        if (options && !strcmp(argv[i], "--")) {
            options = 0;
        } else if (options && !strcmp(argv[i], "--file")) {
            if (path || ++i == argc) goto usage;
            path = argv[i];
        } else {
            if (reason || (options && argv[i][0] == '-')) goto usage;
            reason = argv[i];
        }
    }

    char default_path[64];
    if (!path) path = trigger_default_path(default_path);
    if (trigger_write(path, reason ? reason : "manual") == -1) {
        fprintf(stderr, "fencepost-trigger: %s: %s\n", path, strerror(errno));
        return 1;
    }
    return 0;

usage:
    fputs("usage: fencepost-trigger [reason] [--file PATH]\n", stderr);
    return 2;
}
