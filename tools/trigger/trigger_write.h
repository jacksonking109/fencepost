#ifndef FENCEPOST_TRIGGER_WRITE_H
#define FENCEPOST_TRIGGER_WRITE_H

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

static inline const char *trigger_default_path(char buffer[64])
{
    const char *path = getenv("FENCEPOST_TRIGGER_FILE");
    if (path && *path) return path;
    snprintf(buffer, 64, "/tmp/fencepost-trigger-%u", getuid());
    return buffer;
}

static inline const char *capture_default_path(char buffer[64])
{
    const char *path = getenv("FENCEPOST_CAPTURE_FILE");
    if (path && *path) return path;
    snprintf(buffer, 64, "/tmp/fencepost-capture-%u", getuid());
    return buffer;
}

static inline int trigger_time_newer(struct timespec a, struct timespec b)
{
    return a.tv_sec > b.tv_sec || (a.tv_sec == b.tv_sec && a.tv_nsec > b.tv_nsec);
}

static inline int trigger_advance_mtime(int fd, const char *path)
{
    struct stat old, temp;
    if (stat(path, &old) == -1) return errno == ENOENT ? 0 : -1;
    if (fstat(fd, &temp) == -1) return -1;
    if (trigger_time_newer(temp.st_mtimespec, old.st_mtimespec)) return 0;

    struct timespec times[2] = {temp.st_atimespec, old.st_mtimespec};
    if (++times[1].tv_nsec == 1000000000L) {
        times[1].tv_sec++;
        times[1].tv_nsec = 0;
    }
    if (futimens(fd, times) == -1 || fstat(fd, &temp) == -1) return -1;
    if (trigger_time_newer(temp.st_mtimespec, old.st_mtimespec)) return 0;

    /* Some filesystems round timestamps down, including to two-second increments. */
    times[1].tv_sec = old.st_mtimespec.tv_sec + 2;
    times[1].tv_nsec = 0;
    if (futimens(fd, times) == -1 || fstat(fd, &temp) == -1) return -1;
    if (trigger_time_newer(temp.st_mtimespec, old.st_mtimespec)) return 0;
    errno = EIO;
    return -1;
}

static inline int trigger_write(const char *path, const char *reason, pid_t pid)
{
    if (!*path) {
        errno = EINVAL;
        return -1;
    }
    const char *slash = strrchr(path, '/');
    char *directory = slash ? strndup(path, (size_t)(slash - path) + 1) : strdup("./");
    if (!directory) return -1;

    char *temp = NULL;
    int result = -1, fd = -1, lock = -1, created = 0;
    if (asprintf(&temp, "%s.fencepost-trigger.XXXXXX", directory) == -1) goto done;
    fd = mkstemp(temp);
    if (fd == -1) goto done;
    created = 1;
    int length = (int)strcspn(reason, "\n");
    if (dprintf(fd, "%.*s\n", length, reason) < 0) goto done;
    if (pid > 0 && dprintf(fd, "pid=%d\n", pid) < 0) goto done;

    /* Serialize the timestamp check and rename among helper processes without a lockfile. */
    lock = open(directory, O_RDONLY | O_CLOEXEC);
    if (lock == -1 || flock(lock, LOCK_EX) == -1) goto done;
    if (trigger_advance_mtime(fd, path) == -1) goto done;
    int closed = close(fd);
    fd = -1;
    if (closed == -1) goto done;
    result = rename(temp, path);

done:;
    int error = errno;
    if (fd != -1) close(fd);
    if (created && result == -1) unlink(temp);
    if (lock != -1) close(lock);
    free(temp);
    free(directory);
    errno = error;
    return result;
}

#endif
