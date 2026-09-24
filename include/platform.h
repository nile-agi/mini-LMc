#ifndef MINILMC_PLATFORM_H
#define MINILMC_PLATFORM_H

#include <stddef.h>

/*
 * platform.h — the only portability seam in mini-LMc.
 *
 * Rule: no file other than platform.c may #include <sys/mman.h>,
 * <unistd.h>, <fcntl.h>, or <windows.h>, or call open()/mmap()/
 * CreateFileA() directly. Every other translation unit talks to the
 * filesystem exclusively through the two functions declared here.
 */

typedef struct {
  const void *data; /* read-only view of the whole file; NULL if unmapped */
  size_t size;      /* exact file size in bytes                          */
  void *impl_; /* opaque OS handle(s) — never touched outside platform.c */
} mlmc_mapped_file_t;

/* Opens and maps `path` read-only in one step.
 * Returns 0 and fills *out on success; returns -1 and zeroes *out on
 * failure. */
int mlmc_map_file(const char *path, mlmc_mapped_file_t *out);

/* Releases everything opened by mlmc_map_file(). Safe to call on a
 * zeroed struct, and safe to call twice. */
void mlmc_unmap_file(mlmc_mapped_file_t *f);

#endif /* MINILMC_PLATFORM_H */