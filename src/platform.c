#include "platform.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef struct {
  HANDLE file_handle;
  HANDLE mapping_handle;
} win_impl_t;

int mlmc_map_file(const char *path, mlmc_mapped_file_t *out) {
  memset(out, 0, sizeof(*out));

  win_impl_t *impl = (win_impl_t *)calloc(1, sizeof(win_impl_t));
  if (!impl) {
    return -1;
  }
  impl->file_handle = INVALID_HANDLE_VALUE;
  impl->mapping_handle = NULL;

  impl->file_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (impl->file_handle == INVALID_HANDLE_VALUE) {
    free(impl);
    return -1;
  }

  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(impl->file_handle, &file_size) ||
      file_size.QuadPart <= 0) {
    CloseHandle(impl->file_handle);
    free(impl);
    return -1;
  }

  impl->mapping_handle =
      CreateFileMappingA(impl->file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!impl->mapping_handle) {
    CloseHandle(impl->file_handle);
    free(impl);
    return -1;
  }

  void *view = MapViewOfFile(impl->mapping_handle, FILE_MAP_READ, 0, 0, 0);
  if (!view) {
    CloseHandle(impl->mapping_handle);
    CloseHandle(impl->file_handle);
    free(impl);
    return -1;
  }

  out->data = view;
  out->size = (size_t)file_size.QuadPart;
  out->impl_ = impl;
  return 0;
}

void mlmc_unmap_file(mlmc_mapped_file_t *f) {
  if (!f || !f->impl_) {
    return;
  }
  win_impl_t *impl = (win_impl_t *)f->impl_;
  if (f->data) {
    UnmapViewOfFile(f->data);
  }
  if (impl->mapping_handle) {
    CloseHandle(impl->mapping_handle);
  }
  if (impl->file_handle != INVALID_HANDLE_VALUE) {
    CloseHandle(impl->file_handle);
  }
  free(impl);
  memset(f, 0, sizeof(*f));
}

#else

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  int fd;
} posix_impl_t;

int mlmc_map_file(const char *path, mlmc_mapped_file_t *out) {
  memset(out, 0, sizeof(*out));

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    return -1;
  }

  struct stat st;
  if (fstat(fd, &st) < 0 || st.st_size <= 0) {
    close(fd);
    return -1;
  }
  size_t size = (size_t)st.st_size;

  void *base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) {
    close(fd);
    return -1;
  }

  posix_impl_t *impl = (posix_impl_t *)malloc(sizeof(posix_impl_t));
  if (!impl) {
    munmap(base, size);
    close(fd);
    return -1;
  }
  impl->fd = fd;

  out->data = base;
  out->size = size;
  out->impl_ = impl;
  return 0;
}

void mlmc_unmap_file(mlmc_mapped_file_t *f) {
  if (!f || !f->impl_) {
    return;
  }
  posix_impl_t *impl = (posix_impl_t *)f->impl_;
  if (f->data) {
    munmap((void *)f->data, f->size);
  }
  if (impl->fd >= 0) {
    close(impl->fd);
  }
  free(impl);
  memset(f, 0, sizeof(*f));
}

#endif /* _WIN32 */