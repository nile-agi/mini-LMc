#include "gguf.h"

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
 * Bounds-safe primitive readers.
 *
 * Every read checks "do I have enough bytes left?" BEFORE touching
 * memory, never after. All values are read assuming little-endian
 * on-disk layout, which is what GGUF v3 uses by default.
 * ==================================================================== */

static int read_u8_safe(const uint8_t **cursor, size_t *remaining,
                        uint8_t *out) {
  if (*remaining < 1) {
    return -1;
  }
  *out = **cursor;
  *cursor += 1;
  *remaining -= 1;
  return 0;
}

static int read_u16_safe(const uint8_t **cursor, size_t *remaining,
                         uint16_t *out) {
  if (*remaining < 2) {
    return -1;
  }
  memcpy(out, *cursor, 2);
  *cursor += 2;
  *remaining -= 2;
  return 0;
}

static int read_u32_safe(const uint8_t **cursor, size_t *remaining,
                         uint32_t *out) {
  if (*remaining < 4) {
    return -1;
  }
  memcpy(out, *cursor, 4);
  *cursor += 4;
  *remaining -= 4;
  return 0;
}

static int read_u64_safe(const uint8_t **cursor, size_t *remaining,
                         uint64_t *out) {
  if (*remaining < 8) {
    return -1;
  }
  memcpy(out, *cursor, 8);
  *cursor += 8;
  *remaining -= 8;
  return 0;
}

/* GGUF strings: u64 length prefix, then raw UTF-8, never NUL-terminated
 * on disk. We always allocate len+1 and add our own NUL so the rest of
 * the codebase can treat these as ordinary C strings. */
static char *read_gguf_string_safe(const uint8_t **cursor, size_t *remaining) {
  uint64_t len;
  if (read_u64_safe(cursor, remaining, &len) != 0) {
    return NULL;
  }
  if (len > SIZE_MAX - 1 || len > *remaining) {
    return NULL;
  }

  char *s = (char *)malloc((size_t)len + 1);
  if (!s) {
    return NULL;
  }
  memcpy(s, *cursor, (size_t)len);
  s[len] = '\0';
  *cursor += len;
  *remaining -= (size_t)len;
  return s;
}

static size_t align_up_safe(size_t offset, size_t alignment) {
  if (alignment == 0) {
    return SIZE_MAX;
  }
  size_t rem = offset % alignment;
  if (rem == 0) {
    return offset;
  }
  size_t pad = alignment - rem;
  if (pad > SIZE_MAX - offset) {
    return SIZE_MAX;
  }
  return offset + pad;
}

/* ======================================================================
 * Recursive metadata value parsing and freeing.
 * ==================================================================== */

static void free_value(gguf_value_t *v);

static int parse_value(const uint8_t **cursor, size_t *remaining,
                       enum gguf_metadata_value_type type, gguf_value_t *out) {
  memset(out, 0, sizeof(*out));
  out->type = type;

  switch (type) {
  case GGUF_TYPE_UINT8:
    return read_u8_safe(cursor, remaining, &out->v.v_u8);

  case GGUF_TYPE_INT8: {
    uint8_t raw;
    if (read_u8_safe(cursor, remaining, &raw) != 0) {
      return -1;
    }
    out->v.v_i8 = (int8_t)raw;
    return 0;
  }

  case GGUF_TYPE_UINT16:
    return read_u16_safe(cursor, remaining, &out->v.v_u16);

  case GGUF_TYPE_INT16: {
    uint16_t raw;
    if (read_u16_safe(cursor, remaining, &raw) != 0) {
      return -1;
    }
    out->v.v_i16 = (int16_t)raw;
    return 0;
  }

  case GGUF_TYPE_UINT32:
    return read_u32_safe(cursor, remaining, &out->v.v_u32);

  case GGUF_TYPE_INT32: {
    uint32_t raw;
    if (read_u32_safe(cursor, remaining, &raw) != 0) {
      return -1;
    }
    out->v.v_i32 = (int32_t)raw;
    return 0;
  }

  case GGUF_TYPE_FLOAT32: {
    uint32_t bits;
    if (read_u32_safe(cursor, remaining, &bits) != 0) {
      return -1;
    }
    memcpy(&out->v.v_f32, &bits, sizeof(out->v.v_f32));
    return 0;
  }

  case GGUF_TYPE_BOOL: {
    uint8_t raw;
    if (read_u8_safe(cursor, remaining, &raw) != 0 || raw > 1) {
      return -1;
    }
    out->v.v_bool = raw;
    return 0;
  }

  case GGUF_TYPE_STRING: {
    char *s = read_gguf_string_safe(cursor, remaining);
    if (!s) {
      return -1;
    }
    out->v.v_str = s;
    return 0;
  }

  case GGUF_TYPE_UINT64:
    return read_u64_safe(cursor, remaining, &out->v.v_u64);

  case GGUF_TYPE_INT64: {
    uint64_t raw;
    if (read_u64_safe(cursor, remaining, &raw) != 0) {
      return -1;
    }
    out->v.v_i64 = (int64_t)raw;
    return 0;
  }

  case GGUF_TYPE_FLOAT64: {
    uint64_t bits;
    if (read_u64_safe(cursor, remaining, &bits) != 0) {
      return -1;
    }
    memcpy(&out->v.v_f64, &bits, sizeof(out->v.v_f64));
    return 0;
  }

  case GGUF_TYPE_ARRAY: {
    uint32_t elem_type_raw;
    uint64_t len;
    if (read_u32_safe(cursor, remaining, &elem_type_raw) != 0) {
      return -1;
    }
    if (elem_type_raw > GGUF_TYPE_FLOAT64) {
      return -1; /* unknown element type tag */
    }
    if (read_u64_safe(cursor, remaining, &len) != 0) {
      return -1;
    }

    out->v.v_arr.elem_type = (enum gguf_metadata_value_type)elem_type_raw;
    out->v.v_arr.len = len;
    out->v.v_arr.items = NULL;

    if (len > 0) {
      if (len > SIZE_MAX / sizeof(gguf_value_t)) {
        return -1;
      }
      out->v.v_arr.items =
          (gguf_value_t *)calloc((size_t)len, sizeof(gguf_value_t));
      if (!out->v.v_arr.items) {
        return -1;
      }

      for (uint64_t i = 0; i < len; i++) {
        if (parse_value(cursor, remaining, out->v.v_arr.elem_type,
                        &out->v.v_arr.items[i]) != 0) {
          for (uint64_t j = 0; j < i; j++) {
            free_value(&out->v.v_arr.items[j]);
          }
          free(out->v.v_arr.items);
          out->v.v_arr.items = NULL;
          return -1;
        }
      }
    }
    return 0;
  }

  default:
    return -1;
  }
}

static void free_value(gguf_value_t *v) {
  if (!v) {
    return;
  }
  if (v->type == GGUF_TYPE_STRING) {
    free(v->v.v_str);
  } else if (v->type == GGUF_TYPE_ARRAY && v->v.v_arr.items) {
    for (uint64_t i = 0; i < v->v.v_arr.len; i++) {
      free_value(&v->v.v_arr.items[i]);
    }
    free(v->v.v_arr.items);
  }
  memset(v, 0, sizeof(*v));
}

/* ======================================================================
 * gguf_load(): header -> metadata -> tensor descriptors -> alignment
 * -> validated tensor byte ranges. Any failure at any stage frees
 * everything allocated so far and returns -1.
 * ==================================================================== */

int gguf_load(const char *path, gguf_file_t *out) {
  memset(out, 0, sizeof(*out));

  if (mlmc_map_file(path, &out->file) != 0) {
    fprintf(stderr, "gguf_load: cannot open/map %s\n", path);
    return -1;
  }

  const uint8_t *cursor = (const uint8_t *)out->file.data;
  size_t remaining = out->file.size;

  /* --- header (24 bytes) --- */
  uint32_t magic, version;
  uint64_t tensor_count, kv_count;
  if (read_u32_safe(&cursor, &remaining, &magic) != 0 ||
      read_u32_safe(&cursor, &remaining, &version) != 0 ||
      read_u64_safe(&cursor, &remaining, &tensor_count) != 0 ||
      read_u64_safe(&cursor, &remaining, &kv_count) != 0) {
    fprintf(stderr, "gguf_load: header truncated\n");
    gguf_free(out);
    return -1;
  }
  if (magic != 0x46554747u) { /* "GGUF" little-endian */
    fprintf(stderr, "gguf_load: bad magic 0x%08" PRIX32 "\n", magic);
    gguf_free(out);
    return -1;
  }
  if (version != 3) {
    fprintf(stderr,
            "gguf_load: unsupported version %" PRIu32 " (only v3 handled)\n",
            version);
    gguf_free(out);
    return -1;
  }
  out->version = version;
  out->tensor_count = tensor_count;
  out->metadata_kv_count = kv_count;

  /* --- metadata --- */
  if (kv_count > 0) {
    if (kv_count > SIZE_MAX / sizeof(gguf_kv_t)) {
      fprintf(stderr, "gguf_load: metadata_kv_count too large\n");
      gguf_free(out);
      return -1;
    }
    out->kv = (gguf_kv_t *)calloc((size_t)kv_count, sizeof(gguf_kv_t));
    if (!out->kv) {
      gguf_free(out);
      return -1;
    }
  }

  for (uint64_t i = 0; i < kv_count; i++) {
    char *key = read_gguf_string_safe(&cursor, &remaining);
    if (!key || key[0] == '\0') {
      free(key);
      fprintf(stderr,
              "gguf_load: bad/empty metadata key at index %" PRIu64 "\n", i);
      gguf_free(out);
      return -1;
    }
    uint32_t vtype;
    if (read_u32_safe(&cursor, &remaining, &vtype) != 0) {
      free(key);
      gguf_free(out);
      return -1;
    }
    out->kv[i].key = key;
    if (vtype > GGUF_TYPE_FLOAT64 ||
        parse_value(&cursor, &remaining, (enum gguf_metadata_value_type)vtype,
                    &out->kv[i].value) != 0) {
      fprintf(stderr, "gguf_load: bad value for key '%s'\n", key);
      gguf_free(out);
      return -1;
    }
  }

  /* --- tensor descriptors --- */
  if (tensor_count > 0) {
    if (tensor_count > SIZE_MAX / sizeof(gguf_tensor_info_t)) {
      fprintf(stderr, "gguf_load: tensor_count too large\n");
      gguf_free(out);
      return -1;
    }
    out->tensors = (gguf_tensor_info_t *)calloc((size_t)tensor_count,
                                                sizeof(gguf_tensor_info_t));
    if (!out->tensors) {
      gguf_free(out);
      return -1;
    }
  }

  for (uint64_t i = 0; i < tensor_count; i++) {
    gguf_tensor_info_t *t = &out->tensors[i];

    t->name = read_gguf_string_safe(&cursor, &remaining);
    if (!t->name || t->name[0] == '\0') {
      fprintf(stderr, "gguf_load: bad/empty tensor name at index %" PRIu64 "\n",
              i);
      gguf_free(out);
      return -1;
    }
    if (strlen(t->name) > 64) {
      fprintf(stderr, "gguf_load: tensor name longer than 64 bytes\n");
      gguf_free(out);
      return -1;
    }

    uint32_t n_dims;
    if (read_u32_safe(&cursor, &remaining, &n_dims) != 0 || n_dims == 0 ||
        n_dims > 4) {
      fprintf(stderr, "gguf_load: tensor '%s' has invalid n_dims\n", t->name);
      gguf_free(out);
      return -1;
    }
    t->n_dims = n_dims;
    for (uint32_t d = 0; d < 4; d++) {
      t->dims[d] = 1;
    }
    for (uint32_t d = 0; d < n_dims; d++) {
      if (read_u64_safe(&cursor, &remaining, &t->dims[d]) != 0 ||
          t->dims[d] == 0) {
        fprintf(stderr, "gguf_load: tensor '%s' has invalid dimension %u\n",
                t->name, d);
        gguf_free(out);
        return -1;
      }
    }

    uint32_t type_raw;
    if (read_u32_safe(&cursor, &remaining, &type_raw) != 0) {
      gguf_free(out);
      return -1;
    }
    t->type = (enum ggml_type)type_raw;

    size_t storage_size;
    if (!ggml_tensor_storage_size(t->type, t->n_dims, t->dims, &storage_size)) {
      fprintf(stderr, "gguf_load: invalid shape/type for tensor '%s'\n",
              t->name);
      gguf_free(out);
      return -1;
    }

    if (read_u64_safe(&cursor, &remaining, &t->offset) != 0) {
      gguf_free(out);
      return -1;
    }
  }

  /* --- alignment: default 32, must be a nonzero power of 2 --- */
  out->alignment = 32;
  const gguf_value_t *align = gguf_find(out, "general.alignment");
  if (align) {
    if (align->type != GGUF_TYPE_UINT32 || align->v.v_u32 == 0 ||
        (align->v.v_u32 & (align->v.v_u32 - 1)) != 0) {
      fprintf(stderr, "gguf_load: general.alignment invalid\n");
      gguf_free(out);
      return -1;
    }
    out->alignment = align->v.v_u32;
  }

  /* --- data section offset ---
   * A file with zero tensors has no data section at all, so there is
   * nothing to pad or align to — the file may legitimately end right
   * after the last metadata value. Only require alignment and
   * apply the "must fit in the file" check when a data section
   * actually exists (tensor_count > 0). */
  size_t header_end = (size_t)(cursor - (const uint8_t *)out->file.data);
  if (tensor_count > 0) {
    size_t data_start = align_up_safe(header_end, out->alignment);
    if (data_start == SIZE_MAX || data_start > out->file.size) {
      fprintf(stderr, "gguf_load: tensor data offset exceeds file size\n");
      gguf_free(out);
      return -1;
    }
    out->tensor_data_offset = data_start;
  } else {
    out->tensor_data_offset = header_end;
  }

  /* --- per-tensor range validation --- */
  for (uint64_t i = 0; i < tensor_count; i++) {
    const gguf_tensor_info_t *t = &out->tensors[i];

    if (t->offset % out->alignment != 0) {
      fprintf(stderr, "gguf_load: tensor '%s' offset not aligned\n", t->name);
      gguf_free(out);
      return -1;
    }

    size_t storage_size;
    if (!ggml_tensor_storage_size(t->type, t->n_dims, t->dims, &storage_size)) {
      gguf_free(out);
      return -1;
    }

    if (t->offset > SIZE_MAX - out->tensor_data_offset) {
      fprintf(stderr, "gguf_load: tensor '%s' offset overflow\n", t->name);
      gguf_free(out);
      return -1;
    }
    size_t abs_offset = out->tensor_data_offset + (size_t)t->offset;

    if (storage_size > SIZE_MAX - abs_offset ||
        abs_offset + storage_size > out->file.size) {
      fprintf(stderr, "gguf_load: tensor '%s' extends past end of file\n",
              t->name);
      gguf_free(out);
      return -1;
    }
  }

  return 0;
}

/* ======================================================================
 * Cleanup, lookup, and the tensor-data accessor.
 * ==================================================================== */

void gguf_free(gguf_file_t *f) {
  if (!f) {
    return;
  }
  if (f->kv) {
    for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
      free(f->kv[i].key);
      free_value(&f->kv[i].value);
    }
    free(f->kv);
  }
  if (f->tensors) {
    for (uint64_t i = 0; i < f->tensor_count; i++) {
      free(f->tensors[i].name);
    }
    free(f->tensors);
  }
  mlmc_unmap_file(&f->file);
  memset(f, 0, sizeof(*f));
}

const gguf_value_t *gguf_find(const gguf_file_t *f, const char *key) {
  if (!f || !f->kv || !key) {
    return NULL;
  }
  for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
    if (f->kv[i].key && strcmp(f->kv[i].key, key) == 0) {
      return &f->kv[i].value;
    }
  }
  return NULL;
}

const void *gguf_tensor_data(const gguf_file_t *f, const gguf_tensor_info_t *t,
                             size_t *out_size) {
  if (!f || !t || !f->file.data) {
    return NULL;
  }

  size_t storage_size;
  if (!ggml_tensor_storage_size(t->type, t->n_dims, t->dims, &storage_size)) {
    return NULL;
  }
  if (t->offset > SIZE_MAX - f->tensor_data_offset) {
    return NULL;
  }
  size_t abs_offset = f->tensor_data_offset + (size_t)t->offset;
  if (storage_size > SIZE_MAX - abs_offset ||
      abs_offset + storage_size > f->file.size) {
    return NULL;
  }

  if (out_size) {
    *out_size = storage_size;
  }
  return (const uint8_t *)f->file.data + abs_offset;
}

/* ======================================================================
 * Summary printer.
 * ==================================================================== */

void gguf_print_summary(const gguf_file_t *f, int max_tensors_shown) {
  printf("GGUF v%" PRIu32 " -- %" PRIu64 " tensors, %" PRIu64
         " metadata keys, alignment %" PRIu32 "\n",
         f->version, f->tensor_count, f->metadata_kv_count, f->alignment);
  printf("tensor data starts at file offset %zu\n\n", f->tensor_data_offset);

  uint64_t shown =
      (max_tensors_shown < 0) ? f->tensor_count : (uint64_t)max_tensors_shown;
  if (shown > f->tensor_count) {
    shown = f->tensor_count;
  }

  for (uint64_t i = 0; i < shown; i++) {
    const gguf_tensor_info_t *t = &f->tensors[i];
    printf("%-40s %-10s [", t->name, ggml_type_name(t->type));
    for (uint32_t d = 0; d < t->n_dims; d++) {
      printf("%" PRIu64 "%s", t->dims[d], d + 1 < t->n_dims ? " x " : "");
    }
    printf("]\n");
  }
  if (shown < f->tensor_count) {
    printf("... (%" PRIu64 " more tensors)\n", f->tensor_count - shown);
  }
}