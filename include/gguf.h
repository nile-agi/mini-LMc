#ifndef GGUF_H
#define GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ggml_types.h"
#include "platform.h"

enum gguf_metadata_value_type {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12
};

typedef struct gguf_value gguf_value_t;
struct gguf_value {
  enum gguf_metadata_value_type type;
  union {
    uint8_t v_u8;
    int8_t v_i8;
    uint16_t v_u16;
    int16_t v_i16;
    uint32_t v_u32;
    int32_t v_i32;
    uint64_t v_u64;
    int64_t v_i64;
    float v_f32;
    double v_f64;
    uint8_t v_bool;
    char *v_str;
    struct {
      enum gguf_metadata_value_type elem_type;
      uint64_t len;
      gguf_value_t *items;
    } v_arr;
  } v;
};

typedef struct {
  char *key;
  gguf_value_t value;
} gguf_kv_t;

typedef struct {
  uint32_t n_dims;
  uint64_t dims[4];
  enum ggml_type type;
  uint64_t offset;
  char *name;
} gguf_tensor_info_t;

typedef struct {
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;

  gguf_kv_t *kv;
  gguf_tensor_info_t *tensors;

  uint32_t alignment;
  size_t tensor_data_offset;

  mlmc_mapped_file_t file;
} gguf_file_t;

int gguf_load(const char *path, gguf_file_t *out);
void gguf_free(gguf_file_t *f);
const gguf_value_t *gguf_find(const gguf_file_t *f, const char *key);
const void *gguf_tensor_data(const gguf_file_t *f, const gguf_tensor_info_t *t,
                             size_t *out_size);
void gguf_print_summary(const gguf_file_t *f, int max_tensors_shown);

#endif /* GGUF_H */