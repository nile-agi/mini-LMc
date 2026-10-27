#include "dequant.h"

#include <stddef.h>
#include <string.h>

// Q8_0 block: fp16 scale followed by 32 signed bytes.
#define Q8_0_BLOCK_ELEMS 32
#define Q8_0_BLOCK_BYTES 34
#define Q8_0_SCALE_BYTES 2

static uint16_t read_u16(const uint8_t *p) {
  uint16_t value;
  memcpy(&value, p, sizeof(value));
  return value;
}

static float read_f32(const uint8_t *p) {
  float value;
  memcpy(&value, p, sizeof(value));
  return value;
}

static float bits_to_float(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

float mlmc_fp16_to_fp32(uint16_t half_bits) {
  uint32_t sign = (uint32_t)(half_bits & 0x8000u) << 16;
  int32_t exponent = (half_bits >> 10) & 0x1F;
  uint32_t mantissa = half_bits & 0x3FFu;

  if (exponent == 0) {
    if (mantissa == 0) {
      return bits_to_float(sign);
    }
    exponent = 1;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x3FFu;
  } else if (exponent == 31) {
    return bits_to_float(sign | 0x7F800000u | (mantissa << 13));
  }

  uint32_t float_exponent = (uint32_t)(exponent + 112);
  return bits_to_float(sign | (float_exponent << 23) | (mantissa << 13));
}

float mlmc_bf16_to_fp32(uint16_t bf16_bits) {
  return bits_to_float((uint32_t)bf16_bits << 16);
}

static void dequant_f32(const uint8_t *src, uint64_t n, float *dst) {
  memcpy(dst, src, (size_t)n * sizeof(float));
}

static void dequant_f16(const uint8_t *src, uint64_t n, float *dst) {
  for (uint64_t i = 0; i < n; i++) {
    dst[i] = mlmc_fp16_to_fp32(read_u16(src + i * 2));
  }
}

static void dequant_bf16(const uint8_t *src, uint64_t n, float *dst) {
  for (uint64_t i = 0; i < n; i++) {
    dst[i] = mlmc_bf16_to_fp32(read_u16(src + i * 2));
  }
}

static void dequant_q8_0(const uint8_t *src, uint64_t n, float *dst) {
  uint64_t block_count = n / Q8_0_BLOCK_ELEMS;
  for (uint64_t b = 0; b < block_count; b++) {
    const uint8_t *block = src + b * Q8_0_BLOCK_BYTES;
    float scale = mlmc_fp16_to_fp32(read_u16(block));
    const int8_t *quants = (const int8_t *)(block + Q8_0_SCALE_BYTES);

    for (uint32_t i = 0; i < Q8_0_BLOCK_ELEMS; i++) {
      dst[b * Q8_0_BLOCK_ELEMS + i] = scale * (float)quants[i];
    }
  }
}

bool mlmc_type_is_supported(enum ggml_type type) {
  switch (type) {
  case GGML_TYPE_F32:
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
  case GGML_TYPE_Q8_0:
    return true;
  default:
    return false;
  }
}

bool mlmc_dequantize_row(enum ggml_type type, const void *src,
                         uint64_t n_elements, float *dst) {
  ggml_type_traits_t traits;
  if (!src || !dst || !mlmc_type_is_supported(type) ||
      !ggml_get_type_traits(type, &traits) ||
      n_elements % traits.block_size != 0) {
    return false;
  }

  const uint8_t *bytes = (const uint8_t *)src;
  switch (type) {
  case GGML_TYPE_F32:
    dequant_f32(bytes, n_elements, dst);
    return true;
  case GGML_TYPE_F16:
    dequant_f16(bytes, n_elements, dst);
    return true;
  case GGML_TYPE_BF16:
    dequant_bf16(bytes, n_elements, dst);
    return true;
  case GGML_TYPE_Q8_0:
    dequant_q8_0(bytes, n_elements, dst);
    return true;
  default:
    return false;
  }
}

typedef float (*dot_row_fn)(const uint8_t *row, uint64_t n, const float *x);

static float dot_f32(const uint8_t *row, uint64_t n, const float *x) {
  float sum = 0.0f;
  for (uint64_t i = 0; i < n; i++) {
    sum += read_f32(row + i * 4) * x[i];
  }
  return sum;
}

static float dot_f16(const uint8_t *row, uint64_t n, const float *x) {
  float sum = 0.0f;
  for (uint64_t i = 0; i < n; i++) {
    sum += mlmc_fp16_to_fp32(read_u16(row + i * 2)) * x[i];
  }
  return sum;
}

static float dot_bf16(const uint8_t *row, uint64_t n, const float *x) {
  float sum = 0.0f;
  for (uint64_t i = 0; i < n; i++) {
    sum += mlmc_bf16_to_fp32(read_u16(row + i * 2)) * x[i];
  }
  return sum;
}

static float dot_q8_0(const uint8_t *row, uint64_t n, const float *x) {
  float sum = 0.0f;
  uint64_t block_count = n / Q8_0_BLOCK_ELEMS;

  for (uint64_t b = 0; b < block_count; b++) {
    const uint8_t *block = row + b * Q8_0_BLOCK_BYTES;
    float scale = mlmc_fp16_to_fp32(read_u16(block));
    const int8_t *quants = (const int8_t *)(block + Q8_0_SCALE_BYTES);
    const float *xb = x + b * Q8_0_BLOCK_ELEMS;
    float block_sum = 0.0f;

    for (uint32_t i = 0; i < Q8_0_BLOCK_ELEMS; i++) {
      block_sum += (float)quants[i] * xb[i];
    }
    sum += scale * block_sum;
  }
  return sum;
}

static dot_row_fn find_dot_kernel(enum ggml_type type) {
  switch (type) {
  case GGML_TYPE_F32:
    return dot_f32;
  case GGML_TYPE_F16:
    return dot_f16;
  case GGML_TYPE_BF16:
    return dot_bf16;
  case GGML_TYPE_Q8_0:
    return dot_q8_0;
  default:
    return NULL;
  }
}

bool mlmc_gemv(enum ggml_type type, const void *weights, uint64_t n_rows,
               uint64_t n_cols, const float *x, float *y) {
  dot_row_fn dot = find_dot_kernel(type);
  if (!dot || !weights || !x || !y) {
    return false;
  }

  size_t row_bytes = 0;
  size_t matrix_bytes = 0;
  uint64_t row_dims[1] = {n_cols};
  uint64_t matrix_dims[2] = {n_cols, n_rows};
  if (!ggml_tensor_storage_size(type, 1, row_dims, &row_bytes) ||
      !ggml_tensor_storage_size(type, 2, matrix_dims, &matrix_bytes)) {
    return false;
  }

  const uint8_t *bytes = (const uint8_t *)weights;
  for (uint64_t r = 0; r < n_rows; r++) {
    y[r] = dot(bytes + (size_t)r * row_bytes, n_cols, x);
  }
  return true;
}
