#ifndef MINILMC_DEQUANT_H
#define MINILMC_DEQUANT_H

#include <stdbool.h>
#include <stdint.h>

#include "ggml_types.h"

// Converts one IEEE half (binary16) bit pattern to float32.
float mlmc_fp16_to_fp32(uint16_t half_bits);

// Converts one bfloat16 bit pattern to float32.
float mlmc_bf16_to_fp32(uint16_t bf16_bits);

// True if this type has a dequantization kernel.
bool mlmc_type_is_supported(enum ggml_type type);

// Converts n_elements stored values into floats.
// n_elements must be a whole number of blocks.
// Returns false and writes nothing for an unsupported type or a bad count.
bool mlmc_dequantize_row(enum ggml_type type, const void *src,
                         uint64_t n_elements, float *dst);

// Fused matrix-vector product: y[r] = dot(row r of W, x).
// W is n_rows rows of n_cols stored values. Rows are decoded block by
// block while multiplying, so the full matrix is never turned into floats.
// Returns false for an unsupported type or a shape that is not block-aligned.
bool mlmc_gemv(enum ggml_type type, const void *weights, uint64_t n_rows,
               uint64_t n_cols, const float *x, float *y);

#endif // MINILMC_DEQUANT_H
