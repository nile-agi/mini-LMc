#include "ggml_types.h"

#include <limits.h>

/*
 * Block layouts, one row per active type: (name, elements/block, bytes/block).
 * Every number here is derived from the block struct's own field
 * sizes, not quoted from memory — before trusting a bits-per-weight
 * figure anywhere, compute it yourself: bpw = type_size * 8 / block_size.
 * Slots not listed here (the removed/reserved gaps) are zero-initialized
 * by the C standard and rejected by ggml_get_type_traits().
 */
static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {"F32", 1, 4},
    [GGML_TYPE_F16] = {"F16", 1, 2},
    [GGML_TYPE_Q4_0] = {"Q4_0", 32, 18},
    [GGML_TYPE_Q4_1] = {"Q4_1", 32, 20},
    [GGML_TYPE_Q5_0] = {"Q5_0", 32, 22},
    [GGML_TYPE_Q5_1] = {"Q5_1", 32, 24},
    [GGML_TYPE_Q8_0] = {"Q8_0", 32, 34},
    [GGML_TYPE_Q8_1] = {"Q8_1", 32, 36},
    [GGML_TYPE_Q2_K] = {"Q2_K", 256, 84},
    [GGML_TYPE_Q3_K] = {"Q3_K", 256, 110},
    [GGML_TYPE_Q4_K] = {"Q4_K", 256, 144},
    [GGML_TYPE_Q5_K] = {"Q5_K", 256, 176},
    [GGML_TYPE_Q6_K] = {"Q6_K", 256, 210},
    [GGML_TYPE_Q8_K] = {"Q8_K", 256, 292},
    [GGML_TYPE_IQ2_XXS] = {"IQ2_XXS", 256, 66},
    [GGML_TYPE_IQ2_XS] = {"IQ2_XS", 256, 74},
    [GGML_TYPE_IQ3_XXS] = {"IQ3_XXS", 256, 98},
    [GGML_TYPE_IQ1_S] = {"IQ1_S", 256, 50},
    [GGML_TYPE_IQ4_NL] = {"IQ4_NL", 32, 18},
    [GGML_TYPE_IQ3_S] = {"IQ3_S", 256, 110},
    [GGML_TYPE_IQ2_S] = {"IQ2_S", 256, 82},
    [GGML_TYPE_IQ4_XS] = {"IQ4_XS", 256, 136},
    [GGML_TYPE_I8] = {"I8", 1, 1},
    [GGML_TYPE_I16] = {"I16", 1, 2},
    [GGML_TYPE_I32] = {"I32", 1, 4},
    [GGML_TYPE_I64] = {"I64", 1, 8},
    [GGML_TYPE_F64] = {"F64", 1, 8},
    [GGML_TYPE_IQ1_M] = {"IQ1_M", 256, 56},
    [GGML_TYPE_BF16] = {"BF16", 1, 2},
    [GGML_TYPE_TQ1_0] = {"TQ1_0", 256, 54},
    [GGML_TYPE_TQ2_0] = {"TQ2_0", 256, 66},
    [GGML_TYPE_MXFP4] = {"MXFP4", 32, 17},
    [GGML_TYPE_NVFP4] = {"NVFP4", 64, 36},
    [GGML_TYPE_Q1_0] = {"Q1_0", 128, 18},
    [GGML_TYPE_Q2_0] = {"Q2_0", 64, 18},
};

bool ggml_get_type_traits(enum ggml_type type, ggml_type_traits_t *out) {
  if ((unsigned)type >= GGML_TYPE_COUNT || type_traits[type].block_size == 0 ||
      type_traits[type].type_size == 0) {
    return false;
  }
  if (out) {
    *out = type_traits[type];
  }
  return true;
}

const char *ggml_type_name(enum ggml_type type) {
  if (!ggml_get_type_traits(type, NULL)) {
    return NULL;
  }
  return type_traits[type].name;
}

bool ggml_tensor_storage_size(enum ggml_type type, uint32_t n_dims,
                              const uint64_t *dimensions, size_t *out_size) {
  ggml_type_traits_t traits;
  if (!out_size || n_dims > 4 || (n_dims != 0 && !dimensions) ||
      !ggml_get_type_traits(type, &traits)) {
    return false;
  }

  uint64_t row_elements = (n_dims != 0) ? dimensions[0] : 1;
  if (row_elements % traits.block_size != 0) {
    return false; /* a partial block means the file is malformed */
  }

  size_t block_count = (size_t)(row_elements / traits.block_size);
  for (uint32_t i = 1; i < n_dims; ++i) {
    if (dimensions[i] != 0 && block_count > SIZE_MAX / dimensions[i]) {
      return false; /* would overflow size_t */
    }
    block_count *= (size_t)dimensions[i];
  }

  if (block_count > SIZE_MAX / traits.type_size) {
    return false;
  }
  *out_size = block_count * traits.type_size;
  return true;
}