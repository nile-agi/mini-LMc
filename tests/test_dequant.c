#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dequant.h"

static int failures = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond);       \
      failures++;                                                              \
    }                                                                          \
  } while (0)

// Small deterministic generator so failures are reproducible.
static uint32_t rng_state = 12345u;
static uint32_t next_random(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

static float random_unit_float(void) {
  return ((float)(next_random() % 20001u) - 10000.0f) / 10000.0f;
}

static int close_enough(double a, double b) {
  double tolerance = 1e-4 * (fabs(b) + 1.0);
  return fabs(a - b) <= tolerance;
}

static void test_fp16_conversion(void) {
  CHECK(mlmc_fp16_to_fp32(0x3C00) == 1.0f);
  CHECK(mlmc_fp16_to_fp32(0xC000) == -2.0f);
  CHECK(mlmc_fp16_to_fp32(0x3800) == 0.5f);
  CHECK(mlmc_fp16_to_fp32(0x7BFF) == 65504.0f);
  CHECK(mlmc_fp16_to_fp32(0x0000) == 0.0f);
  CHECK(signbit(mlmc_fp16_to_fp32(0x8000)) != 0);

  // Smallest normal, smallest subnormal, largest subnormal.
  CHECK(mlmc_fp16_to_fp32(0x0400) == 1.0f / 16384.0f);
  CHECK(mlmc_fp16_to_fp32(0x0001) == 1.0f / 16777216.0f);
  CHECK(mlmc_fp16_to_fp32(0x03FF) == 1023.0f / 16777216.0f);

  CHECK(mlmc_fp16_to_fp32(0x7C00) == HUGE_VALF);
  CHECK(mlmc_fp16_to_fp32(0xFC00) == -HUGE_VALF);
  float not_a_number = mlmc_fp16_to_fp32(0x7E00);
  CHECK(not_a_number != not_a_number);
}

static void test_bf16_conversion(void) {
  CHECK(mlmc_bf16_to_fp32(0x3F80) == 1.0f);
  CHECK(mlmc_bf16_to_fp32(0xC000) == -2.0f);
  CHECK(mlmc_bf16_to_fp32(0x0000) == 0.0f);
}

static void test_f32_row_and_gemv(void) {
  float weights[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float x[2] = {1.0f, 1.0f};
  float y[2] = {0.0f, 0.0f};

  CHECK(mlmc_gemv(GGML_TYPE_F32, weights, 2, 2, x, y));
  CHECK(y[0] == 3.0f && y[1] == 7.0f);

  float decoded[4];
  CHECK(mlmc_dequantize_row(GGML_TYPE_F32, weights, 4, decoded));
  CHECK(memcmp(decoded, weights, sizeof(weights)) == 0);
}

static void test_q8_0_known_block(void) {
  uint8_t block[34];
  uint16_t scale_bits = 0x3800;
  memcpy(block, &scale_bits, 2);
  for (int i = 0; i < 32; i++) {
    block[2 + i] = (uint8_t)(int8_t)(i - 16);
  }

  float decoded[32];
  CHECK(mlmc_dequantize_row(GGML_TYPE_Q8_0, block, 32, decoded));
  for (int i = 0; i < 32; i++) {
    CHECK(decoded[i] == 0.5f * (float)(i - 16));
  }
}

static void test_q8_0_gemv_matches_reference(void) {
  enum { ROWS = 3, COLS = 64, BLOCKS_PER_ROW = COLS / 32 };
  const uint16_t scales[3] = {0x3800, 0x3400, 0x3C00};

  uint8_t matrix[ROWS * BLOCKS_PER_ROW * 34];
  for (int block_index = 0; block_index < ROWS * BLOCKS_PER_ROW;
       block_index++) {
    uint8_t *block = matrix + block_index * 34;
    memcpy(block, &scales[block_index % 3], 2);
    for (int i = 0; i < 32; i++) {
      block[2 + i] = (uint8_t)(int8_t)((int)(next_random() % 255u) - 127);
    }
  }

  float x[COLS];
  for (int i = 0; i < COLS; i++) {
    x[i] = random_unit_float();
  }

  float y[ROWS];
  CHECK(mlmc_gemv(GGML_TYPE_Q8_0, matrix, ROWS, COLS, x, y));

  // Reference: fully dequantize each row, then dot in double precision.
  for (int r = 0; r < ROWS; r++) {
    float row[COLS];
    CHECK(mlmc_dequantize_row(GGML_TYPE_Q8_0,
                              matrix + r * BLOCKS_PER_ROW * 34, COLS, row));
    double expected = 0.0;
    for (int c = 0; c < COLS; c++) {
      expected += (double)row[c] * (double)x[c];
    }
    CHECK(close_enough((double)y[r], expected));
  }
}

static void test_rejections(void) {
  float decoded[32] = {0};
  uint8_t block[34] = {0};

  // Q4_K has no kernel yet, so it must refuse instead of guessing.
  CHECK(!mlmc_type_is_supported(GGML_TYPE_Q4_K));
  CHECK(!mlmc_dequantize_row(GGML_TYPE_Q4_K, block, 256, decoded));

  // 31 elements is not a whole Q8_0 block.
  CHECK(!mlmc_dequantize_row(GGML_TYPE_Q8_0, block, 31, decoded));

  float x[31] = {0};
  float y[1] = {0};
  CHECK(!mlmc_gemv(GGML_TYPE_Q8_0, block, 1, 31, x, y));
}

int main(void) {
  test_fp16_conversion();
  test_bf16_conversion();
  test_f32_row_and_gemv();
  test_q8_0_known_block();
  test_q8_0_gemv_matches_reference();
  test_rejections();

  if (failures == 0) {
    printf("ALL DEQUANT TESTS PASSED\n");
  } else {
    printf("%d FAILURES\n", failures);
  }
  return failures != 0;
}
