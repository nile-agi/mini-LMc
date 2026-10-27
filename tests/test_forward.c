#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "model.h"

static int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #condition);  \
      failures++;                                                              \
    }                                                                          \
  } while (0)

typedef struct {
  const char *name;
  uint32_t n_dims;
  uint64_t dims[2];
  const float *values;
  size_t value_count;
} test_tensor_t;

static void put_u32(uint8_t *buffer, size_t *position, uint32_t value) {
  memcpy(buffer + *position, &value, sizeof(value));
  *position += sizeof(value);
}

static void put_u64(uint8_t *buffer, size_t *position, uint64_t value) {
  memcpy(buffer + *position, &value, sizeof(value));
  *position += sizeof(value);
}

static void put_f32(uint8_t *buffer, size_t *position, float value) {
  memcpy(buffer + *position, &value, sizeof(value));
  *position += sizeof(value);
}

static void put_string(uint8_t *buffer, size_t *position, const char *text) {
  size_t length = strlen(text);
  put_u64(buffer, position, length);
  memcpy(buffer + *position, text, length);
  *position += length;
}

static void put_header(uint8_t *buffer, size_t *position, uint64_t tensors,
                       uint64_t metadata) {
  put_u32(buffer, position, 0x46554747u);
  put_u32(buffer, position, 3);
  put_u64(buffer, position, tensors);
  put_u64(buffer, position, metadata);
}

static void put_u32_metadata(uint8_t *buffer, size_t *position, const char *key,
                             uint32_t value) {
  put_string(buffer, position, key);
  put_u32(buffer, position, GGUF_TYPE_UINT32);
  put_u32(buffer, position, value);
}

static void put_f32_metadata(uint8_t *buffer, size_t *position, const char *key,
                             float value) {
  put_string(buffer, position, key);
  put_u32(buffer, position, GGUF_TYPE_FLOAT32);
  put_f32(buffer, position, value);
}

static size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

static void put_tensor_descriptor(uint8_t *buffer, size_t *position,
                                  const test_tensor_t *tensor,
                                  uint64_t offset) {
  put_string(buffer, position, tensor->name);
  put_u32(buffer, position, tensor->n_dims);
  for (uint32_t dim = 0; dim < tensor->n_dims; dim++) {
    put_u64(buffer, position, tensor->dims[dim]);
  }
  put_u32(buffer, position, GGML_TYPE_F32);
  put_u64(buffer, position, offset);
}

static const char *write_test_model(void) {
  static char path[64];
  static int file_count = 0;
  uint8_t buffer[8192] = {0};
  size_t position = 0;

  const float identity[16] = {
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
  };
  const float embedding[12] = {
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
  };
  const float output[12] = {
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
  };
  const float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  const float query_bias[4] = {0.1f, 0.0f, 0.0f, 0.0f};
  const float key_bias[4] = {0.0f, 0.1f, 0.0f, 0.0f};
  const float value_bias[4] = {0.05f, 0.0f, 0.0f, 0.0f};

  const test_tensor_t tensors[] = {
      {"token_embd.weight", 2, {4, 3}, embedding, 12},
      {"output.weight", 2, {4, 3}, output, 12},
      {"output_norm.weight", 1, {4, 0}, ones, 4},
      {"blk.0.attn_norm.weight", 1, {4, 0}, ones, 4},
      {"blk.0.attn_q.weight", 2, {4, 4}, identity, 16},
      {"blk.0.attn_q.bias", 1, {4, 0}, query_bias, 4},
      {"blk.0.attn_k.weight", 2, {4, 4}, identity, 16},
      {"blk.0.attn_k.bias", 1, {4, 0}, key_bias, 4},
      {"blk.0.attn_v.weight", 2, {4, 4}, identity, 16},
      {"blk.0.attn_v.bias", 1, {4, 0}, value_bias, 4},
      {"blk.0.attn_output.weight", 2, {4, 4}, identity, 16},
      {"blk.0.ffn_norm.weight", 1, {4, 0}, ones, 4},
      {"blk.0.ffn_gate.weight", 2, {4, 4}, identity, 16},
      {"blk.0.ffn_up.weight", 2, {4, 4}, identity, 16},
      {"blk.0.ffn_down.weight", 2, {4, 4}, identity, 16},
  };

  put_header(buffer, &position, sizeof(tensors) / sizeof(tensors[0]), 9);
  put_string(buffer, &position, "general.architecture");
  put_u32(buffer, &position, GGUF_TYPE_STRING);
  put_string(buffer, &position, "qwen2");
  put_u32_metadata(buffer, &position, "qwen2.context_length", 4);
  put_u32_metadata(buffer, &position, "qwen2.block_count", 1);
  put_u32_metadata(buffer, &position, "qwen2.embedding_length", 4);
  put_u32_metadata(buffer, &position, "qwen2.feed_forward_length", 4);
  put_u32_metadata(buffer, &position, "qwen2.attention.head_count", 1);
  put_u32_metadata(buffer, &position, "qwen2.attention.head_count_kv", 1);
  put_f32_metadata(buffer, &position, "qwen2.rope.freq_base", 10000.0f);
  put_f32_metadata(buffer, &position, "qwen2.attention.layer_norm_rms_epsilon",
                   1e-5f);

  size_t offset = 0;
  for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++) {
    offset = align_up(offset, 32);
    put_tensor_descriptor(buffer, &position, &tensors[i], offset);
    offset += tensors[i].value_count * sizeof(float);
  }

  size_t data_start = align_up(position, 32);
  size_t data_offset = 0;
  for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++) {
    data_offset = align_up(data_offset, 32);
    size_t bytes = tensors[i].value_count * sizeof(float);
    memcpy(buffer + data_start + data_offset, tensors[i].values, bytes);
    data_offset += bytes;
  }

  snprintf(path, sizeof(path), "forward_test_%d.gguf", file_count++);
  FILE *file = fopen(path, "wb");
  if (!file || data_start + data_offset > sizeof(buffer)) {
    if (file) {
      fclose(file);
    }
    return NULL;
  }
  if (fwrite(buffer, 1, data_start + data_offset, file) !=
      data_start + data_offset) {
    fclose(file);
    return NULL;
  }
  fclose(file);
  return path;
}

static void test_forward_and_kv_cache(void) {
  const char *path = write_test_model();
  CHECK(path != NULL);

  mlmc_qwen2_model_t model = {0};
  CHECK(mlmc_qwen2_model_load(path, &model));
  CHECK(model.config.vocab_size == 3);
  CHECK(model.config.embedding_length == 4);
  CHECK(model.config.block_count == 1);

  mlmc_qwen2_state_t state = {0};
  CHECK(mlmc_qwen2_state_init(&model, 2, &state));
  float first_logits[3] = {0};
  float second_logits[3] = {0};
  float reset_logits[3] = {0};

  CHECK(mlmc_qwen2_forward(&model, &state, 0, first_logits));
  CHECK(state.token_count == 1);
  CHECK(fabsf(state.keys[0]) > 0.0f);
  CHECK(isfinite(first_logits[0]) && isfinite(first_logits[1]) &&
        isfinite(first_logits[2]));
  CHECK(first_logits[0] > first_logits[1]);
  CHECK(first_logits[0] > first_logits[2]);

  CHECK(mlmc_qwen2_forward(&model, &state, 1, second_logits));
  CHECK(state.token_count == 2);
  CHECK(!mlmc_qwen2_forward(&model, &state, 2, second_logits));

  mlmc_qwen2_state_reset(&state);
  CHECK(state.token_count == 0);
  CHECK(mlmc_qwen2_forward(&model, &state, 1, reset_logits));
  CHECK(fabsf(second_logits[0] - reset_logits[0]) > 1e-5f ||
        fabsf(second_logits[1] - reset_logits[1]) > 1e-5f ||
        fabsf(second_logits[2] - reset_logits[2]) > 1e-5f);
  CHECK(!mlmc_qwen2_forward(&model, &state, 3, reset_logits));

  mlmc_qwen2_state_free(&state);
  mlmc_qwen2_model_free(&model);
  remove(path);
}

int main(void) {
  test_forward_and_kv_cache();

  if (failures == 0) {
    printf("ALL FORWARD TESTS PASSED\n");
  } else {
    printf("%d FORWARD TESTS FAILED\n", failures);
  }
  return failures != 0;
}
