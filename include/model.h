#ifndef MINILMC_MODEL_H
#define MINILMC_MODEL_H

#include <stdbool.h>
#include <stdint.h>

#include "gguf.h"

// Qwen2 model dimensions and numerical settings read from GGUF metadata.
typedef struct {
  uint32_t context_length;
  uint32_t block_count;
  uint32_t embedding_length;
  uint32_t feed_forward_length;
  uint32_t head_count;
  uint32_t head_count_kv;
  uint32_t vocab_size;
  float rope_freq_base;
  float rms_epsilon;
} mlmc_qwen2_config_t;

// Owns the mapped GGUF file for one validated Qwen2 model.
typedef struct {
  gguf_file_t gguf;
  mlmc_qwen2_config_t config;
} mlmc_qwen2_model_t;

// Holds one key and one value vector per layer for every cached token.
typedef struct {
  uint64_t token_count;
  uint32_t capacity;
  uint32_t block_count;
  uint32_t key_value_length;
  float *keys;
  float *values;
} mlmc_qwen2_state_t;

// Loads and validates the Qwen2 tensors required for a forward pass.
bool mlmc_qwen2_model_load(const char *path, mlmc_qwen2_model_t *model);

// Releases the model's GGUF mapping and clears its configuration.
void mlmc_qwen2_model_free(mlmc_qwen2_model_t *model);

// Allocates a KV cache that can hold capacity tokens.
bool mlmc_qwen2_state_init(const mlmc_qwen2_model_t *model, uint32_t capacity,
                           mlmc_qwen2_state_t *state);

// Clears cached tokens while retaining the allocated KV cache.
void mlmc_qwen2_state_reset(mlmc_qwen2_state_t *state);

// Releases the KV cache and clears the state.
void mlmc_qwen2_state_free(mlmc_qwen2_state_t *state);

// Runs one token through the Qwen2 decoder and writes vocab_size logits.
// The state accumulates keys and values, so calls must be in token order.
bool mlmc_qwen2_forward(const mlmc_qwen2_model_t *model,
                        mlmc_qwen2_state_t *state, uint32_t token_id,
                        float *logits);

#endif // MINILMC_MODEL_H
