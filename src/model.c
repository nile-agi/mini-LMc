#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dequant.h"

typedef struct {
  float *hidden;
  float *norm;
  float *query;
  float *key;
  float *value;
  float *attention;
  float *projection;
  float *gate;
  float *up;
  float *ffn;
  float *scores;
} forward_workspace_t;

static bool multiply_size(size_t left, size_t right, size_t *out) {
  if (left != 0 && right > SIZE_MAX / left) {
    return false;
  }
  *out = left * right;
  return true;
}

static float read_f32(const uint8_t *bytes) {
  float value;
  memcpy(&value, bytes, sizeof(value));
  return value;
}

static bool metadata_u32(const gguf_file_t *gguf, const char *key,
                         uint32_t *out) {
  const gguf_value_t *value = gguf_find(gguf, key);
  if (!value || value->type != GGUF_TYPE_UINT32) {
    return false;
  }
  *out = value->v.v_u32;
  return true;
}

static bool metadata_f32(const gguf_file_t *gguf, const char *key, float *out) {
  const gguf_value_t *value = gguf_find(gguf, key);
  if (!value || value->type != GGUF_TYPE_FLOAT32) {
    return false;
  }
  *out = value->v.v_f32;
  return true;
}

static bool load_config(mlmc_qwen2_model_t *model) {
  const gguf_value_t *architecture =
      gguf_find(&model->gguf, "general.architecture");
  if (!architecture || architecture->type != GGUF_TYPE_STRING ||
      strcmp(architecture->v.v_str, "qwen2") != 0) {
    return false;
  }

  mlmc_qwen2_config_t *config = &model->config;
  if (!metadata_u32(&model->gguf, "qwen2.context_length",
                    &config->context_length) ||
      !metadata_u32(&model->gguf, "qwen2.block_count", &config->block_count) ||
      !metadata_u32(&model->gguf, "qwen2.embedding_length",
                    &config->embedding_length) ||
      !metadata_u32(&model->gguf, "qwen2.feed_forward_length",
                    &config->feed_forward_length) ||
      !metadata_u32(&model->gguf, "qwen2.attention.head_count",
                    &config->head_count) ||
      !metadata_u32(&model->gguf, "qwen2.attention.head_count_kv",
                    &config->head_count_kv) ||
      !metadata_f32(&model->gguf, "qwen2.rope.freq_base",
                    &config->rope_freq_base) ||
      !metadata_f32(&model->gguf, "qwen2.attention.layer_norm_rms_epsilon",
                    &config->rms_epsilon)) {
    return false;
  }

  if (config->context_length == 0 || config->block_count == 0 ||
      config->embedding_length == 0 || config->feed_forward_length == 0 ||
      config->head_count == 0 || config->head_count_kv == 0 ||
      config->embedding_length % config->head_count != 0 ||
      config->head_count % config->head_count_kv != 0 ||
      (config->embedding_length / config->head_count) % 2 != 0 ||
      config->rope_freq_base <= 0.0f || config->rms_epsilon <= 0.0f) {
    return false;
  }
  return true;
}

static bool is_vector(const gguf_tensor_info_t *tensor, uint32_t length) {
  return tensor && tensor->n_dims == 1 && tensor->dims[0] == length;
}

static bool is_f32_vector(const gguf_tensor_info_t *tensor, uint32_t length) {
  return is_vector(tensor, length) && tensor->type == GGML_TYPE_F32;
}

static bool is_matrix(const gguf_tensor_info_t *tensor, uint32_t columns,
                      uint32_t rows) {
  return tensor && tensor->n_dims == 2 && tensor->dims[0] == columns &&
         tensor->dims[1] == rows && mlmc_type_is_supported(tensor->type);
}

static const gguf_tensor_info_t *layer_tensor(const mlmc_qwen2_model_t *model,
                                              uint32_t layer,
                                              const char *suffix) {
  char name[64];
  int written = snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
  if (written < 0 || (size_t)written >= sizeof(name)) {
    return NULL;
  }
  return gguf_find_tensor(&model->gguf, name);
}

static bool validate_layer(const mlmc_qwen2_model_t *model, uint32_t layer) {
  const mlmc_qwen2_config_t *config = &model->config;
  uint32_t hidden = config->embedding_length;
  uint32_t key_value_length =
      hidden / config->head_count * config->head_count_kv;
  uint32_t ffn = config->feed_forward_length;

  return is_f32_vector(layer_tensor(model, layer, "attn_norm.weight"),
                       hidden) &&
         is_matrix(layer_tensor(model, layer, "attn_q.weight"), hidden,
                   hidden) &&
         is_f32_vector(layer_tensor(model, layer, "attn_q.bias"), hidden) &&
         is_matrix(layer_tensor(model, layer, "attn_k.weight"), hidden,
                   key_value_length) &&
         is_f32_vector(layer_tensor(model, layer, "attn_k.bias"),
                       key_value_length) &&
         is_matrix(layer_tensor(model, layer, "attn_v.weight"), hidden,
                   key_value_length) &&
         is_f32_vector(layer_tensor(model, layer, "attn_v.bias"),
                       key_value_length) &&
         is_matrix(layer_tensor(model, layer, "attn_output.weight"), hidden,
                   hidden) &&
         is_f32_vector(layer_tensor(model, layer, "ffn_norm.weight"), hidden) &&
         is_matrix(layer_tensor(model, layer, "ffn_gate.weight"), hidden,
                   ffn) &&
         is_matrix(layer_tensor(model, layer, "ffn_up.weight"), hidden, ffn) &&
         is_matrix(layer_tensor(model, layer, "ffn_down.weight"), ffn, hidden);
}

static bool validate_tensors(mlmc_qwen2_model_t *model) {
  uint32_t hidden = model->config.embedding_length;
  const gguf_tensor_info_t *embedding =
      gguf_find_tensor(&model->gguf, "token_embd.weight");
  const gguf_tensor_info_t *output =
      gguf_find_tensor(&model->gguf, "output.weight");
  const gguf_tensor_info_t *output_norm =
      gguf_find_tensor(&model->gguf, "output_norm.weight");
  if (!embedding || embedding->n_dims != 2 || embedding->dims[0] != hidden ||
      embedding->dims[1] == 0 || embedding->dims[1] > UINT32_MAX ||
      !mlmc_type_is_supported(embedding->type)) {
    return false;
  }
  model->config.vocab_size = (uint32_t)embedding->dims[1];
  if (!is_matrix(output, hidden, model->config.vocab_size) ||
      !is_f32_vector(output_norm, hidden)) {
    return false;
  }

  for (uint32_t layer = 0; layer < model->config.block_count; layer++) {
    if (!validate_layer(model, layer)) {
      return false;
    }
  }
  return true;
}

bool mlmc_qwen2_model_load(const char *path, mlmc_qwen2_model_t *model) {
  if (!path || !model) {
    return false;
  }
  memset(model, 0, sizeof(*model));
  if (gguf_load(path, &model->gguf) != 0 || !load_config(model) ||
      !validate_tensors(model)) {
    mlmc_qwen2_model_free(model);
    return false;
  }
  return true;
}

void mlmc_qwen2_model_free(mlmc_qwen2_model_t *model) {
  if (!model) {
    return;
  }
  gguf_free(&model->gguf);
  memset(&model->config, 0, sizeof(model->config));
}

static bool workspace_init(const mlmc_qwen2_model_t *model,
                           forward_workspace_t *workspace) {
  const mlmc_qwen2_config_t *config = &model->config;
  uint32_t hidden = config->embedding_length;
  uint32_t key_value_length =
      hidden / config->head_count * config->head_count_kv;
  uint32_t ffn = config->feed_forward_length;

  memset(workspace, 0, sizeof(*workspace));
  workspace->hidden = (float *)calloc(hidden, sizeof(float));
  workspace->norm = (float *)calloc(hidden, sizeof(float));
  workspace->query = (float *)calloc(hidden, sizeof(float));
  workspace->key = (float *)calloc(key_value_length, sizeof(float));
  workspace->value = (float *)calloc(key_value_length, sizeof(float));
  workspace->attention = (float *)calloc(hidden, sizeof(float));
  workspace->projection = (float *)calloc(hidden, sizeof(float));
  workspace->gate = (float *)calloc(ffn, sizeof(float));
  workspace->up = (float *)calloc(ffn, sizeof(float));
  workspace->ffn = (float *)calloc(ffn, sizeof(float));
  workspace->scores = (float *)calloc(config->context_length, sizeof(float));

  if (!workspace->hidden || !workspace->norm || !workspace->query ||
      !workspace->key || !workspace->value || !workspace->attention ||
      !workspace->projection || !workspace->gate || !workspace->up ||
      !workspace->ffn || !workspace->scores) {
    return false;
  }
  return true;
}

static void workspace_free(forward_workspace_t *workspace) {
  free(workspace->hidden);
  free(workspace->norm);
  free(workspace->query);
  free(workspace->key);
  free(workspace->value);
  free(workspace->attention);
  free(workspace->projection);
  free(workspace->gate);
  free(workspace->up);
  free(workspace->ffn);
  free(workspace->scores);
  memset(workspace, 0, sizeof(*workspace));
}

bool mlmc_qwen2_state_init(const mlmc_qwen2_model_t *model, uint32_t capacity,
                           mlmc_qwen2_state_t *state) {
  if (!model || !state || capacity == 0 ||
      capacity > model->config.context_length) {
    return false;
  }

  uint32_t key_value_length = model->config.embedding_length /
                              model->config.head_count *
                              model->config.head_count_kv;
  size_t count = model->config.block_count;
  size_t bytes;
  if (!multiply_size(count, capacity, &count) ||
      !multiply_size(count, key_value_length, &count) ||
      !multiply_size(count, sizeof(float), &bytes) || count == 0) {
    return false;
  }

  memset(state, 0, sizeof(*state));
  state->keys = (float *)calloc(1, bytes);
  state->values = (float *)calloc(1, bytes);
  if (!state->keys || !state->values) {
    mlmc_qwen2_state_free(state);
    return false;
  }
  state->capacity = capacity;
  state->block_count = model->config.block_count;
  state->key_value_length = key_value_length;
  return true;
}

void mlmc_qwen2_state_reset(mlmc_qwen2_state_t *state) {
  if (!state) {
    return;
  }
  size_t count = state->block_count;
  size_t bytes;
  if (state->keys && state->values &&
      multiply_size(count, state->capacity, &count) &&
      multiply_size(count, state->key_value_length, &count) &&
      multiply_size(count, sizeof(float), &bytes)) {
    memset(state->keys, 0, bytes);
    memset(state->values, 0, bytes);
  }
  state->token_count = 0;
}

void mlmc_qwen2_state_free(mlmc_qwen2_state_t *state) {
  if (!state) {
    return;
  }
  free(state->keys);
  free(state->values);
  memset(state, 0, sizeof(*state));
}

static bool matrix_gemv(const mlmc_qwen2_model_t *model,
                        const gguf_tensor_info_t *tensor, uint32_t columns,
                        uint32_t rows, const float *input, float *output) {
  if (!is_matrix(tensor, columns, rows)) {
    return false;
  }
  const void *data = gguf_tensor_data(&model->gguf, tensor, NULL);
  return data && mlmc_gemv(tensor->type, data, rows, columns, input, output);
}

static bool rms_norm(const mlmc_qwen2_model_t *model,
                     const gguf_tensor_info_t *weights, const float *input,
                     float *output) {
  uint32_t length = model->config.embedding_length;
  if (!is_f32_vector(weights, length)) {
    return false;
  }
  const uint8_t *data =
      (const uint8_t *)gguf_tensor_data(&model->gguf, weights, NULL);
  if (!data) {
    return false;
  }

  float sum = 0.0f;
  for (uint32_t i = 0; i < length; i++) {
    sum += input[i] * input[i];
  }
  float scale = 1.0f / sqrtf(sum / (float)length + model->config.rms_epsilon);
  for (uint32_t i = 0; i < length; i++) {
    output[i] = input[i] * scale * read_f32(data + (size_t)i * sizeof(float));
  }
  return true;
}

static bool add_bias(const mlmc_qwen2_model_t *model,
                     const gguf_tensor_info_t *bias, uint32_t length,
                     float *values) {
  if (!is_f32_vector(bias, length)) {
    return false;
  }
  const uint8_t *data =
      (const uint8_t *)gguf_tensor_data(&model->gguf, bias, NULL);
  if (!data) {
    return false;
  }
  for (uint32_t i = 0; i < length; i++) {
    values[i] += read_f32(data + (size_t)i * sizeof(float));
  }
  return true;
}

static bool copy_embedding(const mlmc_qwen2_model_t *model, uint32_t token_id,
                           float *embedding) {
  const gguf_tensor_info_t *tensor =
      gguf_find_tensor(&model->gguf, "token_embd.weight");
  uint32_t hidden = model->config.embedding_length;
  if (!is_matrix(tensor, hidden, model->config.vocab_size) ||
      token_id >= model->config.vocab_size) {
    return false;
  }
  const uint8_t *data =
      (const uint8_t *)gguf_tensor_data(&model->gguf, tensor, NULL);
  if (!data) {
    return false;
  }

  uint64_t dims[1] = {hidden};
  size_t row_bytes = 0;
  if (!ggml_tensor_storage_size(tensor->type, 1, dims, &row_bytes)) {
    return false;
  }
  return mlmc_dequantize_row(tensor->type, data + (size_t)token_id * row_bytes,
                             hidden, embedding);
}

static void apply_rope(float *values, uint32_t head_count, uint32_t head_dim,
                       uint64_t position, float freq_base) {
  for (uint32_t head = 0; head < head_count; head++) {
    float *head_values = values + (size_t)head * head_dim;
    uint32_t half_dim = head_dim / 2;
    for (uint32_t dim = 0; dim < half_dim; dim++) {
      float frequency = powf(freq_base, -(float)(2 * dim) / (float)head_dim);
      float angle = (float)position * frequency;
      float cosine = cosf(angle);
      float sine = sinf(angle);
      float first = head_values[dim];
      float second = head_values[dim + half_dim];
      head_values[dim] = first * cosine - second * sine;
      head_values[dim + half_dim] = first * sine + second * cosine;
    }
  }
}

static float *cache_position(float *cache, const mlmc_qwen2_state_t *state,
                             uint32_t layer, uint32_t position) {
  size_t layer_offset =
      (size_t)layer * state->capacity * state->key_value_length;
  size_t position_offset = (size_t)position * state->key_value_length;
  return cache + layer_offset + position_offset;
}

static void attention(const mlmc_qwen2_model_t *model,
                      const mlmc_qwen2_state_t *state, uint32_t layer,
                      uint32_t position, const float *query, float *output,
                      float *scores) {
  uint32_t head_count = model->config.head_count;
  uint32_t key_value_heads = model->config.head_count_kv;
  uint32_t head_dim = model->config.embedding_length / head_count;
  uint32_t queries_per_key = head_count / key_value_heads;
  float score_scale = 1.0f / sqrtf((float)head_dim);

  for (uint32_t head = 0; head < head_count; head++) {
    const float *query_head = query + (size_t)head * head_dim;
    uint32_t key_value_head = head / queries_per_key;
    float max_score = -INFINITY;

    for (uint32_t token = 0; token <= position; token++) {
      const float *keys = cache_position(state->keys, state, layer, token);
      const float *key = keys + (size_t)key_value_head * head_dim;
      float score = 0.0f;
      for (uint32_t dim = 0; dim < head_dim; dim++) {
        score += query_head[dim] * key[dim];
      }
      scores[token] = score * score_scale;
      if (scores[token] > max_score) {
        max_score = scores[token];
      }
    }

    float normalizer = 0.0f;
    for (uint32_t token = 0; token <= position; token++) {
      scores[token] = expf(scores[token] - max_score);
      normalizer += scores[token];
    }

    float *output_head = output + (size_t)head * head_dim;
    memset(output_head, 0, (size_t)head_dim * sizeof(float));
    for (uint32_t token = 0; token <= position; token++) {
      const float *values = cache_position(state->values, state, layer, token);
      const float *value = values + (size_t)key_value_head * head_dim;
      float probability = scores[token] / normalizer;
      for (uint32_t dim = 0; dim < head_dim; dim++) {
        output_head[dim] += probability * value[dim];
      }
    }
  }
}

static void add_in_place(float *destination, const float *source,
                         uint32_t length) {
  for (uint32_t i = 0; i < length; i++) {
    destination[i] += source[i];
  }
}

static void swiglu(const float *gate, const float *up, uint32_t length,
                   float *output) {
  for (uint32_t i = 0; i < length; i++) {
    float silu = gate[i] / (1.0f + expf(-gate[i]));
    output[i] = silu * up[i];
  }
}

static bool run_layer(const mlmc_qwen2_model_t *model,
                      mlmc_qwen2_state_t *state, uint32_t layer,
                      uint32_t position, forward_workspace_t *workspace) {
  uint32_t hidden = model->config.embedding_length;
  uint32_t key_value_length = state->key_value_length;
  uint32_t ffn = model->config.feed_forward_length;

  const gguf_tensor_info_t *attn_norm =
      layer_tensor(model, layer, "attn_norm.weight");
  const gguf_tensor_info_t *query_weight =
      layer_tensor(model, layer, "attn_q.weight");
  const gguf_tensor_info_t *key_weight =
      layer_tensor(model, layer, "attn_k.weight");
  const gguf_tensor_info_t *value_weight =
      layer_tensor(model, layer, "attn_v.weight");
  const gguf_tensor_info_t *output_weight =
      layer_tensor(model, layer, "attn_output.weight");
  const gguf_tensor_info_t *ffn_norm =
      layer_tensor(model, layer, "ffn_norm.weight");
  const gguf_tensor_info_t *gate_weight =
      layer_tensor(model, layer, "ffn_gate.weight");
  const gguf_tensor_info_t *up_weight =
      layer_tensor(model, layer, "ffn_up.weight");
  const gguf_tensor_info_t *down_weight =
      layer_tensor(model, layer, "ffn_down.weight");

  if (!rms_norm(model, attn_norm, workspace->hidden, workspace->norm) ||
      !matrix_gemv(model, query_weight, hidden, hidden, workspace->norm,
                   workspace->query) ||
      !matrix_gemv(model, key_weight, hidden, key_value_length, workspace->norm,
                   workspace->key) ||
      !matrix_gemv(model, value_weight, hidden, key_value_length,
                   workspace->norm, workspace->value) ||
      !add_bias(model, layer_tensor(model, layer, "attn_q.bias"), hidden,
                workspace->query) ||
      !add_bias(model, layer_tensor(model, layer, "attn_k.bias"),
                key_value_length, workspace->key) ||
      !add_bias(model, layer_tensor(model, layer, "attn_v.bias"),
                key_value_length, workspace->value)) {
    return false;
  }

  uint32_t head_dim = hidden / model->config.head_count;
  apply_rope(workspace->query, model->config.head_count, head_dim, position,
             model->config.rope_freq_base);
  apply_rope(workspace->key, model->config.head_count_kv, head_dim, position,
             model->config.rope_freq_base);

  float *cached_key = cache_position(state->keys, state, layer, position);
  float *cached_value = cache_position(state->values, state, layer, position);
  memcpy(cached_key, workspace->key, (size_t)key_value_length * sizeof(float));
  memcpy(cached_value, workspace->value,
         (size_t)key_value_length * sizeof(float));
  attention(model, state, layer, position, workspace->query,
            workspace->attention, workspace->scores);

  if (!matrix_gemv(model, output_weight, hidden, hidden, workspace->attention,
                   workspace->projection)) {
    return false;
  }
  add_in_place(workspace->hidden, workspace->projection, hidden);

  if (!rms_norm(model, ffn_norm, workspace->hidden, workspace->norm) ||
      !matrix_gemv(model, gate_weight, hidden, ffn, workspace->norm,
                   workspace->gate) ||
      !matrix_gemv(model, up_weight, hidden, ffn, workspace->norm,
                   workspace->up)) {
    return false;
  }
  swiglu(workspace->gate, workspace->up, ffn, workspace->ffn);
  if (!matrix_gemv(model, down_weight, ffn, hidden, workspace->ffn,
                   workspace->projection)) {
    return false;
  }
  add_in_place(workspace->hidden, workspace->projection, hidden);
  return true;
}

bool mlmc_qwen2_forward(const mlmc_qwen2_model_t *model,
                        mlmc_qwen2_state_t *state, uint32_t token_id,
                        float *logits) {
  if (!model || !state || !logits || !state->keys || !state->values ||
      state->token_count >= state->capacity ||
      state->block_count != model->config.block_count ||
      state->key_value_length != model->config.embedding_length /
                                     model->config.head_count *
                                     model->config.head_count_kv) {
    return false;
  }

  forward_workspace_t workspace;
  if (!workspace_init(model, &workspace) ||
      !copy_embedding(model, token_id, workspace.hidden)) {
    workspace_free(&workspace);
    return false;
  }

  uint32_t position = (uint32_t)state->token_count;
  bool ok = true;
  for (uint32_t layer = 0; layer < model->config.block_count; layer++) {
    if (!run_layer(model, state, layer, position, &workspace)) {
      ok = false;
      break;
    }
  }

  const gguf_tensor_info_t *output_norm =
      gguf_find_tensor(&model->gguf, "output_norm.weight");
  const gguf_tensor_info_t *output =
      gguf_find_tensor(&model->gguf, "output.weight");
  if (ok && (!rms_norm(model, output_norm, workspace.hidden, workspace.norm) ||
             !matrix_gemv(model, output, model->config.embedding_length,
                          model->config.vocab_size, workspace.norm, logits))) {
    ok = false;
  }
  if (ok) {
    state->token_count++;
  }
  workspace_free(&workspace);
  return ok;
}
