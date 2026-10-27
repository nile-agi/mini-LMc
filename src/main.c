#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"
#include "model.h"
#include "tokenizer.h"

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s show <model.gguf> [max_tensors]\n"
          "  %s info <model.gguf> <tensor_name>\n"
          "  %s forward <model.gguf> <token_id> [token_id ...]\n"
          "  %s generate <model.gguf> <prompt> [max_tokens] [temperature] "
          "[top_k]\n"
          "  %s chat <model.gguf> [max_tokens] [temperature] [top_k]\n",
          prog, prog, prog, prog, prog);
}

static int cmd_show(int argc, char **argv) {
  int max_tensors = (argc >= 4) ? atoi(argv[3]) : -1;
  gguf_file_t f;
  if (gguf_load(argv[2], &f) != 0) {
    return 1;
  }
  gguf_print_summary(&f, max_tensors);
  gguf_free(&f);
  return 0;
}

static int cmd_info(int argc, char **argv) {
  if (argc != 4) {
    usage(argv[0]);
    return 1;
  }

  gguf_file_t f;
  if (gguf_load(argv[2], &f) != 0) {
    return 1;
  }

  const gguf_tensor_info_t *t = gguf_find_tensor(&f, argv[3]);
  if (!t) {
    fprintf(stderr, "No tensor named '%s'\n", argv[3]);
    gguf_free(&f);
    return 1;
  }

  size_t size = 0;
  const void *ptr = gguf_tensor_data(&f, t, &size);
  if (!ptr) {
    fprintf(stderr, "Failed to resolve tensor data pointer\n");
    gguf_free(&f);
    return 1;
  }

  printf("%s: %zu bytes, type %s\n", t->name, size, ggml_type_name(t->type));
  printf("first bytes: ");
  size_t preview = size < 16 ? size : 16;
  for (size_t i = 0; i < preview; i++) {
    printf("%02x ", ((const unsigned char *)ptr)[i]);
  }
  printf("\n");

  gguf_free(&f);
  return 0;
}

static uint32_t find_max_logit(const float *logits, uint32_t count) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < count; i++) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return best;
}

static int allocation_size(size_t count, size_t element_size, size_t *size) {
  if (count != 0 && element_size > SIZE_MAX / count) {
    return -1;
  }
  *size = count * element_size;
  return 0;
}

static int parse_uint32(const char *text, uint32_t *value) {
  if (!text || !*text || *text == '-') {
    return -1;
  }
  char *end = NULL;
  errno = 0;
  unsigned long parsed = strtoul(text, &end, 10);
  if (errno != 0 || !end || end == text || *end != '\0' ||
      parsed > UINT32_MAX) {
    return -1;
  }
  *value = (uint32_t)parsed;
  return 0;
}

static int parse_token_id(const char *text, uint32_t *token_id) {
  return parse_uint32(text, token_id);
}

static int parse_temperature(const char *text, float *temperature) {
  if (!text || !*text) {
    return -1;
  }
  char *end = NULL;
  errno = 0;
  float parsed = strtof(text, &end);
  if (errno != 0 || !end || end == text || *end != '\0' || !isfinite(parsed) ||
      parsed < 0.0f) {
    return -1;
  }
  *temperature = parsed;
  return 0;
}

static int read_line(char **line) {
  size_t length = 0;
  size_t capacity = 128;
  char *buffer = (char *)malloc(capacity);
  if (!buffer) {
    return -1;
  }

  int character;
  while ((character = getchar()) != '\n' && character != EOF) {
    if (length + 1 >= capacity) {
      if (capacity > SIZE_MAX / 2) {
        free(buffer);
        return -1;
      }
      capacity *= 2;
      char *resized = (char *)realloc(buffer, capacity);
      if (!resized) {
        free(buffer);
        return -1;
      }
      buffer = resized;
    }
    buffer[length++] = (char)character;
  }
  if (character == EOF && ferror(stdin)) {
    free(buffer);
    return -1;
  }
  if (character == EOF && length == 0) {
    free(buffer);
    return 0;
  }
  buffer[length] = '\0';
  *line = buffer;
  return 1;
}

static bool feed_text(const mlmc_tokenizer_t *tokenizer,
                      const mlmc_qwen2_model_t *model,
                      mlmc_qwen2_state_t *state, const char *text, bool add_bos,
                      float *logits) {
  uint32_t *tokens = NULL;
  size_t token_count = 0;
  if (!mlmc_tokenizer_encode(tokenizer, text, add_bos, &tokens, &token_count)) {
    return false;
  }
  if (token_count > state->capacity - state->token_count) {
    free(tokens);
    return false;
  }
  for (size_t i = 0; i < token_count; i++) {
    if (!mlmc_qwen2_forward(model, state, tokens[i], logits)) {
      free(tokens);
      return false;
    }
  }
  free(tokens);
  return true;
}

static uint32_t find_token_id(const mlmc_tokenizer_t *tokenizer,
                              const char *text) {
  for (uint32_t id = 0; id < tokenizer->token_count; id++) {
    if (strcmp(tokenizer->tokens->v.v_arr.items[id].v.v_str, text) == 0) {
      return id;
    }
  }
  return UINT32_MAX;
}

static int cmd_chat(int argc, char **argv) {
  if (argc < 3 || argc > 6) {
    usage(argv[0]);
    return 1;
  }

  uint32_t max_tokens = 128;
  float temperature = 0.0f;
  uint32_t top_k = 40;
  if ((argc >= 4 && parse_uint32(argv[3], &max_tokens) != 0) ||
      (argc >= 5 && parse_temperature(argv[4], &temperature) != 0) ||
      (argc >= 6 && parse_uint32(argv[5], &top_k) != 0) || max_tokens == 0) {
    fprintf(stderr, "Invalid chat parameters\n");
    return 1;
  }

  mlmc_qwen2_model_t model = {0};
  if (!mlmc_qwen2_model_load(argv[2], &model)) {
    fprintf(stderr, "Failed to load a supported Qwen2 model\n");
    return 1;
  }

  mlmc_tokenizer_t tokenizer = {0};
  if (!mlmc_tokenizer_init(&model.gguf, &tokenizer) ||
      tokenizer.token_count != model.config.vocab_size) {
    fprintf(stderr, "Model has missing or unsupported tokenizer metadata\n");
    mlmc_qwen2_model_free(&model);
    return 1;
  }
  uint32_t im_end_id = find_token_id(&tokenizer, "<|im_end|>");
  uint32_t im_start_id = find_token_id(&tokenizer, "<|im_start|>");
  if (im_end_id == UINT32_MAX || im_start_id == UINT32_MAX) {
    fprintf(stderr, "Model tokenizer does not provide Qwen chat markers\n");
    mlmc_tokenizer_free(&tokenizer);
    mlmc_qwen2_model_free(&model);
    return 1;
  }

  mlmc_qwen2_state_t state = {0};
  if (!mlmc_qwen2_state_init(&model, model.config.context_length, &state)) {
    fprintf(stderr, "Failed to allocate the KV cache\n");
    mlmc_tokenizer_free(&tokenizer);
    mlmc_qwen2_model_free(&model);
    return 1;
  }

  size_t logits_size;
  if (allocation_size(model.config.vocab_size, sizeof(float), &logits_size) !=
      0) {
    fprintf(stderr, "Vocabulary is too large\n");
    mlmc_qwen2_state_free(&state);
    mlmc_tokenizer_free(&tokenizer);
    mlmc_qwen2_model_free(&model);
    return 1;
  }
  float *logits = (float *)malloc(logits_size);
  if (!logits) {
    fprintf(stderr, "Failed to allocate logits\n");
    mlmc_qwen2_state_free(&state);
    mlmc_tokenizer_free(&tokenizer);
    mlmc_qwen2_model_free(&model);
    return 1;
  }

  puts("Chat started. Type /exit to quit.");
  bool first_turn = true;
  bool context_full = false;
  int result = 0;
  uint64_t rng_state = UINT64_C(0x243f6a8885a308d3);
  while (!context_full) {
    fputs("\nYou> ", stdout);
    fflush(stdout);
    char *user_text = NULL;
    int input_status = read_line(&user_text);
    if (input_status == 0) {
      break;
    }
    if (input_status < 0) {
      fprintf(stderr, "Failed to read input\n");
      result = 1;
      break;
    }
    if (strcmp(user_text, "/exit") == 0) {
      free(user_text);
      break;
    }
    if (user_text[0] == '\0') {
      free(user_text);
      continue;
    }

    const char *prefix = first_turn ? "<|im_start|>system\nYou are a helpful "
                                      "assistant.<|im_end|>\n<|im_start|>user\n"
                                    : "<|im_end|>\n<|im_start|>user\n";
    bool prompt_ok =
        feed_text(&tokenizer, &model, &state, prefix,
                  first_turn && tokenizer.add_bos_token, logits) &&
        feed_text(&tokenizer, &model, &state, user_text, false, logits) &&
        feed_text(&tokenizer, &model, &state, "\n<|im_start|>assistant\n",
                  false, logits);
    free(user_text);
    if (!prompt_ok) {
      fprintf(stderr, "Prompt failed or the model context is full\n");
      result = 1;
      break;
    }
    first_turn = false;

    fputs("Assistant> ", stdout);
    fflush(stdout);
    uint32_t generation_limit = max_tokens;
    uint64_t remaining = state.capacity - state.token_count;
    if (remaining < generation_limit) {
      generation_limit = (uint32_t)remaining;
    }
    if (generation_limit == 0) {
      context_full = true;
    }
    for (uint32_t i = 0; i < generation_limit; i++) {
      uint32_t next_token;
      if (!mlmc_sample(logits, model.config.vocab_size, temperature, top_k,
                       &rng_state, &next_token)) {
        fprintf(stderr, "\nToken sampling failed\n");
        result = 1;
        context_full = true;
        break;
      }
      if (next_token == im_end_id ||
          (tokenizer.has_eos_token && next_token == tokenizer.eos_token_id)) {
        break;
      }
      if (!mlmc_tokenizer_write_token(&tokenizer, next_token, stdout) ||
          fflush(stdout) != 0) {
        fprintf(stderr, "\nFailed to write assistant response\n");
        result = 1;
        context_full = true;
        break;
      }
      if (!mlmc_qwen2_forward(&model, &state, next_token, logits)) {
        fprintf(stderr, "\nModel forward pass failed during chat\n");
        result = 1;
        context_full = true;
        break;
      }
      if (state.token_count == state.capacity) {
        context_full = true;
        break;
      }
    }
    fputc('\n', stdout);
    if (context_full) {
      fprintf(stderr,
              "Model context is full; restart chat for a new session.\n");
    }
  }

  free(logits);
  mlmc_qwen2_state_free(&state);
  mlmc_tokenizer_free(&tokenizer);
  mlmc_qwen2_model_free(&model);
  return result;
}

static int cmd_generate(int argc, char **argv) {
  if (argc < 4 || argc > 7) {
    usage(argv[0]);
    return 1;
  }

  uint32_t max_tokens = 128;
  float temperature = 0.0f;
  uint32_t top_k = 40;
  if ((argc >= 5 && parse_uint32(argv[4], &max_tokens) != 0) ||
      (argc >= 6 && parse_temperature(argv[5], &temperature) != 0) ||
      (argc >= 7 && parse_uint32(argv[6], &top_k) != 0) || max_tokens == 0) {
    fprintf(stderr, "Invalid generation parameters\n");
    return 1;
  }

  mlmc_qwen2_model_t model = {0};
  if (!mlmc_qwen2_model_load(argv[2], &model)) {
    fprintf(stderr, "Failed to load a supported Qwen2 model\n");
    return 1;
  }

  mlmc_tokenizer_t tokenizer = {0};
  if (!mlmc_tokenizer_init(&model.gguf, &tokenizer) ||
      tokenizer.token_count != model.config.vocab_size) {
    fprintf(stderr, "Model has missing or unsupported tokenizer metadata\n");
    mlmc_qwen2_model_free(&model);
    return 1;
  }
  uint32_t *prompt_tokens = NULL;
  size_t prompt_count = 0;
  if (!mlmc_tokenizer_encode(&tokenizer, argv[3], tokenizer.add_bos_token,
                             &prompt_tokens, &prompt_count) ||
      prompt_count == 0 || prompt_count > UINT32_MAX ||
      prompt_count > model.config.context_length ||
      max_tokens > model.config.context_length - prompt_count) {
    fprintf(stderr, "Prompt cannot be encoded or exceeds the model context\n");
    free(prompt_tokens);
    mlmc_tokenizer_free(&tokenizer);
    mlmc_qwen2_model_free(&model);
    return 1;
  }

  uint32_t capacity = (uint32_t)(prompt_count + max_tokens);
  mlmc_qwen2_state_t state = {0};
  if (!mlmc_qwen2_state_init(&model, capacity, &state)) {
    fprintf(stderr, "Failed to allocate the KV cache\n");
    free(prompt_tokens);
    mlmc_tokenizer_free(&tokenizer);
    mlmc_qwen2_model_free(&model);
    return 1;
  }

  size_t logits_size;
  if (allocation_size(model.config.vocab_size, sizeof(float), &logits_size) !=
      0) {
    fprintf(stderr, "Vocabulary is too large\n");
    mlmc_qwen2_state_free(&state);
    free(prompt_tokens);
    mlmc_tokenizer_free(&tokenizer);
    mlmc_qwen2_model_free(&model);
    return 1;
  }
  float *logits = (float *)malloc(logits_size);
  if (!logits) {
    fprintf(stderr, "Failed to allocate logits\n");
    mlmc_qwen2_state_free(&state);
    free(prompt_tokens);
    mlmc_tokenizer_free(&tokenizer);
    mlmc_qwen2_model_free(&model);
    return 1;
  }

  int result = 0;
  for (size_t i = 0; i < prompt_count; i++) {
    if (!mlmc_qwen2_forward(&model, &state, prompt_tokens[i], logits)) {
      fprintf(stderr, "Model forward pass failed at prompt token %zu\n", i);
      result = 1;
      goto cleanup;
    }
  }

  uint64_t rng_state = UINT64_C(0x243f6a8885a308d3);
  for (uint32_t i = 0; i < max_tokens; i++) {
    uint32_t next_token;
    if (!mlmc_sample(logits, model.config.vocab_size, temperature, top_k,
                     &rng_state, &next_token)) {
      fprintf(stderr, "Token sampling failed\n");
      result = 1;
      break;
    }
    if (tokenizer.has_eos_token && next_token == tokenizer.eos_token_id) {
      break;
    }
    if (!mlmc_tokenizer_write_token(&tokenizer, next_token, stdout)) {
      fprintf(stderr, "Failed to write generated token\n");
      result = 1;
      break;
    }
    if (i + 1 < max_tokens &&
        !mlmc_qwen2_forward(&model, &state, next_token, logits)) {
      fprintf(stderr, "Model forward pass failed during generation\n");
      result = 1;
      break;
    }
  }
  if (fputc('\n', stdout) == EOF || fflush(stdout) != 0) {
    fprintf(stderr, "Failed to flush generated text\n");
    result = 1;
  }

cleanup:
  free(logits);
  mlmc_qwen2_state_free(&state);
  free(prompt_tokens);
  mlmc_tokenizer_free(&tokenizer);
  mlmc_qwen2_model_free(&model);
  return result;
}

static int cmd_forward(int argc, char **argv) {
  if (argc < 4) {
    usage(argv[0]);
    return 1;
  }

  mlmc_qwen2_model_t model = {0};
  if (!mlmc_qwen2_model_load(argv[2], &model)) {
    fprintf(stderr, "Failed to load a supported Qwen2 model\n");
    return 1;
  }

  mlmc_qwen2_state_t state = {0};
  uint32_t capacity = (uint32_t)(argc - 3);
  if (!mlmc_qwen2_state_init(&model, capacity, &state)) {
    fprintf(stderr, "Failed to allocate the KV cache\n");
    mlmc_qwen2_model_free(&model);
    return 1;
  }

  size_t logits_size;
  if (allocation_size(model.config.vocab_size, sizeof(float), &logits_size) !=
      0) {
    fprintf(stderr, "Vocabulary is too large\n");
    mlmc_qwen2_state_free(&state);
    mlmc_qwen2_model_free(&model);
    return 1;
  }
  float *logits = (float *)malloc(logits_size);
  if (!logits) {
    fprintf(stderr, "Failed to allocate logits\n");
    mlmc_qwen2_state_free(&state);
    mlmc_qwen2_model_free(&model);
    return 1;
  }

  int result = 0;
  for (int i = 3; i < argc; i++) {
    uint32_t token_id;
    if (parse_token_id(argv[i], &token_id) != 0 ||
        !mlmc_qwen2_forward(&model, &state, token_id, logits)) {
      fprintf(stderr, "Failed to run token '%s'\n", argv[i]);
      result = 1;
      break;
    }
    uint32_t best = find_max_logit(logits, model.config.vocab_size);
    printf("token %u: top token %u (logit %.6f)\n", token_id, best,
           logits[best]);
  }

  free(logits);
  mlmc_qwen2_state_free(&state);
  mlmc_qwen2_model_free(&model);
  return result;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 1;
  }

  if (strcmp(argv[1], "show") == 0) {
    return cmd_show(argc, argv);
  }
  if (strcmp(argv[1], "info") == 0) {
    return cmd_info(argc, argv);
  }
  if (strcmp(argv[1], "forward") == 0) {
    return cmd_forward(argc, argv);
  }
  if (strcmp(argv[1], "generate") == 0) {
    return cmd_generate(argc, argv);
  }
  if (strcmp(argv[1], "chat") == 0) {
    return cmd_chat(argc, argv);
  }

  usage(argv[0]);
  return 1;
}
