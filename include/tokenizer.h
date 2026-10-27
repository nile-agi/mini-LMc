#ifndef MINILMC_TOKENIZER_H
#define MINILMC_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "gguf.h"

typedef struct {
  uint32_t left;
  uint32_t right;
  uint32_t merged;
  uint32_t rank;
} mlmc_bpe_merge_t;

typedef struct {
  const gguf_value_t *tokens;
  uint64_t token_count;
  mlmc_bpe_merge_t *merges;
  size_t merge_count;
  uint32_t *token_slots;
  size_t token_slot_count;
  size_t *merge_slots;
  size_t merge_slot_count;
  uint32_t bos_token_id;
  uint32_t eos_token_id;
  bool has_bos_token;
  bool has_eos_token;
  bool add_bos_token;
} mlmc_tokenizer_t;

bool mlmc_tokenizer_init(const gguf_file_t *gguf, mlmc_tokenizer_t *tokenizer);
void mlmc_tokenizer_free(mlmc_tokenizer_t *tokenizer);

// Encodes UTF-8 text with the model's byte-level BPE tokenizer.
bool mlmc_tokenizer_encode(const mlmc_tokenizer_t *tokenizer, const char *text,
                           bool add_bos, uint32_t **token_ids,
                           size_t *token_count);

// Writes one token as UTF-8, translating byte-level BPE symbols back to bytes.
bool mlmc_tokenizer_write_token(const mlmc_tokenizer_t *tokenizer,
                                uint32_t token_id, FILE *stream);

// Temperature 0 is greedy. top_k 0 considers the full vocabulary.
bool mlmc_sample(const float *logits, uint32_t count, float temperature,
                 uint32_t top_k, uint64_t *rng_state, uint32_t *token_id);

#endif
