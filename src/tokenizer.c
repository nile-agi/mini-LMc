#include "tokenizer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint32_t token_id;
  float logit;
} candidate_t;

static uint64_t hash_bytes(const char *text, size_t length) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (size_t i = 0; i < length; i++) {
    hash ^= (unsigned char)text[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static uint64_t hash_pair(uint32_t left, uint32_t right) {
  uint64_t value = ((uint64_t)left << 32) | right;
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

static bool next_power_of_two(size_t value, size_t *out) {
  size_t result = 1;
  while (result < value) {
    if (result > SIZE_MAX / 2) {
      return false;
    }
    result *= 2;
  }
  *out = result;
  return true;
}

static bool string_array(const gguf_value_t *value) {
  if (!value || value->type != GGUF_TYPE_ARRAY ||
      value->v.v_arr.elem_type != GGUF_TYPE_STRING ||
      (value->v.v_arr.len != 0 && !value->v.v_arr.items)) {
    return false;
  }
  for (uint64_t i = 0; i < value->v.v_arr.len; i++) {
    if (!value->v.v_arr.items[i].v.v_str) {
      return false;
    }
  }
  return true;
}

static uint32_t lookup_token(const mlmc_tokenizer_t *tokenizer,
                             const char *text, size_t length) {
  if (length == 0 || tokenizer->token_slot_count == 0) {
    return UINT32_MAX;
  }
  size_t slot =
      (size_t)hash_bytes(text, length) & (tokenizer->token_slot_count - 1);
  for (size_t probes = 0; probes < tokenizer->token_slot_count; probes++) {
    uint32_t id = tokenizer->token_slots[slot];
    if (id == UINT32_MAX) {
      return UINT32_MAX;
    }
    const char *candidate = tokenizer->tokens->v.v_arr.items[id].v.v_str;
    if (strlen(candidate) == length && memcmp(candidate, text, length) == 0) {
      return id;
    }
    slot = (slot + 1) & (tokenizer->token_slot_count - 1);
  }
  return UINT32_MAX;
}

static size_t lookup_merge(const mlmc_tokenizer_t *tokenizer, uint32_t left,
                           uint32_t right) {
  if (tokenizer->merge_slot_count == 0) {
    return SIZE_MAX;
  }
  size_t slot =
      (size_t)hash_pair(left, right) & (tokenizer->merge_slot_count - 1);
  for (size_t probes = 0; probes < tokenizer->merge_slot_count; probes++) {
    size_t index = tokenizer->merge_slots[slot];
    if (index == SIZE_MAX) {
      return SIZE_MAX;
    }
    const mlmc_bpe_merge_t *merge = &tokenizer->merges[index];
    if (merge->left == left && merge->right == right) {
      return index;
    }
    slot = (slot + 1) & (tokenizer->merge_slot_count - 1);
  }
  return SIZE_MAX;
}

static bool metadata_token_id(const gguf_file_t *gguf, const char *key,
                              uint32_t *out, bool *present) {
  const gguf_value_t *value = gguf_find(gguf, key);
  *present = value != NULL;
  if (!value) {
    return true;
  }
  if (value->type != GGUF_TYPE_UINT32) {
    return false;
  }
  *out = value->v.v_u32;
  return true;
}

bool mlmc_tokenizer_init(const gguf_file_t *gguf, mlmc_tokenizer_t *tokenizer) {
  if (!gguf || !tokenizer) {
    return false;
  }
  memset(tokenizer, 0, sizeof(*tokenizer));
  const gguf_value_t *tokens = gguf_find(gguf, "tokenizer.ggml.tokens");
  const gguf_value_t *merges = gguf_find(gguf, "tokenizer.ggml.merges");
  if (!string_array(tokens) || !string_array(merges) ||
      tokens->v.v_arr.len == 0 || tokens->v.v_arr.len > UINT32_MAX ||
      merges->v.v_arr.len > UINT32_MAX || tokens->v.v_arr.len > SIZE_MAX / 2 ||
      merges->v.v_arr.len > (SIZE_MAX - 1) / 2) {
    return false;
  }

  tokenizer->tokens = tokens;
  tokenizer->token_count = tokens->v.v_arr.len;
  tokenizer->merge_count = (size_t)merges->v.v_arr.len;
  if (!next_power_of_two((size_t)tokenizer->token_count * 2,
                         &tokenizer->token_slot_count) ||
      !next_power_of_two(tokenizer->merge_count * 2 + 1,
                         &tokenizer->merge_slot_count) ||
      tokenizer->token_slot_count > SIZE_MAX / sizeof(uint32_t) ||
      tokenizer->merge_slot_count > SIZE_MAX / sizeof(size_t) ||
      tokenizer->merge_count > SIZE_MAX / sizeof(*tokenizer->merges)) {
    mlmc_tokenizer_free(tokenizer);
    return false;
  }
  tokenizer->token_slots =
      (uint32_t *)malloc(tokenizer->token_slot_count * sizeof(uint32_t));
  tokenizer->merges =
      tokenizer->merge_count
          ? (mlmc_bpe_merge_t *)calloc(tokenizer->merge_count,
                                       sizeof(*tokenizer->merges))
          : NULL;
  tokenizer->merge_slots =
      (size_t *)malloc(tokenizer->merge_slot_count * sizeof(size_t));
  if (!tokenizer->token_slots || !tokenizer->merge_slots ||
      (tokenizer->merge_count && !tokenizer->merges)) {
    mlmc_tokenizer_free(tokenizer);
    return false;
  }
  for (size_t i = 0; i < tokenizer->token_slot_count; i++) {
    tokenizer->token_slots[i] = UINT32_MAX;
  }
  for (size_t i = 0; i < tokenizer->merge_slot_count; i++) {
    tokenizer->merge_slots[i] = SIZE_MAX;
  }

  for (uint32_t id = 0; id < (uint32_t)tokenizer->token_count; id++) {
    const char *text = tokens->v.v_arr.items[id].v.v_str;
    size_t length = strlen(text);
    size_t slot =
        (size_t)hash_bytes(text, length) & (tokenizer->token_slot_count - 1);
    while (tokenizer->token_slots[slot] != UINT32_MAX) {
      uint32_t existing = tokenizer->token_slots[slot];
      const char *existing_text = tokens->v.v_arr.items[existing].v.v_str;
      if (strcmp(existing_text, text) == 0) {
        mlmc_tokenizer_free(tokenizer);
        return false;
      }
      slot = (slot + 1) & (tokenizer->token_slot_count - 1);
    }
    tokenizer->token_slots[slot] = id;
  }

  for (uint32_t rank = 0; rank < (uint32_t)tokenizer->merge_count; rank++) {
    const char *pair = merges->v.v_arr.items[rank].v.v_str;
    const char *separator = strchr(pair, ' ');
    if (!separator || separator == pair || separator[1] == '\0') {
      mlmc_tokenizer_free(tokenizer);
      return false;
    }
    size_t left_length = (size_t)(separator - pair);
    size_t right_length = strlen(separator + 1);
    uint32_t left = lookup_token(tokenizer, pair, left_length);
    uint32_t right = lookup_token(tokenizer, separator + 1, right_length);
    if (left == UINT32_MAX || right == UINT32_MAX ||
        left_length > SIZE_MAX - right_length) {
      mlmc_tokenizer_free(tokenizer);
      return false;
    }
    size_t combined_length = left_length + right_length;
    char *combined = (char *)malloc(combined_length + 1);
    if (!combined) {
      mlmc_tokenizer_free(tokenizer);
      return false;
    }
    memcpy(combined, pair, left_length);
    memcpy(combined + left_length, separator + 1, right_length);
    combined[combined_length] = '\0';
    uint32_t merged = lookup_token(tokenizer, combined, combined_length);
    free(combined);
    if (merged == UINT32_MAX) {
      mlmc_tokenizer_free(tokenizer);
      return false;
    }
    tokenizer->merges[rank] = (mlmc_bpe_merge_t){left, right, merged, rank};

    size_t slot =
        (size_t)hash_pair(left, right) & (tokenizer->merge_slot_count - 1);
    while (tokenizer->merge_slots[slot] != SIZE_MAX) {
      size_t previous = tokenizer->merge_slots[slot];
      if (tokenizer->merges[previous].left == left &&
          tokenizer->merges[previous].right == right) {
        mlmc_tokenizer_free(tokenizer);
        return false;
      }
      slot = (slot + 1) & (tokenizer->merge_slot_count - 1);
    }
    tokenizer->merge_slots[slot] = rank;
  }

  if (!metadata_token_id(gguf, "tokenizer.ggml.bos_token_id",
                         &tokenizer->bos_token_id, &tokenizer->has_bos_token) ||
      !metadata_token_id(gguf, "tokenizer.ggml.eos_token_id",
                         &tokenizer->eos_token_id, &tokenizer->has_eos_token)) {
    mlmc_tokenizer_free(tokenizer);
    return false;
  }
  const gguf_value_t *add_bos = gguf_find(gguf, "tokenizer.ggml.add_bos_token");
  if (add_bos) {
    if (add_bos->type != GGUF_TYPE_BOOL) {
      mlmc_tokenizer_free(tokenizer);
      return false;
    }
    tokenizer->add_bos_token = add_bos->v.v_bool != 0;
  }
  if ((tokenizer->has_bos_token &&
       tokenizer->bos_token_id >= tokenizer->token_count) ||
      (tokenizer->has_eos_token &&
       tokenizer->eos_token_id >= tokenizer->token_count) ||
      (tokenizer->add_bos_token && !tokenizer->has_bos_token)) {
    mlmc_tokenizer_free(tokenizer);
    return false;
  }
  return true;
}

void mlmc_tokenizer_free(mlmc_tokenizer_t *tokenizer) {
  if (!tokenizer) {
    return;
  }
  free(tokenizer->merges);
  free(tokenizer->token_slots);
  free(tokenizer->merge_slots);
  memset(tokenizer, 0, sizeof(*tokenizer));
}

static size_t encode_codepoint(uint32_t codepoint, char out[4]) {
  if (codepoint < 0x80) {
    out[0] = (char)codepoint;
    return 1;
  }
  if (codepoint < 0x800) {
    out[0] = (char)(0xc0 | (codepoint >> 6));
    out[1] = (char)(0x80 | (codepoint & 0x3f));
    return 2;
  }
  if (codepoint < 0x10000) {
    out[0] = (char)(0xe0 | (codepoint >> 12));
    out[1] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
    out[2] = (char)(0x80 | (codepoint & 0x3f));
    return 3;
  }
  out[0] = (char)(0xf0 | (codepoint >> 18));
  out[1] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
  out[2] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
  out[3] = (char)(0x80 | (codepoint & 0x3f));
  return 4;
}

static uint32_t codepoint_for_byte(unsigned char byte) {
  if ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
      byte >= 174) {
    return byte;
  }
  return 256 + byte;
}

static size_t decode_codepoint(const unsigned char *text, size_t length,
                               uint32_t *codepoint) {
  if (length == 0) {
    return 0;
  }
  unsigned char first = text[0];
  if (first < 0x80) {
    *codepoint = first;
    return 1;
  }
  size_t bytes = first < 0xe0 ? 2 : (first < 0xf0 ? 3 : 4);
  if (bytes > length) {
    return 0;
  }
  uint32_t value = first & (bytes == 2 ? 0x1f : (bytes == 3 ? 0x0f : 0x07));
  for (size_t i = 1; i < bytes; i++) {
    if ((text[i] & 0xc0) != 0x80) {
      return 0;
    }
    value = (value << 6) | (text[i] & 0x3f);
  }
  *codepoint = value;
  return bytes;
}

static int byte_for_codepoint(uint32_t codepoint) {
  if ((codepoint >= 33 && codepoint <= 126) ||
      (codepoint >= 161 && codepoint <= 172) ||
      (codepoint >= 174 && codepoint <= 255)) {
    return (int)codepoint;
  }
  if (codepoint >= 256 && codepoint <= 288) {
    return (int)(codepoint - 256);
  }
  if (codepoint == 383) {
    return 127;
  }
  if (codepoint == 416) {
    return 160;
  }
  if (codepoint == 429) {
    return 173;
  }
  return -1;
}

static bool append_id(uint32_t **ids, size_t *count, size_t *capacity,
                      uint32_t id) {
  if (*count == *capacity) {
    size_t next = *capacity ? *capacity * 2 : 32;
    if (next < *capacity || next > SIZE_MAX / sizeof(uint32_t)) {
      return false;
    }
    uint32_t *resized = (uint32_t *)realloc(*ids, next * sizeof(uint32_t));
    if (!resized) {
      return false;
    }
    *ids = resized;
    *capacity = next;
  }
  (*ids)[(*count)++] = id;
  return true;
}

static bool encode_segment(const mlmc_tokenizer_t *tokenizer,
                           const unsigned char *text, size_t length,
                           uint32_t **ids, size_t *count, size_t *capacity) {
  uint32_t *pieces = NULL;
  size_t piece_count = 0;
  size_t piece_capacity = 0;
  for (size_t i = 0; i < length; i++) {
    char encoded[4];
    size_t encoded_length =
        encode_codepoint(codepoint_for_byte(text[i]), encoded);
    uint32_t id = lookup_token(tokenizer, encoded, encoded_length);
    if (id == UINT32_MAX ||
        !append_id(&pieces, &piece_count, &piece_capacity, id)) {
      free(pieces);
      return false;
    }
  }

  while (piece_count > 1) {
    size_t best_position = SIZE_MAX;
    size_t best_rank = SIZE_MAX;
    for (size_t i = 0; i + 1 < piece_count; i++) {
      size_t merge_index = lookup_merge(tokenizer, pieces[i], pieces[i + 1]);
      if (merge_index != SIZE_MAX &&
          tokenizer->merges[merge_index].rank < best_rank) {
        best_position = i;
        best_rank = tokenizer->merges[merge_index].rank;
      }
    }
    if (best_position == SIZE_MAX) {
      break;
    }
    size_t merge_index = lookup_merge(tokenizer, pieces[best_position],
                                      pieces[best_position + 1]);
    pieces[best_position] = tokenizer->merges[merge_index].merged;
    memmove(pieces + best_position + 1, pieces + best_position + 2,
            (piece_count - best_position - 2) * sizeof(uint32_t));
    piece_count--;
  }

  bool ok = true;
  for (size_t i = 0; i < piece_count; i++) {
    if (!append_id(ids, count, capacity, pieces[i])) {
      ok = false;
      break;
    }
  }
  free(pieces);
  return ok;
}

typedef enum { CHAR_SPACE, CHAR_LETTER, CHAR_NUMBER, CHAR_OTHER } char_class_t;

static size_t utf8_width(unsigned char first) {
  if (first < 0x80) {
    return 1;
  }
  if (first < 0xe0) {
    return 2;
  }
  if (first < 0xf0) {
    return 3;
  }
  return 4;
}

static uint32_t read_codepoint(const unsigned char *text, size_t width) {
  uint32_t value =
      text[0] &
      (width == 1 ? 0x7f : (width == 2 ? 0x1f : (width == 3 ? 0x0f : 0x07)));
  for (size_t i = 1; i < width; i++) {
    value = (value << 6) | (text[i] & 0x3f);
  }
  return value;
}

static char_class_t classify_codepoint(uint32_t codepoint) {
  if (codepoint == ' ' || (codepoint >= '\t' && codepoint <= '\r') ||
      codepoint == 0x85 || codepoint == 0xa0 || codepoint == 0x1680 ||
      (codepoint >= 0x2000 && codepoint <= 0x200a) || codepoint == 0x2028 ||
      codepoint == 0x2029 || codepoint == 0x202f || codepoint == 0x205f ||
      codepoint == 0x3000) {
    return CHAR_SPACE;
  }
  if (codepoint >= '0' && codepoint <= '9') {
    return CHAR_NUMBER;
  }
  if ((codepoint >= 'A' && codepoint <= 'Z') ||
      (codepoint >= 'a' && codepoint <= 'z')) {
    return CHAR_LETTER;
  }
  if ((codepoint >= 0x2000 && codepoint <= 0x206f) ||
      (codepoint >= 0x2e00 && codepoint <= 0x2e7f) ||
      (codepoint >= 0x3001 && codepoint <= 0x303f) ||
      (codepoint >= 0xfe10 && codepoint <= 0xfe1f) ||
      (codepoint >= 0xff01 && codepoint <= 0xff0f) ||
      (codepoint >= 0xff1a && codepoint <= 0xff20) ||
      (codepoint >= 0xff3b && codepoint <= 0xff40) ||
      (codepoint >= 0xff5b && codepoint <= 0xff65)) {
    return CHAR_OTHER;
  }
  if (codepoint >= 0x80) {
    return CHAR_LETTER;
  }
  return CHAR_OTHER;
}

static size_t codepoint_at(const unsigned char *text, size_t length,
                           size_t offset, char_class_t *class_out,
                           uint32_t *codepoint_out) {
  size_t width = utf8_width(text[offset]);
  if (width > length - offset) {
    return 0;
  }
  uint32_t codepoint = read_codepoint(text + offset, width);
  *class_out = classify_codepoint(codepoint);
  if (codepoint_out) {
    *codepoint_out = codepoint;
  }
  return width;
}

static size_t segment_end(const unsigned char *text, size_t length,
                          size_t start) {
  char_class_t first_class;
  uint32_t first_codepoint;
  size_t first_width =
      codepoint_at(text, length, start, &first_class, &first_codepoint);
  if (first_width == 0) {
    return start;
  }
  size_t offset = start + first_width;
  if (first_class == CHAR_SPACE) {
    if (first_codepoint == ' ' && offset < length && text[offset] != ' ' &&
        text[offset] != '\n' && text[offset] != '\r') {
      return segment_end(text, length, offset);
    }
    while (offset < length) {
      char_class_t cls;
      uint32_t cp;
      size_t width = codepoint_at(text, length, offset, &cls, &cp);
      if (width == 0 || cls != CHAR_SPACE) {
        break;
      }
      offset += width;
      if (cp == '\n' || cp == '\r') {
        break;
      }
    }
    return offset;
  }

  if (first_class == CHAR_NUMBER) {
    uint32_t digits = 1;
    while (offset < length && digits < 3) {
      char_class_t cls;
      size_t width = codepoint_at(text, length, offset, &cls, NULL);
      if (width == 0 || cls != CHAR_NUMBER) {
        break;
      }
      offset += width;
      digits++;
    }
    return offset;
  }

  if (first_class == CHAR_LETTER) {
    while (offset < length) {
      char_class_t cls;
      size_t width = codepoint_at(text, length, offset, &cls, NULL);
      if (width == 0 || cls != CHAR_LETTER) {
        break;
      }
      offset += width;
    }
    return offset;
  }

  while (offset < length) {
    char_class_t cls;
    size_t width = codepoint_at(text, length, offset, &cls, NULL);
    if (width == 0 || cls != CHAR_OTHER) {
      break;
    }
    offset += width;
  }
  while (offset < length && (text[offset] == '\n' || text[offset] == '\r')) {
    offset++;
  }
  return offset;
}

static bool is_contraction(const unsigned char *text, size_t length,
                           size_t offset, size_t *end) {
  static const char *const suffixes[] = {"'s", "'t",  "'re", "'ve",
                                         "'m", "'ll", "'d"};
  for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
    size_t suffix_length = strlen(suffixes[i]);
    if (suffix_length > length - offset) {
      continue;
    }
    bool match = true;
    for (size_t j = 0; j < suffix_length; j++) {
      unsigned char actual = text[offset + j];
      unsigned char expected = (unsigned char)suffixes[i][j];
      if (actual >= 'A' && actual <= 'Z') {
        actual = (unsigned char)(actual + ('a' - 'A'));
      }
      if (actual != expected) {
        match = false;
        break;
      }
    }
    if (match) {
      *end = offset + suffix_length;
      return true;
    }
  }
  return false;
}

bool mlmc_tokenizer_encode(const mlmc_tokenizer_t *tokenizer, const char *text,
                           bool add_bos, uint32_t **token_ids,
                           size_t *token_count) {
  if (!tokenizer || !tokenizer->tokens || !text || !token_ids || !token_count) {
    return false;
  }
  *token_ids = NULL;
  *token_count = 0;
  size_t capacity = 0;
  if (add_bos) {
    if (!tokenizer->has_bos_token ||
        !append_id(token_ids, token_count, &capacity,
                   tokenizer->bos_token_id)) {
      free(*token_ids);
      *token_ids = NULL;
      *token_count = 0;
      return false;
    }
  }

  size_t length = strlen(text);
  size_t offset = 0;
  while (offset < length) {
    const char *special_start = strstr(text + offset, "<|");
    const char *special_end =
        special_start ? strstr(special_start + 2, "|>") : NULL;
    uint32_t special_id = UINT32_MAX;
    size_t special_offset = 0;
    size_t special_length = 0;
    if (special_start && special_end) {
      special_offset = (size_t)(special_start - text);
      special_length = (size_t)(special_end + 2 - special_start);
      special_id = lookup_token(tokenizer, special_start, special_length);
    }

    if (special_id != UINT32_MAX) {
      if (special_offset > offset &&
          !encode_segment(tokenizer, (const unsigned char *)text + offset,
                          special_offset - offset, token_ids, token_count,
                          &capacity)) {
        free(*token_ids);
        *token_ids = NULL;
        *token_count = 0;
        return false;
      }
      if (!append_id(token_ids, token_count, &capacity, special_id)) {
        free(*token_ids);
        *token_ids = NULL;
        *token_count = 0;
        return false;
      }
      offset = special_offset + special_length;
      continue;
    }

    if (special_start) {
      size_t ordinary_length = (size_t)(special_start - (text + offset));
      if (ordinary_length != 0) {
        if (!encode_segment(tokenizer, (const unsigned char *)text + offset,
                            ordinary_length, token_ids, token_count,
                            &capacity)) {
          free(*token_ids);
          *token_ids = NULL;
          *token_count = 0;
          return false;
        }
        offset += ordinary_length;
        continue;
      }
    }

    size_t end;
    if (text[offset] == '\'') {
      if (is_contraction((const unsigned char *)text, length, offset, &end)) {
        if (!encode_segment(tokenizer, (const unsigned char *)text + offset,
                            end - offset, token_ids, token_count, &capacity)) {
          free(*token_ids);
          *token_ids = NULL;
          *token_count = 0;
          return false;
        }
        offset = end;
        continue;
      }
    }
    end = segment_end((const unsigned char *)text, length, offset);
    if (end <= offset ||
        !encode_segment(tokenizer, (const unsigned char *)text + offset,
                        end - offset, token_ids, token_count, &capacity)) {
      free(*token_ids);
      *token_ids = NULL;
      *token_count = 0;
      return false;
    }
    offset = end;
  }
  return true;
}

bool mlmc_tokenizer_write_token(const mlmc_tokenizer_t *tokenizer,
                                uint32_t token_id, FILE *stream) {
  if (!tokenizer || !stream || token_id >= tokenizer->token_count) {
    return false;
  }
  const char *piece = tokenizer->tokens->v.v_arr.items[token_id].v.v_str;
  size_t length = strlen(piece);
  size_t offset = 0;
  while (offset < length) {
    uint32_t codepoint;
    size_t width = decode_codepoint((const unsigned char *)piece + offset,
                                    length - offset, &codepoint);
    if (width == 0) {
      return false;
    }
    int byte = byte_for_codepoint(codepoint);
    if (byte >= 0) {
      if (fputc(byte, stream) == EOF) {
        return false;
      }
    } else if (fwrite(piece + offset, 1, width, stream) != width) {
      return false;
    }
    offset += width;
  }
  return true;
}

static int compare_candidates(const void *left, const void *right) {
  const candidate_t *a = (const candidate_t *)left;
  const candidate_t *b = (const candidate_t *)right;
  if (a->logit > b->logit) {
    return -1;
  }
  if (a->logit < b->logit) {
    return 1;
  }
  return a->token_id < b->token_id ? -1 : (a->token_id > b->token_id ? 1 : 0);
}

static uint64_t next_random(uint64_t *state) {
  uint64_t value = *state;
  if (value == 0) {
    value = UINT64_C(0x9e3779b97f4a7c15);
  }
  value ^= value >> 12;
  value ^= value << 25;
  value ^= value >> 27;
  *state = value;
  return value * UINT64_C(0x2545f4914f6cdd1d);
}

bool mlmc_sample(const float *logits, uint32_t count, float temperature,
                 uint32_t top_k, uint64_t *rng_state, uint32_t *token_id) {
  if (!logits || count == 0 || !isfinite(temperature) || temperature < 0.0f ||
      !token_id || (temperature > 0.0f && !rng_state)) {
    return false;
  }
  if (temperature == 0.0f) {
    uint32_t best = 0;
    for (uint32_t i = 0; i < count; i++) {
      if (!isfinite(logits[i])) {
        return false;
      }
      if (logits[i] > logits[best]) {
        best = i;
      }
    }
    *token_id = best;
    return true;
  }

  size_t candidate_bytes = (size_t)count * sizeof(candidate_t);
  if (candidate_bytes / sizeof(candidate_t) != (size_t)count) {
    return false;
  }
  candidate_t *candidates = (candidate_t *)malloc(candidate_bytes);
  if (!candidates) {
    return false;
  }
  for (uint32_t i = 0; i < count; i++) {
    if (!isfinite(logits[i])) {
      free(candidates);
      return false;
    }
    candidates[i] = (candidate_t){i, logits[i]};
  }
  qsort(candidates, count, sizeof(candidate_t), compare_candidates);
  uint32_t retained = top_k == 0 || top_k > count ? count : top_k;
  float max_logit = candidates[0].logit;
  double total = 0.0;
  for (uint32_t i = 0; i < retained; i++) {
    total += exp((double)(candidates[i].logit - max_logit) / temperature);
  }
  double target =
      ((double)(next_random(rng_state) >> 11) / (double)(UINT64_C(1) << 53)) *
      total;
  for (uint32_t i = 0; i < retained; i++) {
    target -= exp((double)(candidates[i].logit - max_logit) / temperature);
    if (target < 0.0) {
      *token_id = candidates[i].token_id;
      free(candidates);
      return true;
    }
  }
  *token_id = candidates[retained - 1].token_id;
  free(candidates);
  return true;
}
