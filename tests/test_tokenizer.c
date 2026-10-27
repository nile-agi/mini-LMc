#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"
#include "tokenizer.h"

static int failures;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #condition);  \
      failures++;                                                              \
    }                                                                          \
  } while (0)

static void put_u32(FILE *file, uint32_t value) {
  fwrite(&value, sizeof(value), 1, file);
}

static void put_u64(FILE *file, uint64_t value) {
  fwrite(&value, sizeof(value), 1, file);
}

static void put_string(FILE *file, const char *text) {
  size_t length = strlen(text);
  put_u64(file, length);
  fwrite(text, 1, length, file);
}

static void put_string_array(FILE *file, const char *key,
                             const char *const *items, size_t count) {
  put_string(file, key);
  put_u32(file, GGUF_TYPE_ARRAY);
  put_u32(file, GGUF_TYPE_STRING);
  put_u64(file, count);
  for (size_t i = 0; i < count; i++) {
    put_string(file, items[i]);
  }
}

static void put_u32_value(FILE *file, const char *key, uint32_t value) {
  put_string(file, key);
  put_u32(file, GGUF_TYPE_UINT32);
  put_u32(file, value);
}

static void put_bool_value(FILE *file, const char *key, uint8_t value) {
  put_string(file, key);
  put_u32(file, GGUF_TYPE_BOOL);
  fwrite(&value, sizeof(value), 1, file);
}

static bool write_fixture(const char *path) {
  static const char *const tokens[] = {"a",
                                       "b",
                                       "c",
                                       "ab",
                                       "abc",
                                       "<bos>",
                                       "<eos>",
                                       "\xc4\xa0",
                                       ("\xc4\xa0"
                                        "a"),
                                       "<|im_start|>",
                                       "<|im_end|>"};
  static const char *const merges[] = {"a b", "ab c", "\xc4\xa0 a"};
  FILE *file = fopen(path, "wb");
  if (!file) {
    return false;
  }
  put_u32(file, 0x46554747u);
  put_u32(file, 3);
  put_u64(file, 0);
  put_u64(file, 5);
  put_string_array(file, "tokenizer.ggml.tokens", tokens,
                   sizeof(tokens) / sizeof(tokens[0]));
  put_string_array(file, "tokenizer.ggml.merges", merges,
                   sizeof(merges) / sizeof(merges[0]));
  put_u32_value(file, "tokenizer.ggml.bos_token_id", 5);
  put_u32_value(file, "tokenizer.ggml.eos_token_id", 6);
  put_bool_value(file, "tokenizer.ggml.add_bos_token", 1);
  bool ok = fclose(file) == 0;
  return ok;
}

static void test_encode_and_decode(void) {
  const char *path = "tokenizer_test.gguf";
  CHECK(write_fixture(path));

  gguf_file_t gguf = {0};
  CHECK(gguf_load(path, &gguf) == 0);
  mlmc_tokenizer_t tokenizer = {0};
  CHECK(mlmc_tokenizer_init(&gguf, &tokenizer));

  uint32_t *ids = NULL;
  size_t count = 0;
  CHECK(mlmc_tokenizer_encode(&tokenizer, "abc", true, &ids, &count));
  CHECK(count == 2);
  CHECK(ids && ids[0] == 5 && ids[1] == 4);
  free(ids);

  ids = NULL;
  count = 0;
  CHECK(
      mlmc_tokenizer_encode(&tokenizer, "a<|im_start|>b", false, &ids, &count));
  CHECK(count == 3 && ids && ids[0] == 0 && ids[1] == 9 && ids[2] == 1);
  free(ids);

  ids = NULL;
  count = 0;
  CHECK(mlmc_tokenizer_encode(&tokenizer, " a", false, &ids, &count));
  CHECK(count == 1 && ids && ids[0] == 8);
  FILE *output = tmpfile();
  CHECK(output != NULL);
  CHECK(output && ids && count == 1 &&
        mlmc_tokenizer_write_token(&tokenizer, ids[0], output));
  if (output) {
    rewind(output);
    char decoded[8] = {0};
    CHECK(fread(decoded, 1, sizeof(decoded), output) == 2);
    CHECK(strcmp(decoded, " a") == 0);
    fclose(output);
  }
  free(ids);

  mlmc_tokenizer_free(&tokenizer);
  gguf_free(&gguf);
  remove(path);
}

static void test_sampling(void) {
  const float logits[] = {1.0f, 3.0f, 2.0f};
  uint32_t token = UINT32_MAX;
  uint64_t rng = 123;
  CHECK(mlmc_sample(logits, 3, 0.0f, 3, &rng, &token));
  CHECK(token == 1);
  CHECK(mlmc_sample(logits, 3, 0.8f, 1, &rng, &token));
  CHECK(token == 1);

  const float invalid[] = {0.0f, NAN};
  CHECK(!mlmc_sample(invalid, 2, 0.0f, 0, &rng, &token));
}

int main(void) {
  test_encode_and_decode();
  test_sampling();
  if (failures != 0) {
    fprintf(stderr, "%d tokenizer test(s) failed\n", failures);
    return 1;
  }
  puts("ALL TOKENIZER TESTS PASSED");
  return 0;
}
