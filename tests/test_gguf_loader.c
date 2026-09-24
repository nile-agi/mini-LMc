#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

static int test_count = 0;
static int passed = 0;
static int failed = 0;

#define TEST(name)                                                             \
  do {                                                                         \
    test_count++;                                                              \
    printf("\nTest %d: %s\n", test_count, name);                               \
  } while (0)

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (cond) {                                                                \
      printf("  ok  %s\n", msg);                                               \
      passed++;                                                                \
    } else {                                                                   \
      printf("  FAIL %s\n", msg);                                              \
      failed++;                                                                \
    }                                                                          \
  } while (0)

static const char *write_tmp(const uint8_t *data, size_t size) {
  static char path[256];
  static int counter = 0;
  snprintf(path, sizeof(path), "gguf_test_%d.gguf", counter++);

  FILE *f = fopen(path, "wb");
  if (!f) {
    return NULL;
  }
  if (size > 0 && fwrite(data, 1, size, f) != size) {
    fclose(f);
    return NULL;
  }
  fclose(f);
  return path;
}

static void put_u32(uint8_t *buf, size_t *pos, uint32_t v) {
  memcpy(buf + *pos, &v, 4);
  *pos += 4;
}
static void put_u64(uint8_t *buf, size_t *pos, uint64_t v) {
  memcpy(buf + *pos, &v, 8);
  *pos += 8;
}
static void put_str(uint8_t *buf, size_t *pos, const char *s) {
  uint64_t len = (uint64_t)strlen(s);
  put_u64(buf, pos, len);
  memcpy(buf + *pos, s, (size_t)len);
  *pos += (size_t)len;
}
static void put_header(uint8_t *buf, size_t *pos, uint32_t version,
                       uint64_t tensor_count, uint64_t kv_count) {
  put_u32(buf, pos, 0x46554747u);
  put_u32(buf, pos, version);
  put_u64(buf, pos, tensor_count);
  put_u64(buf, pos, kv_count);
}

static void test_minimal_valid(void) {
  TEST("Minimal valid GGUF (0 metadata, 0 tensors)");
  uint8_t buf[24];
  size_t pos = 0;
  put_header(buf, &pos, 3, 0, 0);

  const char *path = write_tmp(buf, pos);
  CHECK(path != NULL, "wrote test file");

  gguf_file_t f;
  int rc = gguf_load(path, &f);
  CHECK(rc == 0, "gguf_load succeeded");
  CHECK(f.version == 3, "version == 3");
  CHECK(f.tensor_count == 0, "tensor_count == 0");
  CHECK(f.alignment == 32, "default alignment == 32");
  gguf_free(&f);
  remove(path);
}

static void test_truncated_header(void) {
  TEST("Truncated header (16 bytes)");
  uint8_t buf[16] = {0};
  const char *path = write_tmp(buf, sizeof(buf));
  gguf_file_t f;
  CHECK(gguf_load(path, &f) != 0, "gguf_load correctly failed");
  gguf_free(&f);
  remove(path);
}

static void test_bad_magic(void) {
  TEST("Bad magic number");
  uint8_t buf[24];
  size_t pos = 0;
  put_u32(buf, &pos, 0xDEADBEEFu);
  put_u32(buf, &pos, 3);
  put_u64(buf, &pos, 0);
  put_u64(buf, &pos, 0);
  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  CHECK(gguf_load(path, &f) != 0, "gguf_load correctly rejected bad magic");
  gguf_free(&f);
  remove(path);
}

static void test_bad_version(void) {
  TEST("Unsupported version (v2)");
  uint8_t buf[24];
  size_t pos = 0;
  put_header(buf, &pos, 2, 0, 0);
  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  CHECK(gguf_load(path, &f) != 0, "gguf_load correctly rejected v2");
  gguf_free(&f);
  remove(path);
}

static void test_string_metadata(void) {
  TEST("Single string metadata key");
  uint8_t buf[128];
  size_t pos = 0;
  put_header(buf, &pos, 3, 0, 1);
  put_str(buf, &pos, "general.architecture");
  put_u32(buf, &pos, GGUF_TYPE_STRING);
  put_str(buf, &pos, "llama");

  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  int rc = gguf_load(path, &f);
  CHECK(rc == 0, "gguf_load succeeded");
  const gguf_value_t *v = gguf_find(&f, "general.architecture");
  CHECK(v != NULL, "key found");
  CHECK(v && v->type == GGUF_TYPE_STRING, "value type is STRING");
  CHECK(v && v->v.v_str && strcmp(v->v.v_str, "llama") == 0,
        "value == \"llama\"");
  CHECK(gguf_find(&f, "nonexistent.key") == NULL, "missing key returns NULL");
  gguf_free(&f);
  remove(path);
}

static void test_array_metadata(void) {
  TEST("uint32 array metadata (3 elements)");
  uint8_t buf[128];
  size_t pos = 0;
  put_header(buf, &pos, 3, 0, 1);
  put_str(buf, &pos, "test.array");
  put_u32(buf, &pos, GGUF_TYPE_ARRAY);
  put_u32(buf, &pos, GGUF_TYPE_UINT32);
  put_u64(buf, &pos, 3);
  put_u32(buf, &pos, 10);
  put_u32(buf, &pos, 20);
  put_u32(buf, &pos, 30);

  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  CHECK(gguf_load(path, &f) == 0, "gguf_load succeeded");
  const gguf_value_t *v = gguf_find(&f, "test.array");
  CHECK(v && v->type == GGUF_TYPE_ARRAY, "value type is ARRAY");
  CHECK(v && v->v.v_arr.len == 3, "array length == 3");
  CHECK(v && v->v.v_arr.items[0].v.v_u32 == 10 &&
            v->v.v_arr.items[1].v.v_u32 == 20 &&
            v->v.v_arr.items[2].v.v_u32 == 30,
        "array contents correct");
  gguf_free(&f);
  remove(path);
}

static void test_valid_f32_tensor(void) {
  TEST("Valid F32 tensor: shape [4], data accessible");
  uint8_t buf[256];
  size_t pos = 0;
  put_header(buf, &pos, 3, 1, 0);

  put_str(buf, &pos, "weights");
  put_u32(buf, &pos, 1);
  put_u64(buf, &pos, 4);
  put_u32(buf, &pos, GGML_TYPE_F32);
  put_u64(buf, &pos, 0);

  size_t aligned = ((pos + 31) / 32) * 32;
  for (size_t i = pos; i < aligned; i++) {
    buf[i] = 0;
  }
  pos = aligned;

  float known_values[4] = {1.0f, -2.5f, 3.25f, 0.0f};
  memcpy(buf + pos, known_values, sizeof(known_values));
  pos += sizeof(known_values);

  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  CHECK(gguf_load(path, &f) == 0, "gguf_load succeeded");
  CHECK(f.tensor_count == 1, "tensor_count == 1");

  size_t data_size = 0;
  const void *ptr = gguf_tensor_data(&f, &f.tensors[0], &data_size);
  CHECK(ptr != NULL, "gguf_tensor_data returned non-NULL");
  CHECK(data_size == 16, "data_size == 16 bytes (4 x F32)");
  CHECK(ptr && memcmp(ptr, known_values, sizeof(known_values)) == 0,
        "tensor bytes match what was written");
  gguf_free(&f);
  remove(path);
}

static void test_tensor_past_eof(void) {
  TEST("Tensor offset + size extends past EOF");
  uint8_t buf[256];
  size_t pos = 0;
  put_header(buf, &pos, 3, 1, 0);
  put_str(buf, &pos, "weights");
  put_u32(buf, &pos, 1);
  put_u64(buf, &pos, 100);
  put_u32(buf, &pos, GGML_TYPE_F32);
  put_u64(buf, &pos, 100000);

  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  CHECK(gguf_load(path, &f) != 0,
        "gguf_load correctly rejected out-of-range tensor");
  gguf_free(&f);
  remove(path);
}

static void test_bad_quant_alignment(void) {
  TEST("Q4_0 tensor with dims[0]=31 (not a multiple of block size 32)");
  uint8_t buf[256];
  size_t pos = 0;
  put_header(buf, &pos, 3, 1, 0);
  put_str(buf, &pos, "weights");
  put_u32(buf, &pos, 1);
  put_u64(buf, &pos, 31);
  put_u32(buf, &pos, GGML_TYPE_Q4_0);
  put_u64(buf, &pos, 0);

  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  CHECK(gguf_load(path, &f) != 0,
        "gguf_load correctly rejected non-block-aligned shape");
  gguf_free(&f);
  remove(path);
}

static void test_custom_alignment(void) {
  TEST("general.alignment = 64 is honored");
  uint8_t buf[256];
  size_t pos = 0;
  put_header(buf, &pos, 3, 0, 1);
  put_str(buf, &pos, "general.alignment");
  put_u32(buf, &pos, GGUF_TYPE_UINT32);
  put_u32(buf, &pos, 64);

  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  CHECK(gguf_load(path, &f) == 0, "gguf_load succeeded");
  CHECK(f.alignment == 64, "alignment == 64");
  gguf_free(&f);
  remove(path);
}

static void test_invalid_alignment(void) {
  TEST("general.alignment = 63 (not a power of 2) is rejected");
  uint8_t buf[256];
  size_t pos = 0;
  put_header(buf, &pos, 3, 0, 1);
  put_str(buf, &pos, "general.alignment");
  put_u32(buf, &pos, GGUF_TYPE_UINT32);
  put_u32(buf, &pos, 63);

  const char *path = write_tmp(buf, pos);
  gguf_file_t f;
  CHECK(gguf_load(path, &f) != 0, "gguf_load correctly rejected");
  gguf_free(&f);
  remove(path);
}

int main(void) {
  printf("=== gguf_load() test suite ===\n");

  test_minimal_valid();
  test_truncated_header();
  test_bad_magic();
  test_bad_version();
  test_string_metadata();
  test_array_metadata();
  test_valid_f32_tensor();
  test_tensor_past_eof();
  test_bad_quant_alignment();
  test_custom_alignment();
  test_invalid_alignment();

  printf("\n=== Summary: %d passed, %d failed (of %d checks) ===\n", passed,
         failed, passed + failed);
  return failed != 0;
}