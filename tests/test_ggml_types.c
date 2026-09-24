#include <stdint.h>
#include <stdio.h>

#include "ggml_types.h"

static int failures = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond);       \
      failures++;                                                              \
    }                                                                          \
  } while (0)

int main(void) {
  ggml_type_traits_t traits;

  CHECK(ggml_get_type_traits(GGML_TYPE_Q4_K, &traits));
  CHECK(traits.block_size == 256 && traits.type_size == 144);

  CHECK(ggml_get_type_traits(GGML_TYPE_Q2_K, &traits));
  CHECK(traits.block_size == 256 && traits.type_size == 84);

  CHECK(!ggml_get_type_traits((enum ggml_type)4, NULL));
  CHECK(!ggml_get_type_traits((enum ggml_type)33, NULL));
  CHECK(!ggml_get_type_traits((enum ggml_type)GGML_TYPE_COUNT, NULL));

  CHECK(ggml_type_name(GGML_TYPE_F32) != NULL);
  CHECK(ggml_type_name((enum ggml_type)4) == NULL);

  uint64_t dims1[1] = {512};
  size_t size = 0;
  CHECK(ggml_tensor_storage_size(GGML_TYPE_F32, 1, dims1, &size));
  CHECK(size == 2048);

  uint64_t bad_dims[1] = {31};
  CHECK(!ggml_tensor_storage_size(GGML_TYPE_Q4_0, 1, bad_dims, &size));

  uint64_t dims2[2] = {64, 3};
  CHECK(ggml_tensor_storage_size(GGML_TYPE_Q8_0, 2, dims2, &size));
  CHECK(size == 2 * 3 * 34);

  CHECK(ggml_tensor_storage_size(GGML_TYPE_F32, 0, NULL, &size));
  CHECK(size == 4);

  uint64_t dims5[4] = {1, 1, 1, 1};
  CHECK(!ggml_tensor_storage_size(GGML_TYPE_F32, 5, dims5, &size));

  if (failures == 0) {
    printf("ALL GGML_TYPES TESTS PASSED\n");
  } else {
    printf("%d FAILURES\n", failures);
  }
  return failures != 0;
}