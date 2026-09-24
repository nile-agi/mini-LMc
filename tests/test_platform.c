#include <stdio.h>
#include <string.h>

#include "platform.h"

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond);       \
      failures++;                                                              \
    }                                                                          \
  } while (0)

int main(void) {
  FILE *f = fopen("platform_test.tmp", "wb");
  CHECK(f != NULL);
  const char *payload = "hello, mini-LMc";
  if (f) {
    fwrite(payload, 1, strlen(payload), f);
    fclose(f);
  }

  mlmc_mapped_file_t mapped;
  CHECK(mlmc_map_file("platform_test.tmp", &mapped) == 0);
  CHECK(mapped.size == strlen(payload));
  CHECK(mapped.data != NULL);
  if (mapped.data) {
    CHECK(memcmp(mapped.data, payload, strlen(payload)) == 0);
  }
  mlmc_unmap_file(&mapped);
  CHECK(mapped.data == NULL);

  mlmc_mapped_file_t missing;
  CHECK(mlmc_map_file("this_file_does_not_exist.tmp", &missing) == -1);
  CHECK(missing.data == NULL);

  remove("platform_test.tmp");

  if (failures == 0) {
    printf("ALL PLATFORM TESTS PASSED\n");
  } else {
    printf("%d FAILURES\n", failures);
  }
  return failures != 0;
}