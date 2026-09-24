#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s show <model.gguf> [max_tensors]\n"
          "  %s info <model.gguf> <tensor_name>\n",
          prog, prog);
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

  const gguf_tensor_info_t *t = NULL;
  for (uint64_t i = 0; i < f.tensor_count; i++) {
    if (strcmp(f.tensors[i].name, argv[3]) == 0) {
      t = &f.tensors[i];
      break;
    }
  }
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

  usage(argv[0]);
  return 1;
}