#include <stdint.h>
#include <stdio.h>
#include <string.h>

static FILE *g_f;
static void wu32(uint32_t v) { fwrite(&v, 4, 1, g_f); }
static void wu64(uint64_t v) { fwrite(&v, 8, 1, g_f); }
static void wstr(const char *s) {
  uint64_t len = (uint64_t)strlen(s);
  wu64(len);
  fwrite(s, 1, (size_t)len, g_f);
}

int main(void) {
  g_f = fopen("models/synthetic.gguf", "wb");

  wu32(0x46554747u); /* "GGUF" */
  wu32(3);           /* version */
  wu64(1);           /* tensor_count */
  wu64(2);           /* metadata_kv_count */

  wstr("general.architecture");
  wu32(8);
  wstr("llama");
  wstr("general.name");
  wu32(8);
  wstr("mini-LMc-test");

  wstr("token_embd.weight");
  wu32(1);
  wu64(8);
  wu32(0);
  wu64(0);

  long pos = ftell(g_f);
  long aligned = ((pos + 31) / 32) * 32;
  for (long i = pos; i < aligned; i++)
    fputc(0, g_f);

  for (int i = 0; i < 8; i++) {
    float v = (float)i * 1.5f;
    fwrite(&v, 4, 1, g_f);
  }

  fclose(g_f);
  printf("wrote models/synthetic.gguf\n");
  return 0;
}