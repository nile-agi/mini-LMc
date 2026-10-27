#include "dequant.h"

#include <string.h>

static float bits_to_float(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

float mlmc_fp16_to_fp32(uint16_t half_bits) {
  uint32_t sign = (uint32_t)(half_bits & 0x8000u) << 16;
  int32_t exponent = (half_bits >> 10) & 0x1F;
  uint32_t mantissa = half_bits & 0x3FFu;

  if (exponent == 0) {
    if (mantissa == 0) {
      return bits_to_float(sign);
    }
    exponent = 1;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x3FFu;
  } else if (exponent == 31) {
    return bits_to_float(sign | 0x7F800000u | (mantissa << 13));
  }

  uint32_t float_exponent = (uint32_t)(exponent + 112);
  return bits_to_float(sign | (float_exponent << 23) | (mantissa << 13));
}

float mlmc_bf16_to_fp32(uint16_t bf16_bits) {
  return bits_to_float((uint32_t)bf16_bits << 16);
}
