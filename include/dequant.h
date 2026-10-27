#ifndef MINILMC_DEQUANT_H
#define MINILMC_DEQUANT_H

#include <stdint.h>

// Converts one IEEE half (binary16) bit pattern to float32.
float mlmc_fp16_to_fp32(uint16_t half_bits);

// Converts one bfloat16 bit pattern to float32.
float mlmc_bf16_to_fp32(uint16_t bf16_bits);

#endif // MINILMC_DEQUANT_H
