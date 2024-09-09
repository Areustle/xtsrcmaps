#pragma once

#include <cuchar>

// Set SIMD width based on the architecture
#if defined(__AVX512F__)
constexpr size_t simd_width = 8; // AVX-512 has 8 floats per vector
#elif defined(__AVX__) || defined(__AVX2__)
constexpr size_t simd_width
    = 8; // AVX and AVX2 also support 8 floats (but 256-bit)
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) \
    || defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__)
constexpr size_t simd_width = 4; // SSE has 4 floats per vector
#elif defined(__ARM_NEON)
constexpr size_t simd_width = 4; // NEON has 4 floats per vector
#else
constexpr size_t simd_width = 1; // Fallback to scalar processing if no SIMD
#endif
