#pragma once

#include "xtsrcmaps/misc/misc.hxx"

#include <array>
#include <map>
#include <utility>

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

namespace Fermi {
//
/** ===================================================================================
 * Generic, SIMD aware PSF Pixel convolution with collapsed-point Table lookups.
 *
 * This function performs a PSF convolution on a pixel grid using cubature
 * points. It maps each cubature point to a lookup table (LUT) index based on
 * the Spherical distance to a point-source. It linearly interpolates the PSF
 * LUT to compute the contribution from cubature points in that logarithmically
 * spaced columns of the table. Finally it accumulates a weighted sum for the
 * pixel values per energy bin in the model map.
 * ====================================================================================
 */

template <typename T>
void
convolve_pixel_psf(T* __restrict__ model_map,
                   size_t const Ne,
                   T const* const __restrict__ points_weights,
                   T const* const __restrict__ psf_lut,
                   std::array<T, 3> const& source) {

    // ========================================
    // map
    // This is efficient for small map sizes.
    auto sum_wts = std::map<uint16_t, std::pair<T, T>>();

    uint8_t p    = 0;
    for (; p < 48; p += 4) {
        // Spherical Distance
        T        x = points_weights[p] - source[0];
        T        y = points_weights[p + 1] - source[1];
        T        z = points_weights[p + 2] - source[2];
        T        w = points_weights[p + 3];
        T        d = std::sqrt(x * x + y * y + z * z);
        T        s = 1e4f * 2.0f * rad2deg * std::asin(0.5f * d);
        bool     c = s < 1.f;
        T        j = c ? s : 1.f + recipstep * std::log(s);
        uint16_t i = static_cast<uint16_t>(j);
        i          = i > 0x190 ? 0x190 : i; // 0x190 == 400
        T t        = c ? (j - i) * xmedm1 * std::exp(i * _ss) : j;
        sum_wts[i].first += (1.0f - t) * w;
        sum_wts[i].second += t * w;
    }

    for (const auto& [idx, wgts] : sum_wts) {

        T const va                     = wgts.first;
        T const vb                     = wgts.second;
        T const* const __restrict__ Y0 = &(psf_lut[Ne * idx]);
        T const* const __restrict__ Y1 = &(psf_lut[Ne * (idx + 1UZ)]);

        size_t e                       = 0UZ;
        // Process `simd_width` elements at a time
        for (; e <= Ne - simd_width; e += simd_width) {
            for (size_t lane = 0; lane < simd_width; ++lane) {
                auto const i = e + lane;
                model_map[i] += (va * Y0[i]) + (vb * Y1[i]);
            }
        }
        for (; e < Ne; ++e) {
            model_map[e] += (wgts.first * Y0[e]) + (wgts.second * Y1[e]);
        }
    }
}


} // namespace Fermi
