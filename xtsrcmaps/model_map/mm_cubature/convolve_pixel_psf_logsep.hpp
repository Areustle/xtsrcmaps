#pragma once

#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/misc/simdwidth.hpp"

#include <array>
#include <map>
#include <utility>

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

template <typename T, size_t N>
void
convolve_pixel_psf_logsep(T* __restrict__ model_map,
                          size_t const Ne,
                          T const* const __restrict__ points_weights,
                          T const* const __restrict__ psf_lut,
                          std::array<T, 3> const& source) {

    // ========================================
    // Map index into PSF lookup table with weights (interpolation & cubature)
    // This is efficient for small map sizes.
    //
    auto sum_wts = std::map<uint16_t, std::pair<T, T>>();

    for (size_t p = 0; p < 4 * N; p += 4) {
        // Spherical Distance
        T        x = points_weights[p] - source[0];
        T        y = points_weights[p + 1] - source[1];
        T        z = points_weights[p + 2] - source[2];
        T        w = points_weights[p + 3];
        T        d = std::sqrt(x * x + y * y + z * z);
        T        s = 2.0 * rad2deg * std::asin(0.5 * d);
        bool     c = s < 1e-4;
        T        j = c ? 1.
                       : 1.
                      + recipstep
                            * (9.2103403719761827360719658187374568304044059545150919 // ln(1e4)
                               + std::log(s));
        uint16_t i = 1 + static_cast<uint16_t>(j);
        i          = i > 0x190 ? 0x190 : i; // 0x190 == 400
        T t        = _redm1 * (s * 1e4 * std::exp(sep_step * (2. - i)) - 1.);
        sum_wts[i].first += (1.0 - t) * w;
        sum_wts[i].second += t * w;
    }

    for (const auto& [idx, wgts] : sum_wts) {

        T const va                     = wgts.first;
        T const vb                     = wgts.second;
        T const* const __restrict__ YL = &(psf_lut[Ne * (idx - 1)]);
        T const* const __restrict__ YR = &(psf_lut[Ne * idx]);

        size_t e                       = 0;
        // Process `simd_width` elements at a time
        for (; e <= Ne - simd_width; e += simd_width) {
            for (size_t lane = 0; lane < simd_width; ++lane) {
                auto const i = e + lane;
                model_map[i] += (va * YL[i]) + (vb * YR[i]);
            }
        }
        for (; e < Ne; ++e) {
            model_map[e] += (wgts.first * YL[e]) + (wgts.second * YR[e]);
        }
    }
}


} // namespace Fermi
