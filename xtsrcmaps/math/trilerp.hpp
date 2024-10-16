#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"

namespace Fermi::math {

template <typename T>
void
trilerpEHW(Fermi::Tensor<T, 3> const& src, Fermi::Tensor<T, 3>& dest) {

    // Dimensions of the input tensor
    size_t srcE = src.extent(0);
    size_t srcH = src.extent(1);
    size_t srcW = src.extent(2);

    // Dimensions of the output tensor
    size_t dstE = dest.extent(0);
    size_t dstH = dest.extent(1);
    size_t dstW = dest.extent(2);

    // Scale factors for dimensional offset
    T scale_e   = static_cast<T>(srcE - 1) / (dstE - 1);
    T scale_h   = static_cast<T>(srcH - 1) / (dstH - 1);
    T scale_w   = static_cast<T>(srcW - 1) / (dstW - 1);

    // Iterate over each voxel in the target tensor
#pragma omp parallel for schedule(static, 16)
    for (size_t k = 0; k < dstE; ++k) {
        T      e  = k * scale_e;
        size_t e0 = static_cast<size_t>(e);
        size_t e1 = std::min(e0 + 1, srcE - 1);
        T      fe = e - e0;
        T      ce = 1. - fe;
        for (size_t i = 0; i < dstH; ++i) {
            T      h  = i * scale_h;
            size_t h0 = static_cast<size_t>(h);
            size_t h1 = std::min(h0 + 1, srcH - 1);
            T      fh = h - h0;
            T      ch = 1. - fh;
            for (size_t j = 0; j < dstW; ++j) {
                T      w      = j * scale_w;
                size_t w0     = static_cast<size_t>(w);
                size_t w1     = std::min(w0 + 1, srcW - 1);
                T      fw     = w - w0;
                T      cw     = 1. - fw;

                T C00         = ch * src[e0, h0, w0] + fh * src[e0, h1, w0];
                T C01         = ch * src[e1, h0, w0] + fh * src[e1, h1, w0];
                T C10         = ch * src[e0, h0, w1] + fh * src[e0, h1, w1];
                T C11         = ch * src[e1, h0, w1] + fh * src[e1, h1, w1];

                T C0          = cw * C00 + fw * C10;
                T C1          = cw * C01 + fw * C11;

                dest[k, i, j] = ce * C0 + fe * C1;
            }
        }
    }
}
} // namespace Fermi::math
