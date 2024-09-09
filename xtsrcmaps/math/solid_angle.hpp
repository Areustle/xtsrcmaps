#pragma once

#include "xtsrcmaps/tensor/tensor.hpp"
#include <array>

namespace Fermi::Math {

// Cross product of two 3D vectors
template <typename T>
inline auto
cross(std::array<T, 3> const& L,
      std::array<T, 3> const& R) -> std::array<T, 3> {
    return { L[1] * R[2] - L[2] * R[1],
             L[2] * R[0] - L[0] * R[2],
             L[0] * R[1] - L[1] * R[0] };
}

// Norm of a 3D vector
template <typename T>
inline auto
norm(std::array<T, 3> const& V) -> T {
    return std::sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
}

// Normalize a 3D vector
template <typename T>
inline auto
normalize(std::array<T, 3> const& V) -> std::array<T, 3> {
    T length = norm(V);
    return { V[0] / length, V[1] / length, V[2] / length };
}

// Directional difference between two 3D vectors
template <typename T>
inline auto
dir_diff(std::array<T, 3> const& L, std::array<T, 3> const& R) -> T {
    std::array<T, 3> tmp        = { L[0] - R[0], L[1] - R[1], L[2] - R[2] };
    T                norm_value = norm(tmp);
    return 2.0 * std::asin(0.5 * norm_value);
}

template <typename T = double, typename F>
auto
solid_angle(size_t const Nh,
            size_t const Nw,
            F&&          skygeom) -> Fermi::Tensor<T, 2> {

    Fermi::Tensor<T, 2> phi(Nh, Nw);

    for (size_t h = 0; h < Nh; ++h) {
        T ph = 1. + h;
        for (size_t w = 0; w < Nw; ++w) {
            T pw         = 1. + w;

            // Adapted from FermiTools CountsMap.cxx:612 and FitsImage.cxx:108
            auto const A = skygeom.pix2dir({ ph, pw });
            auto const B = skygeom.pix2dir({ ph, pw + 1.f });
            auto const C = skygeom.pix2dir({ ph + 1.f, pw + 1.f });
            auto const D = skygeom.pix2dir({ ph + 1.f, pw });

            T dOmega1
                = dir_diff(A, B) * dir_diff(A, D)
                  * norm(cross(normalize(std::array<T, 3> {
                                   A[0] - B[0], A[1] - B[1], A[2] - B[2] }),
                               normalize(std::array<T, 3> {
                                   A[0] - D[0], A[1] - D[1], A[2] - D[2] })));

            T dOmega2
                = dir_diff(C, B) * dir_diff(C, D)
                  * norm(cross(normalize(std::array<T, 3> {
                                   C[0] - B[0], C[1] - B[1], C[2] - B[2] }),
                               normalize(std::array<T, 3> {
                                   C[0] - D[0], C[1] - D[1], C[2] - D[2] })));

            phi[h, w] = 0.5 * (dOmega1 + dOmega2);
        }
    }
    return phi;
}
} // namespace Fermi::Math
