#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

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

auto
solid_angle(Fermi::Tensor<float, 3> const& points,
            Fermi::SkyGeom<float> const&   skygeom) -> Fermi::Tensor<float, 2> {
    Fermi::Tensor<float, 2> phi(points.extent(0), points.extent(1));
    for (size_t h = 0; h < points.extent(0); ++h) {
        for (size_t w = 0; w < points.extent(1); ++w) {
            // Adapted from FermiTools CountsMap.cxx:612 and FitsImage.cxx:108
            auto const A
                = skygeom.pix2dir({ points[h, w, 0], points[h, w, 1] });
            auto const B
                = skygeom.pix2dir({ points[h, w, 0], points[h, w, 1] + 1.f });
            auto const C = skygeom.pix2dir(
                { points[h, w, 0] + 1.f, points[h, w, 1] + 1.f });
            auto const D
                = skygeom.pix2dir({ points[h, w, 0] + 1.f, points[h, w, 1] });

            float dOmega1
                = dir_diff(A, B) * dir_diff(A, D)
                  * norm(cross(normalize(std::array<float, 3> {
                                   A[0] - B[0], A[1] - B[1], A[2] - B[2] }),
                               normalize(std::array<float, 3> {
                                   A[0] - D[0], A[1] - D[1], A[2] - D[2] })));

            float dOmega2
                = dir_diff(C, B) * dir_diff(C, D)
                  * norm(cross(normalize(std::array<float, 3> {
                                   C[0] - B[0], C[1] - B[1], C[2] - B[2] }),
                               normalize(std::array<float, 3> {
                                   C[0] - D[0], C[1] - D[1], C[2] - D[2] })));

            phi[h, w] = 0.5 * (dOmega1 + dOmega2);
        }
    }
    return phi;
}

void
Fermi::ModelMap::scale_map_by_solid_angle(Tensor<float, 4>&     model_map,
                                          SkyGeom<float> const& skygeom) {
    size_t const Ns = model_map.extent(0);
    size_t const Nh = model_map.extent(1);
    size_t const Nw = model_map.extent(2);
    size_t const Ne = model_map.extent(3);

    /* Tensor3d const init_points = get_init_centers(Nh, Nw); */
    Tensor<float, 3> init_points(Nh, Nw, 2);
    for (size_t h = 0; h < Nh; ++h) {
        for (size_t w = 0; w < Nw; ++w) {
            init_points[h, w, 0] = 1. + h;
            init_points[h, w, 0] = 1. + w;
        }
    }
    // Compute solid angle for the pixel center points and scale PSF by it.
    auto SA = solid_angle(init_points, skygeom);

#pragma omp parallel for schedule(static, 16)
    for (size_t s = 0; s < Ns; ++s) {
        for (size_t h = 0; h < Nh; ++h) {
            for (size_t w = 0; w < Nw; ++w) {
                for (size_t e = 0; e < Ne; ++e) {
                    model_map[s, h, w, e] *= SA[h, w];
                }
            }
        }
    }
}
