#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/math/solid_angle.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

void
Fermi::ModelMap::scale_map_by_solid_angle(Tensor<double, 4>&     model_map,
                                          SkyGeom<double> const& skygeom) {
    size_t const Ns = model_map.extent(0);
    size_t const Nh = model_map.extent(1);
    size_t const Nw = model_map.extent(2);
    size_t const Ne = model_map.extent(3);

    // Compute solid angle for the pixel center points and scale PSF by it.
    auto Omega      = Fermi::Math::solid_angle(Nh, Nw, skygeom);

#pragma omp parallel for schedule(static, 16)
    for (size_t s = 0; s < Ns; ++s) {
        for (size_t h = 0; h < Nh; ++h) {
            for (size_t w = 0; w < Nw; ++w) {
                for (size_t e = 0; e < Ne; ++e) {
                    model_map[s, h, w, e] *= Omega[h, w];
                }
            }
        }
    }
}


// auto
// solid_angle(Fermi::Tensor<double, 3> const& points,
//             Fermi::SkyGeom<double> const& skygeom) -> Fermi::Tensor<double,
//             2> {
//     Fermi::Tensor<double, 2> phi(points.extent(0), points.extent(1));
//     for (size_t h = 0; h < points.extent(0); ++h) {
//         for (size_t w = 0; w < points.extent(1); ++w) {
//             // Adapted from FermiTools CountsMap.cxx:612 and
//             FitsImage.cxx:108 auto const A
//                 = skygeom.pix2dir({ points[h, w, 0], points[h, w, 1] });
//             auto const B
//                 = skygeom.pix2dir({ points[h, w, 0], points[h, w, 1] + 1.f
//                 });
//             auto const C = skygeom.pix2dir(
//                 { points[h, w, 0] + 1.f, points[h, w, 1] + 1.f });
//             auto const D
//                 = skygeom.pix2dir({ points[h, w, 0] + 1.f, points[h, w, 1]
//                 });
//
//             double dOmega1
//                 = dir_diff(A, B) * dir_diff(A, D)
//                   * norm(cross(normalize(std::array<double, 3> {
//                                    A[0] - B[0], A[1] - B[1], A[2] - B[2] }),
//                                normalize(std::array<double, 3> {
//                                    A[0] - D[0], A[1] - D[1], A[2] - D[2]
//                                    })));
//
//             double dOmega2
//                 = dir_diff(C, B) * dir_diff(C, D)
//                   * norm(cross(normalize(std::array<double, 3> {
//                                    C[0] - B[0], C[1] - B[1], C[2] - B[2] }),
//                                normalize(std::array<double, 3> {
//                                    C[0] - D[0], C[1] - D[1], C[2] - D[2]
//                                    })));
//
//             phi[h, w] = 0.5 * (dOmega1 + dOmega2);
//         }
//     }
//     return phi;
// }
