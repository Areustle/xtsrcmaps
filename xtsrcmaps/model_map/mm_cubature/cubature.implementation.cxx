#include "xtsrcmaps/model_map/mm_cubature/convolve_pixel_psf.hpp"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <omp.h>

#include <array>
#include <cassert>

namespace Fermi {

/********************************************************************************
 * Convolve the PSF via lookup table, with each of the pixels for every source.
 * ----
 * Split the spherical pixels into 2 spherical triangles and apply the ARPIST[1]
 * algorithm with a degree 4 triangular cubature scheme[2].
 *
 * [1] Yipeng Li 1, Xiangmin Jiao, ARPIST: Provably accurate and stable
 * numerical integration over spherical triangles. J. Comp. & Ap. Math. (2023)
 * [2]  G.R. Cowper, Gaussian quadrature formulas for triangles, Internat.
 * J. Numer. Methods Engrg. 7 (1973).
 * https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE/rules_triangle.html
 ********************************************************************************/

auto
convolve_psf_with_map_deg4x2_arpist(
    size_t const             Nh,
    size_t const             Nw,
    Tensor<double, 2> const& src_sphcrds,
    Tensor<float, 3> const&  psf_lut, // psf lookup table [Source, Seps, Energy]
    SkyGeom<float> const&    skygeom) -> Tensor<float, 4> {

    size_t const Ns = src_sphcrds.extent(0);
    assert(src_sphcrds.extent(1) == 2);
    size_t const Ne = psf_lut.extent(2);

    Tensor<float, 4> model_map({ Ns, Nh, Nw, Ne });
    model_map.clear();

    /********************************************************************************
     * Populate the Cubature points in all of the pixels.
     * Per pixel: 2 triangles * 6 points * 4 values (x,y,z,w) = 48 floats
     * https://doi.org/10.1016/j.cam.2022.114822
     ********************************************************************************/

    Tensor<float, 3> points_weights({ Nh, Nw, 48 });

    constexpr size_t      CARD                 = 6;
    constexpr size_t      PN                   = CARD * 2;
    std::array<float, PN> cubature_deg4_points = {
        //
        4.45948490915965001235576892213430e-01,
        4.45948490915964779190971967182122e-01,
        //
        4.45948490915965001235576892213430e-01,
        1.08103018168070233451238948418904e-01,
        //
        1.08103018168070219573451140604448e-01,
        4.45948490915965001235576892213430e-01,
        //
        9.15762135097707846709269574603240e-02,
        9.15762135097707846709269574603240e-02,
        //
        9.15762135097708124265025730892376e-02,
        8.16847572980458402902570469450438e-01,
        //
        8.16847572980458624947175394481746e-01,
        9.15762135097706875264123027591268e-02,
    };

    std::array<float, CARD> cubature_deg4_weights
        = { 1.11690794839005735905601568447310e-01,
            1.11690794839005735905601568447310e-01,
            1.11690794839005749783389376261766e-01,
            5.49758718276609978370395026558981e-02,
            5.49758718276609423258882713980711e-02,
            5.49758718276608937536309440474724e-02 };


    for (size_t h = 0; h < Nw; ++h) {
        for (size_t w = 0; w < Nw; ++w) {

            float vh = 1. + h;
            float vw = 1. + w;

            // UL -------- UR
            //   |      / |       x1 -- x3
            //   |     /  |         | /
            //   |    /   |         x2
            //   |   /    |
            //   |  /     |
            //   | /      |
            // LL -------- LR

            std::array<float, 2> const UL { vh - 0.5f, vw - 0.5f };
            std::array<float, 2> const UR { vh - 0.5f, vw + 0.5f };
            std::array<float, 2> const LL { vh + 0.5f, vw - 0.5f };
            std::array<float, 2> const LR { vh + 0.5f, vw + 0.5f };

            auto const x1    = skygeom.pix2dir(UL);
            auto const x2    = skygeom.pix2dir(LL);
            auto const x3    = skygeom.pix2dir(UR);
            auto const x4    = skygeom.pix2dir(LR);

            // determinant from cross product
            // |a b c|   |x1[0] x1[1] x2[2]|
            // |d e f| = |x2[0] x2[1] x2[2]|
            // |g h i|   |x3[0] x3[1] x3[2]|
            float const det1 = x1[0] * (x2[1] * x3[2] - x2[2] * x3[1])
                               - x1[1] * (x2[0] * x3[2] - x2[2] * x3[0])
                               + x1[2] * (x2[0] * x3[1] - x2[1] * x3[0]);

            // determinant from cross product
            // |a b c|   |x4[0] x4[1] x4[2]|
            // |d e f| = |x3[0] x3[1] x3[2]|
            // |g h i|   |x2[0] x2[1] x2[2]|
            float const det2 = x4[0] * (x3[1] * x2[2] - x3[2] * x2[1])
                               - x4[1] * (x3[0] * x2[2] - x3[2] * x2[0])
                               + x4[2] * (x3[0] * x2[1] - x3[1] * x2[0]);

            // Populate ARPIST cubature weights and points for both triangular
            // halves of the divided pixe.
            for (size_t p = 0; p < CARD; ++p) {
                float const ksi = cubature_deg4_points[p];
                float const eta = cubature_deg4_points[p + 1];
                float const wgt = cubature_deg4_weights[p];

                float xx
                    = x1[0] + ksi * (x2[0] - x1[0]) + eta * (x3[0] - x1[0]);
                float xy
                    = x1[1] + ksi * (x2[1] - x1[1]) + eta * (x3[1] - x1[1]);
                float xz
                    = x1[2] + ksi * (x2[2] - x1[2]) + eta * (x3[2] - x1[2]);
                float invnorm = 1.0 / std::sqrt(xx * xx + xy * xy + xz * xz);
                int   q       = 4 * p;
                points_weights[h, w, q]     = xx * invnorm;
                points_weights[h, w, q + 1] = xy * invnorm;
                points_weights[h, w, q + 2] = xz * invnorm;
                points_weights[h, w, q + 3] = det1 * wgt * std::pow(invnorm, 3);

                xx      = x4[0] + eta * (x2[0] - x4[0]) + ksi * (x3[0] - x4[0]);
                xy      = x4[1] + eta * (x2[1] - x4[1]) + ksi * (x3[1] - x4[1]);
                xz      = x4[2] + eta * (x2[2] - x4[2]) + ksi * (x3[2] - x4[2]);
                invnorm = 1.0 / std::sqrt(xx * xx + xy * xy + xz * xz);
                q += (4 * CARD);
                points_weights[h, w, q]     = xx * invnorm;
                points_weights[h, w, q + 1] = xy * invnorm;
                points_weights[h, w, q + 2] = xy * invnorm;
                points_weights[h, w, q + 3] = det2 * wgt * std::pow(invnorm, 3);
            }
        }
    }

#pragma omp parallel for schedule(static, 16)
    for (size_t s = 0; s < Ns; ++s) {
        std::array<float, 3> source = skygeom.sph2dir(
            { src_sphcrds[s, 0], src_sphcrds[s, 1] }); // CLHEP Style 3
        for (size_t h = 0; h < Nw; ++h) {
            for (size_t w = 0; w < Nw; ++w) {
                convolve_pixel_psf<float>(&(model_map[s, h, w, 0]),
                                          Ne,
                                          &(points_weights[h, w, 0]),
                                          &(psf_lut[s, 0, 0]),
                                          source);
            }
        }
    }
    return model_map;
}
} // namespace Fermi
