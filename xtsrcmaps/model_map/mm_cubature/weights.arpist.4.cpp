#include "xtsrcmaps/model_map/mm_cubature/cubature.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

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
 * Populate the Cubature points in all of the pixels.
 * Per pixel: 2 triangles * 6 points * 4 values (x,y,z,w) = 48 floats
 * https://doi.org/10.1016/j.cam.2022.114822
 ********************************************************************************/
auto
Fermi::deg4x2_arpist_ptswts(size_t const          Nh,
                            size_t const          Nw,
                            SkyGeom<double> const& skygeom) -> Tensor<double, 3> {

    Tensor<double, 3> points_weights({ Nh, Nw, 48 });

    constexpr size_t      CARD                     = 6;
    constexpr size_t      PN                       = CARD * 2;
    std::array<double, PN> tri_cubature_deg4_points = {
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

    std::array<double, CARD> tri_cubature_deg4_weights
        = { 1.11690794839005735905601568447310e-01,
            1.11690794839005735905601568447310e-01,
            1.11690794839005749783389376261766e-01,
            5.49758718276609978370395026558981e-02,
            5.49758718276609423258882713980711e-02,
            5.49758718276608937536309440474724e-02 };


    for (size_t h = 0; h < Nw; ++h) {
        for (size_t w = 0; w < Nw; ++w) {

            double ph = 1. + h;
            double pw = 1. + w;

            // UL -------- UR
            //   |      / |       x1 -- x3
            //   |     /  |         | /
            //   |    /   |         x2
            //   |   /    |
            //   |  /     |
            //   | /      |
            // LL -------- LR

            std::array<double, 2> const LL { ph - 0.5, pw - 0.5 };
            std::array<double, 2> const LR { ph - 0.5, pw + 0.5 };
            std::array<double, 2> const UL { ph + 0.5, pw - 0.5 };
            std::array<double, 2> const UR { ph + 0.5, pw + 0.5 };

            auto const x1 = skygeom.pix2dir(UL);
            auto const x2 = skygeom.pix2dir(LL);
            auto const x3 = skygeom.pix2dir(UR);
            auto const x4 = skygeom.pix2dir(LR);

            // determinant from cross product
            // |a b c|   |x1[0] x1[1] x2[2]|
            // |d e f| = |x2[0] x2[1] x2[2]|
            // |g h i|   |x3[0] x3[1] x3[2]|
            double const det1
                = std::abs(x1[0] * (x2[1] * x3[2] - x2[2] * x3[1])
                           - x1[1] * (x2[0] * x3[2] - x2[2] * x3[0])
                           + x1[2] * (x2[0] * x3[1] - x2[1] * x3[0]));

            // determinant from cross product
            // |a b c|   |x4[0] x4[1] x4[2]|
            // |d e f| = |x3[0] x3[1] x3[2]|
            // |g h i|   |x2[0] x2[1] x2[2]|
            double const det2
                = std::abs(x4[0] * (x3[1] * x2[2] - x3[2] * x2[1])
                           - x4[1] * (x3[0] * x2[2] - x3[2] * x2[0])
                           + x4[2] * (x3[0] * x2[1] - x3[1] * x2[0]));

            // Populate ARPIST cubature weights and points for both triangular
            // halves of the divided pixe.
            for (size_t p = 0; p < CARD; ++p) {
                double const ksi = tri_cubature_deg4_points[2*p];
                double const eta = tri_cubature_deg4_points[2*p + 1];
                double const wgt = tri_cubature_deg4_weights[p];

                double xx
                    = x1[0] + ksi * (x2[0] - x1[0]) + eta * (x3[0] - x1[0]);
                double xy
                    = x1[1] + ksi * (x2[1] - x1[1]) + eta * (x3[1] - x1[1]);
                double xz
                    = x1[2] + ksi * (x2[2] - x1[2]) + eta * (x3[2] - x1[2]);
                /* double invnorm = 1.0 / std::sqrt(xx * xx + xy * xy + xz * xz); */
                int   q       = 4 * p;
                points_weights[h, w, q]     = xx; // * invnorm;
                points_weights[h, w, q + 1] = xy; // * invnorm;
                points_weights[h, w, q + 2] = xz; // * invnorm;
                points_weights[h, w, q + 3] = det1 * wgt; // * std::pow(invnorm, 3);

                xx      = x4[0] + eta * (x2[0] - x4[0]) + ksi * (x3[0] - x4[0]);
                xy      = x4[1] + eta * (x2[1] - x4[1]) + ksi * (x3[1] - x4[1]);
                xz      = x4[2] + eta * (x2[2] - x4[2]) + ksi * (x3[2] - x4[2]);
                q += (4 * CARD);
                points_weights[h, w, q]     = xx; // * invnorm;
                points_weights[h, w, q + 1] = xy; // * invnorm;
                points_weights[h, w, q + 2] = xy; // * invnorm;
                points_weights[h, w, q + 3] = det2 * wgt; // * std::pow(invnorm, 3);
            }
        }
    }
    return points_weights;
}
