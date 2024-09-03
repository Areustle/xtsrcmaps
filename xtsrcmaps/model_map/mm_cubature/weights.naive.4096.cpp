#include "xtsrcmaps/model_map/mm_cubature/cubature.hpp"
#include "xtsrcmaps/tensor/tensor.hpp"

/********************************************************************************
 * Convolve the PSF via lookup table, with each of the pixels for every source.
 * ----
 ********************************************************************************/
auto
Fermi::naive_4096_ptswts(size_t const           Nh,
                         size_t const           Nw,
                         SkyGeom<double> const& skygeom) -> Tensor<double, 3> {

    constexpr size_t  NPIX  = 64;
    constexpr double  dstep = 1.0 / static_cast<double>(NPIX);
    constexpr double  wt    = 1.0 / (NPIX * NPIX);
    Tensor<double, 3> points_weights({ Nh, Nw, 4 * NPIX * NPIX });

    for (size_t h = 0; h < Nw; ++h) {
        for (size_t w = 0; w < Nw; ++w) {

            double ph = 1. + h;
            double pw = 1. + w;
            /**/
            /* // x3 UL -------- UR x4 */
            /* //      |        | */
            /* //      |        | */
            /* //      |        | */
            /* // x1 LL -------- LR x2 */
            /**/
            /* std::array<double, 2> const LL { ph - 0.5f, pw - 0.5f }; */
            /* std::array<double, 2> const LR { ph - 0.5f, pw + 0.5f }; */
            /* std::array<double, 2> const UL { ph + 0.5f, pw - 0.5f }; */
            /* std::array<double, 2> const UR { ph + 0.5f, pw + 0.5f }; */
            /**/
            /* auto const ll = skygeom.pix2dir(LL); */
            /* auto const lr = skygeom.pix2dir(LR); */
            /* auto const ul = skygeom.pix2dir(UL); */
            /* auto const ur = skygeom.pix2dir(UR); */
            /* m_dir = Hep3Vector( cos(ra)*cos(dec), sin(ra)*cos(dec) , sin(dec)
             * );         */

            for (size_t p = 0; p < NPIX * NPIX; ++p) {

                size_t i = p / NPIX;
                size_t j = p % NPIX;

                double x = ph + (-0.5 + (i + 0.5) * dstep);
                double y = pw + (-0.5 + (j + 0.5) * dstep);

                auto dir = skygeom.pix2dir({ x, y });

                points_weights[h, w, 4 * p] ///
                    = dir[0];
                points_weights[h, w, 4 * p + 1] = dir[1];
                points_weights[h, w, 4 * p + 2] = dir[2];
                points_weights[h, w, 4 * p + 3] = wt;
            }
        }
    }
    return points_weights;
}
