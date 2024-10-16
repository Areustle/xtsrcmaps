#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/misc/simdwidth.hpp"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"



auto
Fermi::Psf::make_kernel(size_t const             Nk,
                        SkyGeom<double> const&   skygeom,
                        Tensor<double, 3> const& uPsf) -> Tensor<double, 3> {
    size_t const Ns         = uPsf.extent(0);
    size_t const Ne         = uPsf.extent(2);
    /* size_t       Nk   = Nh <= Nw ? Nh : Nw; */
    /* Nk                = Nk % 2 ? Nk - 1 : Nk; // Ensure odd kernel size */
    auto kernel             = Tensor<double, 3>(Ne, Nk, Nk);
    auto totals             = std::vector<double>(Ne, 0.0);
    double* __restrict__ TT = &(totals[0]);

    auto const refdir       = skygeom.refdir();

    for (size_t h = 0; h < Nk; ++h) {
        double const ph = h + 1.;
        for (size_t w = 0; w < Nk; ++w) {
            double const pw  = w + 1.;
            auto const   dir = skygeom.pix2dir({ ph, pw });
            double const s   = skygeom.srcpixoff(refdir, dir);
            bool         c   = s < 1e-4;
            double       j
                = c ? 1.
                    : 1.
                          + recipstep
                                * (9.2103403719761827360719658187374568304044059545150919 // ln(1e4)
                                   + std::log(s));
            uint16_t i = 1 + static_cast<uint16_t>(j);
            i          = i > 0x190 ? 0x190 : i; // 0x190 == 400
            double t  = _redm1 * (s * 1e4 * std::exp(sep_step * (2. - i)) - 1.);
            double ct = 1.0 - t;

            // Fast access to psf memory.
            /* double* __restrict__ KK             = &(kernel[h, w, 0]); */
            double const* const __restrict__ YL = &(uPsf[0, (i - 1), 0]);
            double const* const __restrict__ YR = &(uPsf[0, i, 0]);

            size_t e                            = 0uz;
            // Process `simd_width` elements at a time
            for (; e <= Ne - simd_width; e += simd_width) {
                for (size_t lane = 0; lane < simd_width; ++lane) {
                    auto i          = e + lane;
                    auto val        = (t * YL[i]) + (ct * YR[i]);
                    kernel[i, h, w] = val;
                    TT[i] += val;
                }
            }
            for (; e < Ne; ++e) {
                auto val        = (t * YL[e]) + (ct * YR[e]);
                kernel[e, h, w] = val;
                TT[e] += val;
            }
        }
    }

    // Invert the totals for normalization.
    size_t e = 0uz;
    // Process `simd_width` elements at a time
    for (; e <= Ne - simd_width; e += simd_width) {
        for (size_t lane = 0; lane < simd_width; ++lane) {
            auto i = e + lane;
            TT[i]  = 1.0 / TT[i];
        }
    }
    for (; e < Ne; ++e) { TT[e] = 1.0 / TT[e]; }

    // Normalize the kernel along energy chanels
    for (size_t h = 0; h < Nk; ++h) {
        for (size_t w = 0; w < Nk; ++w) {
            size_t e = 0uz;
            // Process `simd_width` elements at a time
            for (; e <= Ne - simd_width; e += simd_width) {
                for (size_t lane = 0; lane < simd_width; ++lane) {
                    auto i = e + lane;
                    kernel[i, h, w] *= TT[i];
                }
            }
            for (; e < Ne; ++e) { kernel[e, h, w] *= TT[e]; }
        }
    }

    return kernel;
};
