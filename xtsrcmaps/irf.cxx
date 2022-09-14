#include "xtsrcmaps/irf.hxx"

#include "xtsrcmaps/gauss_legendre.hxx"

#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>

#include "experimental/mdspan"
#include <fmt/format.h>
#include <xtsrcmaps/fmt_source.hxx>

using std::pair;
using std::sin;
using std::string;
using std::vector;
using std::experimental::mdspan;
using std::experimental::submdspan;

//************************************************************************************
// Convert the raw arrays read in from an IRF FITS file into a useable set of AEFF
// data tensors for future sampling.
//************************************************************************************
auto
Fermi::prepare_irf_data(fits::IrfGrid const& pars) -> IrfData3
{

    // Get sizing params
    size_t const M_t_base = pars.costhe_lo.size();
    size_t const M_t      = M_t_base + 2;

    size_t const M_e_base = pars.energy_lo.size();
    size_t const M_e      = M_e_base + 2;

    // scale and pad the energy data
    std::vector<double> cosths(M_t, 0.0);
    for (size_t k(0); k < M_t_base; k++)
    {
        // Arithmetic mean of cosine bins
        cosths[1 + k] = 0.5 * (pars.costhe_lo[k] + pars.costhe_hi[k]);
    }
    // padded cosine bin values.
    cosths.front() = 0.0;
    cosths.back()  = 1.0;

    // scale and pad the energy data
    std::vector<double> logEs(M_e, 0.0);
    for (size_t k(0); k < M_e_base; k++)
    {
        // Geometric mean of energy bins
        logEs[1 + k] = std::log10(std::sqrt(pars.energy_lo[k] * pars.energy_hi[k]));
    }

    // padded energy bin values.
    logEs.front() = 0.0;
    logEs.back()  = 10.0;

    auto params   = vector<double>(M_t * M_e * pars.Ngrids);
    auto pv       = mdspan(params.data(), M_t, M_e, pars.Ngrids);

    /// First let's assign the data values into the params block structure.
    /// Pad with value duplication.
    for (size_t t = 0; t < pv.extent(0); ++t) // costheta
    {
        size_t const t_ = t == 0 ? 0 : t >= M_t_base ? M_t_base - 1 : t - 1;
        for (size_t e = 0; e < pv.extent(1); ++e) // energy
        {
            size_t const e_ = e == 0 ? 0 : e >= M_e_base ? M_e_base - 1 : e - 1;
            for (size_t p = 0; p < pv.extent(2); ++p) // params
            {
                auto parview = mdspan(pars.params[p].data(), M_t_base, M_e_base);
                pv(t, e, p)  = parview(t_, e_);
            }
        }
    }

    return { cosths,
             logEs,
             mdarray3(params, pv.extent(0), pv.extent(1), pv.extent(2)) };
}


inline constexpr auto
psf_base_function(double u, double gamma) -> double
{
    // ugly kluge because of sloppy programming in handoff_response
    // when setting boundaries of fit parameters for the PSF.
    if (gamma == 1) { gamma = 1.001; }
    return (1. - 1. / gamma) * std::pow(1. + u / gamma, -gamma);
}

//
inline constexpr auto
evaluate_king(double const sep, double const scale_factor, auto const pars) -> double
{
    double const ncore = pars[0];
    double const ntail = pars[1];
    double const score = pars[2] * scale_factor;
    double const stail = pars[3] * scale_factor;
    double const gcore = pars[4];
    double const gtail = pars[5];

    double rc          = sep / score;
    double uc          = rc * rc / 2.;

    double rt          = sep / stail;
    double ut          = rt * rt / 2.;
    return (ncore * psf_base_function(uc, gcore)
            + ntail * ncore * psf_base_function(ut, gtail));
}

inline constexpr auto
psf2_psf_base_integral(double u, double gamma) -> double
{
    double arg(1. + u / gamma);
    if (arg < 0) { throw std::runtime_error("neg. arg to pow"); }
    return 1. - std::pow(arg, 1. - gamma);
}

inline constexpr auto
psf3_psf_base_integral(double const radius, double const scale_factor, auto const pars)
    -> double
{
    double const ncore = pars[0];
    double const ntail = pars[1];
    double const score = pars[2] * scale_factor;
    double const stail = pars[3] * scale_factor;
    double const gcore = pars[4];
    double const gtail = pars[5];

    double sep         = radius * M_PI / 180.;
    double rc          = sep / score;
    double uc          = rc * rc / 2.;

    double rt          = sep / stail;
    double ut          = rt * rt / 2.;

    if (gcore < 0. || gtail < 0.) { throw std::runtime_error("gamma < 0"); }

    return (ncore * psf2_psf_base_integral(uc, gcore) * 2. * M_PI * score * score
            + ntail * ncore * psf2_psf_base_integral(ut, gtail) * 2. * M_PI * stail
                  * stail);
}

auto
Fermi::normalize_irf_data(IrfData3& data, fits::IrfScale const& scale) -> void
{

    // span the IrfData params (should pass mdarray)
    // auto pv = mdspan(data.params.data(), data.extent0, data.extent1, data.extent2);
    // Next normalize and scale.

    auto scaleFactor = [sp0 = (scale.scale0 * scale.scale0),
                        sp1 = (scale.scale1 * scale.scale1),
                        si  = scale.scale_index](double const energy) {
        double const tt = std::pow(energy / 100., si);
        return std::sqrt(sp0 * tt * tt + sp1);
    };

    // An integration is required below, so let's precompute the orthogonal legendre
    // polynomials here for future use.
    auto const polypars = legendre_poly_rw<64>(1e-15);

    auto pv             = mdspan(data.params.data(),
                     data.params.extent(0),
                     data.params.extent(1),
                     data.params.extent(2));

    for (size_t t = 0; t < pv.extent(0); ++t) // costheta
    {
        for (size_t e = 0; e < pv.extent(1); ++e) // energy
        {
            double const energy = std::pow(10, data.logEs[e]);
            double const sf     = scaleFactor(energy);
            double const norm
                = energy < 120. //
                      ? gauss_legendre_integral(
                          0.0,
                          90.,
                          polypars,
                          [&](auto const& v) -> double {
                              auto x = evaluate_king(
                                  v * M_PI / 180, sf, submdspan(pv, t, e, pair(0, 6)));
                              auto y = sin(v * M_PI / 180.) * 2. * M_PI * M_PI / 180.;
                              return x * y;
                          })
                      : psf3_psf_base_integral(
                          90.0, sf, submdspan(pv, t, e, pair(0, 6)));

            pv(t, e, 0) /= norm;
        }
    }
};

//************************************************************************************
// Convert the raw arrays read in from an IRF PSF FITS file into a useable set of PSF
// data tensors for future sampling.
//************************************************************************************
// auto
// Fermi::prepare_psf_data(fits::PsfParamData const& pars) -> IrfData
// {
//
//     // Get sizing params
//     size_t const M_t_base = pars.costhe_lo.size();
//     size_t const M_t      = M_t_base + 2;
//
//     size_t const M_e_base = pars.energy_lo.size();
//     size_t const M_e      = M_e_base + 2;
//
//     // scale and pad the energy data
//     std::vector<double> cosths(M_t, 0.0);
//     for (size_t k(0); k < M_t_base; k++)
//     {
//         // Arithmetic mean of cosine bins
//         cosths[1 + k] = 0.5 * (pars.costhe_lo[k] + pars.costhe_hi[k]);
//     }
//     // padded cosine bin values.
//     cosths.front() = -1.0;
//     cosths.back()  = 1.0;
//
//     // scale and pad the energy data
//     std::vector<double> logEs(M_e, 0.0);
//     for (size_t k(0); k < M_e_base; k++)
//     {
//         // Geometric mean of energy bins
//         logEs[1 + k] = std::log10(std::sqrt(pars.energy_lo[k] * pars.energy_hi[k]));
//     }
//     // padded energy bin values.
//     logEs.front()   = 0.0;
//     logEs.back()    = 10.0;
//
//     auto params     = vector<double>(M_t * M_e * 6);
//     auto pv         = mdspan(params.data(), M_t, M_e, 6);
//     auto ncore_view = mdspan(pars.ncore.data(), M_t_base, M_e_base);
//     auto ntail_view = mdspan(pars.ntail.data(), M_t_base, M_e_base);
//     auto score_view = mdspan(pars.score.data(), M_t_base, M_e_base);
//     auto stail_view = mdspan(pars.stail.data(), M_t_base, M_e_base);
//     auto gcore_view = mdspan(pars.gcore.data(), M_t_base, M_e_base);
//     auto gtail_view = mdspan(pars.gtail.data(), M_t_base, M_e_base);
//
//     /// First let's assign the data values into the params block structure.
//     /// Pad with value duplication.
//     for (size_t t = 0; t < pv.extent(0); ++t) // costheta
//     {
//         size_t const t_ = t == 0 ? 0 : t >= M_t_base ? M_t_base - 1 : t - 1;
//         for (size_t e = 0; e < pv.extent(1); ++e) // energy
//         {
//             size_t const e_ = e == 0 ? 0 : e >= M_e_base ? M_e_base - 1 : e - 1;
//             pv(t, e, 0)     = ncore_view(t_, e_);
//             pv(t, e, 1)     = ntail_view(t_, e_);
//             pv(t, e, 2)     = score_view(t_, e_);
//             pv(t, e, 3)     = stail_view(t_, e_);
//             pv(t, e, 4)     = gcore_view(t_, e_);
//             pv(t, e, 5)     = gtail_view(t_, e_);
//         }
//     }
//
//     // Next normalize and scale.
//
//     auto scaleFactor = [sp0 = (pars.scale0 * pars.scale0),
//                         sp1 = (pars.scale1 * pars.scale1),
//                         si  = pars.scale_index](double const energy) {
//         double const tt = std::pow(energy / 100., si);
//         return std::sqrt(sp0 * tt * tt + sp1);
//     };
//
//     // An integration is required below, so let's precompute the orthogonal legendre
//     // polynomials here for future use.
//     auto const polypars = legendre_poly_rw<64>(1e-15);
//
//     for (size_t t = 0; t < pv.extent(0); ++t) // costheta
//     {
//         for (size_t e = 0; e < pv.extent(1); ++e) // energy
//         {
//             double const energy = std::pow(10, logEs[e]);
//             double const sf     = scaleFactor(energy);
//             double const norm
//                 = energy < 120. //
//                       ? gauss_legendre_integral(
//                           0.0,
//                           90.,
//                           polypars,
//                           [&](auto const& v) -> double {
//                               auto x = evaluate_king(
//                                   v * M_PI / 180, sf, submdspan(pv, t, e, pair(0,
//                                   6)));
//                               auto y = sin(v * M_PI / 180.) * 2. * M_PI * M_PI /
//                               180.; return x * y;
//                           })
//                       : psf3_psf_base_integral(
//                           90.0, sf, submdspan(pv, t, e, pair(0, 6)));
//
//             pv(t, e, 0) /= norm;
//         }
//     }
//
//     return {  cosths,logEs, params };
// }
