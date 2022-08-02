#include "xtsrcmaps/psf.hxx"

#include "xtsrcmaps/gauss_legendre.hxx"

#include <cmath>

#include "experimental/mdspan"
#include <fmt/format.h>
#include <xtsrcmaps/fmt_source.hxx>

using std::string;
using std::vector;
using std::experimental::mdspan;


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
evaluate_moffat(double const sep,
                double const scale_factor,
                double const ncore,
                double const ntail,
                double       score,
                double       stail,
                double const gcore,
                double const gtail) -> double
{
    score *= scale_factor;
    stail *= scale_factor;

    double rc = sep / score;
    double uc = rc * rc / 2.;

    double rt = sep / stail;
    double ut = rt * rt / 2.;
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
psf3_psf_base_integral(double const radius,
                       double const scale_factor,
                       double const ncore,
                       double const ntail,
                       double       score,
                       double       stail,
                       double const gcore,
                       double const gtail) -> double
{
    score *= scale_factor;
    stail *= scale_factor;

    double sep = radius * M_PI / 180.;
    double rc  = sep / score;
    double uc  = rc * rc / 2.;

    double rt  = sep / stail;
    double ut  = rt * rt / 2.;

    if (gcore < 0 || gtail < 0) { throw std::runtime_error("gamma < 0"); }

    return (ncore * psf2_psf_base_integral(uc, gcore) * 2. * M_PI * score * score
            + ntail * ncore * psf2_psf_base_integral(ut, gtail) * 2. * M_PI * stail
                  * stail);
}

//************************************************************************************
// Convert the raw arrays read in from an IRF PSF FITS file into a useable set of PSF
// data tensors for future sampling.
//************************************************************************************
auto
Fermi::prepare_psf_data(fits::PsfParamData const& pars) -> PsfData
{

    // scale and reformat the data
    std::vector<double> Energies(pars.energy_lo.size() + 2, 0.0);
    std::vector<double> logEs(pars.energy_lo.size() + 2, 0.0);
    for (size_t k(0); k < pars.energy_lo.size(); k++)
    {
        Energies[1 + k] = std::sqrt(pars.energy_lo[k] * pars.energy_hi[k]);
        logEs[1 + k]    = std::log10(Energies[1 + k]);
    }
    Energies.front() = 1.0;
    Energies.back()  = 1e10;
    logEs.front()    = 0.0;
    logEs.back()     = 10.0;

    std::vector<double> cosths(pars.costhe_lo.size() + 2, 0.0);
    for (size_t k(0); k < pars.costhe_lo.size(); k++)
    {
        cosths[1 + k] = 0.5 * (pars.costhe_lo[k] + pars.costhe_hi[k]);
    }
    cosths.front()   = -1.0;
    cosths.back()    = 1.0;

    auto params      = vector<double>(/*23 * 8*/ 184 * 6);
    auto pv          = mdspan(params.data(), 23, 8, 6);
    auto ncore_view  = mdspan(pars.ncore.data(), 23, 8);
    auto ntail_view  = mdspan(pars.ntail.data(), 23, 8);
    auto score_view  = mdspan(pars.score.data(), 23, 8);
    auto stail_view  = mdspan(pars.stail.data(), 23, 8);
    auto gcore_view  = mdspan(pars.gcore.data(), 23, 8);
    auto gtail_view  = mdspan(pars.gtail.data(), 23, 8);

    auto scaleFactor = [s0  = pars.scale0,
                        sp1 = (pars.scale1 * pars.scale1),
                        sp2 = (pars.scale2 * pars.scale2)](double const energy) {
        double tt = std::pow(energy / 100., s0);
        return std::sqrt(sp1 * tt * tt + sp2);
    };

    auto psf_value = [&](double const sep,
                         double const scale_factor,
                         size_t const i,
                         size_t const j) -> double {
        double const s = sep * M_PI / 180;
        return evaluate_moffat(s,
                               scale_factor,
                               ncore_view(i, j),
                               ntail_view(i, j),
                               score_view(i, j),
                               stail_view(i, j),
                               gcore_view(i, j),
                               gtail_view(i, j));
    };


    auto polypars = legendre_poly_rw<8>(1e-15);

    for (size_t i = 0; i < pv.extent(0); ++i) // energy
    {
        double const energy = Energies[i + 1];
        double const sf     = scaleFactor(energy);

        for (size_t j = 0; j < pv.extent(1); ++j) // costheta
        {
            double const norm
                = energy < 120.
                      ? gauss_legendre_integral(0.0,
                                                std::acos(cosths[j]) * 180. / M_PI,
                                                polypars,
                                                [&](auto const& v) -> double {
                                                    auto x = psf_value(v, sf, i, j);
                                                    auto y = std::sin(v * M_PI * 180.)
                                                             * 2. * M_PI * M_PI / 180.;
                                                    return x * y;
                                                })
                      : psf3_psf_base_integral(90.0,
                                               sf,
                                               ncore_view(i, j),
                                               ntail_view(i, j),
                                               score_view(i, j),
                                               stail_view(i, j),
                                               gcore_view(i, j),
                                               gtail_view(i, j));

            pv(i, j, 0)     = ncore_view(i, j) / norm;
            pv(i, j, 1)     = ntail_view(i, j) / norm;
            pv(i, j, 2)     = (score_view(i, j) / norm) * sf;
            pv(i, j, 3)     = (stail_view(i, j) / norm) * sf;
            double const gc = gcore_view(i, j) / norm;
            double const gt = gtail_view(i, j) / norm;
            pv(i, j, 4)     = gc == 1. ? 1.001 : gc;
            pv(i, j, 5)     = gt == 1. ? 1.001 : gt;
        }
    }

    return {
        std::move(logEs), std::move(Energies), std::move(cosths), std::move(params)
    };
}
