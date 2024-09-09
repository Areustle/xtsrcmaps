#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/irf/_irf_private.hpp"

#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <cassert>

#include "fmt/format.h"

//************************************************************************************
//************************************************************************************

auto
read_opt(auto&&             f,
         std::string const& filename,
         std::string const& tablename) -> auto {
    auto opt_irf = f(filename, tablename);
    if (!opt_irf) fmt::print("Cannot read {} table {}!\n", filename, tablename);
    return opt_irf;
}
//************************************************************************************
//************************************************************************************
auto
Fermi::Irf::load_aeff(std::string const& filename)
    -> std::optional<Irf::aeff::Pass8FB> {
    auto f   = fits::read_irf_pars;
    auto o0f = read_opt(f, filename, "EFFECTIVE AREA_FRONT");
    auto o1f = read_opt(f, filename, "PHI_DEPENDENCE_FRONT");
    auto o2f = read_opt(f, filename, "EFFICIENCY_PARAMS_FRONT");
    auto o0b = read_opt(f, filename, "EFFECTIVE AREA_BACK");
    auto o1b = read_opt(f, filename, "PHI_DEPENDENCE_BACK");
    auto o2b = read_opt(f, filename, "EFFICIENCY_PARAMS_BACK");

    if (o0f && o1f && o2f && o0b && o1b && o2b) {
        return { irf_private::prepare_aeff_data(o0f.value(),
                                                o1f.value(),
                                                o2f.value(),
                                                o0b.value(),
                                                o1b.value(),
                                                o2b.value()) };
    } else {
        return std::nullopt;
    }
}

auto
Fermi::Irf::load_psf(std::string const& filename)
    -> std::optional<Irf::psf::Pass8FB> {
    auto f   = fits::read_irf_pars;
    auto o0f = read_opt(f, filename, "RPSF_FRONT");
    auto o1f = read_opt(f, filename, "PSF_SCALING_PARAMS_FRONT");
    auto o2f = read_opt(f, filename, "FISHEYE_CORRECTION_FRONT");
    auto o0b = read_opt(f, filename, "RPSF_BACK");
    auto o1b = read_opt(f, filename, "PSF_SCALING_PARAMS_BACK");
    auto o2b = read_opt(f, filename, "FISHEYE_CORRECTION_BACK");

    if (o0f && o1f && o2f && o0b && o1b && o2b) {
        return { irf_private::prepare_psf_data(o0f.value(),
                                               o1f.value(),
                                               o2f.value(),
                                               o0b.value(),
                                               o1b.value(),
                                               o2b.value()) };
    } else {
        return std::nullopt;
    }
}

auto
Fermi::Irf::livetime_efficiency_factors(std::vector<double> const& logEs,
                                        IrfEffic const&            effic)
    -> Tensor<double, 2> /* [2, Ne] */ {
    // pair<vector<double>, vector<double>> {

    auto single_factor = [](auto const& p) -> auto {
        auto const&  a0     = p.at(0);
        auto const&  b0     = p.at(1);
        auto const&  a1     = p.at(2);
        auto const&  logEb1 = p.at(3);
        auto const&  a2     = p.at(4);
        auto const&  logEb2 = p.at(5);
        double const b1     = (a0 - a1) * logEb1 + b0;
        double const b2     = (a1 - a2) * logEb2 + b1;

        return [=](double const logE) -> double {
            return logE < logEb1   ? a0 * logE + b0
                   : logE < logEb2 ? a1 * logE + b1
                                   : a2 * logE + b2;
        };
    };

    /* auto lf1 = vector<double>(logEs.size(), 0.); */
    /* auto lf0 = vector<double>(logEs.size(), 0.); */
    Tensor<double, 2> ltf(2UZ, logEs.size());
    // Intentionally ordered that way. ¯\_(ツ)_/¯
    std::transform(logEs.cbegin(),
                   logEs.cend(),
                   ltf.begin_at(1, 0),
                   single_factor(effic.p0));
    std::transform(logEs.cbegin(),
                   logEs.cend(),
                   ltf.begin_at(0, 0),
                   single_factor(effic.p1));

    return ltf; // [2, Ne]
    // Intentionally ordered that way. ¯\_(ツ)_/¯
    /* return { lf1, lf0 }; */
}
