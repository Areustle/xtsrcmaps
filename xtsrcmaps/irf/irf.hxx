#pragma once

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/irf/irf_types.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

namespace Fermi {

auto livetime_efficiency_factors(std::vector<double> const& logEs,
                                 IrfEffic const&            effic)
    -> Tensor<double, 2> /* [2, Ne] */;
// std::pair<std::vector<double>, std::vector<double>>;

auto load_aeff(std::string const&) -> std::optional<irf::aeff::Pass8FB>;

auto load_psf(std::string const&) -> std::optional<irf::psf::Pass8FB>;

auto collect_irf_data(XtCfg const& cfg, XtObs const& obs) -> XtIrf;

// Define a concept for types that support the subscript operator
/* template <typename C> */
/* concept Subscriptable = requires(C t, std::size_t i) { */
/*     { t[i] } -> std::convertible_to<typename C::value_type>; */
/* }; */

template <typename T, typename C>
inline auto
evaluate_king(T const sep, T const scale_factor, C const& pars) -> T {
    auto f = [](T const u, T gamma) -> T {
        // ugly kluge because of sloppy programming in handoff_response
        // when setting boundaries of fit parameters for the PSF.
        if (gamma == 1) { gamma = 1.001; }
        return (1. - 1. / gamma) * std::pow(1. + u / gamma, -gamma);
    };

    T const ncore = pars[0];
    T const ntail = pars[1];
    T const score = pars[2] * scale_factor;
    T const stail = pars[3] * scale_factor;
    T const gcore = pars[4];
    T const gtail = pars[5];

    T rc          = sep / score;
    T uc          = rc * rc / 2.;

    T rt          = sep / stail;
    T ut          = rt * rt / 2.;
    return (ncore * f(uc, gcore) + ntail * ncore * f(ut, gtail));
}


template <typename T, typename C>
inline auto
psf3_psf_base_integral(T const  radius,
                       T const  scale_factor,
                       C const& pars) -> T {
    auto f = [](T u, T gamma) -> T {
        T arg(1. + u / gamma);
        if (arg < 0) { throw std::runtime_error("neg. arg to pow"); }
        return 1. - std::pow(arg, 1. - gamma);
    };

    T const ncore = pars[0];
    T const ntail = pars[1];
    T const score = pars[2] * scale_factor;
    T const stail = pars[3] * scale_factor;
    T const gcore = pars[4];
    T const gtail = pars[5];

    T sep         = radius * deg2rad;
    T rc          = sep / score;
    T uc          = rc * rc / 2.;

    T rt          = sep / stail;
    T ut          = rt * rt / 2.;

    if (gcore < 0. || gtail < 0.) { throw std::runtime_error("gamma < 0"); }

    return (ncore * f(uc, gcore) * twopi * score * score
            + ntail * ncore * f(ut, gtail) * twopi * stail * stail);
}


auto aeff_value(std::vector<double> const& costhet,
                std::vector<double> const& logEs,
                IrfData3 const&            AeffData) -> Tensor<double, 2>;
} // namespace Fermi
