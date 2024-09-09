#pragma once

#include <cassert>
#include <vector>

#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/irf/irf_types.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

namespace Fermi::Psf {

struct XtPsf {
    Tensor<double, 3> uPsf;
    Tensor<double, 3> partial_psf_integral;
};


constexpr unsigned short sep_arr_len = 401;

using SepArr                         = std::array<double, sep_arr_len>;

template <size_t N = sep_arr_len - 1, typename T = double>
constexpr auto
separations(double const xmin = 1e-4, double const xmax = 70.) -> SepArr {
    auto   sep   = std::array<double, N + 1> { 0.0 };
    double xstep = std::log(xmax / xmin) / (N - 1.);
    for (size_t i = 0; i < N; ++i) { sep[i + 1] = xmin * std::exp(i * xstep); }
    return sep;
};

constexpr auto
separation(double const x) -> double {
    assert(x >= 0.);
    return 1e-4 * (x >= 1. ? std::pow(sep_delta, (x - 1.)) : x);
}

constexpr auto
inverse_separation(double const s) -> double {
    assert(s >= 0.);
    return s < 1e-4 ? 1e4 * s : 1. + std::log(s * 1e4) / sep_step;
}

auto fast_separation_lower_index(Tensor<double, 1> seps) -> Tensor<int, 1>;


///////////////////////////////////////////////////////////////////////////////////////
/// Given a PSF IRF grid and a set of separations, compute the King/Moffat
/// results for every entry in the table and every separation.
///////////////////////////////////////////////////////////////////////////////////////
auto king(Irf::psf::Data const& data) -> Tensor<double, 3>;

auto bilerp(std::vector<double> const& costhetas,
            std::vector<double> const& logEs,
            Tensor<double, 1> const&   par_cosths,
            Tensor<double, 1> const&   par_logEs,
            Tensor<double, 3> const&   kings) -> Tensor<double, 3>;

auto
corrected_exposure_psf(Tensor<double, 3> const& obs_psf,
                       Tensor<double, 2> const& obs_aeff,
                       Tensor<double, 2> const& src_exposure_cosbins,
                       Tensor<double, 2> const& src_weighted_exposure_cosbins,
                       Tensor<double, 2> const& front_LTF /*[Ne]*/
                       ) -> Tensor<double, 3>;

// [Ns, Ne, Nd]
auto mean_psf(                                    //
    Tensor<double, 3> const& front_corrected_psf, /*[Ne, Nc, Nd]*/
    Tensor<double, 3> const& back_corrected_psf,  /*[Ne, Nc, Nd]*/
    Tensor<double, 2> const& exposures /*[Nsrc, Ne]*/)
    -> Tensor<double, 3> /*[Ns, Nd, Ne]*/;

auto partial_total_integral(Tensor<double, 3> const& mean_psf /*[Ns, Ne, Nd]*/)
    -> std::pair<Tensor<double, 3>,
                 Tensor<double, 2>>; // <[Ns, Ne, Nd], [Nsrc, Ne]>


auto normalize(Tensor<double, 3>&       mean_psf, /* [Ns, Ne, Nd] */
               Tensor<double, 2> const& total_integrals) -> void;


auto psf_lookup_table_and_partial_integrals(
    Irf::psf::Pass8FB const&   data,
    std::vector<double> const& costhetas,
    std::vector<double> const& logEs,
    Tensor<double, 2> const&   front_aeff,
    Tensor<double, 2> const&   back_aeff,
    Tensor<double, 2> const&   src_exposure_cosbins,
    Tensor<double, 2> const&   src_weighted_exposure_cosbins,
    Tensor<double, 2> const&   front_LTF, /*[2, Ne]*/
    Tensor<double, 2> const&   exposure   /*[Nsrc, Ne]*/
    ) -> std::tuple<Tensor<double, 3>, Tensor<double, 3>>;

auto compute_psf_data(Obs::XtObs const&        obs,
                      Fermi::Irf::XtIrf const& irf,
                      Exposure::XtExp const&   exp) -> XtPsf;

} // namespace Fermi::Psf
