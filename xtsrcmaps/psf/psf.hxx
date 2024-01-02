#pragma once

#include <vector>

#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/irf/irf_types.hxx"
#include "xtsrcmaps/math/tensor_types.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"

namespace Fermi::PSF {

// ln(7e5)/399
constexpr double sep_step = 0.033731417579011382769913057686378448039204491;

// exp(ln(7e5)/399)
constexpr double sep_delta
    = 1.0343067728020497121475244691736145227943806311378006312643513;

// sep_delta - 1.
constexpr double edm1
    = 0.0343067728020497121475244691736145227943806311378006312643513;

constexpr unsigned short sep_arr_len = 401;

using SepArr                         = std::array<double, sep_arr_len>;

template <size_t N = sep_arr_len - 1, typename T = double>
auto
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

auto fast_separation_lower_index(Tensor1d seps) -> Tensor1i;

// constexpr auto
// linear_inverse_separation(double const x) -> double
// {
//     assert(x >= 0.);
//     double y_ = inverse_separation(x);
//     double y0 = std::floor(y_);
//     double y1 = std::ceil(y_);
//     double x0 = separation(y0);
//     double x1 = separation(y1);
//     return (x - x0) * ((y1 - y0) / (x1 - x0)) + y0;
// }

///////////////////////////////////////////////////////////////////////////////////////
/// Given a PSF IRF grid and a set of separations, compute the King/Moffat
/// results for every entry in the table and every separation.
///////////////////////////////////////////////////////////////////////////////////////
auto king(irf::psf::Data const& data) -> Tensor3d;

auto bilerp(std::vector<double> const& costhetas,
            std::vector<double> const& logEs,
            Tensor1d const&            par_cosths,
            Tensor1d const&            par_logEs,
            Tensor3d const&            kings) -> Tensor3d;

auto
corrected_exposure_psf(Tensor3d const& obs_psf,
                       Tensor2d const& obs_aeff,
                       Tensor2d const& src_exposure_cosbins,
                       Tensor2d const& src_weighted_exposure_cosbins,
                       std::pair<std::vector<double>,
                                 std::vector<double>> const& front_LTF /*[Ne]*/
                       ) -> Tensor3d;

// [Nd, Ne, Ns]
auto mean_psf(                           //
    Tensor3d const& front_corrected_psf, /*[Nd, Nc, Ne]*/
    Tensor3d const& back_corrected_psf,  /*[Nd, Nc, Ne]*/
    Tensor2d const& exposures /*[Ne, Nsrc]*/) -> Tensor3d;

auto partial_total_integral(Tensor3d const& mean_psf /*[Nd, Ne, Ns]*/)
    -> std::pair<Tensor3d, Tensor2d>; // <[D,E,S], [E,S]>


auto normalize(Tensor3d& mean_psf, Tensor2d const& total_integrals) -> void;

auto peak_psf(Tensor3d const& mean_psf) -> Tensor2d;


auto psf_lookup_table_and_partial_integrals(
    irf::psf::Pass8FB const&   data,
    std::vector<double> const& costhetas,
    std::vector<double> const& logEs,
    Tensor2d const&            front_aeff,
    Tensor2d const&            back_aeff,
    Tensor2d const&            src_exposure_cosbins,
    Tensor2d const&            src_weighted_exposure_cosbins,
    std::pair<std::vector<double>, std::vector<double>> const&
                    front_LTF, /*[Ne]*/
    Tensor2d const& exposure   /*[Ne, Nsrc]*/
    ) -> std::tuple<Tensor3d, Tensor3d>;


struct XtPsf {
    Tensor3d uPsf;
    Tensor3d partial_psf_integral;
};

auto
compute_psf_data(XtObs const& obs, XtIrf const& irf, XtExp const exp) -> XtPsf;


} // namespace Fermi::PSF
