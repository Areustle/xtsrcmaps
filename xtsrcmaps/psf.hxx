#pragma once

#include <vector>

#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/tensor_types.hxx"

namespace Fermi::PSF
{

// ln(7e5)/399
constexpr double sep_step = 0.033731417579011382769913057686378448039204491;

// exp(ln(7e5)/399)
constexpr double sep_delta
    = 1.0343067728020497121475244691736145227943806311378006312643513;

// sep_delta - 1.
constexpr double edm1 = 0.0343067728020497121475244691736145227943806311378006312643513;

using SepArr          = std::array<double, 401>;

///////////////////////////////////////////////////////////////////////////////////////
/// Given a PSF IRF grid and a set of separations, compute the King/Moffat results
/// for every entry in the table and every separation.
///////////////////////////////////////////////////////////////////////////////////////
auto
king(SepArr const& deltas, irf::psf::Data const& data) -> mdarray3; //[Nd, Me, Mc]


// constexpr auto
// separations(double const xmin = 1e-4, double const xmax = 70., size_t const N = 400)
//     -> std::vector<double>;
constexpr auto
separation(double const x) -> double
{
    assert(x >= 0.);
    return 1e-4 * (x >= 1. ? std::pow(sep_delta, (x - 1.)) : x);
}


constexpr auto
inverse_separation(double const s) -> double
{
    assert(s >= 0.);
    return s < 1e-4 ? 1e4 * s : 1. + std::log(s * 1e4) / sep_step;
}

constexpr auto
linear_inverse_separation(double const x) -> double
{
    assert(x >= 0.);
    double y_ = inverse_separation(x);
    double y0 = std::floor(y_);
    double y1 = std::ceil(y_);
    double x0 = separation(y0);
    double x1 = separation(y1);
    return (x - x0) * ((y1 - y0) / (x1 - x0)) + y0;
}


template <size_t N = 400, typename T = double>
constexpr auto
separations(double const xmin = 1e-4, double const xmax = 70.) -> SepArr
{
    auto   sep   = std::array<double, N + 1> { 0.0 };
    double xstep = std::log(xmax / xmin) / (N - 1.);
    for (size_t i = 0; i < N; ++i) { sep[i + 1] = xmin * std::exp(i * xstep); }
    return sep;
};
// constexpr auto
// inverse_separations(double const s) -> double;

// auto
// psf_fixed_grid(std::vector<double> const& deltas, IrfData3 const& pars)
//     -> std::vector<double>;
//
auto
bilerp(std::vector<double> const& costhetas,
       std::vector<double> const& logEs,
       std::vector<double> const& par_cosths,
       std::vector<double> const& par_logEs,
       mdarray3 const&            kings) -> mdarray3;

auto
corrected_exposure_psf(
    mdarray3 const& obs_psf,
    mdarray2 const& obs_aeff,
    mdarray2 const& src_exposure_cosbins,
    mdarray2 const& src_weighted_exposure_cosbins,
    std::pair<std::vector<double>, std::vector<double>> const& front_LTF /*[Ne]*/
    ) -> mdarray3;

auto
mean_psf(                                //
    mdarray3 const& front_corrected_psf, /*[Nd, Nc, Ne]*/
    mdarray3 const& back_corrected_psf,  /*[Nd, Nc, Ne]*/
    mdarray2 const& exposure /*[Ns, Ne]*/) -> mdarray3;

auto
partial_total_integral(std::vector<double> const& deltas, mdarray3 const& mean_psf)
    -> std::pair<mdarray3, mdarray2>;

auto
integral(std::vector<double> deltas,
         mdarray3 const&     partial_integrals,
         mdarray3 const&     mean_psf) -> mdarray3;

auto
normalize(mdarray3& mean_psf, mdarray2 const& total_integrals) -> void;

auto
peak_psf(mdarray3 const& mean_psf) -> mdarray2;

} // namespace Fermi::PSF
