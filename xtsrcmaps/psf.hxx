#pragma once

#include <vector>

#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/tensor_types.hxx"

namespace Fermi::PSF
{

///////////////////////////////////////////////////////////////////////////////////////
/// Given a PSF IRF grid and a set of separations, compute the King/Moffat results for
/// every entry in the table and every separation.
///////////////////////////////////////////////////////////////////////////////////////
auto
king(std::vector<double> const& deltas, irf::psf::Data const& data)
    -> mdarray3; //[Nd, Me, Mc]


auto
separations(double const xmin, double const xmax, size_t const N)
    -> std::vector<double>;

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
} // namespace Fermi::PSF
