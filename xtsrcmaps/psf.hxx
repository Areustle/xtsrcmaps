#pragma once

#include <cwchar>
#include <vector>

#include "xtsrcmaps/irf.hxx"

namespace Fermi::PSF
{

auto
separations(double const xmin, double const xmax, size_t const N)
    -> std::vector<double>;

// auto
// psf_fixed_grid(std::vector<double> const& deltas, IrfData3 const& pars)
//     -> std::vector<double>;
//
auto
bilerp(std::vector<double> const& cosBins,
       std::vector<double> const& logEs,
       std::vector<double> const& par_cosths,
       std::vector<double> const& par_logEs,
       mdarray3 const&            kings) -> mdarray3;

} // namespace Fermi::PSF
