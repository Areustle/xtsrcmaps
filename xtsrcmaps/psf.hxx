#pragma once

#include <cwchar>
#include <vector>

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf.hxx"

namespace Fermi
{

auto
separations(double const xmin, double const xmax, size_t const N)
    -> std::vector<double>;

auto
psf_fixed_grid(std::vector<double> const& deltas, IrfData3 const& pars)
    -> std::vector<double>;

auto
bilerp(std::vector<double> const& kings,
       std::vector<double> const& energies,
       std::vector<double> const& cosBins,
       IrfData3 const&             pars) -> std::vector<double>;

} // namespace Fermi
