#pragma once

#include <vector>

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf.hxx"

namespace Fermi
{


auto
psf_fixed_grid(IrfData const& pars) -> std::vector<double>;

auto
bilerp(std::vector<double> const& kings,
       std::vector<double> const& energies,
       std::vector<double> const& cosBins,
       IrfData const&             pars) -> std::vector<double>;

} // namespace Fermi
