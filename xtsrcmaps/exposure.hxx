#pragma once

#include "xtsrcmaps/irf.hxx"

#include <vector>

namespace Fermi
{

auto
aeff_value(std::vector<double> const&, std::vector<double> const&, IrfData3 const&)
    -> mdarray2;

auto
phi_mod(std::vector<double> const&, std::vector<double> const&, IrfData3 const&, bool)
    -> mdarray2;

auto
exposure(mdarray2 const& aeff, mdarray2 const& phi, std::vector<double> costhe)
    -> mdarray1;


} // namespace Fermi
