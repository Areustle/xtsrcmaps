#pragma once

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include <vector>

namespace Fermi
{

struct ExposureMap
{
    std::size_t nside;
    std::size_t nbins;
    mdarray2    params; // Healpix, CosineBin
};

auto
exp_map(fits::ExposureCubeData const&) -> ExposureMap;

auto
src_exp_cosbins(std::vector<std::pair<double, double>> const&, ExposureMap const&)
    -> mdarray2;

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
