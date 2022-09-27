#pragma once

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include <vector>

namespace Fermi
{

struct LiveTimeExposure
{

    std::size_t nside;
    mdarray2    params; // Healpix, CosineBin

    auto
    mdspan() -> mdspan2
    {
        return mdspan2(params.data(), params.extent(0), params.extent(1));
    }
};

auto
lt_exposure(std::optional<fits::LiveTimeCubeData> const& data)
    -> std::optional<Fermi::LiveTimeExposure>;

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
