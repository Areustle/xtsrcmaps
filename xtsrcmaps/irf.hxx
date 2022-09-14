#pragma once

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include "experimental/mdspan"

#include <vector>

namespace Fermi
{

struct IrfData3
{
    std::vector<double> cosths;
    std::vector<double> logEs;
    mdarray3            params;

    auto
    mdspan() -> mdspan3
    {
        return mdspan3(
            params.data(), params.extent(0), params.extent(1), params.extent(2));
    }
};

namespace Psf
{

struct Data
{
    IrfData3 rpsf;
    IrfData3 psf_scaling_params;
    IrfData3 fisheye_correction;
};

struct Pass8
{
    Psf::Data front;
    Psf::Data back;
};

} // namespace Psf

namespace Aeff
{

struct Data
{
    IrfData3 effective_area;
    IrfData3 phi_dependence;
    IrfData3 efficiency_params;
};

struct Pass8
{
    Aeff::Data front;
    Aeff::Data back;
};
} // namespace Aeff

auto
prepare_irf_data(fits::IrfGrid const& pars) -> IrfData3;

auto
normalize_irf_data(IrfData3&, fits::IrfScale const&) -> void;

} // namespace Fermi
