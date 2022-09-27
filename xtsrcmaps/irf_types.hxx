#pragma once

#include "xtsrcmaps/tensor_types.hxx"

#include "experimental/mdspan"

#include <array>
#include <cmath>
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

struct IrfScale
{
    float scale0;
    float scale1;
    float scale_index;
};

struct IrfEffic
{
    std::array<float, 6> p0;
    std::array<float, 6> p1;
};

namespace Psf
{

struct Data
{
    IrfData3 rpsf;
    IrfScale psf_scaling_params;
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
    IrfEffic efficiency_params;
};

struct Pass8
{
    Aeff::Data front;
    Aeff::Data back;
};
} // namespace Aeff

} // namespace Fermi