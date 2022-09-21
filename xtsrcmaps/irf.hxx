#pragma once

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include "experimental/mdspan"

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

auto
prepare_grid(fits::TablePars const& pars) -> IrfData3;

auto
prepare_scale(fits::TablePars const& pars) -> IrfScale;

auto
normalize_rpsf(Psf::Data&) -> void;

auto
prepare_psf_data(fits::TablePars const&,
                 fits::TablePars const&,
                 fits::TablePars const&,
                 fits::TablePars const&,
                 fits::TablePars const&,
                 fits::TablePars const&) -> Psf::Pass8;

auto
prepare_aeff_data(fits::TablePars const&,
                  fits::TablePars const&,
                  fits::TablePars const&,
                  fits::TablePars const&,
                  fits::TablePars const&,
                  fits::TablePars const&) -> Aeff::Pass8;

auto
load_aeff(std::string const&) -> std::optional<Aeff::Pass8>;

auto
load_psf(std::string const&) -> std::optional<Psf::Pass8>;

} // namespace Fermi
