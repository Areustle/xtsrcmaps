#pragma once

#include <vector>

#include "xtsrcmaps/fitsfuncs.hxx"

namespace Fermi
{

struct IrfData
{
    std::vector<double> cosths;
    std::vector<double> logEs;
    std::vector<double> params;
    size_t              extent0;
    size_t              extent1;
    size_t              extent2;
};

auto
prepare_psf_data(fits::PsfParamData const& pars) -> IrfData;

auto
irf_fixed_grid(IrfData const& pars) -> std::vector<double>;

auto
prepare_irf_data(fits::IrfGrid const& pars) -> IrfData;

auto
normalize_irf_data(IrfData&, fits::IrfScale const&) -> void;



} // namespace Fermi
