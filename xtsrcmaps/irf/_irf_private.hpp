#pragma once

#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/irf/irf_types.hxx"

#include "fmt/format.h"

namespace irf_private {

// Forward declarations of private functions for IRF component

auto prepare_grid(Fermi::fits::TablePars const& pars) -> Fermi::IrfData3;

auto prepare_scale(Fermi::fits::TablePars const& pars) -> Fermi::IrfScale;

auto prepare_effic(Fermi::fits::TablePars const& pars) -> Fermi::IrfEffic;

auto normalize_rpsf(Fermi::irf::psf::Data& psfdata) -> void;

auto prepare_psf_data(Fermi::fits::TablePars const& front_rpsf,
                      Fermi::fits::TablePars const& front_scaling,
                      Fermi::fits::TablePars const& front_fisheye,
                      Fermi::fits::TablePars const& back_rpsf,
                      Fermi::fits::TablePars const& back_scaling,
                      Fermi::fits::TablePars const& back_fisheye)
    -> Fermi::irf::psf::Pass8FB;

auto prepare_aeff_data(Fermi::fits::TablePars const& front_eff_area,
                       Fermi::fits::TablePars const& front_phi_dep,
                       Fermi::fits::TablePars const& front_effici,
                       Fermi::fits::TablePars const& back_eff_area,
                       Fermi::fits::TablePars const& back_phi_dep,
                       Fermi::fits::TablePars const& back_effici)
    -> Fermi::irf::aeff::Pass8FB;

} // namespace irf_private
