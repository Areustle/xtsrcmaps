#pragma once

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf_types.hxx"

#include "experimental/mdspan"

#include <cmath>
#include <vector>

namespace Fermi
{

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
