#pragma once

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/irf/irf_types.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"

#include <optional>
#include <vector>

namespace Fermi {

auto livetime_efficiency_factors(std::vector<double> const& logEs,
                                 IrfEffic const&            effic)
    -> std::pair<std::vector<double>, std::vector<double>>;

auto load_aeff(std::string const&) -> std::optional<irf::aeff::Pass8FB>;

auto load_psf(std::string const&) -> std::optional<irf::psf::Pass8FB>;

auto collect_irf_data(XtCfg const& cfg, XtObs const& obs) -> XtIrf;

auto aeff_value(std::vector<double> const& costhet,
                std::vector<double> const& logEs,
                IrfData3 const&            AeffData) -> Tensor2d;
} // namespace Fermi
