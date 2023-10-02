#pragma once

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf_types.hxx"

#include <cmath>
#include <optional>
#include <vector>

namespace Fermi {

auto
livetime_efficiency_factors(std::vector<double> const& logEs, IrfEffic const& effic)
    -> std::pair<std::vector<double>, std::vector<double>>;

auto load_aeff(std::string const&) -> std::optional<irf::aeff::Pass8FB>;

auto load_psf(std::string const&) -> std::optional<irf::psf::Pass8FB>;

} // namespace Fermi
