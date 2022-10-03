#pragma once

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf_types.hxx"

#include "experimental/mdspan"

#include <cmath>
#include <optional>
#include <vector>

namespace Fermi
{

auto
lt_effic_factors(std::vector<double> const& logEs, IrfEffic const& effic)
    -> std::pair<std::vector<double>, std::vector<double>>;

auto
load_aeff(std::string const&) -> std::optional<Aeff::Pass8>;

auto
load_psf(std::string const&) -> std::optional<Psf::Pass8>;

} // namespace Fermi
