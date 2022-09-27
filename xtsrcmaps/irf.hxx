#pragma once

#include "xtsrcmaps/irf_types.hxx"

#include "experimental/mdspan"

#include <cmath>
#include <optional>
#include <vector>

namespace Fermi
{

auto
load_aeff(std::string const&) -> std::optional<Aeff::Pass8>;

auto
load_psf(std::string const&) -> std::optional<Psf::Pass8>;

} // namespace Fermi
