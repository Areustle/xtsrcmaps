#pragma once

#include "xtsrcmaps/tensor_types.hxx"

namespace Fermi
{

auto
psf_lerp_slow(Tensor2d const& lut, long const eidx, double const offset) -> double;

} // namespace Fermi
