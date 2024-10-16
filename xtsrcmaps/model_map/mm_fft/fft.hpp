#pragma once

#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

namespace Fermi::ModelMap::FFT {
auto convolve_map_psf(size_t const                    Nh,
                      size_t const                    Nw,
                      Fermi::Tensor<double, 2> const& src_sphcrds,
                      Fermi::Tensor<double, 3> const&
                          psf_lut, // psf lookup table [Source, Seps, Energy]
                      Fermi::SkyGeom<double> const& skygeom)
    -> Fermi::Tensor<double, 4>;

} // namespace Fermi::ModelMap::FFT
