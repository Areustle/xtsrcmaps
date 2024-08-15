#pragma once

#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

namespace Fermi {
auto convolve_psf_with_map_deg4x2_arpist(
    size_t const             Nh,
    size_t const             Nw,
    Tensor<double, 2> const& src_sphcrds,
    Tensor<float, 3> const&  psf_lut, // psf lookup table [Source, Seps, Energy]
    SkyGeom<float> const&    skygeom) -> Tensor<float, 4>; //[SHWE]
}
