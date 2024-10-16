#pragma once

#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

namespace Fermi {

template <typename T = double, std::size_t R = 2>
struct SkyImage {
    Tensor<T, R>        data     = {};
    SkyGeom<double>     skygeom  = {};
    std::vector<double> energies = { 100.0 };
};

} // namespace Fermi
