#pragma once

#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

namespace Fermi {

template <typename T = double, std::size_t R = 2>
class SkyImage {
  private:
    Tensor<T, R>        _data     = {};
    SkyGeom<double>     _skygeom  = {};
    std::vector<double> _energies = { 100.0 };

  public:
    SkyImage() {};

    SkyImage(Tensor<T, R> const&        data,
             SkyGeom<double> const&     skygeom,
             std::vector<double> const& energies)
        : _data(data), _skygeom(skygeom), _energies(energies) {}
};

} // namespace Fermi
