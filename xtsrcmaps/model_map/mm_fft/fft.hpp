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

// 1. ✅ Load cropped galdiffuse file
// 2. ✅ Decide resample dimensions - update skygeom
//      ~~a. ✅ separate solid angle computation into math module~~
// 3. ✅ Resample diffuse image into the higher resolution block.
//      a. ✅ Write trilerp function
//      b. 📥 Apply trilerp to diffuse image block
// 4. ✅ Multiply Diffuse Image by Exposure.
//      a. Fermi::ModelMap should have some version of this.
// 5. ✅ Generate a PSF block as kernel for the upsampled diffuse block.
//      a. ✅ Create Psf::make_kernel function.
//      b. 📥 Build appropriate SkyGeom object for center of kernel using cdelt
//      resolution of the upscaled diffuse map.
// 6. ✅  FFT the upsampled block.
//      a. 👎 Accelerate Framework BNNS
//      b. 👎 Intel oneDNN
//      c. ✅ fallback to FFTW.
// 8. 📥 Turn into a model map somehow.
} // namespace Fermi::ModelMap::FFT
