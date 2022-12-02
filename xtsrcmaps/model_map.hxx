#pragma once

#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include "unsupported/Eigen/CXX11/Tensor"

namespace Fermi::ModelMap
{

auto
get_init_points(long const Nh, long const Nw) -> Tensor3d;

auto
spherical_direction_of_pixels(Tensor3d const& points, SkyGeom const& skygeom)
    -> Array3Xd;

auto
psf_fast_lut(Array3Xd const& points3, ArrayXd const& src_d, Tensor2d const& tuPsf_ED)
    -> Tensor3d;

auto
point_src_model_map_wcs(long const      Nx,
                        long const      Ny,
                        vpd const&      src_dirs,
                        Tensor3d const& uPsf,
                        SkyGeom const&  skygeom,
                        double const    ftol_threshold = 1e-3) -> Tensor4d;


} // namespace Fermi::ModelMap
