#pragma once

#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/math/tensor_types.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"

#include "unsupported/Eigen/CXX11/Tensor"

namespace Fermi::ModelMap {

auto compute_srcmaps(XtObs const& obs, XtExp const& exp, XtPsf const& psf)
    -> Tensor4f;

auto point_src_model_map_wcs(long const                      Nh,
                             long const                      Nw,
                             Obs::sphcrd_v_t const&          src_sph,
                             std::vector<std::string> const& src_names,
                             Tensor3d const&                 uPsf,
                             SkyGeom const&                  skygeom,
                             Tensor2d const&                 exposures,
                             Tensor3d const& partial_integrals, /* [D,E,S] */
                             double const    ftol_threshold = 1e-3) -> Tensor4f;

auto get_init_points(long const Nh, long const Nw) -> Tensor3d;

/* auto spherical_direction_of_genz_pixels(Tensor3d const& points, */
/*                                         SkyGeom const&  skygeom) -> Array3Xd;
 */

auto psf_fast_lut(Array3Xd const& points3,
                  ArrayXd const&  src_d,
                  Tensor2d const& tuPsf_ED) -> Tensor3d;

auto pixel_mean_psf_genz(long const                      Nh,
                         long const                      Nw,
                         Obs::sphcrd_v_t const&          src_sph,
                         std::vector<std::string> const& src_names,
                         Tensor3d const&                 psf_lut,
                         SkyGeom const&                  skygeom,
                         double const ftol_threshold = 1e-3) -> Tensor4d;

auto create_offset_map(long const                       Nh,
                       long const                       Nw,
                       std::pair<double, double> const& dir,
                       Fermi::SkyGeom const&            skygeom) -> Tensor2d;

/* auto */
/* pixel_mean_psf_riemann(long const      Nh, */
/*                        long const      Nw, */
/*                        Obs::sphcrd_v_t const&      src_sph, */
/*                        Tensor3d const& psf_lut, */
/*                        Tensor2d const& psf_peak, */
/*                        SkyGeom const&  skygeom, */
/*                        double const    ftol_threshold = 1e-3) -> Tensor4d; */

auto
solid_angle(Tensor3d const& points, Fermi::SkyGeom const& skygeom) -> Tensor2d;

void scale_map_by_solid_angle(Tensor4d& model_map, SkyGeom const& skygeom);

void scale_map_by_exposure(Tensor4d& model_map, Tensor2d const& exposures);

auto map_correction_factor(Tensor2d const& MapInteg,
                           Tensor1d const& psf_radius,
                           Tensor1b const& is_in_fov,
                           Tensor3d const& mean_psf,         /* [D,E,S] */
                           Tensor3d const& partial_integrals /* [D,E,S] */
                           ) -> Tensor2d;

void scale_map_by_correction_factors(Tensor4d&       model_map, /*[E,H,W,S]*/
                                     Tensor2d const& factor /*[E,S]*/);

auto
psf_boundary_radius(long const             Nh,
                    long const             Nw,
                    Obs::sphcrd_v_t const& src_sph,
                    SkyGeom const& skygeom) -> std::pair<Tensor1d, Tensor1b>;

auto map_integral(Tensor4d const&        model_map,
                  Obs::sphcrd_v_t const& src_sph,
                  SkyGeom const&         skygeom,
                  Tensor1d const&        psf_radius,
                  Tensor1b const&        is_in_fov) -> Tensor2d;

auto integral(Tensor1d const& angles,
              Tensor3d const& mean_psf,         /* [D,E,S] */
              Tensor3d const& partial_integrals /* [D,E,S] */
              ) -> Tensor2d;


} // namespace Fermi::ModelMap