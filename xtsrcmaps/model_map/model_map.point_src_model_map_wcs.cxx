#include "xtsrcmaps/model_map/model_map.hxx"

#include "xtsrcmaps/math/tensor_ops.hxx"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"

#include "unsupported/Eigen/CXX11/Tensor"
#include <Eigen/Dense>

auto
Fermi::ModelMap::point_src_model_map_wcs(
    long const             Nh,
    long const             Nw,
    Obs::sphcrd_v_t const& src_sphcrds,
    Tensor3d const&        uPsf,
    SkyGeom const&         skygeom,
    Tensor2d const&        exposures,
    Tensor3d const&        partial_integrals, /* [D,E,S] */
    double const           ftol_threshold) -> Tensor4f {

    // Use Genz Malik Integration Scheme to compute the per-pixel average PSF
    // value for every source and energy level.
    Tensor4d model_map = pixel_mean_psf_genz(
        Nh, Nw, src_sphcrds, uPsf, skygeom, ftol_threshold);

    // Scale the model_map by the central solid angle of every pixel.
    scale_map_by_solid_angle(model_map, skygeom);

    // Compute the sources in the FOV and the PSF boundary radius;
    auto [full_psf_radius, is_in_fov]
        = psf_boundary_radius(Nh, Nw, src_sphcrds, skygeom);
    Tensor1d const psf_radius = filter_in(full_psf_radius, is_in_fov);

    // Compute the MapIntegral scalar for each source & energy given source
    // location.
    Tensor2d const inv_mapinteg
        = map_integral(model_map, src_sphcrds, skygeom, psf_radius, is_in_fov);

    // Scale each map value by the exposure for this source.
    scale_map_by_exposure(model_map, exposures);

    // Apply map_correction_factor
    Tensor2d const correction_factor = map_correction_factor(
        inv_mapinteg, psf_radius, is_in_fov, uPsf, partial_integrals);

    scale_map_by_correction_factors(model_map, correction_factor);

    return model_map.cast<float>();
}
