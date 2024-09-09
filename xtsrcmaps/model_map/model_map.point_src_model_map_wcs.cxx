#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/model_map/mm_cubature/cubature.hpp"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

auto
Fermi::ModelMap::point_src_model_map_wcs(
    size_t const                    Nh,
    size_t const                    Nw,
    SkyGeom<double> const&          skygeom,
    Tensor<double, 2> const&        src_sph,
    std::vector<std::string> const& src_names,
    Tensor<double, 2> const&        exposures,
    Tensor<double, 3> const&        uPsf,             // [SDE]
    Tensor<double, 3> const&        partial_integrals // [SDE]
    ) -> Tensor<double, 4> {

    // Use the cubature scheme to convolve the PSF for each source with the
    // energy map.
    Tensor<double, 4> model_map
        = cubature_convolve_map_psf(Nh, Nw, src_sph, uPsf, skygeom);

    // Scale the model_map by the central solid angle of every pixel.
    scale_map_by_solid_angle(model_map, skygeom);

    // Compute the sources in the FOV and the PSF boundary radius;
    auto [psf_radius, is_in_fov]
        = psf_boundary_radius(Nh, Nw, src_sph, skygeom);

    // Compute the MapIntegral scalar for each source & energy given source
    // location.
    Tensor<double, 2> const inv_mapinteg
        = map_integral(model_map, src_sph, skygeom, psf_radius, is_in_fov);

    // Scale each map value by the exposure for this source.
    scale_map_by_exposure(model_map, exposures);

    // Apply map_correction_factor
    Tensor<double, 2> const correction_factor = map_correction_factor(
        inv_mapinteg, psf_radius, is_in_fov, uPsf, partial_integrals);

    scale_map_by_correction_factors(model_map, correction_factor);

    return model_map;
}
