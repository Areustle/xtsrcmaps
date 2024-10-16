#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/math/trilerp.hpp"
#include "xtsrcmaps/model_map/mm_fft/fft.hpp"
#include "xtsrcmaps/sky_geom/sky_geom.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

auto
Fermi::ModelMap::diffuse_src_model_map_wcs(
    size_t const                    Nh,
    size_t const                    Nw,
    SkyGeom<double> const&          skygeom,
    Tensor<double, 2> const&        src_sph,
    std::vector<std::string> const& src_names,
    Tensor<double, 2> const&        exposures,
    Tensor<double, 3> const&        uPsf,              // [SDE]
    Tensor<double, 3> const&        partial_integrals, // [SDE]
    double const                    target_pix_size) -> Tensor<double, 4> {

    size_t const Ns = src_sph.extent(0);
    size_t const Ne = uPsf.extent(2);

    // 1. ✅ Load cropped galdiffuse file:
    // 2. ✅ Decide resample dimensions - update skygeom
    //      ~~a. ✅ separate solid angle computation into math module~~
    // 3. ✅ Resample diffuse image into the higher resolution block.
    //      a. ✅ Write trilerp function
    //      b. 📥 Apply trilerp to diffuse image block
    // 4. ✅ Multiply Diffuse Image by Exposure.
    //      a. Fermi::ModelMap should have some version of this.
    // 5. ✅ Generate a PSF block as kernel for the upsampled diffuse block.
    //      a. ✅ Create Psf::make_kernel function.
    //      b. 📥 Build appropriate SkyGeom object for center of kernel using
    //      cdelt resolution of the upscaled diffuse map.
    // 6. ✅ FFT the upsampled block.
    //      a. 👎 Accelerate Framework BNNS
    //      b. 👎 Intel oneDNN
    //      c. ✅ fallback to FFTW.
    // 8. ✅ Turn into a model map somehow.
    ////////////////////////////////////////////////////////////////
    /// Read cropped galdiffuse file
    auto galdiffuse = fits::read_allsky_cropped(skygeom, src_names[0]);
    ////////////////////////////////////////////////////////////////
    /// Multiply by Exposures [Ne] ?
    ////////////////////////////////////////////////////////////////
    /// Compute Rebin Factor
    double pixsz    = (target_pix_size < std::max(galdiffuse.skygeom.cdelt()[0],
                                               galdiffuse.skygeom.cdelt()[1]))
                          ? target_pix_size
                          : galdiffuse.skygeom.cdelt()[0];
    auto   upscale_diffuse = rebin_skygeom(galdiffuse.skygeom, pixsz);
    size_t const Nhp       = upscale_diffuse.naxes()[1];
    size_t const Nwp       = upscale_diffuse.naxes()[0];
    size_t       Nk        = std::min(Nhp, Nwp);
    ////////////////////////////////////////////////////////////////
    /// Upscale cropped diffuse image by rebin factor (Trilerp || Omega)
    auto highresgaldiff    = Tensor<float, 3>(Ne, Nhp, Nwp);
    Fermi::math::trilerpEHW(galdiffuse.data, highresgaldiff);
    ////////////////////////////////////////////////////////////////
    /// Generate PSF Block
    Nk -= (Nk % 2) ? 0uz : 1uz; // Ensure Odd so central pixel aligns with
                                // memory target.
    auto kernel = Fermi::Psf::make_kernel(Nk, upscale_diffuse, uPsf);
    ////////////////////////////////////////////////////////////////
    /// Convolve
    Tensor<double, 4> model_map = FFT::convolve_map_psf(highresgaldiff, kernel);
    /* Nh, Nw, src_sph, uPsf, skygeom); */
    ////////////////////////////////////////////////////////////////
    /// 📥 Downsample
    ////////////////////////////////////////////////////////////////
    /// 📥 Perform final scalings
    ////////////////////////////////////////////////////////////////


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
