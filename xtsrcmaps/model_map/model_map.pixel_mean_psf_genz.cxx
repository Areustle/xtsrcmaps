#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/math/genz_malik.hxx"

#include "indicators/block_progress_bar.hpp"
/* #include "fmt/color.h" */

auto
spherical_direction_of_genz_pixels(Tensor3d const&       points,
                                   Fermi::SkyGeom const& skygeom) -> Array3Xd {
    Array3Xd dir_points(3, points.dimension(1) * points.dimension(2));
    for (long j = 0; j < points.dimension(2); ++j) {
        for (long i = 0; i < Fermi::Genz::Ncnt; ++i) {
            Vector3d p = skygeom.pix2dir({ points(0, i, j), points(1, i, j) });
            dir_points(Eigen::all, i + Fermi::Genz::Ncnt * j) = p;
        }
    }
    return dir_points;
};


auto
Fermi::ModelMap::pixel_mean_psf_genz(long const             Nh,
                                     long const             Nw,
                                     Obs::sphcrd_v_t const& src_sphcrds,
                                     std::vector<std::string> const& src_names,
                                     Tensor3d const&                 psf_lut,
                                     SkyGeom const&                  skygeom,
                                     double const ftol_threshold) -> Tensor4d {
    long const Ns    = src_sphcrds.size();
    long const Nd    = psf_lut.dimension(0);
    long const Ne    = psf_lut.dimension(1);
    long const Nevts = Nh * Nw;

    indicators::BlockProgressBar bar {
        indicators::option::BarWidth { 60 },
        indicators::option::PrefixText { "Convolving Pixels with Source PSF " },
        indicators::option::ForegroundColor { indicators::Color::magenta },
        /* indicators::option::ShowRemainingTime { true }, */
        /* indicators::option::FontStyles { std::vector<indicators::FontStyle> {
         */
        /*     indicators::FontStyle::bold } }, */
        indicators::option::MaxProgress { Ns },
    };

    Tensor4d model_map(Ne, Nh, Nw, Ns);
    model_map.setZero();

    // Compute initial (H,W) pairs of pixel center points.
    Tensor3d const init_points              = get_init_centers(Nh, Nw);

    // Pixel centers, and doubles halfwidth, volume
    auto const [centers, halfwidth, volume] = Genz::pixel_region(init_points);

    // Pixel points with minor Genz purturbations.
    Tensor3d const genz_points              = Genz::fullsym(centers,
                                               halfwidth * Genz::alpha2,
                                               halfwidth * Genz::alpha4,
                                               halfwidth * Genz::alpha5);

    // Transform functor for genz_points to spherical 3-points
    auto get_dir_points = [&skygeom](Tensor3d const& p) -> Array3Xd {
        return spherical_direction_of_genz_pixels(p, skygeom);
    };

    Array3Xd const dir_points = get_dir_points(genz_points);

    for (long s = 0; s < Ns; ++s) {
        // Update the progress bar source name
        bar.set_option(indicators::option::PostfixText { src_names[s] });

        // A slice of the PSF table just for this source's segment of the table.
        Tensor2d const tuPsf_ED
            = psf_lut.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 })
                  .reshape(Idx2 { Nd, Ne })
                  .shuffle(Idx2 { 1, 0 });

        // Get the sources coordinate in 3-direction space.
        Vector3d src_dir = skygeom.sph2dir(src_sphcrds[s]); // CLHEP Style 3

        /******************************************************************
         * Psf Energy values for sample points in direction space
         ******************************************************************/
        auto integrand   = [&src_dir, &tuPsf_ED](Array3Xd const& points3) {
            return psf_fast_lut(points3, src_dir.array(), tuPsf_ED);
        };

        // View of the results buffer
        Map<MatrixXd> result_value(
            model_map.data() + s * Ne * Nevts, Ne, Nevts);

        // The Genz Malik Integration rule adapted for this problem.
        Genz::integrate_region(integrand,
                               get_dir_points,
                               result_value,
                               centers,
                               halfwidth,
                               volume,
                               dir_points,
                               ftol_threshold);

        // Tick the bar
        bar.tick();
    }

    bar.mark_as_completed();

    return model_map;
}
