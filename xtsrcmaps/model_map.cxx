#include "xtsrcmaps/model_map.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/fmt_source.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"

#include "fmt/format.h"
#include "unsupported/Eigen/CXX11/Tensor"

auto
Fermi::ModelMap::pix_dirs_with_padding(SkyGeom const& skygeom,
                                       long const     Nw,
                                       long const     Nh) -> MatCoord3
{
    long const Nw_pad = Nw + 2;
    long const Nh_pad = Nh + 2;

    MatCoord3 pdirs(Nw_pad, Nh_pad);
    for (long ph = 0; ph < Nh_pad; ++ph)
    {
        for (long pw = 0; pw < Nw_pad; ++pw)
        {
            pdirs(pw, ph) = skygeom.pix2dir({ ph, pw });
        }
    }

    return pdirs;
}

auto
Fermi::ModelMap::pixel_angular_offset_from_source_with_padding(
    SkyGeom::coord3 const& src_dir,
    SkyGeom::coord2 const& src_pix,
    double const           ref_size,
    SkyGeom const&         skygeom,
    long const             Nw,
    long const             Nh) -> Eigen::MatrixXd
{
    long const Nw_pad = Nw + 2;
    long const Nh_pad = Nh + 2;

    Eigen::MatrixXd Offsets(Nw_pad, Nh_pad);
    for (long ph = 0; ph < Nh_pad; ++ph)
    {
        for (long pw = 0; pw < Nw_pad; ++pw)
        {
            auto   pdir    = skygeom.pix2dir({ ph, pw });
            double pix_sep = ref_size
                             * std::sqrt(std::pow(src_pix.first - ph, 2)
                                         + std::pow(src_pix.second - pw, 2));
            double ang_sep  = skygeom.srcpixoff(src_dir, pdir);
            Offsets(pw, ph) = pix_sep > 1E-6 ? ang_sep / pix_sep - 1. : 0.0;
        }
    }

    return Offsets;
}

auto
Fermi::ModelMap::pixel_angular_offset_from_source_with_padding(
    MatCoord3 const&       pdirs,
    SkyGeom::coord3 const& src_dir,
    SkyGeom::coord2 const& src_pix,
    double const           ref_size,
    SkyGeom const&         skygeom) -> Eigen::MatrixXd
{
    long const Nw_pad = pdirs.rows();
    long const Nh_pad = pdirs.cols();

    Eigen::MatrixXd Offsets(Nw_pad, Nh_pad);
    for (long ph = 0; ph < Nh_pad; ++ph)
    {
        for (long pw = 0; pw < Nw_pad; ++pw)
        {
            double pix_sep = ref_size
                             * std::sqrt(std::pow(src_pix.first - ph, 2)
                                         + std::pow(src_pix.second - pw, 2));
            double ang_sep  = skygeom.srcpixoff(src_dir, pdirs(pw, ph));
            Offsets(pw, ph) = pix_sep > 1E-6 ? ang_sep / pix_sep - 1. : 0.0;
        }
    }

    return Offsets;
}

// Index in PSF Lookup Table of Pixel given by dispacement from source
auto
Fermi::ModelMap::psf_idx_sep(SkyGeom::coord3 const& src_dir,
                             MatCoord3 const&       pdirs,
                             SkyGeom const&         skygeom) -> Eigen::MatrixXd
{
    long const Nw = pdirs.rows() - 2;
    long const Nh = pdirs.cols() - 2;

    Eigen::MatrixXd Displacements(Nw, Nh);
    for (long h = 0; h < Nh; ++h)
    {
        for (long w = 0; w < Nw; ++w)
        {
            Displacements(w, h) = Fermi::PSF::linear_inverse_separation(
                skygeom.srcpixoff(src_dir, pdirs(w + 1, h + 1)));
        }
    }

    return Displacements;
}


auto
Fermi::ModelMap::is_integ_psf_converged(Eigen::Ref<Eigen::MatrixXd const> const& v0,
                                        Eigen::Ref<Eigen::MatrixXd const> const& v1,
                                        double const ftol_threshold) -> bool
{
    return (((v1 - v0).array() / v0.array()).abs() < ftol_threshold).all();
}



template <>
auto
Fermi::ModelMap::integrate_psf_recursive<64u>(
    long const                               pw,
    long const                               ph,
    SkyGeom::coord3 const                    src_dir,
    SkyGeom const&                           skygeom,
    double const                             ftol_threshold,
    Eigen::MatrixXd const&                   suPsf, // Nd, Ne
    Eigen::Ref<Eigen::MatrixXd const> const& v0) -> Eigen::MatrixXd
{
    (void)(ftol_threshold);
    (void)(v0);
    return integrate_psf_<64u>(pw, ph, src_dir, skygeom, suPsf);
}



auto
Fermi::ModelMap::pixels_to_integrate(
    Eigen::Ref<Eigen::MatrixXd const> const& mean_psf_v0,
    Eigen::Ref<Eigen::VectorXd const> const& uPeak,
    double const                             peak_threshold,
    long const                               Nw,
    long const                               Nh) -> std::vector<std::pair<long, long>>
{
    Eigen::MatrixX<bool> needs_more
        = (((mean_psf_v0.array().rowwise() / uPeak.transpose().array())
            >= peak_threshold)
               .rowwise()
               .any())
              .reshaped(Nw, Nh);

    std::vector<std::pair<long, long>> Idxs {};
    Fermi::ModelMap::visit_lambda(needs_more, [&Idxs](bool v, long w, long h) {
        if (v) { Idxs.push_back({ w, h }); }
    });
    Idxs.shrink_to_fit();
    return Idxs;
}


auto
Fermi::ModelMap::point_src_model_map_wcs(long const      Nw,
                                         long const      Nh,
                                         vpd const&      dirs,
                                         Tensor3d const& uPsf,
                                         Tensor2d const& uPeak,
                                         SkyGeom const&  skygeom) -> Tensor4d
{
    long const Ns               = dirs.size();
    long const Nd               = uPsf.dimension(0);
    long const Ne               = uPsf.dimension(1);

    double const peak_threshold = 1e-6;
    double const ftol_threshold = 1e-3;
    auto         pdirs          = pix_dirs_with_padding(skygeom, Nw, Nh);

    Tensor4d xtpsfEst(Nw, Nh, Ne, Ns);

    for (long s = 0; s < Ns; ++s)
    {
        auto src_dir = skygeom.sph2dir(dirs[s]); // CLHEP Style 3
        // auto         src_pix  = skygeom.sph2pix(dirs[s]); // Grid Style 2
        // double const ref_size = 0.2;

        /////
        Tensor2d const tuPsf
            = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 }).reshape(Idx2 { Nd, Ne });
        Tensor1d const tuPeak
            = uPeak.slice(Idx2 { 0, s }, Idx2 { Ne, 1 }).reshape(Idx1 { Ne });

        Eigen::Map<Eigen::MatrixXd const> const suPsf(tuPsf.data(), Nd, Ne);
        Eigen::Map<Eigen::VectorXd const> const suPeak(tuPeak.data(), Ne);

        // Eigen::MatrixXd const Offsets =
        // pixel_angular_offset_from_source_with_padding(
        //     pdirs, src_dir, src_pix, ref_size, skygeom);

        Eigen::ArrayXd isep = psf_idx_sep(src_dir, pdirs, skygeom).reshaped().array();

        /// Compute the initial mean psf for every pixel vs this source.
        Eigen::MatrixXd psfEst = psf_lut(isep, suPsf);

        /// Which pixels need further psf integration because their psf value is too
        /// close to the peak psf?
        auto pxs_int = pixels_to_integrate(psfEst, suPeak, peak_threshold, Nw, Nh);

        /// Integrate the necessary pixels.
        for (auto const& p : pxs_int)
        {
            long const w  = p.first;
            long const h  = p.second;

            long const pw = p.first;
            long const ph = p.second;

            Eigen::MatrixXd v1
                = integrate_psf_recursive(pw,
                                          ph,
                                          src_dir,
                                          skygeom,
                                          ftol_threshold,
                                          suPsf,
                                          psfEst.block(w + h * Nw, 0, 1, Ne));
            psfEst.block(w + h * Nw, 0, 1, Ne) = v1;
        }
    }

    return xtpsfEst;
}
