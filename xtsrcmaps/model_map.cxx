#include "xtsrcmaps/model_map.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/fmt_source.hxx"
#include "xtsrcmaps/genz_malik.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"

#include <cmath>

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
psf_sample_full_energy(ArrayXd const& src_d, Tensor2d const& tuPsf_ED) -> auto
{

    long const Ne = tuPsf_ED.dimension(0);

    /******************************************************************
     * Psf Energy values for sample points in direction space
     ******************************************************************/
    return [&, Ne](Array3Xd const& points3) -> Tensor3d {
        long const Npts     = points3.size() / 3;
        // std::cout << Npts << " " << std::flush;
        // Given sample points on the sphere in 3-direction-space, compute the
        // separation.
        auto           diff = points3.colwise() - src_d;
        auto           mag  = diff.colwise().norm();
        auto           off  = 2. * rad2deg * Eigen::asin(0.5 * mag);
        ArrayXXd const separation_index
            = (off < 1e-4).select(1e4 * off, 1. + ((off * 1e4).log() / sep_step));
        TensorMap<Tensor1d const> const idxs(separation_index.data(), Npts);
        // std::cout << std::endl
        //           << idxs.reshape(Idx2 { 17, Npts / 17 })
        //                  .slice(Idx2 { 0, 1 }, Idx2 { 17, 1 })
        //                  .reshape(Idx2 { 1, 17 })
        //           << std::endl;

        Tensor3d vals(Ne, Fermi::Genz::Ncnt, Npts / Fermi::Genz::Ncnt);

        long i = 0;
        while (i < Npts)
        {
            long         d  = 1;
            double const x1 = std::floor(idxs(i));
            while ((i + d < Npts) && x1 == std::floor(idxs(i + d))) { ++d; }
            TensorMap<Tensor1d const> const ss(idxs.data() + i, d);
            Tensor2d                        alpha(d, 2);
            TensorMap<Tensor1d>(alpha.data(), d) = ss - x1;
            TensorMap<Tensor1d>(alpha.data() + d, d)
                = 1. - TensorMap<Tensor1d>(alpha.data(), d);
            TensorMap<Tensor2d const> const psf(tuPsf_ED.data() + long(x1) * Ne, Ne, 2);
            Tensor2d const vv = psf.contract(alpha, IdxPair1 { { { 1, 1 } } });
            TensorMap<Tensor2d>(vals.data() + i * Ne, Ne, d) = vv;
            i += d;
        }
        // std::cout << std::endl
        //           << vals.slice(Idx3 { 0, 0, 1 }, Idx3 { Ne, 17, 1 })
        //                  .reshape(Idx2 { Ne, 17 })
        //           << std::endl;
        return vals;
    };
}


auto
psf_sample_single_energy(ArrayXd const& src_d, Tensor2d const& tuPsf_DE) -> auto
{

    long const Nd = tuPsf_DE.dimension(0);

    /******************************************************************
     * Psf Energy values for sample points in direction space
     ******************************************************************/
    return [&, Nd](Array3Xd const& points3, long const& ei) -> Tensor3d {
        long const Npts     = points3.size() / 3;
        // std::cout << Npts << " " << std::flush;
        // Given sample points on the sphere in 3-direction-space, compute the
        // separation.
        auto           diff = points3.colwise() - src_d;
        auto           mag  = diff.colwise().norm();
        auto           off  = 2. * rad2deg * Eigen::asin(0.5 * mag);
        ArrayXXd const separation_index
            = (off < 1e-4).select(1e4 * off, 1. + ((off * 1e4).log() / sep_step));
        TensorMap<Tensor1d const> const idxs(separation_index.data(), Npts);

        Tensor3d vals(1, Fermi::Genz::Ncnt, Npts / Fermi::Genz::Ncnt); // 1, 17, Nevts

        long i = 0;
        while (i < Npts)
        {
            long         d  = 1;
            double const x1 = std::floor(idxs(i));
            while ((i + d < Npts) && x1 == std::floor(idxs(i + d))) { ++d; }
            TensorMap<Tensor1d const> const ss(idxs.data() + i, d);
            Tensor2d                        alpha(d, 2);
            TensorMap<Tensor1d>(alpha.data(), d) = ss - x1;
            TensorMap<Tensor1d>(alpha.data() + d, d)
                = 1. - TensorMap<Tensor1d>(alpha.data(), d);

            TensorMap<Tensor2d const> const psf(
                tuPsf_DE.data() + long(x1) + Nd * ei, 2, 1);

            Tensor2d const vv = psf.contract(alpha, IdxPair1 { { { 0, 1 } } });

            TensorMap<Tensor2d>(vals.data() + i, 1, d) = vv;
            i += d;
        }
        return vals;
    };
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
    long const Nevts            = Nw * Nh;

    // double const peak_threshold = 1e-6;
    double const ftol_threshold = 1e-3;

    auto get_dir_points         = [&](Tensor3d const& points) -> Array3Xd {
        Array3Xd dir_points(3, points.dimension(1) * points.dimension(2));
        for (long j = 0; j < points.dimension(2); ++j)
        {
            for (long i = 0; i < Genz::Ncnt; ++i)
            {
                coord3 p = skygeom.pix2dir({ points(0, i, j), points(1, i, j) });
                dir_points(0, i + Genz::Ncnt * j) = std::get<0>(p);
                dir_points(1, i + Genz::Ncnt * j) = std::get<1>(p);
                dir_points(2, i + Genz::Ncnt * j) = std::get<2>(p);
            }
        }
        return dir_points;
    };

    Tensor4d xtpsfEst(Nw, Nh, Ne, Ns);

    Tensor3d low(2, Nh, Nw);
    Tensor3d high(2, Nh, Nw);

    for (long w = 0; w < Nw; ++w)
    {
        for (long h = 0; h < Nh; ++h)
        {
            low(0, h, w)  = (1 + h) - 0.5;
            low(1, h, w)  = (1 + w) - 0.5;
            high(0, h, w) = (1 + h) + 0.5;
            high(1, h, w) = (1 + w) + 0.5;
        }
    }

    // long const Nsamp                 = Genz::Ncnt * Nevts;

    auto [center, halfwidth, volume] = Genz::region(low, high, Nevts);
    Tensor3d       points            = Genz::fullsym(center,
                                    halfwidth * Genz::alpha2,
                                    halfwidth * Genz::alpha4,
                                    halfwidth * Genz::alpha5);
    Array3Xd const dir_points        = get_dir_points(points);

    for (long s = 0; s < Ns; ++s)
    {
        // Index of events.
        Eigen::VectorX<Eigen::Index> evtidx
            = Eigen::VectorX<Eigen::Index>::LinSpaced(Nevts, 0, Nevts - 1);

        MatrixXd result_value(Ne, Nevts);
        MatrixXd result_error(Ne, Nevts);
        result_value.setZero();
        result_error.setZero();

        Tensor2d const tuPsf_ED = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 })
                                      .reshape(Idx2 { Nd, Ne })
                                      .shuffle(Idx2 { 1, 0 });
        Tensor2d const tuPsf_DE
            = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 }).reshape(Idx2 { Nd, Ne });

        auto           src_dir = skygeom.sph2dir(dirs[s]); // CLHEP Style 3
        Eigen::ArrayXd src_d(3, 1);
        src_d << std::get<0>(src_dir), std::get<1>(src_dir), std::get<2>(src_dir);

        /******************************************************************
         * Psf Energy values for sample points in direction space
         ******************************************************************/
        // auto sphere_pix_upsf = psf_sample_full_energy(src_d, tuPsf_ED);
        auto sphere_pix_upsf = psf_sample_single_energy(src_d, tuPsf_DE);

        if (!(s % 1) || s == Ns - 1) { std::cout << s << " " << std::endl; }
        for (long ei = 0; ei < 1; ++ei)
        {

            std::cout << dir_points.block<3, 17>(0, 17 * 101) << std::endl << std::endl;
            Tensor3d integrand_evals = sphere_pix_upsf(dir_points, ei);
            // [Ne, Nevts]
            auto [value, error]      = Genz::result_err(integrand_evals);

            //                  .slice(Idx2 { 0, 1 }, Idx2 { 17, 1 })
            std::cout << integrand_evals.slice(Idx3 { 0, 0, 101 }, Idx3 { 1, 17, 1 })
                      << std::endl;
            std::cout << value.slice(Idx2 { 0, 101 }, Idx2 { 1, 1 }) << std::endl;
            std::cout << error.slice(Idx2 { 0, 101 }, Idx2 { 1, 1 }) << std::endl;

            // size_t not_converged_count = not_converged.size();
            size_t iteration_depth = 1;
            while (iteration_depth < 2)
            {
                // Determine which regions are converged
                auto [converged, not_converged]
                    = Genz::converged_indices(value, error, ftol_threshold);

                // // Accumulate converged region results into correct event
                // Map<MatrixXd> valM(value.data(), Ne, value.dimension(1));
                // Map<MatrixXd> errM(error.data(), Ne, error.dimension(1));
                // result_value(Eigen::all, evtidx(converged))
                //     += valM(Eigen::all, converged);
                // result_error(Eigen::all, evtidx(converged))
                //     += errM(Eigen::all, converged);

                long const Nucnv = not_converged.size();
                // std::cout << Nucnv << " " << error.maximum() << std::endl;
                if (Nucnv == 0) { break; }

                // # nmask.shape [ regions_events ]
                // nmask = ~cmask
                // center, halfwidth, vol = region.split(
                //     center[:, nmask], halfwidth[:, nmask], vol[nmask],
                //     split_dim[nmask]
                // )

                Tensor2d hwUcnv(2, Nucnv);
                Map<MatrixXd>(hwUcnv.data(), 2, Nucnv) = Map<MatrixXd>(
                    halfwidth.data(), 2, Nevts)(Eigen::all, not_converged);

                Tensor2d volUcnv(1, Nucnv);
                Map<MatrixXd>(volUcnv.data(), 1, Nucnv)
                    = Map<MatrixXd const>(volume.data(), 1, Nevts)(0, not_converged);

                Tensor1byt split_dim = Genz::split_dims(
                    integrand_evals, error, hwUcnv, volUcnv, not_converged);

                Tensor2d centerUcnv(2, Nucnv);
                Map<MatrixXd>(centerUcnv.data(), 2, Nucnv)
                    = Map<MatrixXd>(center.data(), 2, Nevts)(Eigen::all, not_converged);

                // evtidx = np.tile(evtidx[nmask], 2)
                Eigen::VectorX<Eigen::Index> uevx = evtidx(not_converged);
                evtidx.resize(Nucnv * 2);
                evtidx << uevx, uevx;

                center.resize(2, Nucnv * 2);
                halfwidth.resize(2, Nucnv * 2);
                volume.resize(1, Nucnv * 2);

                Genz::region_split(
                    center, halfwidth, volume, split_dim, centerUcnv, hwUcnv, volUcnv);

                points                 = Genz::fullsym(center,
                                       halfwidth * Genz::alpha2,
                                       halfwidth * Genz::alpha4,
                                       halfwidth * Genz::alpha5);
                integrand_evals        = sphere_pix_upsf(get_dir_points(points), ei);
                std::tie(value, error) = Genz::result_err(integrand_evals, volume);

                ++iteration_depth;
            }
            std::cout << " [" << iteration_depth << "] " << std::endl;

            break;
        }
        break;
    }

    return xtpsfEst;
}
