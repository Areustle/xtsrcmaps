#include "xtsrcmaps/model_map.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/fmt_source.hxx"
// #include "xtsrcmaps/genz_malik.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"

#include <cmath>

#include "fmt/format.h"
#include "unsupported/Eigen/CXX11/Tensor"

// auto
// psf_sample_full_energy(ArrayXd const& src_d, Tensor2d const& tuPsf_ED) -> auto
// {
//
//     long const Ne = tuPsf_ED.dimension(0);
//
//     /******************************************************************
//      * Psf Energy values for sample points in direction space
//      ******************************************************************/
//     return [&, Ne](Array3Xd const& points3) -> Tensor3d {
//         long const Npts     = points3.size() / 3;
//         // Given sample points on the sphere in 3-direction-space, compute the
//         // separation.
//         auto           diff = points3.colwise() - src_d;
//         auto           mag  = diff.colwise().norm();
//         auto           off  = 2. * rad2deg * Eigen::asin(0.5 * mag);
//         ArrayXXd const separation_index
//             = (off < 1e-4).select(1e4 * off, 1. + ((off * 1e4).log() / sep_step));
//         TensorMap<Tensor1d const> const idxs(separation_index.data(), Npts);
//         Tensor3d vals(Ne, Fermi::Genz::Ncnt, Npts / Fermi::Genz::Ncnt);
//
//         long i = 0;
//         while (i < Npts)
//         {
//             long         d  = 1;
//             double const x1 = std::floor(idxs(i));
//             while ((i + d < Npts) && x1 == std::floor(idxs(i + d))) { ++d; }
//             TensorMap<Tensor1d const> const ss(idxs.data() + i, d);
//             Tensor2d                        alpha(d, 2);
//             TensorMap<Tensor1d>(alpha.data(), d) = ss - x1;
//             TensorMap<Tensor1d>(alpha.data() + d, d)
//                 = 1. - TensorMap<Tensor1d>(alpha.data(), d);
//             TensorMap<Tensor2d const> const psf(tuPsf_ED.data() + long(x1) * Ne,//
//                                                 Ne, 2);
//             Tensor2d const vv = psf.contract(alpha, IdxPair1 { { { 1, 1 } } });
//             TensorMap<Tensor2d>(vals.data() + i * Ne, Ne, d) = vv;
//             i += d;
//         }
//         return vals;
//     };
// }


// auto
// psf_sample_single_energy(ArrayXd const& src_d, Tensor2d const& tuPsf_DE) -> auto
// {
//
//     long const Nd = tuPsf_DE.dimension(0);
//
//     /******************************************************************
//      * Psf Energy values for sample points in direction space
//      ******************************************************************/
//     return [&src_d, &tuPsf_DE, Nd](
//                Array3Xd const& points3, long const ei, long const Nn = 1)
//                -> Tensor2d {
//         long const Npts     = points3.size() / 3;
//         // Given sample points on the sphere in 3-direction-space, compute the
//         // separation.
//         auto           diff = points3.colwise() - src_d;
//         auto           mag  = diff.colwise().norm();
//         auto           off  = 2. * rad2deg * Eigen::asin(0.5 * mag);
//         ArrayXXd const separation_index
//             = (off < 1e-4).select(1e4 * off, 1. + ((off * 1e4).log() / sep_step));
//         TensorMap<Tensor1d const> const idxs(separation_index.data(), Npts);
//
//         Tensor2d vals(Nn, Npts / Nn); // Nn, Nevts
//
//         long i = 0;
//         while (i < Npts)
//         {
//             long         d  = 1;
//             double const x1 = std::floor(idxs(i));
//             while ((i + d < Npts) && x1 == std::floor(idxs(i + d))) { ++d; }
//             TensorMap<Tensor1d const> const ss(idxs.data() + i, d);
//             Tensor2d                        alpha(d, 2);
//             TensorMap<Tensor1d>(alpha.data(), d) = ss - x1;
//             TensorMap<Tensor1d>(alpha.data() + d, d)
//                 = 1. - TensorMap<Tensor1d>(alpha.data(), d);
//
//             TensorMap<Tensor2d const> const psf(
//                 tuPsf_DE.data() + long(x1) + Nd * ei, 2, 1);
//
//             Tensor2d const vv = psf.contract(alpha, IdxPair1 { { { 0, 1 } } });
//
//             TensorMap<Tensor2d>(vals.data() + i, 1, d) = vv;
//             i += d;
//         }
//         return vals;
//     };
// }

auto
Fermi::ModelMap::separation(Array3Xd const& points3, ArrayXd const& src_d) -> ArrayXd
{
    // Given sample points on the sphere in 3-direction-space, compute the
    // separation.
    auto    diff = points3.colwise() - src_d;
    auto    mag  = diff.colwise().norm();
    ArrayXd off  = 2. * rad2deg * Eigen::asin(0.5 * mag);
    return off;
}

auto
Fermi::ModelMap::index_from_sep(ArrayXd const& off) -> Tensor1d
{
    long const Npts = off.size();
    ArrayXXd   sep_idx
        = (off < 1e-4).select(1e4 * off, 1. + (1e4 * off).log() / sep_step);
    TensorMap<Tensor1d> idxs(sep_idx.data(), Npts);
    Tensor1d            rval = idxs;
    return rval;
}

auto
Fermi::ModelMap::separation_indices(Array3Xd const& points3, ArrayXd const& src_d)
    -> Tensor1d
{
    return index_from_sep(separation(points3, src_d));
}

auto
Fermi::ModelMap::psf_single_energy(Tensor1d const& idxs,
                                   Tensor2d const& tuPsf_DE,
                                   long const      ei,
                                   long const      Nn) -> Tensor2d
{
    long const Nd   = tuPsf_DE.dimension(0);
    long const Npts = idxs.size();
    // Use separation indexes to sample psf
    Tensor2d vals(Nn, Npts / Nn); // Nn, Nevts

    long i = 0;
    while (i < Npts)
    {
        long         d  = 1;
        double const x1 = std::floor(idxs(i));
        while ((i + d < Npts) && x1 == std::floor(idxs(i + d))) { ++d; }
        TensorMap<Tensor1d const> const ss(idxs.data() + i, d);
        Tensor2d                        alpha(d, 2);
        TensorMap<Tensor1d>(alpha.data() + d, d) = ss - x1;
        TensorMap<Tensor1d>(alpha.data(), d)
            = 1. - TensorMap<Tensor1d>(alpha.data() + d, d);

        TensorMap<Tensor2d const> const psf(tuPsf_DE.data() + long(x1) + Nd * ei, 2, 1);

        Tensor2d const vv = psf.contract(alpha, IdxPair1 { { { 0, 1 } } });

        TensorMap<Tensor2d>(vals.data() + i, 1, d) = vv;

        // std::cout << "i " << i << " "
        //           << "d " << d << std::endl
        //           << "ss " << std::endl
        //           << ss.reshape(Idx2 { 1, d }) << std::endl
        //           << "alpha " << std::endl
        //           << alpha.shuffle(Idx2 { 1, 0 }) << std::endl
        //           << "psf " << std::endl
        //           << psf << std::endl
        //           << "vv " << std::endl
        //           << vv << std::endl
        //           << " "
        //           //
        //           << std::endl;
        i += d;
    }

    // std::cout << "=================================================" << std::endl;
    return vals;
}

auto
Fermi::ModelMap::get_dir_points(Tensor3d const& points, Fermi::SkyGeom const& skygeom)
    -> Array3Xd
{
    long const Nn    = points.dimension(1);
    long const Nevts = points.dimension(2);
    Array3Xd   dir_points(3, Nn * Nevts);
    for (long j = 0; j < Nevts; ++j)
    {
        for (long i = 0; i < Nn; ++i)
        {
            coord3 p = skygeom.pix2dir({ points(0, i, j), points(1, i, j) });
            dir_points(0, i + Nn * j) = std::get<0>(p);
            dir_points(1, i + Nn * j) = std::get<1>(p);
            dir_points(2, i + Nn * j) = std::get<2>(p);
        }
    }
    return dir_points;
}

auto
Fermi::ModelMap::sub_pixel_points(MatrixXd const& points, short const iteration_depth)
    -> Tensor3d
{
    assert(points.rows() == 2);
    long const Nevts = points.cols();
    long const N     = 1 << iteration_depth;
    auto       steps = std::vector<double>(N);
    for (short i = 0; i < N; ++i) { steps[i] = (2. * i - double(N - 1)) / (2. * N); }

    Tensor3d sub_points(2, N * N, Nevts);
    for (long k = 0; k < Nevts; ++k)
    {
        for (long i = 0; i < N; ++i)
        {
            for (long j = 0; j < N; ++j)
            {
                sub_points(0, j + i * N, k) = points(0, k) + steps[j];
                sub_points(1, j + i * N, k) = points(1, k) + steps[i];
            }
        }
    }
    return sub_points;
}

auto
step_point_count(long const iteration_depth) -> long
{
    return (1 << (2 * iteration_depth));
}

auto
cumulative_point_count(long const iteration_depth) -> long
{
    return ((1 << (2 * (iteration_depth + 1))) - 1) / 3;
}

auto
relative_error(Tensor1d const& S0, Tensor1d const& S1, long const iteration_depth)
    -> Tensor1d
{
    long const Nnp1 = cumulative_point_count(iteration_depth);
    long const Nn   = cumulative_point_count(iteration_depth - 1);

    return (1. - ((Nnp1 * S0) / (Nn * S1))).abs();
}


auto
converged_indices(Tensor1d const& relerr, double const ftol_threshold)
    -> std::tuple<std::vector<long>, std::vector<long>>
{
    auto converged     = std::vector<long> {};
    auto not_converged = std::vector<long> {};

    for (long i = 0; i < relerr.size(); ++i)
    {
        if (relerr(i) > ftol_threshold) { not_converged.push_back(i); }
        else { converged.push_back(i); }
    }

    return { converged, not_converged };
}

auto
Fermi::ModelMap::get_init_points(long const Nh, long const Nw) -> MatrixXd
{
    MatrixXd init_points(2, Nh * Nw);
    for (long w = 0; w < Nw; ++w)
    {
        for (long h = 0; h < Nh; ++h)
        {
            init_points(0, h + w * Nh) = 1. + h;
            init_points(1, h + w * Nh) = 1. + w;
        }
    }

    return init_points;
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

    Tensor4d xtpsfEst(Nw, Nh, Ne, Ns);

    MatrixXd const init_points = get_init_points(Nh, Nw);
    Array3Xd const init_x0 = get_dir_points(sub_pixel_points(init_points, 0), skygeom);

    for (long s = 0; s < 64; ++s)
    {

        MatrixXd result_value(Ne, Nevts);
        result_value.setZero();

        // Tensor2d const tuPsf_ED = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 })
        //                               .reshape(Idx2 { Nd, Ne })
        //                               .shuffle(Idx2 { 1, 0 });
        Tensor2d const tuPsf_DE
            = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 }).reshape(Idx2 { Nd, Ne });

        auto           src_dir = skygeom.sph2dir(dirs[s]); // CLHEP Style 3
        Eigen::ArrayXd src_d(3, 1);
        src_d << std::get<0>(src_dir), std::get<1>(src_dir), std::get<2>(src_dir);

        /******************************************************************
         * Psf Energy values for sample points in direction space
         ******************************************************************/

        if (!(s % 25) || s == Ns - 1) { std::cout << s << " " << std::endl; }
        for (long ei = 0; ei < Ne; ++ei)
        {
            // Index of events.
            IdxVec   evtidx        = IdxVec::LinSpaced(Nevts, 0, Nevts - 1);
            MatrixXd points        = init_points;
            Tensor1d idxs0         = separation_indices(init_x0, src_d);
            Tensor2d y0            = psf_single_energy(idxs0, tuPsf_DE, ei, 1);
            Tensor1d S0            = y0.sum(Idx1 { 0 });

            size_t iteration_depth = 1;
            while (iteration_depth < 7)
            {
                long const Nn_ = step_point_count(iteration_depth);
                Array3Xd x1 = get_dir_points(sub_pixel_points(points, iteration_depth),
                                             skygeom);
                Tensor1d idxs1 = separation_indices(x1, src_d);
                Tensor2d y1    = psf_single_energy(idxs1, tuPsf_DE, ei, Nn_);
                Tensor1d S1    = y1.sum(Idx1 { 0 });
                S1 += S0;
                Tensor1d relerror = relative_error(S0, S1, iteration_depth);
                // Determine which regions are converged
                auto [converged, not_converged]
                    = converged_indices(relerror, ftol_threshold);
                // Accumulate converged region results into correct event
                Map<VectorXd> valM(S1.data(), S1.dimension(0));
                result_value(ei, evtidx(converged)) = valM(converged) / double(Nn_);
                long const Nucnv                    = not_converged.size();
                if (Nucnv == 0) { break; }
                S0.resize(not_converged.size());
                Map<VectorXd> S0M(S0.data(), S0.dimension(0));
                Map<VectorXd> S1M(S1.data(), S1.dimension(0));
                S0M                       = S1M(not_converged);

                IdxVec unconverged_evtidx = evtidx(not_converged);
                evtidx.resize(unconverged_evtidx.size());
                evtidx << unconverged_evtidx;

                MatrixXd unconverged_points = points(Eigen::all, not_converged);
                points.resize(2, unconverged_points.cols());
                points << unconverged_points;

                ++iteration_depth;
            }
        }
    }

    return xtpsfEst;
}
