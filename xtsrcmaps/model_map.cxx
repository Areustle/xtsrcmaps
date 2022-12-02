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
#include <Eigen/Dense>

auto
Fermi::ModelMap::get_init_points(long const Nh, long const Nw) -> Tensor3d
{
    Tensor3d init_points(2, Nh, Nw);
    for (long w = 0; w < Nw; ++w)
    {
        for (long h = 0; h < Nh; ++h)
        {
            init_points(0, h, w) = 1. + h;
            init_points(1, h, w) = 1. + w;
        }
    }

    return init_points;
}

// astro::SkyDir A(*lon, *lat, proj);
// astro::SkyDir B(*lon, *(lat+1), proj);
// astro::SkyDir C(*(lon+1), *(lat+1), proj);
// astro::SkyDir D(*(lon+1), *lat, proj);
//
// double solidAngle(st_facilities::FitsImage::solidAngle(A, B, C, D));
// // Approximation to the pixel solid angle:  Divide into two
// // triangles and compute the area as if the space were flat.
//    double dOmega1 = A.difference(B)*A.difference(D)
//       *((A()-B()).unit().cross((A() - D()).unit())).mag();
//
//    double dOmega2 = C.difference(B)*C.difference(D)
//       *((C()-B()).unit().cross((C() - D()).unit())).mag();
//
//    return (dOmega1 + dOmega2)/2.;

auto
solid_angle(Tensor3d const& points, Fermi::SkyGeom const& skygeom) -> Tensor2d
{
    Tensor2d phi(points.dimension(1), points.dimension(2));
    for (int w = 0; w < points.dimension(2); ++w)
    {
        for (int h = 0; h < points.dimension(1); ++h)
        {
            // Adapted from FermiTools CountsMap.cxx:612 and FitsImage.cxx:108
            coord3 a = skygeom.pix2dir({ points(0, h, w), points(1, h, w) });
            coord3 b = skygeom.pix2dir({ points(0, h, w), points(1, h, w) + 1 });
            coord3 c = skygeom.pix2dir({ points(0, h, w) + 1, points(1, h, w) + 1 });
            coord3 d = skygeom.pix2dir({ points(0, h, w) + 1, points(1, h, w) });

            // Eigen::Vector3d const A(std::get<0>(a), std::get<1>(a), std::get<2>(a));
            // Eigen::Vector3d const B(std::get<0>(b), std::get<1>(b), std::get<2>(b));
            // Eigen::Vector3d const C(std::get<0>(c), std::get<1>(c), std::get<2>(c));
            // Eigen::Vector3d const D(std::get<0>(d), std::get<1>(d), std::get<2>(d));

            Map<Eigen::Vector3d> A(&std::get<0>(a));
            Map<Eigen::Vector3d> B(&std::get<0>(b));
            Map<Eigen::Vector3d> C(&std::get<0>(c));
            Map<Eigen::Vector3d> D(&std::get<0>(d));

            auto diff
                = [](Eigen::Vector3d const& L, Eigen::Vector3d const& R) -> double {
                return 2. * asin(0.5 * (L - R).norm());
            };

            double dOmega1 = diff(A, B) * diff(A, D)
                             * (A - B).normalized().cross((A - D).normalized()).norm();

            double dOmega2 = diff(C, B) * diff(C, D)
                             * (C - B).normalized().cross((C - D).normalized()).norm();
            phi(h, w) = 0.5 * (dOmega1 - dOmega2);
        }
    }
    return phi;
}

auto
Fermi::ModelMap::spherical_direction_of_pixels(Tensor3d const& points,
                                               SkyGeom const&  skygeom) -> Array3Xd
{
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

auto
Fermi::ModelMap::psf_fast_lut(Array3Xd const& points3,
                              ArrayXd const&  src_d,
                              Tensor2d const& tuPsf_ED) -> Tensor3d
{
    // Dimensions
    long const Npts = points3.cols();
    long const Ne   = tuPsf_ED.dimension(0);

    // Given sample points on the sphere in 3-direction-space, compute the
    // separation.
    auto diff       = points3.colwise() - src_d;
    auto mag        = diff.colwise().norm();
    auto scaled_off = 1e4 * 2. * rad2deg * Eigen::asin(0.5 * mag);
    // From the separation, use the logarithmic interpolation trick to get the index
    // value
    ArrayXXd const separation_index
        = (scaled_off < 1.).select(scaled_off, 1. + (scaled_off.log() / sep_step));
    TensorMap<Tensor1d const> const idxs(separation_index.data(), Npts);

    // Sample the PSF lookup table by finding all neighboring elements which share a
    // Table column by virtue of having the same separation index. Then use tensor
    // contraction to contract all of these together with the correct alpha multiplier
    // of the psf values.

    // Allocate a result buffer
    Tensor3d vals(Ne, Fermi::Genz::Ncnt, Npts / Fermi::Genz::Ncnt);

    // iterate over every point
    long i = 0;
    while (i < Npts)
    {

        // Lookup table's separation index.
        double const index = std::floor(idxs(i));
        // run length of points which share a separation index.
        long len           = 1;
        // Iterate sequential points until a new index value is seen
        while ((i + len < Npts) && index == std::floor(idxs(i + len))) { ++len; }
        // Get a view Linear of the same-separation points.
        TensorMap<Tensor1d const> const ss(idxs.data() + i, len);
        // Compute the interpolation weights for every ss point.
        Tensor2d weights(len, 2);
        TensorMap<Tensor1d>(weights.data() + len, len) = ss - index;
        TensorMap<Tensor1d>(weights.data(), len)
            = 1. - TensorMap<Tensor1d>(weights.data() + len, len);

        // Get a view of the psf lookup table.
        TensorMap<Tensor2d const> const psf(tuPsf_ED.data() + long(index) * Ne, Ne, 2);
        // Contract the weights with the lookup table entries, thereby computing the
        // PSF values for every energy in the table and every ss point.
        // Write the Energies into the result buffer via a veiw.
        TensorMap<Tensor2d>(vals.data() + i * Ne, Ne, len)
            = psf.contract(weights, IdxPair1 { { { 1, 1 } } });

        // Shift the target point by the length of ss points to ensure we start at an
        // unseen point
        i += len;
    }
    return vals;
}


auto
Fermi::ModelMap::point_src_model_map_wcs(long const      Nw,
                                         long const      Nh,
                                         vpd const&      dirs,
                                         Tensor3d const& uPsf,
                                         SkyGeom const&  skygeom,
                                         double const    ftol_threshold) -> Tensor4d
{
    long const Ns    = dirs.size();
    long const Nd    = uPsf.dimension(0);
    long const Ne    = uPsf.dimension(1);
    long const Nevts = Nw * Nh;

    Tensor4d xtpsfEst(Ne, Nh, Nw, Ns);
    xtpsfEst.setZero();

    Tensor3d const init_points              = get_init_points(Nh, Nw);
    // Pixel centers, and doubles halfwidth, volume
    auto const [centers, halfwidth, volume] = Genz::pixel_region(init_points);

    // Pixel points with minor Genz purturbations.
    Tensor3d const genz_points              = Genz::fullsym(centers,
                                               halfwidth * Genz::alpha2,
                                               halfwidth * Genz::alpha4,
                                               halfwidth * Genz::alpha5);

    // Transform functor for genz_points to spherical 3-points
    auto get_dir_points = [&skygeom](Tensor3d const& p) -> Array3Xd {
        return spherical_direction_of_pixels(p, skygeom);
    };

    Array3Xd const dir_points = get_dir_points(genz_points);

    for (long s = 0; s < Ns; ++s)
    {
        // Get a slice of the PSF lookup table just for this sources piece of the
        // table.
        Tensor2d const tuPsf_ED = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 })
                                      .reshape(Idx2 { Nd, Ne })
                                      .shuffle(Idx2 { 1, 0 });

        // Get the sources coordinate in 3-direction space.
        auto           src_dir = skygeom.sph2dir(dirs[s]); // CLHEP Style 3
        Eigen::ArrayXd src_d(3, 1);
        src_d << std::get<0>(src_dir), std::get<1>(src_dir), std::get<2>(src_dir);

        /******************************************************************
         * Psf Energy values for sample points in direction space
         ******************************************************************/
        auto integrand = [&src_d, &tuPsf_ED](Array3Xd const& points3) {
            return psf_fast_lut(points3, src_d, tuPsf_ED);
        };

        // View of the results buffer
        Map<MatrixXd> result_value(xtpsfEst.data() + s * Ne * Nevts, Ne, Nevts);

        // The Genz Malik Integration rule adapted for this problem.
        Genz::integrate_region(integrand,
                               get_dir_points,
                               result_value,
                               centers,
                               halfwidth,
                               volume,
                               dir_points,
                               ftol_threshold);
    }

    return xtpsfEst;
}
