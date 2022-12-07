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
                              Tensor2d const& table) -> Tensor3d
{
    // Dimensions
    long const Npts           = points3.cols();
    long const Ne             = table.dimension(0);
    long const Nevts          = Npts / Genz::Ncnt;

    // Given sample points on the sphere in 3-direction-space, compute the
    // separation.
    auto diff                 = points3.colwise() - src_d;
    auto mag                  = diff.colwise().norm();
    auto off                  = 2. * rad2deg * Eigen::asin(0.5 * mag);

    // From the spherical offset, use logarithmic interpolation to get the index val
    auto           scaled_off = 1e4 * off;
    ArrayXXd const separation_index
        = (scaled_off < 1.).select(scaled_off, 1. + (scaled_off.log() / sep_step));
    TensorMap<Tensor1d const> const idxs(separation_index.data(), Npts);

    // Sample the PSF lookup table by finding sequential elements which share a
    // table column (by virtue of having the same separation index). Then use tensor
    // contraction to contract all of these together with the correct alpha multiplier
    // of the psf values.

    // Allocate a result buffer [Ne, 17, Nevts]
    // Tensor3d vals(Ne, Genz::Ncnt, Npts / Genz::Ncnt);
    // Allocate a result buffer [17, Ne, Nevts]
    Tensor3d vals(Genz::Ncnt, Ne, Nevts);

    // iterate over every point
    long i = 0;
    while (i < Npts)
    {

        // Lookup table's separation index.
        double const index = std::floor(idxs(i));
        // run length of points which share a separation index.
        long Nlen          = 1;
        // Iterate sequential points until a new index value is seen
        while ((i + Nlen < Npts) && index == std::floor(idxs(i + Nlen))) { ++Nlen; }
        // Get a view Linear of the same-separation points.
        TensorMap<Tensor1d const> const ss(idxs.data() + i, Nlen);
        // Compute the interpolation weights for every ss point.
        Tensor2d weights(Nlen, 2);
        TensorMap<Tensor1d>(weights.data() + Nlen, Nlen) = ss - index;
        TensorMap<Tensor1d>(weights.data(), Nlen)
            = 1. - TensorMap<Tensor1d>(weights.data() + Nlen, Nlen);

        // Get a view of the psf lookup table.
        TensorMap<Tensor2d const> const lut(table.data() + long(index) * Ne, Ne, 2);
        // Contract the weights with the lookup table entries, thereby computing the
        // PSF values for every energy in the table and every ss point.
        // [Ne, Nlen]
        Tensor2d vv = lut.contract(weights, IdxPair1 { { { 1, 1 } } });

        // // Write the Energies into the result buffer via a veiw.
        // TensorMap<Tensor2d>(vals.data() + i * Ne, Ne, Nlen) = vv;
        for (long j = 0; j < Nlen; ++j)
        {
            long evoff = (i + j) / Genz::Ncnt;
            long gzoff = (i + j) % Genz::Ncnt;

            // (vals.data() + (evoff * Ne * Genz::Ncnt) + gzoff) = 0.;
            for (long k = 0; k < Ne; ++k) { vals(gzoff, k, evoff) = vv(k, j); }
        }

        // Shift the target point by the length of ss points to ensure we start at an
        // unseen point
        i += Nlen;
    }
    return vals;
}


auto
Fermi::ModelMap::pixel_mean_psf(long const      Nh,
                                long const      Nw,
                                vpd const&      dirs,
                                Tensor3d const& psf_lut,
                                SkyGeom const&  skygeom,
                                double const    ftol_threshold) -> Tensor4d
{
    long const Ns    = dirs.size();
    long const Nd    = psf_lut.dimension(0);
    long const Ne    = psf_lut.dimension(1);
    long const Nevts = Nh * Nw;

    Tensor4d xtpsf(Ne, Nh, Nw, Ns);
    xtpsf.setZero();

    // Compute initial (lon,lat) pairs of pixel center points.
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
        // auto t0                 = std::chrono::high_resolution_clock::now();
        // A slice of the PSF table just for this source's segment of the table.
        Tensor2d const tuPsf_ED = psf_lut.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 })
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
        Map<MatrixXd> result_value(xtpsf.data() + s * Ne * Nevts, Ne, Nevts);

        // auto t1 = std::chrono::high_resolution_clock::now();
        // The Genz Malik Integration rule adapted for this problem.
        Genz::integrate_region(integrand,
                               get_dir_points,
                               result_value,
                               centers,
                               halfwidth,
                               volume,
                               dir_points,
                               ftol_threshold);
        // auto t2  = std::chrono::high_resolution_clock::now();
        // auto d10 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        // auto d21 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        // std::cout << " mm: [" << d10 << " " << d21 << "] " << std::flush;
        // std::cout << std::endl;
    }

    return xtpsf;
}

auto
sph_diff(Eigen::Vector3d const& L, Eigen::Vector3d const& R) -> double
{
    return 2. * asin(0.5 * (L - R).norm());
};

auto
sph_diff(Eigen::Vector2d const& L,
         Eigen::Vector2d const& R,
         Fermi::SkyGeom const&  skygeom) -> double
{
    coord3 a = skygeom.pix2dir({ L(0), L(1) });
    coord3 b = skygeom.pix2dir({ R(0), R(1) });
    return sph_diff({ std::get<0>(a), std::get<1>(a), std::get<2>(a) },
                    { std::get<0>(b), std::get<1>(b), std::get<2>(b) });
};

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

            Eigen::Vector3d const A(std::get<0>(a), std::get<1>(a), std::get<2>(a));
            Eigen::Vector3d const B(std::get<0>(b), std::get<1>(b), std::get<2>(b));
            Eigen::Vector3d const C(std::get<0>(c), std::get<1>(c), std::get<2>(c));
            Eigen::Vector3d const D(std::get<0>(d), std::get<1>(d), std::get<2>(d));

            double dOmega1 = sph_diff(A, B) * sph_diff(A, D)
                             * (A - B).normalized().cross((A - D).normalized()).norm();

            double dOmega2 = sph_diff(C, B) * sph_diff(C, D)
                             * (C - B).normalized().cross((C - D).normalized()).norm();
            phi(h, w) = 0.5 * (dOmega1 - dOmega2);
        }
    }
    return phi;
}

void
Fermi::ModelMap::scale_psf_by_solid_angle(Tensor4d& pixpsf, SkyGeom const& skygeom)
{
    long const Ne              = pixpsf.dimension(0);
    long const Nh              = pixpsf.dimension(1);
    long const Nw              = pixpsf.dimension(2);
    long const Ns              = pixpsf.dimension(3);

    Tensor3d const init_points = get_init_points(Nh, Nw);
    // Compute solid angle for the pixel center points and scale PSF by it.
    pixpsf *= solid_angle(init_points, skygeom)
                  .reshape(Idx4 { 1, Nh, Nw, 1 })
                  .broadcast(Idx4 { Ne, 1, 1, Ns });
}

inline auto
point_nearest_to_source_on_segment(Eigen::Vector2d const& v,
                                   Eigen::Vector2d const& p0,
                                   Eigen::Vector2d const& p1,
                                   double const&          c1,
                                   double const&          c2) -> Eigen::Vector2d
{
    // v = p1 - p0
    // if ((w.v)=c1 <= 0) then before P0 return p0
    // if ((v.v)=c2 <= c1) then after P1 return p1
    // pb = p0+b*v
    return c1 <= 0. ? p0 : c2 <= c1 ? p1 : (p0 + (c1 / c2) * v);
}

auto
Fermi::ModelMap::compute_psf_map_corrections(Tensor4d const& pixpsf,
                                             vpd const&      src_dirs,
                                             SkyGeom const&  skygeom)
    -> std::tuple<Tensor2d, std::vector<double>>
{
    long const Ne = pixpsf.dimension(0);
    long const Nh = pixpsf.dimension(1);
    long const Nw = pixpsf.dimension(2);
    long const Ns = pixpsf.dimension(3);

    // ................
    // ...a ----- d....
    // ...|ooooooo|....
    // ...|ooooooo|....
    // ...b ----- c....
    // ................

    // Determine which sources are in the PSF radius
    auto a        = skygeom.sph2pix({ 0.5, 0.5 });
    auto b        = skygeom.sph2pix({ Nh + 0.5, 0.5 });
    auto c        = skygeom.sph2pix({ Nh + 0.5, Nw + 0.5 });
    auto d        = skygeom.sph2pix({ 0.5, Nw + 0.5 });

    Vector2d const A(std::get<0>(a), std::get<1>(a));
    Vector2d const B(std::get<0>(b), std::get<1>(b));
    Vector2d const C(std::get<0>(c), std::get<1>(c));
    Vector2d const D(std::get<0>(d), std::get<1>(d));

    Vector2d const AB  = B - A;
    Vector2d const AD  = D - A;
    Vector2d const CB  = B - C;
    Vector2d const CD  = D - C;

    double const lenAB = AB.dot(AB);
    double const lenAD = AD.dot(AD);
    double const lenCB = CB.dot(CB);
    double const lenCD = CD.dot(CD);

    Tensor2d MapIntegral(Ne, Ns);
    MapIntegral.setZero();
    std::vector<double> min_psf_radius(Ns);
    auto                src_pts_pix = skygeom.sph2pix(src_dirs);
    for (long s = 0; s < Ns; ++s)
    {
        // Simple geometric trick to determine if the source point is bounded by the
        // convex hull of our spherically warped field of view.
        Vector2d const S(std::get<0>(src_pts_pix[s]), std::get<1>(src_pts_pix[s]));
        Vector2d const AS        = S - A;
        Vector2d const CS        = S - C;
        double const   AS_AB     = AS.dot(AB);
        double const   AS_AD     = AS.dot(AD);
        double const   CS_CB     = CS.dot(CB);
        double const   CS_CD     = CS.dot(CD);
        bool const     is_in_fov = 0. <= AS_AB && AS_AB <= lenAB //
                               && 0. <= AS_AD && AS_AD <= lenAD  //
                               && 0. <= CS_CB && CS_CB <= lenCB  //
                               && 0. <= CS_CD && CS_CD <= lenCD; //

        // Source isn't in the field of view so no psf correction.
        if (!is_in_fov) { break; }

        // Points on boundary of FOV nearest to the source.
        Vector2d pSAB    = point_nearest_to_source_on_segment(AB, A, B, AS_AB, lenAB);
        Vector2d pSAD    = point_nearest_to_source_on_segment(AD, A, D, AS_AD, lenAD);
        Vector2d pSCB    = point_nearest_to_source_on_segment(CB, C, B, CS_CB, lenCB);
        Vector2d pSCD    = point_nearest_to_source_on_segment(CD, C, D, CS_CD, lenCD);

        // Distance between the source and the boundary lines of the field of view;
        double const dAB = sph_diff(S, pSAB, skygeom);
        double const dAD = sph_diff(S, pSAD, skygeom);
        double const dCB = sph_diff(S, pSCB, skygeom);
        double const dCD = sph_diff(S, pSCD, skygeom);

        double min_rad   = dAB < dAD ? dAB : dAD;
        min_rad          = min_rad < dCB ? min_rad : dCB;
        min_rad          = min_rad < dCD ? min_rad : dCD;
        min_rad *= R2D;

        // Accumulate
        for (long w = 0; w < Nw; ++w)
        {
            for (long h = 0; h < Nh; ++h)
            {
                if (sph_diff(S, Vector2d(h + 1., w + 1.), skygeom) * R2D <= min_rad)
                {
                    MapIntegral.slice(Idx2 { 0, s }, Idx2 { Ne, 1 })
                        .reshape(Idx4 { Ne, 1, 1, 1 })
                        += pixpsf.slice(Idx4 { 0, h, w, s }, Idx4 { Ne, 1, 1, 1 });
                }
            }
        }

        min_psf_radius[s] = min_rad;
    }

    return { MapIntegral, min_psf_radius };
}

auto
Fermi::ModelMap::point_src_model_map_wcs(long const      Nh,
                                         long const      Nw,
                                         vpd const&      src_dirs,
                                         Tensor3d const& uPsf,
                                         SkyGeom const&  skygeom,
                                         double const    ftol_threshold) -> Tensor4d
{
    Tensor4d xtpsf = pixel_mean_psf(Nh, Nw, src_dirs, uPsf, skygeom, ftol_threshold);
    // scale_psf_by_solid_angle(xtpsf, skygeom);

    return xtpsf;
}
