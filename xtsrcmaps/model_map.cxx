#include "xtsrcmaps/model_map.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/fmt_source.hxx"
#include "xtsrcmaps/genz_malik.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

#include <cmath>

#include "fmt/format.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include <Eigen/Dense>

auto
Fermi::ModelMap::point_src_model_map_wcs(
    long const      Nh,
    long const      Nw,
    vpd const&      dirs,
    Tensor3d const& uPsf,
    SkyGeom const&  skygeom,
    Tensor2d const& exposure,
    Tensor3d const& partial_integrals, /* [D,E,S] */
    double const    ftol_threshold) -> Tensor4d
{
    // Use Genz Malik Integration Scheme to compute the per-pixel average PSF value for
    // every source and energy level.
    Tensor4d model_map = pixel_mean_psf(Nh, Nw, dirs, uPsf, skygeom, ftol_threshold);
    // Scale the model_map by the central solid angle of every pixel.
    scale_map_by_solid_angle(model_map, skygeom);
    // Compute the sources in the FOV and the PSF boundary radius;
    auto [full_psf_radius, is_in_fov] = psf_boundary_radius(Nh, Nw, dirs, skygeom);
    Tensor1d const psf_radius         = filter_in(full_psf_radius, is_in_fov);
    // Compute the MapIntegral scalar for each source & energy given source location.
    Tensor2d const inv_mapinteg
        = map_integral(model_map, dirs, skygeom, psf_radius, is_in_fov);
    // Scale each map value by the exposure for this source.
    scale_map_by_exposure(model_map, exposure);

    Tensor2d const correction_factor = map_correction_factor(
        inv_mapinteg, psf_radius, is_in_fov, uPsf, partial_integrals);

    scale_map_by_correction_factors(model_map, correction_factor);

    return model_map;
}

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
            Vector3d p = skygeom.pix2dir({ points(0, i, j), points(1, i, j) });
            dir_points(Eigen::all, i + Genz::Ncnt * j) = p;
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

    // From the spherical offset, use logarithmic interpolation to get the index val.
    // Similar to implementation of Fermi::PSF::fast_separation_lower_index
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

    Tensor4d model_map(Ne, Nh, Nw, Ns);
    model_map.setZero();

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
        Vector3d src_dir = skygeom.sph2dir(dirs[s]); // CLHEP Style 3

        /******************************************************************
         * Psf Energy values for sample points in direction space
         ******************************************************************/
        auto integrand   = [&src_dir, &tuPsf_ED](Array3Xd const& points3) {
            return psf_fast_lut(points3, src_dir.array(), tuPsf_ED);
        };

        // View of the results buffer
        Map<MatrixXd> result_value(model_map.data() + s * Ne * Nevts, Ne, Nevts);

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

    return model_map;
}

// inline auto
// sph_diff(Eigen::Vector3d const& L, Eigen::Vector3d const& R) -> double
// {
//     return 2. * asin(0.5 * (L - R).norm());
// };
//
// auto
// sph_diff(Vector2d const& L, Vector2d const& R, Fermi::SkyGeom const& skygeom) ->
// double
// {
//     coord3 a = skygeom.pix2dir({ L(0), L(1) });
//     coord3 b = skygeom.pix2dir({ R(0), R(1) });
//     return sph_diff({ std::get<0>(a), std::get<1>(a), std::get<2>(a) },
//                     { std::get<0>(b), std::get<1>(b), std::get<2>(b) });
// };



auto
Fermi::ModelMap::solid_angle(Tensor3d const& points, Fermi::SkyGeom const& skygeom)
    -> Tensor2d
{
    Tensor2d phi(points.dimension(1), points.dimension(2));
    for (int w = 0; w < points.dimension(2); ++w)
    {
        for (int h = 0; h < points.dimension(1); ++h)
        {
            // Adapted from FermiTools CountsMap.cxx:612 and FitsImage.cxx:108
            Vector3d const A = skygeom.pix2dir({ points(0, h, w), points(1, h, w) });
            Vector3d const B
                = skygeom.pix2dir({ points(0, h, w), points(1, h, w) + 1. });
            Vector3d const C
                = skygeom.pix2dir({ points(0, h, w) + 1., points(1, h, w) + 1. });
            Vector3d const D
                = skygeom.pix2dir({ points(0, h, w) + 1., points(1, h, w) });

            double dOmega1 = dir_diff(A, B) * dir_diff(A, D)
                             * (A - B).normalized().cross((A - D).normalized()).norm();

            double dOmega2 = dir_diff(C, B) * dir_diff(C, D)
                             * (C - B).normalized().cross((C - D).normalized()).norm();
            phi(h, w) = 0.5 * (dOmega1 + dOmega2);
        }
    }
    return phi;
}

void
Fermi::ModelMap::scale_map_by_solid_angle(Tensor4d& model_map, SkyGeom const& skygeom)
{
    long const Ne              = model_map.dimension(0);
    long const Nh              = model_map.dimension(1);
    long const Nw              = model_map.dimension(2);
    long const Ns              = model_map.dimension(3);

    Tensor3d const init_points = get_init_points(Nh, Nw);
    // Compute solid angle for the pixel center points and scale PSF by it.
    model_map *= solid_angle(init_points, skygeom)
                     .reshape(Idx4 { 1, Nh, Nw, 1 })
                     .broadcast(Idx4 { Ne, 1, 1, Ns });
}

void
Fermi::ModelMap::scale_map_by_exposure(Tensor4d& model_map, Tensor2d const& exposure)
{
    long const Ne = model_map.dimension(0);
    long const Nh = model_map.dimension(1);
    long const Nw = model_map.dimension(2);
    long const Ns = model_map.dimension(3);

    assert(Ne == exposure.dimension(0));
    assert(Ns == exposure.dimension(1));

    model_map
        *= exposure.reshape(Idx4 { Ne, 1, 1, Ns }).broadcast(Idx4 { 1, Nh, Nw, 1 });
}


inline auto
point_nearest_to_source_on_segment(Vector2d const& v,
                                   Vector2d const& p0,
                                   Vector2d const& p1,
                                   double const&   c1,
                                   double const&   c2) -> Eigen::Vector2d
{
    // v = p1 - p0
    // if ((w.v)=c1 <= 0) then before P0 return p0
    // if ((v.v)=c2 <= c1) then after P1 return p1
    // pb = p0+b*v
    return c1 <= 0. ? p0 : c2 <= c1 ? p1 : (p0 + (c1 / c2) * v);
}

inline auto
point_nearest_to_source_on_segment(Vector2d const& p,
                                   Vector2d const& p0,
                                   Vector2d const& p1) -> Vector2d
{
    // v = p1 - p0
    Vector2d const v = p1 - p0;
    // w = p - p0
    Vector2d const w = p - p0;
    // if ((w.v)=c1 <= 0) then before P0 return p0
    double const c1  = w.dot(v);
    if (c1 <= 0)
    { /*before P0*/
        return p0;
    }
    // if ((v.v)=c2 <= c1) then after P1 return p1
    double const c2 = v.dot(v);
    if (c2 <= c1)
    { /*after P1*/
        return p1;
    }
    // b = c1 / c2
    double const b    = c1 / c2;
    // pb = p0+b*v
    Vector2d const pb = p0 + b * v;
    return pb;
    // return c1 <= 0. ? p0 : c2 <= c1 ? p1 : (p0 + (c1 / c2) * v);
}


// // ................
// // ...a ----- d....
// // ...|ooooooo|....
// // ...|ooooooo|....
// // ...b ----- c....
// // ................
//
// // Determine which sources are in the PSF radius
// // Vector2d const A(1.0, 1.0);
// // Vector2d const B(Nh, 1.0);
// // Vector2d const C(Nh, Nw);
// // Vector2d const D(1.0, Nw);
// //
// // Vector2d const AB  = B - A;
// // Vector2d const AD  = D - A;
// // Vector2d const CB  = B - C;
// // Vector2d const CD  = D - C;
// //
// // double const lenAB = AB.dot(AB);
// // double const lenAD = AD.dot(AD);
// // double const lenCB = CB.dot(CB);
// // double const lenCD = CD.dot(CD);
//
// // Tensor1d radius(Ns);
// Tensor1d radius(Ns);
// Tensor1b is_in_fov(Ns);
// // is_in_fov.setConstant(false);
//
// // auto src_pts_pix = skygeom.sph2pix(src_dirs);
//
// for (long s = 0; s < Ns; ++s)
// {
//     // // Simple geometric trick to determine if the source point is bounded by the
//     // // convex hull of our spherically warped field of view.
//     // Vector2d const S(std::get<0>(src_pts_pix[s]), std::get<1>(src_pts_pix[s]));
//     Vector2d const S(std::get<0>(src_dirs[s]), std::get<1>(src_dirs[s]));
//     // Vector2d const AS     = S - A;
//     // Vector2d const CS     = S - C;
//     // double const   AS_AB  = AS.dot(AB);
//     // double const   AS_AD  = AS.dot(AD);
//     // double const   CS_CB  = CS.dot(CB);
//     // double const   CS_CD  = CS.dot(CD);
//     // bool const     in_fov = 0. <= AS_AB && AS_AB <= lenAB //
//     //                     && 0. <= AS_AD && AS_AD <= lenAD  //
//     //                     && 0. <= CS_CB && CS_CB <= lenCB  //
//     //                     && 0. <= CS_CD && CS_CD <= lenCD; //
//     //
//     // is_in_fov(s)     = in_fov;
//     //
//     // // // Source isn't in the field of view so no psf correction.
//     // // if (!in_fov) { continue; }
//     //
//     // // Points on boundary of FOV nearest to the source.
//     // Vector2d pSAB    = point_nearest_to_source_on_segment(AB, A, B, AS_AB,
//     // lenAB); Vector2d pSAD    = point_nearest_to_source_on_segment(AD, A, D,
//     // AS_AD, lenAD); Vector2d pSCB    = point_nearest_to_source_on_segment(CB, C,
//     // B, CS_CB, lenCB); Vector2d pSCD    = point_nearest_to_source_on_segment(CD,
//     // C, D, CS_CD, lenCD);
//     //
//     // // Distance between the source and the boundary lines of the field of view;
//     // double const dAB = sph_diff(S, pSAB.array().round(), skygeom);
//     // double const dAD = sph_diff(S, pSAD.array().round(), skygeom);
//     // double const dCB = sph_diff(S, pSCB.array().round(), skygeom);
//     // double const dCD = sph_diff(S, pSCD.array().round(), skygeom);
//     //
//     // double min_rad   = dAB < dAD ? dAB : dAD;
//     // min_rad          = min_rad < dCB ? min_rad : dCB;
//     // min_rad          = min_rad < dCD ? min_rad : dCD;
//     // min_rad *= R2D;

// Compute the psf radius of each source relative to the field of view by computing the
// minimal distance between the source and each boundary segment. Sources outside
// the field of view are not included in the radius vector, and are set to false in the
// boolean vector
auto
Fermi::ModelMap::psf_boundary_radius(long const     Nh,
                                     long const     Nw,
                                     vpd const&     src_dirs,
                                     SkyGeom const& skygeom)
    -> std::pair<Tensor1d, Tensor1b>
{
    long const Ns = src_dirs.size();
    // ................
    // ...a ----- d....
    // ...|ooooooo|....
    // ...|ooooooo|....
    // ...b ----- c....
    // ................

    Tensor1d radius(Ns);
    Tensor1b is_in_fov(Ns);
    is_in_fov.setConstant(false);
    double const pix_buffer = 3.5;

    for (long s = 0; s < Ns; ++s)
    {
        double      min_deg = 360.;
        auto const& ss      = src_dirs[s];
        auto        ps      = skygeom.sph2pix(ss);
        is_in_fov(s)        = ps.first > pix_buffer && ps.first < (Nh - pix_buffer)
                       && ps.second > pix_buffer && ps.second < (Nw - pix_buffer);

        for (long h = 0; h <= Nh; ++h)
        {
            double d = sph_pix_diff(ss, Vector2d(h + 0.5, 0.5), skygeom) * R2D;
            min_deg  = min_deg < d ? min_deg : d;
            d        = sph_pix_diff(ss, Vector2d(h + 0.5, Nw + 0.5), skygeom) * R2D;
            min_deg  = min_deg < d ? min_deg : d;
        }
        for (long w = 0; w <= Nw; ++w)
        {
            double d = sph_pix_diff(ss, Vector2d(0.5, w + 0.5), skygeom) * R2D;
            min_deg  = min_deg < d ? min_deg : d;
            d        = sph_pix_diff(ss, Vector2d(Nh + 0.5, w + 0.5), skygeom) * R2D;
            min_deg  = min_deg < d ? min_deg : d;
        }

        radius(s) = min_deg;
    }
    return { radius, is_in_fov };
}

auto
Fermi::ModelMap::map_integral(Tensor4d const& model_map,
                              vpd const&      src_dirs,
                              SkyGeom const&  skygeom,
                              Tensor1d const& psf_radius,
                              Tensor1b const& is_in_fov) -> Tensor2d
{
    long const Ne = model_map.dimension(0);
    long const Nh = model_map.dimension(1);
    long const Nw = model_map.dimension(2);
    long const Ns = model_map.dimension(3);
    long const Nf = psf_radius.dimension(0);

    Tensor2d MapIntegral(Ne, Nf);
    MapIntegral.setZero();

    // Annoyingly nested, but hard to declarize because of the skygeom dependency.
    long i = 0;
    for (long s = 0; s < Ns; ++s)
    {
        if (!is_in_fov(s)) { continue; }

        double const rad = psf_radius(i);
        for (long w = 0; w < Nw; ++w)
        {
            for (long h = 0; h < Nh; ++h)
            {
                if (sph_pix_diff(src_dirs[s], Vector2d(h + 1., w + 1.), skygeom) * R2D
                    <= rad)
                {
                    MapIntegral.slice(Idx2 { 0, i }, Idx2 { Ne, 1 })
                        += model_map.slice(Idx4 { 0, h, w, s }, Idx4 { Ne, 1, 1, 1 })
                               .reshape(Idx2 { Ne, 1 });
                }
            }
        }
        ++i;
    }

    Tensor2d zeros          = MapIntegral.constant(0.0);
    Tensor2d invMapIntegral = MapIntegral.inverse();
    MapIntegral             = (MapIntegral == 0.).select(zeros, invMapIntegral);

    return MapIntegral;
}

auto
Fermi::ModelMap::map_correction_factor(Tensor2d const& inv_map_integ, /* [E, F] */
                                       Tensor1d const& psf_radius,
                                       Tensor1b const& is_in_fov,
                                       Tensor3d const& mean_psf,         /* [D,E,S] */
                                       Tensor3d const& partial_integrals /* [D,E,S] */
                                       ) -> Tensor2d
{
    long const Nd = mean_psf.dimension(0);
    long const Ne = mean_psf.dimension(1);
    long const Ns = mean_psf.dimension(2);
    long const Nf = psf_radius.dimension(0);

    Tensor3d filtered_psf(Nd, Ne, Nf);
    Tensor3d filtered_parint(Nd, Ne, Nf);

    long f = 0;
    for (long s = 0; s < Ns; ++s)
    {
        if (!is_in_fov(s)) { continue; }
        filtered_psf.slice(Idx3 { 0, 0, f }, Idx3 { Nd, Ne, 1 })
            = mean_psf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 });
        filtered_parint.slice(Idx3 { 0, 0, f }, Idx3 { Nd, Ne, 1 })
            = partial_integrals.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 });
        ++f;
    }

    Tensor2d const rad_integ
        = Fermi::ModelMap::integral(psf_radius, filtered_psf, filtered_parint);
    Tensor2d cor_fac = rad_integ * inv_map_integ;

    Tensor2d correction_factor(Ne, Ns);
    correction_factor.setConstant(1.);

    f = 0;
    for (long s = 0; s < Ns; ++s)
    {
        if (!is_in_fov(s)) { continue; }
        correction_factor.slice(Idx2 { 0, s }, Idx2 { Ne, 1 })
            = cor_fac.slice(Idx2 { 0, f }, Idx2 { Ne, 1 });
        ++f;
    }

    return correction_factor;
}

void
Fermi::ModelMap::scale_map_by_correction_factors(Tensor4d& model_map, /*[E,H,W,S]*/
                                                 Tensor2d const& factor /*[E,S]*/)
{
    long const Ne = model_map.dimension(0);
    long const Nh = model_map.dimension(1);
    long const Nw = model_map.dimension(2);
    long const Ns = model_map.dimension(3);

    assert(Ne == factor.dimension(0));
    assert(Ns == factor.dimension(1));

    model_map *= factor.reshape(Idx4 { Ne, 1, 1, Ns }).broadcast(Idx4 { 1, Nh, Nw, 1 });;
}

auto
Fermi::ModelMap::integral(Tensor1d const& angles,
                          Tensor3d const& mean_psf,         /* [D,E,S] */
                          Tensor3d const& partial_integrals /* [D,E,S] */
                          ) -> Tensor2d
{
    // Apply the same Midpoint rule found in Fermitools, but take advantage of the
    // fact that the energy sample occurs at exactly every energy entry.
    //
    //     size_t k(std::upper_bound(energies.begin(), energies.end(), energy)
    //              - energies.begin() - 1);
    //     if (k < 0 || k > static_cast<int>(energies.size() - 1))
    //     {
    //         std::ostringstream what;
    //         what << "MeanPsf::integral: energy " << energy << " out-of-range. "
    //              << energies.front() << '-' << energies.back() << std::endl;
    //         throw std::out_of_range(what.str());
    //     }
    //
    //     if (angle < s_separations.front()) { return 0; }
    //     else if (angle >= s_separations.back()) { return 1; }
    //     size_t j(std::upper_bound(s_separations.begin(), s_separations.end(), angle)
    //              - s_separations.begin() - 1);
    //
    // long const Nd = mean_psf.dimension(0);
    long const Ne          = mean_psf.dimension(1);
    long const Ns          = mean_psf.dimension(2);

    PSF::SepArr const seps = PSF::separations();

    assert(angles.size() == Ns);

    Tensor1i const sep_idxs = Fermi::PSF::fast_separation_lower_index(angles);

    Tensor3d ang_psf(2, Ne, Ns);
    Tensor3d X(2, 1, Ns); // theta
    Tensor3d PartInt(1, Ne, Ns);
    //
    Idx3 const o0 = { 0, 0, 0 };
    Idx3 const o1 = { 1, 0, 0 };
    Idx3 const b0 = { 1, Ne, 1 };
    Idx3 const e1 = { 1, 1, Ns };
    Idx3 const e2 = { 1, Ne, Ns };

    //     size_t index(k * s_separations.size() + j);
    //     double theta1(s_separations[j] * M_PI / 180.);
    //     double theta2(s_separations[j + 1] * M_PI / 180.);
    for (long i = 0; i < Ns; ++i)
    {
        auto ix = sep_idxs(i);
        ix      = ix >= PSF::sep_arr_len - 2
                      ? PSF::sep_arr_len - 2
                      : ix; // Prevent overflow. Must select out later.
        ang_psf.slice(Idx3 { 0, 0, i }, Idx3 { 2, Ne, 1 })
            = mean_psf.slice(Idx3 { ix, 0, i }, Idx3 { 2, Ne, 1 });
        X(0, 0, i) = seps[ix] * deg2rad;
        X(1, 0, i) = seps[ix + 1] * deg2rad;
        PartInt.slice(Idx3 { 0, 0, i }, Idx3 { 1, Ne, 1 })
            = partial_integrals.slice(Idx3 { ix, 0, i }, Idx3 { 1, Ne, 1 });
    }

    //     double y1(2. * M_PI * m_psfValues.at(index) * std::sin(theta1));
    //     double y2(2. * M_PI * m_psfValues.at(index + 1) * std::sin(theta2));
    // [2, Ne, Ns]
    Tensor3d Y     = ang_psf * X.unaryExpr([](double t) {
                                return twopi * std::sin(t);
                            }).broadcast(Idx3 { 1, Ne, 1 });

    //     double slope((y2 - y1) / (theta2 - theta1));
    Tensor3d DY    = Y.slice(o1, e2) - Y.slice(o0, e2); // [1, Ne, Ns] = e2
    Tensor3d DX    = X.slice(o1, e1) - X.slice(o0, e1); // [1, 1, Ns]  = e1
    Tensor3d M     = DY / DX.broadcast(b0);             // [1, Ne, Ns] = e2

    //     double intercept(y1 - theta1 * slope);
    // 1, Ne, Ns = e2
    Tensor3d B     = Y.slice(o0, e2) - (M * X.slice(o0, e1).broadcast(b0));
    //     double theta(angle * M_PI / 180.);
    // 1, 1, Ns = e1
    Tensor3d Th    = angles.reshape(e1) * deg2rad;
    Tensor3d ST    = Th + X.slice(o0, e1);
    Tensor3d DT    = Th - X.slice(o0, e1);
    //     double value(slope * (theta * theta - theta1 * theta1) / 2.
    //                  + intercept * (theta - theta1));
    // 1, Ne, Ns = e2
    Tensor3d V     = (0.5 * M * ST.broadcast(b0) + B) * DT.broadcast(b0);
    //     double integral1(m_partialIntegrals.at(k).at(j) + value);
    //
    // 1, Ne, Ns = e2
    Tensor3d Integ = V + PartInt; // 1, Ne, Ns = e2
    //
    // Final cleanup in case the angle was beyond the lookup table. In that case the
    // Normalized Partial Integral Maximum was 1., so the Integral = V + PartInt was
    // greater than one. We use this trick to quickly upper_bound all the values to be
    // 1.0;
    Integ          = Integ.cwiseMax(1.0);

    return Integ.reshape(Idx2 { Ne, Ns });
}
