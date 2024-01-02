#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/misc/misc.hxx"

// Compute the psf radius of each source relative to the field of view by
// computing the minimal distance between the source and each boundary segment.
// Sources outside the field of view are not included in the radius vector, and
// are set to false in the boolean vector
auto
Fermi::ModelMap::psf_boundary_radius(long const             Nh,
                                     long const             Nw,
                                     Obs::sphcrd_v_t const& src_dirs,
                                     SkyGeom const&         skygeom)
    -> std::pair<Tensor1d, Tensor1b> {
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

    for (long s = 0; s < Ns; ++s) {
        double      min_deg = 360.;
        auto const& ss      = src_dirs[s];
        auto        ps      = skygeom.sph2pix(ss);
        is_in_fov(s) = ps.first > pix_buffer && ps.first < (Nh - pix_buffer)
                       && ps.second > pix_buffer
                       && ps.second < (Nw - pix_buffer);

        for (long h = 0; h <= Nh; ++h) {
            double d = sph_pix_diff(ss, Vector2d(h + 0.5, 0.5), skygeom) * R2D;
            min_deg  = min_deg < d ? min_deg : d;
            d = sph_pix_diff(ss, Vector2d(h + 0.5, Nw + 0.5), skygeom) * R2D;
            min_deg = min_deg < d ? min_deg : d;
        }
        for (long w = 0; w <= Nw; ++w) {
            double d = sph_pix_diff(ss, Vector2d(0.5, w + 0.5), skygeom) * R2D;
            min_deg  = min_deg < d ? min_deg : d;
            d = sph_pix_diff(ss, Vector2d(Nh + 0.5, w + 0.5), skygeom) * R2D;
            min_deg = min_deg < d ? min_deg : d;
        }

        radius(s) = min_deg;
    }
    return { radius, is_in_fov };
}


// inline auto
// point_nearest_to_source_on_segment(Vector2d const& v,
//                                    Vector2d const& p0,
//                                    Vector2d const& p1,
//                                    double const&   c1,
//                                    double const&   c2) -> Eigen::Vector2d {
//     // v = p1 - p0
//     // if ((w.v)=c1 <= 0) then before P0 return p0
//     // if ((v.v)=c2 <= c1) then after P1 return p1
//     // pb = p0+b*v
//     return c1 <= 0. ? p0 : c2 <= c1 ? p1 : (p0 + (c1 / c2) * v);
// }
//
// inline auto
// point_nearest_to_source_on_segment(Vector2d const& p,
//                                    Vector2d const& p0,
//                                    Vector2d const& p1) -> Vector2d {
//     // v = p1 - p0
//     Vector2d const v = p1 - p0;
//     // w = p - p0
//     Vector2d const w = p - p0;
//     // if ((w.v)=c1 <= 0) then before P0 return p0
//     double const c1  = w.dot(v);
//     if (c1 <= 0) { /*before P0*/
//         return p0;
//     }
//     // if ((v.v)=c2 <= c1) then after P1 return p1
//     double const c2 = v.dot(v);
//     if (c2 <= c1) { /*after P1*/
//         return p1;
//     }
//     // b = c1 / c2
//     double const b    = c1 / c2;
//     // pb = p0+b*v
//     Vector2d const pb = p0 + b * v;
//     return pb;
//     // return c1 <= 0. ? p0 : c2 <= c1 ? p1 : (p0 + (c1 / c2) * v);
// }
//
//
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
//     // // Simple geometric trick to determine if the source point is bounded
//     by the
//     // // convex hull of our spherically warped field of view.
//     // Vector2d const S(std::get<0>(src_pts_pix[s]),
//     std::get<1>(src_pts_pix[s])); Vector2d const S(std::get<0>(src_dirs[s]),
//     std::get<1>(src_dirs[s]));
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
//     // lenAB); Vector2d pSAD    = point_nearest_to_source_on_segment(AD, A,
//     D,
//     // AS_AD, lenAD); Vector2d pSCB    =
//     point_nearest_to_source_on_segment(CB, C,
//     // B, CS_CB, lenCB); Vector2d pSCD    =
//     point_nearest_to_source_on_segment(CD,
//     // C, D, CS_CD, lenCD);
//     //
//     // // Distance between the source and the boundary lines of the field of
//     view;
//     // double const dAB = sph_diff(S, pSAB.array().round(), skygeom);
//     // double const dAD = sph_diff(S, pSAD.array().round(), skygeom);
//     // double const dCB = sph_diff(S, pSCB.array().round(), skygeom);
//     // double const dCD = sph_diff(S, pSCD.array().round(), skygeom);
//     //
//     // double min_rad   = dAB < dAD ? dAB : dAD;
//     // min_rad          = min_rad < dCB ? min_rad : dCB;
//     // min_rad          = min_rad < dCD ? min_rad : dCD;
//     // min_rad *= R2D;
