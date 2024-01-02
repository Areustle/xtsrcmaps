#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/misc/misc.hxx"

// inline auto
// sph_diff(Eigen::Vector3d const& L, Eigen::Vector3d const& R) -> double {
//     return 2. * asin(0.5 * (L - R).norm());
// };
//
//
// auto
// reimann_integ(double const    dh,
//               double const    dw,
//               Vector2d const& src_pix,
//               Tensor2d const& pixelOffsets,
//               Tensor2d const& tuPsf_ED,
//               long const      Ne) -> Tensor1d
// {
//     bool     converged = false;
//     long     Nsub      = 1;
//     double   sumNsub   = 0.0;
//     Tensor1d mean_psf(Ne);
//     mean_psf.setZero();
//
//     // pixel center displacement vector. We add this to the weights tensor
//     // as an intermediate step to compute the norm.
//     Tensor2d pix_src_disp(1, 2);
//     pix_src_disp.setValues({ { dh - src_pix(0) }, { dw - src_pix(1) } });
//
//     while (!converged && Nsub <= 64)
//     {
//         std::cout << Nsub << std::endl;
//
//         double   dsub = 1.0 / Nsub;
//         Tensor2d dsteps(Nsub, 1);
//         for (long i = 0; i < Nsub; ++i) { dsteps(i, 0) = (i + 0.5) * dsub; }
//
//         Tensor2d lerp_wts(Nsub, 2);
//         lerp_wts.slice(Idx2 { 0, 0 }, Idx2 { Nsub, 1 }) = dsteps;
//         lerp_wts.slice(Idx2 { 0, 1 }, Idx2 { Nsub, 1 }) = 1.0 - dsteps;
//
//         std::cout << "lerp_wts: " << std::endl
//                   << lerp_wts.swap_layout() << std::endl
//                   << std::endl;
//
//         Tensor2d t1    = subpoff.contract(lerp_wts, IdxPair1 { { { 1, 1 } }
//         }); Tensor2d scale = 1.0 + t1.contract(lerp_wts, IdxPair1 { { { 0, 1
//         } } });
//
//         std::cout << "scale: " << std::endl
//                   << scale.swap_layout() << std::endl
//                   << std::endl;
//
//         Tensor2d dispwts(Nsub, 2);
//         dispwts = dsteps.broadcast(Idx2 { 1, 2 });
//         dispwts += pix_src_disp.broadcast(Idx2 { Nsub, 1 });
//         std::cout << "dispwts: " << std::endl
//                   << dispwts.swap_layout() << std::endl
//                   << std::endl;
//
//         // std::cout << "contract dispwts: " << std::endl
//         //           << dispwts.contract(dispwts, IdxPair1 { { { 1, 1 } } })
//         <<
//         //           std::endl
//         //           << std::endl;
//
//         // Tensor2d pix_offset // Nsub, Nsub
//         //     = 0.2 * dispwts.contract(dispwts, IdxPair1 { { { 1, 1 } }
//         }).sqrt(); Tensor2d pix_offset(Nsub, Nsub); for (long i = 0; i <
//         Nsub; ++i)
//         {
//             for (long j = 0; j < Nsub; ++j)
//             {
//                 pix_offset(j, i)
//                     = 0.2 * (dispwts(j, 0) + dispwts(i, 0) * dispwts(i, 0));
//             }
//         }
//         std::cout << "pix_offset: " << std::endl
//                   << pix_offset << std::endl
//                   << std::endl;
//
//         pix_offset = pix_offset.sqrt();
//
//         std::cout << "pix_offset: " << std::endl
//                   << pix_offset << std::endl
//                   << std::endl;
//
//         // Tensor2d displacement(Nsub, Nsub);
//         Tensor2d displacement = scale * pix_offset;
//
//         std::cout << "Disp: \n" << displacement << std::endl << std::endl;
//
//         long const Npts       = Nsub * Nsub;
//         auto       scaled_off = 1e4 * displacement;
//         Tensor2d   separation_index
//             = (scaled_off < 1.).select(scaled_off, 1. + (scaled_off.log() /
//             sep_step));
//         TensorMap<Tensor1d const> const idxs(separation_index.data(), Npts);
//
//         // std::cout << "idxs: " << idxs.reshape(Idx2 { 1, Npts }) <<
//         std::endl
//         //           << std::endl;
//
//         Tensor1d itr_psf(Ne);
//         itr_psf.setZero();
//
//         // iterate over every point
//         long i = 0;
//         while (i < Npts)
//         {
//
//             // Lookup table's separation index.
//             double const index = std::floor(idxs(i));
//             // run length of points which share a separation index.
//             long Nlen          = 1;
//             // Iterate sequential points until a new index value is seen
//             while ((i + Nlen < Npts) && index == std::floor(idxs(i + Nlen)))
//             {
//             ++Nlen; }
//             // Get a view Linear of the same-separation points.
//             TensorMap<Tensor1d const> const ss(idxs.data() + i, Nlen);
//             // Compute the interpolation weights for every ss point.
//             Tensor2d weights(Nlen, 2);
//             TensorMap<Tensor1d>(weights.data() + Nlen, Nlen) = ss - index;
//             TensorMap<Tensor1d>(weights.data(), Nlen)
//                 = 1. - TensorMap<Tensor1d>(weights.data() + Nlen, Nlen);
//
//
//
//             // Get a view of the psf lookup table.
//             TensorMap<Tensor2d const> const lut(
//                 tuPsf_ED.data() + long(index) * Ne, Ne, 2);
//             // Contract the weights with the lookup table entries, thereby
//             computing the
//             // PSF values for every energy in the table and every ss point.
//             // [Ne, Nlen]
//             Tensor2d vv = lut.contract(weights, IdxPair1 { { { 1, 1 } } });
//
//             // Write the Energies into the result buffer via a veiw.
//             // TensorMap<Tensor2d>(vals.data() + i * Ne, Ne, Nlen) = vv;
//             itr_psf += vv.sum(Idx1 { 1 });
//
//             // Shift the target point by the length of ss points to ensure we
//             start at
//             // an unseen point
//             i += Nlen;
//         }
//
//         Tensor1d new_mean_psf
//             = ((mean_psf * double(sumNsub)) + itr_psf) / double(Nsub +
//             sumNsub);
//         Tensor1d relerr = (new_mean_psf - mean_psf).abs() / mean_psf.abs();
//
//         std::cout << relerr.reshape(Idx2 { 1, Ne }) << std::endl;
//
//         Tensor0b cvgd = (relerr < 1E-3).all();
//         converged     = cvgd(0);
//         mean_psf      = new_mean_psf;
//         sumNsub += Nsub;
//         Nsub *= 2;
//     }
//     return mean_psf;
// }

double
bilinear_on_grid_slow(double x, double y, Tensor2d const& grid) {
    long const Nx = grid.dimension(0);
    long const Ny = grid.dimension(1);

    long   ix = std::min(long(std::floor(std::clamp(x, 0.0, Nx - 1.))), Nx - 2);
    long   iy = std::min(long(std::floor(std::clamp(y, 0.0, Ny - 1.))), Ny - 2);
    double rx = x - ix;
    double ry = y - iy;
    double sx = 1. - rx;
    double sy = 1. - ry;
    return sx * sy * grid(x, y) + sx * ry * grid(x, y + 1)
           + rx * sy * grid(x + 1, y) + rx * ry * grid(x + 1, y + 1);
}

auto
Fermi::ModelMap::create_offset_map(long const                       Nh,
                                   long const                       Nw,
                                   std::pair<double, double> const& dir,
                                   Fermi::SkyGeom const& skygeom) -> Tensor2d {
    // Get the sources coordinate in 3-direction space.
    Vector3d src_dir = skygeom.sph2dir(dir); // CLHEP Style 3
    Vector2d src_pix = skygeom.sph2pix(Vector2d(dir.first, dir.second));

    // Compute the pixel offsets map used for distance scaling computation.
    // Create Offset Map
    Tensor2d pixelOffsets(Nh, Nw);
    for (long w = 0; w < Nw; ++w) {
        for (long h = 0; h < Nh; ++h) {
            auto pixpix        = Vector2d(h + 1, w + 1);
            auto pixdir        = skygeom.pix2dir(pixpix);
            auto ang_sep       = Fermi::dir_diff(pixdir, src_dir) * rad2deg;

            // auto pix_sep = 0.2
            //                * std::sqrt(std::pow(src_pix(0) - (h + 1), 2)
            //                            + std::pow(src_pix(1) - (w + 1), 2));
            auto pix_sep       = 0.2 * (src_pix - pixpix).norm();

            auto pixelOffset   = ang_sep / pix_sep - 1.0;
            pixelOffsets(h, w) = pix_sep > 1E-6 ? pixelOffset : 0.0;
        }
    }
    return pixelOffsets;
}


/* auto */
/* riemann_slow(Tensor2d const& tuPsf_ED, */
/*              long const      e, */
/*              long const      h, */
/*              long const      w, */
/*              long const      Nsub, */
/*              Vector2d const  src_pix, */
/*              Tensor2d const& pixelOffsets) -> double */
/* { */
/*     double psf_value(0); */
/*     double dstep(1. / Nsub); */
/*     for (long i(0); i < Nsub; i++) */
/*     { */
/*         double x((h + 1) + i * dstep - 0.5 + 0.5 * dstep); */
/*         for (long j(0); j < Nsub; j++) */
/*         { */
/*             double y((w + 1) + j * dstep - 0.5 + 0.5 * dstep); */
/*             double pix_offset = 0.2 */
/*                                 * std::sqrt(std::pow(src_pix(0) - x, 2) */
/*                                             + std::pow(src_pix(1) - y, 2));
 */
/*             // We have to subtract off 1 to get from the FITS convention to
 * the */
/*             // array indices */
/*             // double scale = bilinear_on_grid(y - 1, x - 1, pixelOffsets);
 */
/*             double scale = bilinear_on_grid_slow(y - 1, x - 1, pixelOffsets);
 */
/**/
/*             // psf_value += meanPsf(e, pix_offset * (1 + scale)); */
/*             psf_value += Fermi::psf_lerp_slow(tuPsf_ED, e, pix_offset * (1. +
 * scale)); */
/*         } */
/*     } */
/*     return psf_value / static_cast<double>(Nsub * Nsub); */
/* } */


/* auto */
/* Fermi::ModelMap::pixel_mean_psf_riemann(long const      Nh, */
/*                                         long const      Nw, */
/*                                       Obs::sphcrd_v_t const& src_sphcrds, */
/*                                         Tensor3d const& psf_lut, */
/*                                         Tensor2d const& psf_peak, */
/*                                         SkyGeom const&  skygeom, */
/*                                         double const    ftol_threshold) ->
 * Tensor4d
 */
/* { */
/*     long const Ns = src_sphcrds.size(); */
/*     long const Nd = psf_lut.dimension(0); */
/*     long const Ne = psf_lut.dimension(1); */
/*     // long const Nevts = Nh * Nw; */
/**/
/*     Tensor4d model_map(Ne, Nh, Nw, Ns); */
/*     model_map.setZero(); */
/**/
/*     for (long s = 0; s < 1; ++s) */
/*     { */
/*         std::cout << "\n" << s << std::flush; */
/**/
/*         // Compute the pixel offsets map used for distance scaling
 * computation. */
/*         // Create Offset Map */
/*         Tensor2d pixelOffsets   = create_offset_map(Nh, Nw, src_sphcrds[s],
 * skygeom);
 */
/**/
/*         // long const source_index_offset = s * Ne * Nevts; */
/*         // A slice of the PSF table just for this source's segment of the
 * table. */
/*         Tensor2d const tuPsf_ED = psf_lut.slice(Idx3 { 0, 0, s }, Idx3 { Nd,
 * Ne, 1 })
 */
/*                                       .reshape(Idx2 { Nd, Ne }) */
/*                                       .shuffle(Idx2 { 1, 0 }); */
/**/
/*         // Get the sources coordinate in 3-direction space. */
/*         Vector3d src_dir = skygeom.sph2dir(src_sphcrds[s]); // CLHEP Style 3
 */
/*         Vector2d src_pix */
/*             = skygeom.sph2pix(Vector2d(src_sphcrds[s].first,
 * src_sphcrds[s].second));
 */
/**/
/**/
/*         for (long w = 0; w < Nw; ++w) */
/*         { */
/*             std::cout << "." << std::flush; */
/*             for (long h = 0; h < Nh; ++h) */
/*             { */
/*                 // Tensor1d rei_u_psf = reimann_integ(dh, dw, src_pix,
 * pixelOffsets,
 */
/*                 // tuPsf_ED, Ne); */
/**/
/**/
/*                 //////////////////////////////////////////////////////////////////////////////////
 */
/*                 /// */
/*                 auto const   pixdir = skygeom.pix2dir(Vector2d(h + 1, w +
 * 1)); */
/*                 double const offset = Fermi::dir_diff(pixdir, src_dir) *
 * rad2deg; */
/*                 for (long e = 0; e < Ne; ++e) */
/*                 { */
/**/
/*                     double peak_val   = psf_peak(e, s); */
/*                     double v0         = Fermi::psf_lerp_slow(tuPsf_ED, e,
 * offset); */
/*                     double v1         = 0.; */
/*                     double ferr       = 0.; */
/*                     double peak_ratio = peak_val ? v0 / peak_val : 0.0; */
/**/
/*                     if (peak_ratio < 1e-3) { v1 = v0; } */
/*                     else */
/*                     { */
/*                         long Nsub = 1; */
/*                         while (true) */
/*                         { */
/*                             Nsub *= 2; */
/*                             v1 = riemann_slow( */
/*                                 tuPsf_ED, e, h, w, Nsub, src_pix,
 * pixelOffsets); */
/*                             double vdiff = v1 - v0; */
/*                             if (vdiff == 0.0 || v1 == 0.0) { break; } */
/**/
/*                             ferr = std::fabs(vdiff / v1); */
/*                             if (ferr < ftol_threshold || Nsub >= 64) { break;
 * } */
/*                             v0 = v1; */
/*                         } */
/*                         // fmt::print("Integral Value {}\n", v1); */
/*                     } */
/*                     model_map(e, h, w, s) = v1; */
/*                 } */
/**/
/*                 //////////////////////////////////////////////////////////////////////////////////
 */
/**/
/**/
/*                 // std::cout << "Done: " << rei_u_psf << std::endl; */
/*                 // */
/*                 // TensorMap<Tensor1d>(model_map.data() + h * Ne + w * Ne *
 * Nw + */
/*                 // source_index_offset, */
/*                 //                     Ne) */
/*                 //     = rei_u_psf; */
/*             } */
/*         } */
/**/
/******************************************************************
 * Psf Energy values for sample points in direction space
 ******************************************************************/
/*     } */
/*     // fmt::print("\n"); */
/**/
/*     return model_map; */
/* } */

// auto
// sph_diff(Vector2d const& L, Vector2d const& R, Fermi::SkyGeom const& skygeom)
// -> double
// {
//     coord3 a = skygeom.pix2dir({ L(0), L(1) });
//     coord3 b = skygeom.pix2dir({ R(0), R(1) });
//     return sph_diff({ std::get<0>(a), std::get<1>(a), std::get<2>(a) },
//                     { std::get<0>(b), std::get<1>(b), std::get<2>(b) });
// };
