
#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"


#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/model_map.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/source_utils.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

#include <algorithm>

template <typename T>
void
req_displacement(long const                               ph,
                 long const                               pw,
                 std::tuple<double, double, double> const src_dir,
                 Fermi::SkyGeom const&                    skygeom,
                 T const                                  SD)
{
    long constexpr N  = SD.rows();
    auto constexpr ds = Fermi::ModelMap::integ_delta_steps<N>();
    for (long w = 0; w < N; ++w)
    {
        for (long h = 0; h < N; ++h)
        {
            double const true_off = skygeom.srcpixoff(
                src_dir, skygeom.pix2dir({ ph + ds[h], pw + ds[w] }));
            REQUIRE_MESSAGE(SD(w, h) == doctest::Approx(true_off).epsilon(1e-3),
                            N << ": (" << pw << " " << ph << ") -> " << w << " " << h
                              << " | " << ds[w] << " " << ds[h] << " | " << pw + ds[w]
                              << " " << ph + ds[h] << " | "
                              << "");
        }
    }
}
//
// // template <typename T>
// // void
// // req_idx_scale(long const                               ph,
// //               long const                               pw,
// //               std::tuple<double, double, double> const src_dir,
// //               Fermi::SkyGeom const&                    skygeom,
// //               T const                                  SD)
// // {
// //     long constexpr N   = SD.rows();
// //     // auto constexpr ds  = Fermi::ModelMap::integ_delta_steps<N>();
// //
// //     auto s_separations = Fermi::PSF::separations();
// //
// //     for (long w = 0; w < N; ++w)
// //     {
// //         for (long h = 0; h < N; ++h)
// //         {
// //
// //             double iv = Fermi::PSF::inverse_separations(SD(w, h));
// //             int    i  = int(iv);
// //             double u  = (SD(w, h) - s_separations.at(i))
// //                        / (s_separations.at(i + 1) - s_separations.at(i));
// //
// //             double uu
// //                 = (SD(w, h) - s_separations[i + 1]) / (s_separations[i] - SD(w,
// h));
// //
// //             double uuu = (iv - 8 * Fermi::PSF::inverse_separations(10) - sep_step
// *
// //             i)
// //                          / sep_step;
// //
// //             CHECK_MESSAGE(u == doctest::Approx(uu),
// //                           w << " " << h << " from: " << SD(w, h)
// //                             << "\n"
// //
// //                                " got:"
// //                             << iv << " (" << s_separations[i] << ","
// //                             << s_separations[i + 1] << ") ");
// //
// //             CHECK_MESSAGE(u == doctest::Approx(uuu),
// //                           w << " " << h << " from: " << SD(w, h)
// //                             << "\n"
// //
// //                                " got:"
// //                             << iv << " (" << s_separations[i] << ","
// //                             << s_separations[i + 1] << ") ");
// //         }
// //     }
// // }
//
// TEST_CASE("Test Model Map Pixel Displacements")
// {
//
//     long const NpW       = 100;
//     long const NpH       = 100;
//     long const NpW_      = NpW + 2;
//     long const NpH_      = NpH + 2;
//     auto       cfg       = Fermi::XtCfg();
//     auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
//     auto const dirs      = Fermi::directions_from_point_sources(srcs);
//
//     auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
//     REQUIRE(opt_ccube);
//     auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
//     Fermi::SkyGeom skygeom(ccube);
//     auto           dir    = dirs[0];
//     // for (auto dir : dirs)
//     // {
//     auto         src_dir  = skygeom.sph2dir(dir); // CLHEP Style 3
//     auto         src_pix  = skygeom.sph2pix(dir); // Grid Style 2
//     double const ref_size = 0.2;
//
//     Eigen::MatrixXd Offsets(NpW_, NpH_);
//     Offsets.setZero();
//     for (long pw = 0; pw < NpW_; ++pw)
//     {
//         for (long ph = 0; ph < NpH_; ++ph)
//         {
//             auto   pdir    = skygeom.pix2dir({ ph, pw });
//             double pix_sep = ref_size
//                              * std::sqrt(std::pow(src_pix.first - (ph), 2)
//                                          + std::pow(src_pix.second - (pw), 2));
//             double ang_sep = skygeom.srcpixoff(
//                 src_dir, { std::get<0>(pdir), std::get<1>(pdir), std::get<2>(pdir)
//                 });
//             Offsets(pw, ph) = pix_sep > 1E-6 ? ang_sep / pix_sep - 1. : 0.0;
//         }
//     }
//
//
//     for (long pw = 1; pw <= NpW; ++pw)
//     {
//         for (long ph = 1; ph <= NpH; ++ph)
//         {
//
//
//             Eigen::Matrix<double, 2, 2> SD;
//             SD.setZero();
//
//             // Using Comb Separations Function.
//             SD = Fermi::ModelMap::rectangular_comb_separations<2>(
//                 pw, ph, ref_size, src_pix, Offsets);
//             req_displacement(ph, pw, src_dir, skygeom, SD);
//
//
//             Eigen::Matrix<double, 4, 4> SD4
//                 = Fermi::ModelMap::rectangular_comb_separations<4>(
//                     pw, ph, ref_size, src_pix, Offsets);
//             req_displacement(ph, pw, src_dir, skygeom, SD4);
//
//
//             Eigen::Matrix<double, 8, 8> SD8
//                 = Fermi::ModelMap::rectangular_comb_separations<8>(
//                     pw, ph, ref_size, src_pix, Offsets);
//             req_displacement(ph, pw, src_dir, skygeom, SD8);
//
//
//             Eigen::Matrix<double, 16, 16> SD16
//                 = Fermi::ModelMap::rectangular_comb_separations<16>(
//                     pw, ph, ref_size, src_pix, Offsets);
//             req_displacement(ph, pw, src_dir, skygeom, SD16);
//
//
//             Eigen::Matrix<double, 32, 32> SD32
//                 = Fermi::ModelMap::rectangular_comb_separations<32>(
//                     pw, ph, ref_size, src_pix, Offsets);
//             req_displacement(ph, pw, src_dir, skygeom, SD32);
//
//
//             Eigen::Matrix<double, 64, 64> SD64
//                 = Fermi::ModelMap::rectangular_comb_separations<64>(
//                     pw, ph, ref_size, src_pix, Offsets);
//             req_displacement(ph, pw, src_dir, skygeom, SD64);
//         }
//     }
//     //     break;
//     // }
// }
//
// TEST_CASE("Test Model Map Pixel Offsets")
// {
//
//     long const NpW       = 100;
//     long const NpH       = 100;
//     // long const NpW_      = NpW + 2;
//     // long const NpH_      = NpH + 2;
//     auto       cfg       = Fermi::XtCfg();
//     auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
//     auto const dirs      = Fermi::directions_from_point_sources(srcs);
//     // long const Ns        = srcs.size();
//
//     auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
//     REQUIRE(opt_ccube);
//     auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
//     Fermi::SkyGeom skygeom(ccube);
//     auto           src_dir  = skygeom.sph2dir(dirs[0]); // CLHEP Style 3
//     auto           src_pix  = skygeom.sph2pix(dirs[0]); // Grid Style 2
//     double const   ref_size = 0.2;
//
//     Tensor2d Offsets(NpW, NpH);
//     Offsets.setZero();
//     for (long pw = 0; pw < NpW; ++pw)
//     {
//         for (long ph = 0; ph < NpH; ++ph)
//         {
//             auto   pdir    = skygeom.pix2dir({ ph + 1, pw + 1 });
//             double pix_sep = ref_size
//                              * std::sqrt(std::pow(src_pix.first - (ph + 1), 2)
//                                          + std::pow(src_pix.second - (pw + 1), 2));
//             double ang_sep = skygeom.srcpixoff(
//                 src_dir, { std::get<0>(pdir), std::get<1>(pdir), std::get<2>(pdir)
//                 });
//             Offsets(pw, ph) = pix_sep > 1E-6 ? ang_sep / pix_sep - 1. : 0.0;
//         }
//     }
//
//     std::ifstream ifs("./xtsrcmaps/tests/expected/src_pixelOffset.bin",
//                       std::ios::in | std::ios::binary);
//
//     size_t const bufsz      = 100 * 100;
//     size_t const bufszbytes = bufsz * sizeof(double);
//
//     std::vector<double> rmbuf(bufsz);
//     ifs.read((char*)(&rmbuf[0]), bufszbytes);
//     Tensor2d const stpixOff
//         = Fermi::row_major_buffer_to_col_major_tensor(rmbuf.data(), NpW, NpH);
//     for (long pw = 0; pw < NpW; ++pw)
//     {
//         for (long ph = 0; ph < NpH; ++ph)
//         {
//             REQUIRE_MESSAGE(Offsets(pw, ph) == doctest::Approx(stpixOff(pw, ph)),
//                             pw << " " << ph);
//         }
//     }
// }
//
TEST_CASE("integ_delta")
{
    std::array<double, 2> ids2 = { -0.25, 0.25 };
    REQUIRE(ids2 == Fermi::ModelMap::integ_delta_steps<2>());
    std::array<double, 4> ids4 = { -0.375, -0.125, 0.125, 0.375 };
    REQUIRE(ids4 == Fermi::ModelMap::integ_delta_steps<4>());
    std::array<double, 8> ids8
        = { -0.4375, -0.3125, -0.1875, -0.0625, 0.0625, 0.1875, 0.3125, 0.4375 };
    REQUIRE(ids8 == Fermi::ModelMap::integ_delta_steps<8>());
    std::array<double, 16> ids16
        = { -0.46875, -0.40625, -0.34375, -0.28125, -0.21875, -0.15625,
            -0.09375, -0.03125, 0.03125,  0.09375,  0.15625,  0.21875,
            0.28125,  0.34375,  0.40625,  0.46875 };
    REQUIRE(ids16 == Fermi::ModelMap::integ_delta_steps<16>());

    std::array<double, 32> ids32 = {
        // clang-format off
            -0.484375, -0.453125, -0.421875, -0.390625, -0.359375, -0.328125,
            -0.296875, -0.265625, -0.234375, -0.203125, -0.171875, -0.140625,
            -0.109375, -0.078125, -0.046875, -0.015625,  0.015625,  0.046875,
             0.078125,  0.109375,  0.140625,  0.171875,  0.203125,  0.234375,
             0.265625,  0.296875,  0.328125,  0.359375,  0.390625,  0.421875,
             0.453125,  0.484375
        // clang-format on
    };
    REQUIRE(ids32 == Fermi::ModelMap::integ_delta_steps<32>());
    std::array<double, 64> ids64 = {
        // clang-format off
            -0.4921875, -0.4765625, -0.4609375, -0.4453125, -0.4296875,
            -0.4140625, -0.3984375, -0.3828125, -0.3671875, -0.3515625,
            -0.3359375, -0.3203125, -0.3046875, -0.2890625, -0.2734375,
            -0.2578125, -0.2421875, -0.2265625, -0.2109375, -0.1953125,
            -0.1796875, -0.1640625, -0.1484375, -0.1328125, -0.1171875,
            -0.1015625, -0.0859375, -0.0703125, -0.0546875, -0.0390625,
            -0.0234375, -0.0078125,  0.0078125,  0.0234375,  0.0390625,
             0.0546875,  0.0703125,  0.0859375,  0.1015625,  0.1171875,
             0.1328125,  0.1484375,  0.1640625,  0.1796875,  0.1953125,
             0.2109375,  0.2265625,  0.2421875,  0.2578125,  0.2734375,
             0.2890625,  0.3046875,  0.3203125,  0.3359375,  0.3515625,
             0.3671875,  0.3828125,  0.3984375,  0.4140625,  0.4296875,
             0.4453125,  0.4609375,  0.4765625,  0.4921875
        // clang-format on
    };
    REQUIRE(ids64 == Fermi::ModelMap::integ_delta_steps<64>());

    REQUIRE(std::array<double, 2> { 1.25, -0.25 }
            == Fermi::ModelMap::integ_delta_lo<2>());
    REQUIRE(std::array<double, 2> { 0.75, 0.25 }
            == Fermi::ModelMap::integ_delta_hi<2>());

    REQUIRE(std::array<double, 4> { 1.375, -0.375, 1.125, -0.125 }
            == Fermi::ModelMap::integ_delta_lo<4>());
    REQUIRE(std::array<double, 4> { 0.875, 0.125, 0.625, 0.375 }
            == Fermi::ModelMap::integ_delta_hi<4>());

    REQUIRE(std::array<double, 8> {
                1.4375, -0.4375, 1.3125, -0.3125, 1.1875, -0.1875, 1.0625, -0.0625 }
            == Fermi::ModelMap::integ_delta_lo<8>());
    REQUIRE(std::array<double, 8> {
                0.9375, 0.0625, 0.8125, 0.1875, 0.6875, 0.3125, 0.5625, 0.4375 }
            == Fermi::ModelMap::integ_delta_hi<8>());

    REQUIRE(std::array<double, 16> { 1.46875,
                                     -0.46875,
                                     1.40625,
                                     -0.40625,
                                     1.34375,
                                     -0.34375,
                                     1.28125,
                                     -0.28125,
                                     1.21875,
                                     -0.21875,
                                     1.15625,
                                     -0.15625,
                                     1.09375,
                                     -0.09375,
                                     1.03125,
                                     -0.03125 }
            == Fermi::ModelMap::integ_delta_lo<16>());
    REQUIRE(std::array<double, 16> { 0.96875,
                                     0.03125,
                                     0.90625,
                                     0.09375,
                                     0.84375,
                                     0.15625,
                                     0.78125,
                                     0.21875,
                                     0.71875,
                                     0.28125,
                                     0.65625,
                                     0.34375,
                                     0.59375,
                                     0.40625,
                                     0.53125,
                                     0.46875 }
            == Fermi::ModelMap::integ_delta_hi<16>());

    REQUIRE(std::array<double, 32> {
                1.484375, -0.484375, 1.453125, -0.453125, 1.421875, -0.421875,
                1.390625, -0.390625, 1.359375, -0.359375, 1.328125, -0.328125,
                1.296875, -0.296875, 1.265625, -0.265625, 1.234375, -0.234375,
                1.203125, -0.203125, 1.171875, -0.171875, 1.140625, -0.140625,
                1.109375, -0.109375, 1.078125, -0.078125, 1.046875, -0.046875,
                1.015625, -0.015625 }
            == Fermi::ModelMap::integ_delta_lo<32>());
    REQUIRE(std::array<double, 32> {
                0.984375, 0.015625, 0.953125, 0.046875, 0.921875, 0.078125, 0.890625,
                0.109375, 0.859375, 0.140625, 0.828125, 0.171875, 0.796875, 0.203125,
                0.765625, 0.234375, 0.734375, 0.265625, 0.703125, 0.296875, 0.671875,
                0.328125, 0.640625, 0.359375, 0.609375, 0.390625, 0.578125, 0.421875,
                0.546875, 0.453125, 0.515625, 0.484375 }
            == Fermi::ModelMap::integ_delta_hi<32>());

    REQUIRE(std::array<double, 64> {
                1.4921875, -0.4921875, 1.4765625, -0.4765625, 1.4609375, -0.4609375,
                1.4453125, -0.4453125, 1.4296875, -0.4296875, 1.4140625, -0.4140625,
                1.3984375, -0.3984375, 1.3828125, -0.3828125, 1.3671875, -0.3671875,
                1.3515625, -0.3515625, 1.3359375, -0.3359375, 1.3203125, -0.3203125,
                1.3046875, -0.3046875, 1.2890625, -0.2890625, 1.2734375, -0.2734375,
                1.2578125, -0.2578125, 1.2421875, -0.2421875, 1.2265625, -0.2265625,
                1.2109375, -0.2109375, 1.1953125, -0.1953125, 1.1796875, -0.1796875,
                1.1640625, -0.1640625, 1.1484375, -0.1484375, 1.1328125, -0.1328125,
                1.1171875, -0.1171875, 1.1015625, -0.1015625, 1.0859375, -0.0859375,
                1.0703125, -0.0703125, 1.0546875, -0.0546875, 1.0390625, -0.0390625,
                1.0234375, -0.0234375, 1.0078125, -0.0078125 }
            == Fermi::ModelMap::integ_delta_lo<64>());

    REQUIRE(std::array<double, 64> {
                0.9921875, 0.0078125, 0.9765625, 0.0234375, 0.9609375, 0.0390625,
                0.9453125, 0.0546875, 0.9296875, 0.0703125, 0.9140625, 0.0859375,
                0.8984375, 0.1015625, 0.8828125, 0.1171875, 0.8671875, 0.1328125,
                0.8515625, 0.1484375, 0.8359375, 0.1640625, 0.8203125, 0.1796875,
                0.8046875, 0.1953125, 0.7890625, 0.2109375, 0.7734375, 0.2265625,
                0.7578125, 0.2421875, 0.7421875, 0.2578125, 0.7265625, 0.2734375,
                0.7109375, 0.2890625, 0.6953125, 0.3046875, 0.6796875, 0.3203125,
                0.6640625, 0.3359375, 0.6484375, 0.3515625, 0.6328125, 0.3671875,
                0.6171875, 0.3828125, 0.6015625, 0.3984375, 0.5859375, 0.4140625,
                0.5703125, 0.4296875, 0.5546875, 0.4453125, 0.5390625, 0.4609375,
                0.5234375, 0.4765625, 0.5078125, 0.4921875 }
            == Fermi::ModelMap::integ_delta_hi<64>());
}

// TEST_CASE("Test Model Map PsfEstimate")
// {
//
//     long const Nw        = 100;
//     long const Nh        = 100;
//     long const Ns        = 263;
//     long const Ne        = 38;
//     long const Nd        = 401;
//     auto       cfg       = Fermi::XtCfg();
//     auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
//     auto const dirs      = Fermi::directions_from_point_sources(srcs);
//
//     auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
//     REQUIRE(opt_ccube);
//     auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
//     Fermi::SkyGeom skygeom(ccube);
//
//     Tensor3d const uPsf = Fermi::row_major_file_to_col_major_tensor(
//         "./xtsrcmaps/tests/expected/uPsf_normalized_SED.bin", Ns, Ne, Nd);
//     REQUIRE(uPsf.dimension(0) == Nd);
//     REQUIRE(uPsf.dimension(1) == Ne);
//     REQUIRE(uPsf.dimension(2) == Ns);
//
//     Tensor2d const uPeak = Fermi::row_major_file_to_col_major_tensor(
//         "./xtsrcmaps/tests/expected/uPsf_peak_SE.bin", Ns, Ne);
//     REQUIRE(uPeak.dimension(0) == Ne);
//     REQUIRE(uPeak.dimension(1) == Ns);
//
//     std::ifstream ifs("./xtsrcmaps/tests/expected/src_psfEstimate.bin",
//                       std::ios::in | std::ios::binary);
//
//     Tensor4d xtpsfEst(Nw, Nh, Ne, Ns);
//
//     // long const   s        = 0;
//     for (long s = 0; s < Ns; ++s)
//     {
//         auto         src_dir  = skygeom.sph2dir(dirs[s]); // CLHEP Style 3
//         auto         src_pix  = skygeom.sph2pix(dirs[s]); // Grid Style 2
//         double const ref_size = 0.2;
//
//         // size_t const bufsz      = Ne * Nw * Nh;
//         // size_t const bufszbytes = bufsz * sizeof(double);
//
//         // std::vector<double> rmbuf(bufsz);
//         // ifs.read((char*)(&rmbuf[0]), bufszbytes);
//         // Tensor3d const STpsfEstimate
//         //     = Fermi::row_major_buffer_to_col_major_tensor(rmbuf.data(), Nw, Nh,
//         Ne);
//         // REQUIRE(STpsfEstimate.dimension(0) == Ne);
//         // REQUIRE(STpsfEstimate.dimension(1) == Nh);
//         // REQUIRE(STpsfEstimate.dimension(2) == Nw);
//
//         /////
//         Tensor2d const tuPsf
//             = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 }).reshape(Idx2 { Nd, Ne
//             });
//         // REQUIRE(tuPsf.dimension(0) == Nd);
//         // REQUIRE(tuPsf.dimension(1) == Ne);
//         Tensor1d const tuPeak
//             = uPeak.slice(Idx2 { 0, s }, Idx2 { Ne, 1 }).reshape(Idx1 { Ne });
//         // REQUIRE(tuPeak.dimension(0) == Ne);
//
//         Eigen::Map<Eigen::MatrixXd const> const suPsf(tuPsf.data(), Nd, Ne);
//         Eigen::Map<Eigen::VectorXd const> const suPeak(tuPeak.data(), Ne);
//         // REQUIRE(suPsf.rows() == Nd);
//         // REQUIRE(suPsf.cols() == Ne);
//         // REQUIRE(suPeak.rows() == Ne);
//         // REQUIRE(suPeak.cols() == 1);
//
//         Eigen::MatrixXd Offsets
//             = Fermi::ModelMap::pixel_angular_offset_from_source_with_padding(
//                 src_dir, src_pix, ref_size, skygeom);
//
//         for (long pw = 1; pw <= Nw; ++pw)
//         {
//             for (long ph = 1; ph <= Nh; ++ph)
//             {
//                 double pix_off
//                     = skygeom.srcpixoff(src_dir, skygeom.pix2dir({ ph, pw }));
//
//                 pix_off                = Fermi::PSF::inverse_separations(pix_off);
//
//                 // Eigen::VectorXd psfEst = Fermi::ModelMap::integrate_psf_adaptive(
//                 //     pw, ph, ref_size, src_pix, pix_off, Offsets, suPsf, suPeak);
//                 //
//                 // xtpsfEst.slice(Idx4 { 0, pw - 1, ph - 1, s }, Idx4 { Ne, 1, 1, 1
//                 })
//                 //     = Eigen::TensorMap<Tensor4d>(psfEst.data(), Ne, 1, 1, 1);
//
//                 // for (long ie = 0; ie < Ne; ++ie)
//                 // {
//                 //     CHECK_MESSAGE(
//                 //         STpsfEstimate(ie, ph - 1, pw - 1)
//                 //             == doctest::Approx(xtpsfEst(ie, pw - 1, ph - 1, s))
//                 //                    .epsilon(1e-2),
//                 //         ie << " " << pw << " " << ph << " " << s);
//                 // }
//             }
//         }
//         // break;
//     }
//     ifs.close();
// }
//
//

TEST_CASE("Test Model Map Execution")
{

    auto       cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::directions_from_point_sources(srcs);

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(opt_ccube);
    auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
    Fermi::SkyGeom skygeom(ccube);
    long const     Nw  = 100;
    long const     Nh  = 100;
    long const     Ns  = dirs.size();
    long const     Nd  = 401;
    long const     Ne  = 38;

    auto const st_off0 = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_initial_offset.bin", Ns, Nw, Nh);
    assert(st_off0.dimension(0) == Nh);
    assert(st_off0.dimension(1) == Nw);
    assert(st_off0.dimension(2) == Ns);

    auto const st_offi0 = Fermi::row_major_file_to_col_major_tensor<unsigned short>(
        "./xtsrcmaps/tests/expected/src_initial_offset_idx.bin", Ns, Nw, Nh);
    assert(st_offi0.dimension(0) == Nh);
    assert(st_offi0.dimension(1) == Nw);
    assert(st_offi0.dimension(2) == Ns);

    auto const st_offs0 = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_initial_offset_scalar.bin", Ns, Nw, Nh);
    assert(st_offs0.dimension(0) == Nh);
    assert(st_offs0.dimension(1) == Nw);
    assert(st_offs0.dimension(2) == Ns);

    auto const st_nmc = Fermi::row_major_file_to_col_major_tensor<char>(
        "./xtsrcmaps/tests/expected/src_needs_more.bin", Ns, Nw, Nh, Ne);
    assert(st_nmc.dimension(0) == Ne);
    assert(st_nmc.dimension(1) == Nh);
    assert(st_nmc.dimension(2) == Nw);
    assert(st_nmc.dimension(3) == Ns);
    Eigen::Tensor<bool, 4> st_nm0 = st_nmc.cast<bool>();

    Tensor4d const st_v0          = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_mean_psf_v0.bin", Ns, Nw, Nh, Ne);
    assert(st_v0.dimension(0) == Ne);
    assert(st_v0.dimension(1) == Nh);
    assert(st_v0.dimension(2) == Nw);
    assert(st_v0.dimension(3) == Ns);

    Tensor4d const st_psfEst = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_psfEstimate.bin", Ns, Nw, Nh, Ne);
    assert(st_psfEst.dimension(0) == Ne);
    assert(st_psfEst.dimension(1) == Nh);
    assert(st_psfEst.dimension(2) == Nw);
    assert(st_psfEst.dimension(3) == Ns);

    Tensor3d const uPsf = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/uPsf_normalized_SED.bin", Ns, Ne, Nd);
    assert(uPsf.dimension(0) == Nd);
    assert(uPsf.dimension(1) == Ne);
    assert(uPsf.dimension(2) == Ns);

    Tensor2d const uPeak = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/uPsf_peak_SE.bin", Ns, Ne);
    assert(uPeak.dimension(0) == Ne);
    assert(uPeak.dimension(1) == Ns);

    auto const uPeakRatio = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_peak_ratio.bin", Ns, Nw, Nh, Ne);

    auto const st_upsf_peak = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_mean_psf_peak.bin", Ns, Ne);
    assert(st_upsf_peak.dimension(0) == Ne);
    assert(st_upsf_peak.dimension(1) == Ns);

    auto const st_upsf_arr = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_mean_psf_values.bin", Ns, Ne, Nd);
    assert(st_upsf_arr.dimension(0) == Nd);
    assert(st_upsf_arr.dimension(1) == Ne);
    assert(st_upsf_arr.dimension(2) == Ns);

    // SUBCASE("Check Upsf Value")
    // {
    //     for (long s = 0; s < Ns; ++s)
    //     {
    //         for (long e = 0; e < Ne; ++e)
    //         {
    //             for (long d = 0; d < Nd; ++d)
    //             {
    //                 CHECK_MESSAGE(st_upsf_arr(d, e, s) == uPsf(d, e, s),
    //                               d << " " << e << " " << s);
    //             }
    //         }
    //     }
    // }

    // SUBCASE("Check Upsf Peak")
    // {
    //     for (long s = 0; s < Ns; ++s)
    //     {
    //         for (long e = 0; e < Ne; ++e)
    //         {
    //             CHECK_MESSAGE(st_upsf_peak(e, s) == uPeak(e, s), e << " " << s);
    //         }
    //     }
    // }


    Fermi::ModelMap::MatCoord3 const pdirs
        = Fermi::ModelMap::pix_dirs_with_padding(skygeom, Nw, Nh);

    // SUBCASE("Test Pixel Directions with Padding")
    // {
    //     REQUIRE(pdirs.rows() == 102);
    //     REQUIRE(pdirs.cols() == 102);
    //
    //     for (long w = 0; w < pdirs.rows(); ++w)
    //     {
    //         for (long h = 0; h < pdirs.cols(); ++h)
    //         {
    //             auto pix = std::make_pair(h, w);
    //             REQUIRE(pdirs(w, h) == skygeom.pix2dir(pix));
    //             REQUIRE_MESSAGE(
    //                 skygeom.pix2sph(pix).first
    //                     == doctest::Approx(skygeom.dir2sph(pdirs(w, h)).first),
    //                 w << " " << h);
    //             REQUIRE_MESSAGE(
    //                 skygeom.pix2sph(pix).second
    //                     == doctest::Approx(skygeom.dir2sph(pdirs(w, h)).second),
    //                 w << " " << h);
    //         }
    //     }
    // }

    // double const ref_size = 0.2;

    for (long s = 0; s < Ns; ++s)
    {
        auto src_dir = skygeom.sph2dir(dirs[s]); // CLHEP Style 3
        // // auto src_pix = skygeom.sph2pix(dirs[s]); // Grid Style 2
        // SUBCASE("Initial Pixel Offset")
        // {
        //     for (long h = 0; h < Nh; ++h)
        //     {
        //         for (long w = 0; w < Nw; ++w)
        //         {
        //             REQUIRE_MESSAGE(
        //                 doctest::Approx(skygeom.srcpixoff(src_dir, pdirs(w+1,h+1)))
        //                         .epsilon(1e-5)
        //                     == st_off0(h, w, s),
        //                 s << " " << h << " " << w << ": ");
        //         }
        //     }
        // }

        /////
        Tensor2d const tuPsf
            = uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 }).reshape(Idx2 { Nd, Ne });
        Tensor1d const tuPeak
            = uPeak.slice(Idx2 { 0, s }, Idx2 { Ne, 1 }).reshape(Idx1 { Ne });

        Eigen::Map<Eigen::MatrixXd const> const suPsf(tuPsf.data(), Nd, Ne);
        Eigen::Map<Eigen::VectorXd const> const suPeak(tuPeak.data(), Ne);
        REQUIRE(suPsf.rows() == Nd);
        REQUIRE(suPsf.cols() == Ne);
        REQUIRE(suPeak.rows() == Ne);
        REQUIRE(suPeak.cols() == 1);

        // SUBCASE("PeakPSF")
        // {
        //     for (long e = 0; e < Ne; ++e)
        //     {
        //         REQUIRE_MESSAGE(uPeak(e, s) == doctest::Approx(suPeak(e)),
        //                         s << " " << e);
        //         REQUIRE_MESSAGE(st_upsf_peak(e, s) == doctest::Approx(suPeak(e)),
        //                         s << " " << e);
        //     }
        // }

        // Eigen::MatrixXd const Offsets
        //     = Fermi::ModelMap::pixel_angular_offset_from_source_with_padding(
        //         pdirs, src_dir, src_pix, ref_size, skygeom);
        // REQUIRE(Offsets.rows() == 102);
        // REQUIRE(Offsets.cols() == 102);

        Eigen::ArrayXd isep
            = Fermi::ModelMap::psf_idx_sep(src_dir, pdirs, skygeom).reshaped().array();

        SUBCASE("Pixel PSF logspace displacement")
        {
            auto sep_v = Fermi::PSF::separations();
            for (long h = 0; h < Nh; ++h)
            {
                for (long w = 0; w < Nw; ++w)
                {

                    long   logidx = long(isep.reshaped(Nw, Nh)(w, h));
                    double resid  = isep.reshaped(Nw, Nh)(w, h) - logidx;

                    double sep    = skygeom.srcpixoff(src_dir, pdirs(w + 1, h + 1));
                    auto   sepi   = std::upper_bound(sep_v.begin(), sep_v.end(), sep);
                    REQUIRE_MESSAGE(logidx == std::distance(sep_v.begin(), (sepi - 1)),
                                    s << " " << h << " " << w << ": "
                                      << isep.reshaped(Nw, Nh)(w, h) << " " << sep
                                      << " [" << *(sepi - 1) << " " << *sepi << "]");
                    REQUIRE_MESSAGE(logidx == st_offi0(h, w, s),
                                    s << " " << h << " " << w << ": "
                                      << isep.reshaped(Nw, Nh)(w, h) << " " << sep
                                      << "  " << resid);
                    REQUIRE_MESSAGE(doctest::Approx(resid) == 1. - st_offs0(h, w, s),
                                    s << " " << h << " " << w << ": "
                                      << isep.reshaped(Nw, Nh)(w, h) << " " << sep
                                      << "  " << logidx);
                }
            }
        }

        /// Compute the initial mean psf for every pixel vs this source.
        Eigen::MatrixXd psfEst   = Fermi::ModelMap::psf_lut(isep, suPsf);
        Eigen::MatrixXd init_psf = psfEst;

        REQUIRE(psfEst.rows() == Nw * Nh);
        REQUIRE(psfEst.cols() == Ne);

        // SUBCASE("Initial Pixel PSF")
        // {
        //     for (long h = 0; h < Nh; ++h)
        //     {
        //         for (long w = 0; w < Nw; ++w)
        //         {
        //             for (long e = 0; e < Ne; ++e)
        //             {
        //                 REQUIRE_MESSAGE(
        //                     doctest::Approx(psfEst(w + h * Nw, e)).epsilon(1e-3)
        //                         == st_v0(e, h, w, s),
        //                     s << " " << w << " " << h << " " << e << "\n");
        //             }
        //         }
        //     }
        // }

        double const peak_threshold = 1e-6;
        double const ftol_threshold = 1e-3;

        /// Which pixels need further psf integration because their psf value is too
        /// close to the peak psf?
        auto pxs_int                = Fermi::ModelMap::pixels_to_integrate(
            psfEst, suPeak, peak_threshold, Nw, Nh);

        Eigen::MatrixX<bool> const needs_more
            = (((init_psf.array().rowwise() / suPeak.transpose().array())
                >= peak_threshold)
                   .rowwise()
                   .any())
                  .reshaped(Nw, Nh);

        // SUBCASE("Check Needs More and pxs_int")
        // {
        //     REQUIRE(needs_more.count() == pxs_int.size());
        //     for (long h = 0; h < Nh; ++h)
        //     {
        //         for (long w = 0; w < Nw; ++w)
        //         {
        //             bool st_nm = false;
        //             for (long e = 0; e < Ne; ++e)
        //             {
        //                 st_nm = st_nm || st_nm0(e, h, w, s);
        //             }
        //             REQUIRE_MESSAGE(needs_more(w, h) == st_nm,
        //                             s << " " << h << " " << w << "\n");
        //
        //             REQUIRE_MESSAGE(
        //                 std::find(pxs_int.begin(), pxs_int.end(), std::make_pair(w,
        //                 h))
        //                     != pxs_int.end(),
        //                 s << " " << h << " " << w << "\n");
        //         }
        //     }
        // }


        /// Integrate the necessary pixels.
        for (auto const& p : pxs_int)
        {
            long const w       = p.first;
            long const h       = p.second;

            // Eigen::Matrix<double, 2, 2> sep
            //     = Fermi::ModelMap::rectangular_true_separations<2>(
            //         w, h, src_dir, skygeom);
            // sep = (sep.array() < 1e-4)
            //           .select(1e4 * sep, 1. + ((sep.array() * 1e4).log() /
            //           sep_step));
            // auto y0 = sep.array().floor();
            // auto y1 = sep.array().ceil();
            // auto x0
            //     = 1e-4
            //       * (y0 >= 1.).select(Eigen::pow(Fermi::PSF::sep_delta, (y0 - 1.)),
            //       y0);
            // auto x1
            //     = 1e-4
            //       * (y1 >= 1.).select(Eigen::pow(Fermi::PSF::sep_delta, (y1 - 1.)),
            //       y1);
            // sep = (sep.array() - x0).array() * ((y1 - y0).array() / (x1 -
            // x0).array())
            //       + y0;
            //
            // auto asep = sep.reshaped().array();
            //
            // Eigen::MatrixXd v1(1, Ne);
            // // Fermi::ModelMap::mean_psf_lut(v1, sep.reshaped().array(), suPsf);
            // Eigen::ArrayXd r = asep - asep.floor();
            //
            // v1 = (suPsf(asep.floor(), Eigen::all).array().colwise() * (1. - r)
            //           + suPsf(asep.floor() + 1, Eigen::all).array().colwise() * r)
            //              .colwise()
            //              .mean();
            // REQUIRE(v1.rows() == 1);
            // REQUIRE(v1.cols() == 38);

            Eigen::MatrixXd v1 = Fermi::ModelMap::integrate_psf_recursive(
                w,
                h,
                src_dir,
                skygeom,
                ftol_threshold,
                suPsf,
                psfEst.block(w + h * Nw, 0, 1, Ne));
            REQUIRE(v1.rows() == 1);
            REQUIRE(v1.cols() == 38);
            psfEst.block(w + h * Nw, 0, 1, Ne) = v1;
            for (long e = 0; e < Ne; ++e)
            {
                REQUIRE_MESSAGE(
                    doctest::Approx(psfEst(w + h * Nw, e)).epsilon(1e-3)
                    // doctest::Approx(v1(0, e)).epsilon(1e-4)
                                == st_psfEst(e, h, w, s),
                    s << " " << h << " " << w << " " << e << "\n"
                      << needs_more(w, h) << " " << st_nm0(e, h, w, s) << "\n"
                      << init_psf(w + h * Nw, e) << " " << st_v0(e, h, w, s) << "\n");
            }
        }

        // SUBCASE("Final Mean PSF")
        // {
        //     // std::cout << needs_more.count() << "\n" << needs_more << std::endl <<
        //     // std::endl;
        //     for (long h = 0; h < Nh; ++h)
        //     {
        //         for (long w = 0; w < Nw; ++w)
        //         {
        //             for (long e = 0; e < Ne; ++e)
        //             {
        //                 REQUIRE_MESSAGE(
        //                     doctest::Approx(psfEst(w + h * Nw, e)).epsilon(1e-3)
        //                         == st_psfEst(e, h, w, s),
        //                     s << " " << h << " " << w << " " << e << "\n"
        //                       << needs_more(w, h) << " " << st_nm0(e, h, w, s) <<
        //                       "\n"
        //                       << init_psf(w + h * Nw, e) << " " << st_v0(e, h, w, s)
        //                       << "\n"
        //                       << needs_more);
        //             }
        //         }
        //     }
        // }
    }
}
