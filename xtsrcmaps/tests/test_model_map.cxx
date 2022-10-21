
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
                 T                                        SD)
{
    long constexpr N  = SD.rows();
    auto constexpr ds = Fermi::ModelMap::integ_delta_steps<N>();
    for (long w = 0; w < N; ++w)
    {
        for (long h = 0; h < N; ++h)
        {

            REQUIRE_MESSAGE(
                SD(w, h)
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds[h], pw + ds[w] }))),
                N << " " << w << " " << h);
        }
    }
}


TEST_CASE("Test Model Map Pixel Displacements")
{

    long const NpW       = 100;
    long const NpH       = 100;
    long const NpW_      = NpW + 2;
    long const NpH_      = NpH + 2;
    auto       cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::directions_from_point_sources(srcs);
    // long const Ns        = srcs.size();

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(opt_ccube);
    auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
    Fermi::SkyGeom skygeom(ccube);
    auto           src_dir  = skygeom.sph2dir(dirs[0]); // CLHEP Style 3
    auto           src_pix  = skygeom.sph2pix(dirs[0]); // Grid Style 2
    double const   ref_size = 0.2;

    Eigen::MatrixXd Offsets(NpW_, NpH_);
    Offsets.setZero();
    for (long pw = 0; pw < NpW_; ++pw)
    {
        for (long ph = 0; ph < NpH_; ++ph)
        {
            auto   pdir    = skygeom.pix2dir({ ph, pw });
            double pix_sep = ref_size
                             * std::sqrt(std::pow(src_pix.first - (ph), 2)
                                         + std::pow(src_pix.second - (pw), 2));
            double ang_sep = skygeom.srcpixoff(
                src_dir, { std::get<0>(pdir), std::get<1>(pdir), std::get<2>(pdir) });
            Offsets(pw, ph) = pix_sep > 1E-6 ? ang_sep / pix_sep - 1. : 0.0;
        }
    }

    // // // Eigen::MatrixXd SepOffsets(NpW_, NpH_);
    // // // SepOffsets.setZero();
    // // // for (long pw = 0; pw < NpW_; ++pw)
    // // // {
    // // //     for (long ph = 0; ph < NpH_; ++ph)
    // // //     {
    // // //         auto   pdir    = skygeom.pix2dir({ ph, pw });
    // // //         double ang_sep = skygeom.srcpixoff(
    // // //             src_dir, { std::get<0>(pdir), std::get<1>(pdir),
    // std::get<2>(pdir) });
    // // //         SepOffsets(pw, ph) = ang_sep;
    // // //     }
    // // // }


    // std::ifstream ifs("./xtsrcmaps/tests/expected/src_pix_offset.bin",
    //                   std::ios::in | std::ios::binary);

    // std::array<long, 7> npts       = { 1, 4, 16, 64, 256, 1024, 4096 };
    // size_t const        bufsz      = 100 * 100 * 5461;
    // size_t const        bufszbytes = bufsz * sizeof(double);

    // for (int s = 0; s < Ns; ++s)
    // {
    // std::vector<double> rmbuf(bufsz);
    // ifs.read((char*)(&rmbuf[0]), bufszbytes);
    // Tensor3d const A
    //     = Fermi::row_major_buffer_to_col_major_tensor(rmbuf.data(), 100, 100, 5461);

    auto constexpr ds2 = Fermi::ModelMap::integ_delta_steps<2>();
    // auto constexpr ds4  = Fermi::ModelMap::integ_delta_steps<4>();
    // auto constexpr ds8  = Fermi::ModelMap::integ_delta_steps<8>();
    // auto constexpr ds16 = Fermi::ModelMap::integ_delta_steps<16>();
    // auto constexpr ds32 = Fermi::ModelMap::integ_delta_steps<32>();
    // auto constexpr ds64 = Fermi::ModelMap::integ_delta_steps<64>();
    for (long pw = 1; pw <= NpW; ++pw)
    {
        for (long ph = 1; ph <= NpH; ++ph)
        {


            // REQUIRE_MESSAGE(skygeom.srcpixoff(src_dir, skygeom.pix2dir({ ph, pw }))
            //                     == doctest::Approx(A(pw - 1, ph - 1, 0)),
            //                 ph << " " << pw);
            // /// 2
            // for (long i = 0; i < npts[1]; ++i)
            // {
            //     double z = skygeom.srcpixoff(
            //         src_dir, skygeom.pix2dir({ ph + ds2[i / 2], pw + ds2[i % 2] }));
            //     REQUIRE_MESSAGE(z == doctest::Approx(A(pw - 1, ph - 1, 1 + i)),
            //                     ph << " " << pw);
            // }
            // Eigen::TensorMap<Tensor2d const> const D(delta_arr2.data(), 3, 2);


            /// Split Lerp Sep copmutation.
            auto constexpr delta_lo2 = Fermi::ModelMap::integ_delta_lo1<2>();
            auto constexpr delta_hi2 = Fermi::ModelMap::integ_delta_hi1<2>();
            Eigen::Map<Eigen::Matrix<double, 2, 1> const> const Dlo(delta_lo2.data());
            Eigen::Map<Eigen::Matrix<double, 2, 1> const> const Dhi(delta_hi2.data());
            REQUIRE(Dlo.rows() == 2);
            REQUIRE(Dlo.cols() == 1);
            REQUIRE(Dhi.rows() == 2);
            REQUIRE(Dhi.cols() == 1);


            Eigen::Matrix<double, 2, 2> SD;
            Eigen::Matrix<double, 3, 1> ID = Offsets.block<3, 2>(pw - 1, ph - 1) * Dlo;
            SD.block<1, 1>(0, 0).noalias() = Dlo.transpose() * ID.block<2, 1>(0, 0);
            SD.block<1, 1>(1, 0).noalias() = Dhi.transpose() * ID.block<2, 1>(1, 0);
            ID.noalias()                   = Offsets.block<3, 2>(pw - 1, ph) * Dhi;
            SD.block<1, 1>(0, 1).noalias() = Dlo.transpose() * ID.block<2, 1>(0, 0);
            SD.block<1, 1>(1, 1).noalias() = Dhi.transpose() * ID.block<2, 1>(1, 0);
            REQUIRE(ref_size
                        * std::sqrt(std::pow(src_pix.first - ph - ds2[0], 2)
                                    + std::pow(src_pix.second - pw - ds2[0], 2))
                        * (1. + SD(0, 0))
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[0], pw + ds2[0] }))));

            REQUIRE(ref_size
                        * std::sqrt(std::pow(src_pix.second - pw - ds2[1], 2)
                                    + std::pow(src_pix.first - ph - ds2[0], 2))
                        * (1. + SD(1, 0))
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[0], pw + ds2[1] }))));

            REQUIRE(ref_size
                        * std::sqrt(std::pow(src_pix.second - pw - ds2[0], 2)
                                    + std::pow(src_pix.first - ph - ds2[1], 2))
                        * (1. + SD(0, 1))
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[1], pw + ds2[0] }))));

            REQUIRE(ref_size
                        * std::sqrt(std::pow(src_pix.first - ph - ds2[1], 2)
                                    + std::pow(src_pix.second - pw - ds2[1], 2))
                        * (1. + SD(1, 1))
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[1], pw + ds2[1] }))));



            /// Combined Lerp Sep copmutation.
            auto constexpr delta_1 = Fermi::ModelMap::integ_delta_1<2>();
            Eigen::Map<Eigen::Matrix<double, 3, 2> const> const D(delta_1.data());
            REQUIRE(D.rows() == 3);
            REQUIRE(D.cols() == 2);

            SD.noalias() = D.transpose() * Offsets.block<3, 3>(pw - 1, ph - 1) * D;
            REQUIRE(ref_size
                        * std::sqrt(std::pow(src_pix.first - ph - ds2[0], 2)
                                    + std::pow(src_pix.second - pw - ds2[0], 2))
                        * (1. + SD(0, 0))
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[0], pw + ds2[0] }))));

            REQUIRE(ref_size
                        * std::sqrt(std::pow(src_pix.second - pw - ds2[1], 2)
                                    + std::pow(src_pix.first - ph - ds2[0], 2))
                        * (1. + SD(1, 0))
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[0], pw + ds2[1] }))));

            REQUIRE(ref_size
                        * std::sqrt(std::pow(src_pix.second - pw - ds2[0], 2)
                                    + std::pow(src_pix.first - ph - ds2[1], 2))
                        * (1. + SD(0, 1))
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[1], pw + ds2[0] }))));

            REQUIRE(ref_size
                        * std::sqrt(std::pow(src_pix.first - ph - ds2[1], 2)
                                    + std::pow(src_pix.second - pw - ds2[1], 2))
                        * (1. + SD(1, 1))
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[1], pw + ds2[1] }))));


            /// Mat Delta computation.
            Eigen::Vector2d spv(src_pix.first - ph, src_pix.second - pw);
            auto constexpr dsmatv = Fermi::ModelMap::integ_delta_lin<2>();
            Eigen::Map<Eigen::Matrix<double, 2, 2 * 2> const> const dsm(
                dsmatv.data(), 2, 4);

            Eigen::Matrix<double, 2, 2> subpixel_offset_size
                = (dsm.colwise() - spv).colwise().norm().reshaped(2, 2);

            SD = ref_size * subpixel_offset_size.array() * (1. + SD.array());

            REQUIRE(SD(0, 0)
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[0], pw + ds2[0] }))));
            REQUIRE(SD(1, 0)
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[0], pw + ds2[1] }))));
            REQUIRE(SD(0, 1)
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[1], pw + ds2[0] }))));
            REQUIRE(SD(1, 1)
                    == doctest::Approx(skygeom.srcpixoff(
                        src_dir, skygeom.pix2dir({ ph + ds2[1], pw + ds2[1] }))));
            SD.setZero();


            // // // SepOffset based computation.
            // // SD.setZero();
            // // SD.noalias() = D.transpose() * SepOffsets.block<3, 3>(pw - 1, ph - 1)
            // // * D;
            // // REQUIRE(SD(0, 0)
            // //         == doctest::Approx(skygeom.srcpixoff(
            // //             src_dir, skygeom.pix2dir({ ph + ds2[0], pw + ds2[0] }))));
            // // REQUIRE(SD(1, 0)
            // //         == doctest::Approx(skygeom.srcpixoff(
            // //             src_dir, skygeom.pix2dir({ ph + ds2[0], pw + ds2[1] }))));
            // // REQUIRE(SD(0, 1)
            // //         == doctest::Approx(skygeom.srcpixoff(
            // //             src_dir, skygeom.pix2dir({ ph + ds2[1], pw + ds2[0] }))));
            // // REQUIRE(SD(1, 1)
            // //         == doctest::Approx(skygeom.srcpixoff(
            // //             src_dir, skygeom.pix2dir({ ph + ds2[1], pw + ds2[1] }))));

            // Using Comb Separations Function.
            SD = Fermi::ModelMap::rectangular_comb_separations<2>(
                pw, ph, ref_size, src_pix, Offsets);
            req_displacement(ph, pw, src_dir, skygeom, SD);


            Eigen::Matrix<double, 4, 4> SD4
                = Fermi::ModelMap::rectangular_comb_separations<4>(
                    pw, ph, ref_size, src_pix, Offsets);
            req_displacement(ph, pw, src_dir, skygeom, SD4);


            Eigen::Matrix<double, 8, 8> SD8
                = Fermi::ModelMap::rectangular_comb_separations<8>(
                    pw, ph, ref_size, src_pix, Offsets);
            req_displacement(ph, pw, src_dir, skygeom, SD8);


            Eigen::Matrix<double, 16, 16> SD16
                = Fermi::ModelMap::rectangular_comb_separations<16>(
                    pw, ph, ref_size, src_pix, Offsets);
            req_displacement(ph, pw, src_dir, skygeom, SD16);


            Eigen::Matrix<double, 32, 32> SD32
                = Fermi::ModelMap::rectangular_comb_separations<32>(
                    pw, ph, ref_size, src_pix, Offsets);
            req_displacement(ph, pw, src_dir, skygeom, SD32);


            Eigen::Matrix<double, 64, 64> SD64
                = Fermi::ModelMap::rectangular_comb_separations<64>(
                    pw, ph, ref_size, src_pix, Offsets);
            req_displacement(ph, pw, src_dir, skygeom, SD64);


            // Eigen::Matrix3d P = Offsets.block<3, 3>(pw - 1, ph - 1);
            // // double dstep(0.5);
            // long k            = 0;
            // for (long h(0); h < 2; h++)
            // {
            //     // double dh = i * dstep - 0.5 + (0.5 * dstep);
            //     double dh = ds2[h];
            //     double nh(ph + dh);
            //     for (long w(0); w < 2; w++)
            //     {
            //         // double dw = j * dstep - 0.5 + (0.5 * dstep);
            //         double dw = ds2[w];
            //         double nw(pw + dw);
            //         double pix_offset
            //             = ref_size
            //               * std::sqrt(std::pow(src_pix.first - ph - dh, 2)
            //                           + std::pow(src_pix.second - pw - dw, 2));
            //         long iw      = dw < 0 ? pw - 1 : pw; // long(std::floor(nw));
            //         long ih      = dh < 0 ? ph - 1 : ph; // long(std::floor(nh));
            //         // double rw = nw - iw; // = (pw + dw) - (pw + (-1))
            //         // double rh = nh - ih; // = (ph + dh) - (ph + (-1))
            //         double rw    = dw < 0 ? dw + 1 : dw;
            //         double rh    = dh < 0 ? dh + 1 : dh;
            //         double sw    = (1. - rw);
            //         double sh    = (1. - rh);
            //         // CHECK_MESSAGE(D(0, h) == (dh < 0 ? sh : 0.0), dh << " " <<
            //         dw);
            //         // CHECK_MESSAGE(D(1, h) == (dh < 0 ? rh : sh), dh << " " << dw);
            //         // CHECK_MESSAGE(D(2, h) == (dh < 0 ? 0.0 : rh), dh << " " <<
            //         dw);
            //         // REQUIRE_MESSAGE(D(0, h) == (dh < 0 ? sh : 0.0), dh << " " <<
            //         dw);
            //         // double z1 = Offsets(iw, ih);
            //         // double z2 = Offsets(iw, ih + 1);
            //         // double z3 = Offsets(iw + 1, ih);
            //         // double z4 = Offsets(iw + 1, ih + 1);
            //         double z1    = P(w, h);
            //         double z2    = P(w, h + 1);
            //         double z3    = P(w + 1, h);
            //         double z4    = P(w + 1, h + 1);
            //         double scale = (z1 * sw * sh   //
            //                         + z2 * sw * rh //
            //                         + z3 * rw * sh //
            //                         + z4 * rw * rh);
            //         // REQUIRE_MESSAGE(scale == doctest::Approx(SD(w,h)) , w << " "
            //         <<
            //         // h);
            //         pix_offset *= 1. + scale;
            //         double true_off = skygeom.srcpixoff(
            //             src_dir, skygeom.pix2dir({ ph + ds2[k / 2], pw + ds2[k % 2]
            //             }));
            //         double pix_off
            //             = skygeom.srcpixoff(src_dir, skygeom.pix2dir({ nh, nw }));
            //         double st_off = A(pw - 1, ph - 1, 1 + h * 2 + w);
            //
            //         REQUIRE_MESSAGE(pix_off == doctest::Approx(true_off),
            //                         pw << " " << ph << ": " << w << " "
            //                            << h
            //                            // << " [" << x << " " << y << "]"
            //                            << " [" << dh << " " << dw << "]"
            //                            << " [" << ds2[k / 2] << " " << ds2[k % 2] <<
            //                            "]"
            //                            << " [" << ph + ds2[k / 2] << " "
            //                            << ph + ds2[k % 2] << "]");
            //         REQUIRE_MESSAGE(pix_off == doctest::Approx(st_off),
            //                         pw << " " << ph << ": " //
            //                            << w << " " << h     //
            //                            << " [" << dw << " " << dh << "]"
            //
            //         );
            //         REQUIRE_MESSAGE(pix_offset == doctest::Approx(true_off),
            //                         "pw ph = " << pw << " " << ph << ": "
            //                                    << " " << h << " " << w << " "
            //                                    << "<" << k << ">"
            //                                    << " (" << dw << " " << dh << ")"
            //                                    << " (" << nh << " " << nw << ")"
            //                                    << " [" << iw << " " << ih << " : " <<
            //                                    rw
            //                                    << " " << rh << "]"
            //                                    << "  ==> " << z1 << " " << z2 << " "
            //                                    << z3 << " " << z4
            //
            //         );
            //         ++k;
            //     }
            // }


            //
            // /// 4
            // for (long i = 0; i < npts[2]; ++i)
            // {
            //     REQUIRE_MESSAGE(skygeom.srcpixoff(src_dir,
            //                                       skygeom.pix2dir({ ph + ds4[i / 4],
            //                                                         pw + ds4[i % 4]
            //                                                         }))
            //                         == doctest::Approx(A(pw - 1, ph - 1, 5 + i)),
            //                     ph << " " << pw);
            // }
            // /// 8
            // for (long i = 0; i < npts[3]; ++i)
            // {
            //     REQUIRE_MESSAGE(skygeom.srcpixoff(src_dir,
            //                                       skygeom.pix2dir({ ph + ds8[i / 8],
            //                                                         pw + ds8[i % 8]
            //                                                         }))
            //                         == doctest::Approx(A(pw - 1, ph - 1, 21 + i)),
            //                     ph << " " << pw);
            // }
            // /// 16
            // for (long i = 0; i < npts[4]; ++i)
            // {
            //     REQUIRE_MESSAGE(
            //         skygeom.srcpixoff(
            //             src_dir,
            //             skygeom.pix2dir({ ph + ds16[i / 16], pw + ds16[i % 16] }))
            //             == doctest::Approx(A(pw - 1, ph - 1, 85 + i)),
            //         ph << " " << pw);
            // }
            // /// 32
            // for (long i = 0; i < npts[5]; ++i)
            // {
            //     REQUIRE_MESSAGE(
            //         skygeom.srcpixoff(
            //             src_dir,
            //             skygeom.pix2dir({ ph + ds32[i / 32], pw + ds32[i % 32] }))
            //             == doctest::Approx(A(pw - 1, ph - 1, 341 + i)),
            //         ph << " " << pw);
            // }
            // /// 64
            // for (long i = 0; i < npts[6]; ++i)
            // {
            //     REQUIRE_MESSAGE(
            //         skygeom.srcpixoff(
            //             src_dir,
            //             skygeom.pix2dir({ ph + ds64[i / 64], pw + ds64[i % 64] }))
            //             == doctest::Approx(A(pw - 1, ph - 1, 1365 + i)),
            //         ph << " " << pw);
            // }
        }
    }
}

TEST_CASE("Test Model Map Pixel Offsets")
{

    long const NpW       = 100;
    long const NpH       = 100;
    // long const NpW_      = NpW + 2;
    // long const NpH_      = NpH + 2;
    auto       cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::directions_from_point_sources(srcs);
    // long const Ns        = srcs.size();

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(opt_ccube);
    auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
    Fermi::SkyGeom skygeom(ccube);
    auto           src_dir  = skygeom.sph2dir(dirs[0]); // CLHEP Style 3
    auto           src_pix  = skygeom.sph2pix(dirs[0]); // Grid Style 2
    double const   ref_size = 0.2;

    Tensor2d Offsets(NpW, NpH);
    Offsets.setZero();
    for (long pw = 0; pw < NpW; ++pw)
    {
        for (long ph = 0; ph < NpH; ++ph)
        {
            auto   pdir    = skygeom.pix2dir({ ph + 1, pw + 1 });
            double pix_sep = ref_size
                             * std::sqrt(std::pow(src_pix.first - (ph + 1), 2)
                                         + std::pow(src_pix.second - (pw + 1), 2));
            double ang_sep = skygeom.srcpixoff(
                src_dir, { std::get<0>(pdir), std::get<1>(pdir), std::get<2>(pdir) });
            Offsets(pw, ph) = pix_sep > 1E-6 ? ang_sep / pix_sep - 1. : 0.0;
        }
    }

    std::ifstream ifs("./xtsrcmaps/tests/expected/src_pixelOffset.bin",
                      std::ios::in | std::ios::binary);

    size_t const bufsz      = 100 * 100;
    size_t const bufszbytes = bufsz * sizeof(double);

    std::vector<double> rmbuf(bufsz);
    ifs.read((char*)(&rmbuf[0]), bufszbytes);
    Tensor2d const stpixOff
        = Fermi::row_major_buffer_to_col_major_tensor(rmbuf.data(), NpW, NpH);
    for (long pw = 0; pw < NpW; ++pw)
    {
        for (long ph = 0; ph < NpH; ++ph)
        {
            REQUIRE_MESSAGE(Offsets(pw, ph) == doctest::Approx(stpixOff(pw, ph)),
                            pw << " " << ph);
        }
    }
}

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
