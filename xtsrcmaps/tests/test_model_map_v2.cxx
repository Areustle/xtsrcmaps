
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

TEST_CASE("Test Model Map pixel mean psf")
{

    auto       cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::directions_from_point_sources(srcs);

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(opt_ccube);
    auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
    Fermi::SkyGeom skygeom(ccube);
    long const     Ne  = 38;
    long const     Nh  = 100;
    long const     Nw  = 100;
    long const     Ns  = dirs.size();
    long const     Nd  = 401;

    Tensor4d st_psfEst = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_psfEstimate.bin", Ns, Nw, Nh, Ne);
    assert(st_psfEst.dimension(0) == Ne);
    assert(st_psfEst.dimension(1) == Nh);
    assert(st_psfEst.dimension(2) == Nw);
    assert(st_psfEst.dimension(3) == Ns);

    Tensor3d norm_uPsf = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/uPsf_normalized_SED.bin", Ns, Ne, Nd);
    assert(norm_uPsf.dimension(0) == Nd);
    assert(norm_uPsf.dimension(1) == Ne);
    assert(norm_uPsf.dimension(2) == Ns);

    // Tensor4d psfEst
    //     = Fermi::ModelMap::pixel_mean_psf(100, 100, dirs, norm_uPsf, { ccube },
    //     1e-3);
    //
    // SUBCASE("Pixel PSF Integral Estimate")
    // {
    //
    //     Tensor4d psfreldiff = (psfEst - st_psfEst).abs() / st_psfEst;
    //
    //     long count_pixels_more_than_1_pct_diff
    //         = std::count_if(psfreldiff.data(),
    //                         psfreldiff.data() + psfreldiff.size(),
    //                         [](auto x) -> bool { return x > 1e-2; });
    //
    //     CHECK_MESSAGE(count_pixels_more_than_1_pct_diff * 100 < psfreldiff.size(),
    //                   "Too many pixels differ from Fermitools Psf Estimates.");
    //
    //     if (count_pixels_more_than_1_pct_diff * 100 > psfreldiff.size())
    //     {
    //         for (long s = 0; s < Ns; ++s)
    //         {
    //             for (long w = 0; w < Nw; ++w)
    //             {
    //                 for (long h = 0; h < Nh; ++h)
    //                 {
    //                     for (long e = 0; e < Ne; ++e)
    //                     {
    //                         CHECK_MESSAGE(
    //                             doctest::Approx(psfEst(e, h, w, s)).epsilon(1e-2)
    //                                 == st_psfEst(e, h, w, s),
    //                             e << " " << h << " " << w << " " << s);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    Tensor3d const init_points = Fermi::ModelMap::get_init_points(Nh, Nw);
    Tensor2d       sang        = Fermi::ModelMap::solid_angle(init_points, skygeom);

    Tensor4d ft_psf_sang       = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/ft_mm_psf_sang.bin", Ns, Nw, Nh, Ne);

    Fermi::ModelMap::scale_map_by_solid_angle(st_psfEst, skygeom);

    SUBCASE("Scaled ModelMap By Solid Angle")
    {
        for (long s = 0; s < Ns; ++s)
        {
            for (long w = 0; w < Nw; ++w)
            {
                for (long h = 0; h < Nh; ++h)
                {
                    for (long e = 0; e < Ne; ++e)
                    {
                        CHECK_MESSAGE(
                            doctest::Approx(st_psfEst(e, h, w, s)).epsilon(1e-2)
                                == ft_psf_sang(e, h, w, s),
                            e << " " << h << " " << w << " " << s);
                    }
                }
            }
        }
    }

    // PSF_boundary_radius
    //

    auto [full_psf_radius, is_in_fov]
        = Fermi::ModelMap::psf_boundary_radius(Nh, Nw, dirs, skygeom);
    Tensor1d psf_radius = Fermi::filter_in(full_psf_radius, is_in_fov);

    long const Nf
        = std::count(is_in_fov.data(), is_in_fov.data() + is_in_fov.size(), true);

    REQUIRE(Nf == 47);
    REQUIRE(psf_radius.dimension(0) == Nf);
    REQUIRE(psf_radius.size() == Nf);

    Tensor1d const ft_psf_brad = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/ft_mm_psf_radius.bin", Ns);

    Tensor1b const ft_in_fov = Fermi::row_major_file_to_col_major_tensor<bool>(
        "./xtsrcmaps/tests/expected/ft_mm_withinBounds.bin", Ns);

    SUBCASE("PSF_boundary_radius")
    {
        for (long s = 0; s < Ns; ++s)
        {
            auto xtr = full_psf_radius(s);
            CHECK_MESSAGE(doctest::Approx(xtr) == ft_psf_brad(s), s);
            CHECK_MESSAGE(is_in_fov(s) == ft_in_fov(s),
                          s << " (" << skygeom.sph2pix(dirs[s]).first << ", "
                            << skygeom.sph2pix(dirs[s]).second << ")");
        }
    }

    // map_integral
    //

    Tensor2d const inv_mapinteg = Fermi::ModelMap::map_integral(
        st_psfEst, dirs, skygeom, psf_radius, is_in_fov);

    REQUIRE(Ne == inv_mapinteg.dimension(0));
    REQUIRE(Nf == inv_mapinteg.dimension(1));

    Tensor2d const ft_map_integ = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/ft_mm_mapIntegrals.bin", Nf, Ne);

    REQUIRE(Ne == ft_map_integ.dimension(0));
    REQUIRE(Nf == ft_map_integ.dimension(1));

    SUBCASE("Modelmap mapIntegrals")
    {
        for (long f = 0; f < Nf; ++f)
        {
            for (long e = 0; e < Ne; ++e)
            {
                CHECK_MESSAGE(doctest::Approx(inv_mapinteg(e, f))
                                  == (1. / ft_map_integ(e, f)),
                              e << " " << f);
            }
        }
    }

    Tensor3d unnorm_uPsf = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/uPsf_SED.bin", Ns, Ne, Nd);
    auto [part_psf_integ, psf_integ] = Fermi::PSF::partial_total_integral(unnorm_uPsf);

    Tensor3d filtered_psf(Nd, Ne, Nf);
    Tensor3d filtered_parint(Nd, Ne, Nf);

    long f = 0;
    for (long s = 0; s < Ns; ++s)
    {
        if (!is_in_fov(s)) { continue; }
        filtered_psf.slice(Idx3 { 0, 0, f }, Idx3 { Nd, Ne, 1 })
            = norm_uPsf.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 });
        filtered_parint.slice(Idx3 { 0, 0, f }, Idx3 { Nd, Ne, 1 })
            = part_psf_integ.slice(Idx3 { 0, 0, s }, Idx3 { Nd, Ne, 1 });
        ++f;
    }

    Tensor2d const rad_integ
        = Fermi::ModelMap::integral(psf_radius, filtered_psf, filtered_parint);

    Tensor2d ft_rad_int = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/ft_mm_upsf_radius_integ.bin", Nf, Ne);

    SUBCASE("Modelmap integral")
    {
        long f = 0;
        for (long s = 0; s < Ns; ++s)
        {
            if (!is_in_fov(s)) { continue; }
            else
            {
                for (long e = 0; e < Ne; ++e)
                {
                    REQUIRE_MESSAGE(doctest::Approx(rad_integ(e, f))
                                        == (ft_rad_int(e, f)),
                                    e << " " << s << " f= " << f);
                }
                ++f;
            }
        }
    }

    Tensor2d const exposure = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/exposure.bin", Ns, Ne);

    Fermi::ModelMap::scale_map_by_exposure(st_psfEst, exposure);

    Tensor2d const correction_factor = Fermi::ModelMap::map_correction_factor(
        inv_mapinteg, psf_radius, is_in_fov, norm_uPsf, part_psf_integ);

    REQUIRE(correction_factor.dimension(0) == Ne);
    REQUIRE(correction_factor.dimension(1) == Ns);

    Tensor2d const ft_corfac = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/ft_mm_correction_factor.bin", Nf, Ne);

    SUBCASE("Modelmap correction_factor")
    {
        long f = 0;
        for (long s = 0; s < Ns; ++s)
        {
            if (!is_in_fov(s))
            {
                for (long e = 0; e < Ne; ++e)
                {
                    REQUIRE_MESSAGE(doctest::Approx(correction_factor(e, s)) == 1.0,
                                    e << " " << s);
                }
            }
            else
            {
                for (long e = 0; e < Ne; ++e)
                {
                    REQUIRE_MESSAGE(doctest::Approx(correction_factor(e, s))
                                        == (ft_corfac(e, f)),
                                    e << " " << s << " f= " << f);
                }
                ++f;
            }
        }
    }

    Fermi::ModelMap::scale_map_by_correction_factors(st_psfEst, correction_factor);

    Tensor4d const ft_modelmap = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/ft_model_map.bin", Ns, Ne, Nw, Nh);

    SUBCASE("Scaled ModelMap Final")
    {
        for (long s = 0; s < Ns; ++s)
        {
            for (long w = 0; w < Nw; ++w)
            {
                for (long h = 0; h < Nh; ++h)
                {
                    for (long e = 0; e < Ne; ++e)
                    {
                        REQUIRE_MESSAGE(
                            doctest::Approx(st_psfEst(e, h, w, s)).epsilon(1e-2)
                                == ft_modelmap(h, w, e, s),
                            e << " " << h << " " << w << " " << s);
                    }
                }
            }
        }
    }
}
