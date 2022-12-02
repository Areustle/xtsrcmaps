
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

TEST_CASE("Test Model Map Execution")
{

    auto       cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::directions_from_point_sources(srcs);

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(opt_ccube);
    auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
    Fermi::SkyGeom skygeom(ccube);
    long const     Nw        = 100;
    long const     Nh        = 100;
    long const     Ns        = dirs.size();
    long const     Nd        = 401;
    long const     Ne        = 38;

    // auto const st_off0 = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/src_initial_offset.bin", Ns, Nw, Nh);
    // assert(st_off0.dimension(0) == Nh);
    // assert(st_off0.dimension(1) == Nw);
    // assert(st_off0.dimension(2) == Ns);
    //
    // auto const st_offi0 = Fermi::row_major_file_to_col_major_tensor<unsigned short>(
    //     "./xtsrcmaps/tests/expected/src_initial_offset_idx.bin", Ns, Nw, Nh);
    // assert(st_offi0.dimension(0) == Nh);
    // assert(st_offi0.dimension(1) == Nw);
    // assert(st_offi0.dimension(2) == Ns);
    //
    // auto const st_offs0 = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/src_initial_offset_scalar.bin", Ns, Nw, Nh);
    // assert(st_offs0.dimension(0) == Nh);
    // assert(st_offs0.dimension(1) == Nw);
    // assert(st_offs0.dimension(2) == Ns);
    //
    // auto const st_nmc = Fermi::row_major_file_to_col_major_tensor<char>(
    //     "./xtsrcmaps/tests/expected/src_needs_more.bin", Ns, Nw, Nh, Ne);
    // assert(st_nmc.dimension(0) == Ne);
    // assert(st_nmc.dimension(1) == Nh);
    // assert(st_nmc.dimension(2) == Nw);
    // assert(st_nmc.dimension(3) == Ns);
    // Eigen::Tensor<bool, 4> st_nm0 = st_nmc.cast<bool>();
    //
    // Tensor4d const st_v0          = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/src_mean_psf_v0.bin", Ns, Nw, Nh, Ne);
    // assert(st_v0.dimension(0) == Ne);
    // assert(st_v0.dimension(1) == Nh);
    // assert(st_v0.dimension(2) == Nw);
    // assert(st_v0.dimension(3) == Ns);

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

    // Tensor2d const uPeak = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/uPsf_peak_SE.bin", Ns, Ne);
    // assert(uPeak.dimension(0) == Ne);
    // assert(uPeak.dimension(1) == Ns);
    //
    // auto const uPeakRatio = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/src_peak_ratio.bin", Ns, Nw, Nh, Ne);
    //
    // auto const st_upsf_peak = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/src_mean_psf_peak.bin", Ns, Ne);
    // assert(st_upsf_peak.dimension(0) == Ne);
    // assert(st_upsf_peak.dimension(1) == Ns);
    //
    // auto const st_upsf_arr = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/src_mean_psf_values.bin", Ns, Ne, Nd);
    // assert(st_upsf_arr.dimension(0) == Nd);
    // assert(st_upsf_arr.dimension(1) == Ne);
    // assert(st_upsf_arr.dimension(2) == Ns);

    Tensor4d psfEst = Fermi::ModelMap::point_src_model_map_wcs(
        100, 100, dirs, uPsf, { ccube }, 1e-3);

    for (long s = 0; s < Ns; ++s)
    {
        for (long w = 0; w < Nw; ++w)
        {
            for (long h = 0; h < Nh; ++h)
            {
                for (long e = 0; e < Ne; ++e)
                {
                    CHECK_MESSAGE(doctest::Approx(psfEst(e, h, w, s)).epsilon(1e-2)
                                        == st_psfEst(e, h, w, s),
                                    s << " " << h << " " << w << " " << e);
                }
            }
        }
    }
}
