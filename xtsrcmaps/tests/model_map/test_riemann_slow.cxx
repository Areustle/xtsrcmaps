
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

#include "xtsrcmaps/tests/fermi_tests.hxx"

TEST_CASE("Test Model Map pixel mean psf")
{
    auto const cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::directions_from_point_sources(srcs);

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(opt_ccube);
    auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
    Fermi::SkyGeom skygeom(ccube);
    long const     Ne  = 38;
    long const     Nh  = 100;
    long const     Nw  = 100;
    // long const     Ns  = dirs.size();
    long const Nd      = 401;

    Tensor4d st_psfEst = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_psfEstimate.bin", 1, Nw, Nh, Ne);

    std::cout << st_psfEst.slice(Idx4 { 0, 0, 0, 0 }, Idx4 { 1, Nh, Nw, 1 }) << "\n";

    Tensor3d norm_uPsf = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/uPsf_normalized_SED.bin", 1, Ne, Nd);

    Tensor2d peak_uPsf = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/uPsf_peak_SE.bin", 1, Ne);

    Tensor4d psfEst = Fermi::ModelMap::pixel_mean_psf_riemann(
        Nh, Nw, { dirs[0] }, norm_uPsf, peak_uPsf, { ccube }, 1e-3);

    std::cout << psfEst.slice(Idx4 { 0, 0, 0, 0 }, Idx4 { 1, Nh, Nw, 1 }) << "\n";

    REQUIRE(allclose(psfEst, st_psfEst, 1e-5, 1e-5));
    // for (long s = 0; s < Ns; ++s)
    // {
    //     // TensorMap<Tensor2d> st_po(st_psf.data() + Nw * Nh * s, Nh, Nw);
    //     // Tensor2d            pixelOffsets
    //     //     = Fermi::ModelMap::create_offset_map(Nh, Nw, dirs[s], skygeom);
    //     // bool close = allclose(pixelOffsets, st_po, 1e-5, 1e-5);
    //     // REQUIRE(close);
    // }
}
