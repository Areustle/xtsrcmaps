
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
    auto       cfg       = Fermi::XtCfg();
    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::directions_from_point_sources(srcs);

    auto const opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(opt_ccube);
    auto const     ccube = good(opt_ccube, "Cannot read counts cube map file!");
    Fermi::SkyGeom skygeom(ccube);
    // long const     Ne  = 38;
    long const Nh      = 100;
    long const Nw      = 100;
    long const Ns      = dirs.size();

    Tensor3d st_pixoff = Fermi::row_major_file_to_col_major_tensor(
        "./xtsrcmaps/tests/expected/src_pixelOffset.bin", Ns, Nw, Nh);
    assert(st_pixoff.dimension(0) == Nh);
    assert(st_pixoff.dimension(1) == Nw);
    assert(st_pixoff.dimension(2) == Ns);

    // st_pixoff = np.fromfile("xtsrcmaps/tests/expected/src_pixelOffset.bin",
    // count=(263*100*100)).reshape(263,100,100)
    //
    for (long s = 0; s < Ns; ++s)
    {
        TensorMap<Tensor2d const> const st_po(st_pixoff.data() + Nw * Nh * s, Nh, Nw);
        Tensor2d                        pixelOffsets
            = Fermi::ModelMap::create_offset_map(Nh, Nw, dirs[s], skygeom);

        Tensor<bool, 2> xfin
            = pixelOffsets.unaryExpr([](double v) -> bool { return std::isfinite(v); });
        Tensor<bool, 2> yfin
            = st_po.unaryExpr([](double v) -> bool { return std::isfinite(v); });

        if (xfin(0) && yfin(0))
        {
            Tensor<double, 2> abserr = (pixelOffsets - st_po).abs();
            Tensor<double, 2> ar_tol  = 1e-5 + 1e-5 * st_po.abs();
        }

        // REQUIRE(allclose<double, 2>(pixelOffsets, st_po, 1e-5, 1e-5));
    }
}
