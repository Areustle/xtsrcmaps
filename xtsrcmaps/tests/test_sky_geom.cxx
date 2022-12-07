#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/source_utils.hxx"

auto
pcmp(std::pair<double, double> const& a, std::pair<double, double> const& b) -> void
{
    CHECK(std::get<0>(a) == doctest::Approx(std::get<0>(b)));
    CHECK(std::get<1>(a) == doctest::Approx(std::get<1>(b)));
}

TEST_CASE("Fermi Sky Geom")
{
    Fermi::fits::CCubePixels cc = {
        {   100,   100,               37},
        {  50.5,  50.5,              1.0},
        {193.98, -5.82,            100.0},
        {  -0.2,   0.2, 25.8844718872141},
        0.0,
        "AIT",
        false
    };

    Fermi::SkyGeom sg(cc);

    std::pair<double, double> src_sph_0  = { 193.98, -5.82 };
    auto                      src_dir_0  = sg.sph2dir(src_sph_0);
    auto                      src_dsph_0 = sg.dir2sph(src_dir_0);

    pcmp(src_sph_0, src_dsph_0);

    auto       cfg                = Fermi::XtCfg();
    auto const srcs               = Fermi::parse_src_xml(cfg.srcmdl);
    auto const src_spherical_dirs = Fermi::directions_from_point_sources(srcs);

    size_t const& Ns              = src_spherical_dirs.size();

    for (size_t s = 0; s < Ns; ++s)
    {
        auto src_dir  = sg.sph2dir(src_spherical_dirs[s]);
        auto src_dsph = sg.dir2sph(src_dir);
        pcmp(src_dsph, src_spherical_dirs[s]);
    }
}
