#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/tests/fits/src_dirs_expected.hxx"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/source_utils.hxx"

TEST_CASE("Test directions from point sources")
{
    auto cfg  = Fermi::XtCfg();
    auto srcs = Fermi::parse_src_xml(cfg.srcmdl);
    auto dirs = Fermi::directions_from_point_sources(srcs);

    REQUIRE(dirs.size() == FermiTest::dirs.size());
    for (size_t i = 0; i < dirs.size(); ++i)
    {
        CHECK(dirs[i].first == doctest::Approx(FermiTest::dirs[i].first));
        CHECK(dirs[i].second == doctest::Approx(FermiTest::dirs[i].second));
    }
}
