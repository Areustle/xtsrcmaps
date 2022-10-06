#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/misc.hxx"

#include "xtsrcmaps/tests/fits/aeff_expected.hxx"

#include "fmt/format.h"


TEST_CASE("Load AEFF IRF")
{
    auto cfg   = Fermi::XtCfg();
    auto oaeff = Fermi::load_aeff(cfg.aeff_name);
    REQUIRE(oaeff);
    auto aeff = oaeff.value();

    CHECK(aeff.front.effective_area.cosths.size() == 34);
    CHECK(aeff.front.effective_area.cosths == FermiTest::Aeff::Front::effarea_cosths);
    for (size_t i = 0; i < FermiTest::Aeff::Front::effarea_cosths.size(); ++i)
        CHECK(aeff.front.effective_area.cosths[i]
              == FermiTest::Aeff::Front::effarea_cosths[i]);
    CHECK(aeff.front.effective_area.logEs.size() == 76);
    CHECK(aeff.front.effective_area.logEs == FermiTest::Aeff::Front::effarea_loge);

    CHECK(aeff.front.efficiency_params.p0 != aeff.front.efficiency_params.p1);

    auto expect_fr_efp_p0 = std::array<float, 6> { -2.210081, 6.0228114,    -0.52524257,
                                                   2.3434312, -0.073382795, 3.5008383 };
    auto expect_fr_efp_p1 = std::array<float, 6> { 1.9994605, -4.4488373, 0.47518694,
                                                   2.3434312, 0.06638942, 3.5008383 };
    CHECK(aeff.front.efficiency_params.p0 == expect_fr_efp_p0);
    CHECK(aeff.front.efficiency_params.p1 == expect_fr_efp_p1);
}

TEST_CASE("Test Irf Efficiency factor Generation.")
{

    auto const cfg   = Fermi::XtCfg();
    auto const oaeff = Fermi::load_aeff(cfg.aeff_name);
    auto const oen   = Fermi::fits::ccube_energies(cfg.cmap);
    REQUIRE(oaeff);
    REQUIRE(oen);
    auto const aeff  = oaeff.value();
    auto const logEs = Fermi::log10_v(oen.value());

    auto const& efp  = aeff.front.efficiency_params;
    auto        LTFs = Fermi::lt_effic_factors(logEs, efp);

    REQUIRE(LTFs.first.size() == LTFs.first.size());
    REQUIRE(LTFs.first.size() == FermiTest::Aeff::livetime_factor1.size());
    REQUIRE(LTFs.second.size() == FermiTest::Aeff::livetime_factor1.size());

    for (size_t i = 0; i < LTFs.first.size(); ++i)
    {
        CHECK(LTFs.first[i] == doctest::Approx(FermiTest::Aeff::livetime_factor1[i]));
        CHECK(LTFs.second[i] == doctest::Approx(FermiTest::Aeff::livetime_factor2[i]));
    }
}
