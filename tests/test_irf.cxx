#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include "tests/fermi_tests.hxx"
#include "tests/fits/aeff_expected.hxx"


TEST_CASE("Load AEFF IRF") {
    auto cfg   = Fermi::XtCfg();
    auto oaeff = Fermi::load_aeff(cfg.aeff_file);
    REQUIRE(oaeff);
    auto aeff = oaeff.value();

    CHECK(aeff.front.effective_area.cosths.extent(0) == 34);
    for (size_t i = 0; i < FermiTest::Aeff::Front::effarea_cosths.size(); ++i)
        CHECK_MESSAGE(aeff.front.effective_area.cosths[i]
                          == FermiTest::Aeff::Front::effarea_cosths[i],
                      i);

    CHECK(aeff.front.effective_area.logEs.extent(0) == 76);
    std::vector<double> loges(aeff.front.effective_area.logEs.data(),
                              aeff.front.effective_area.logEs.data() + 76);
    CHECK(loges == FermiTest::Aeff::Front::effarea_loge);
    for (size_t i = 0; i < FermiTest::Aeff::Front::effarea_loge.size(); ++i)
        CHECK(aeff.front.effective_area.logEs[i]
              == FermiTest::Aeff::Front::effarea_loge[i]);

    CHECK(doctest::Approx(aeff.front.effective_area.minCosTheta) == 0.2);
    CHECK(aeff.front.efficiency_params.p0 != aeff.front.efficiency_params.p1);

    auto expect_fr_efp_p0
        = std::array<float, 6> { -2.210081, 6.0228114,    -0.52524257,
                                 2.3434312, -0.073382795, 3.5008383 };
    auto expect_fr_efp_p1
        = std::array<float, 6> { 1.9994605, -4.4488373, 0.47518694,
                                 2.3434312, 0.06638942, 3.5008383 };

    CHECK(aeff.front.efficiency_params.p0 == expect_fr_efp_p0);
    CHECK(aeff.front.efficiency_params.p1 == expect_fr_efp_p1);

    REQUIRE(aeff.front.effective_area.params.extent(0) == 34); // Mc
    REQUIRE(aeff.front.effective_area.params.extent(1) == 76); // Me
    REQUIRE(aeff.front.effective_area.params.extent(2) == 1);  // Ngrids

    // Test effective_area params are what we expect
    auto expect_fr_effarea
        = Tensor2d(FermiTest::Aeff::Front::effarea_raw_params, 32, 74);

    for (size_t c = 0; c < 32; ++c) {
        for (size_t e = 0; e < 74; ++e) {
            REQUIRE(doctest::Approx(
                        aeff.front.effective_area.params[c + 1, e + 1, 0])
                    == expect_fr_effarea[c, e]);
        }
    }

    ///////////
    // Sides //
    ///////////

    // Top
    for (size_t i = 0; i < 74; ++i) {
        REQUIRE(doctest::Approx(aeff.front.effective_area.params[0, i + 1, 0])
                == expect_fr_effarea[0, i]);
    }
    // bottom
    for (size_t i = 0; i < 74; ++i) {
        REQUIRE(doctest::Approx(aeff.front.effective_area.params[33, i + 1, 0])
                == expect_fr_effarea[31, i]);
    }
    // Left
    for (size_t i = 0; i < 32; ++i) {
        REQUIRE(doctest::Approx(aeff.front.effective_area.params[i + 1, 0, 0])
                == expect_fr_effarea[i, 0]);
    }
    // Right
    for (size_t i = 0; i < 32; ++i) {
        REQUIRE(doctest::Approx(aeff.front.effective_area.params[i + 1, 75, 0])
                == expect_fr_effarea[i, 73]);
    }

    /////////////
    // Corners //
    /////////////

    REQUIRE(doctest::Approx(aeff.front.effective_area.params[0, 0, 0])
            == expect_fr_effarea[0, 0]);
    REQUIRE(doctest::Approx(aeff.front.effective_area.params[0, 75, 0])
            == expect_fr_effarea[0, 73]);
    REQUIRE(doctest::Approx(aeff.front.effective_area.params[33, 0, 0])
            == expect_fr_effarea[31, 0]);
    REQUIRE(doctest::Approx(aeff.front.effective_area.params[33, 75, 0])
            == expect_fr_effarea[31, 73]);
}

TEST_CASE("Test Irf Efficiency factor Generation.") {

    auto const cfg   = Fermi::XtCfg();
    auto const oaeff = Fermi::load_aeff(cfg.aeff_file);
    auto const oen   = Fermi::fits::ccube_energies(cfg.cmap);
    REQUIRE(oaeff);
    REQUIRE(oen);
    auto const aeff  = oaeff.value();
    auto const logEs = Fermi::log10_v(oen.value());

    auto const& efp  = aeff.front.efficiency_params;
    auto        LTFs = Fermi::livetime_efficiency_factors(logEs, efp);

    REQUIRE(LTFs.extent(1) == FermiTest::Aeff::livetime_factor1.size());

    for (size_t i = 0; i < LTFs.extent(1); ++i) {
        CHECK(LTFs[0, i]
              == doctest::Approx(FermiTest::Aeff::livetime_factor1[i]));
        CHECK(LTFs[1, i]
              == doctest::Approx(FermiTest::Aeff::livetime_factor2[i]));
    }
}
