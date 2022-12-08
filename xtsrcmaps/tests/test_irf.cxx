#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

#include "xtsrcmaps/tests/fits/aeff_expected.hxx"

#include "fmt/format.h"


TEST_CASE("Load AEFF IRF")
{
    auto cfg   = Fermi::XtCfg();
    auto oaeff = Fermi::load_aeff(cfg.aeff_name);
    REQUIRE(oaeff);
    auto aeff = oaeff.value();

    CHECK(aeff.front.effective_area.cosths.dimension(0) == 34);
    std::vector<double> escth(aeff.front.effective_area.cosths.data(),
                              aeff.front.effective_area.cosths.data() + 34);
    CHECK(escth == FermiTest::Aeff::Front::effarea_cosths);
    for (size_t i = 0; i < FermiTest::Aeff::Front::effarea_cosths.size(); ++i)
        CHECK(aeff.front.effective_area.cosths[i]
              == FermiTest::Aeff::Front::effarea_cosths[i]);

    CHECK(aeff.front.effective_area.logEs.size() == 76);
    std::vector<double> loges(aeff.front.effective_area.logEs.data(),
                              aeff.front.effective_area.logEs.data() + 76);
    CHECK(loges == FermiTest::Aeff::Front::effarea_loge);
    for (size_t i = 0; i < FermiTest::Aeff::Front::effarea_loge.size(); ++i)
        CHECK(aeff.front.effective_area.logEs[i]
              == FermiTest::Aeff::Front::effarea_loge[i]);

    CHECK(doctest::Approx(aeff.front.effective_area.minCosTheta) == 0.2);
    CHECK(aeff.front.efficiency_params.p0 != aeff.front.efficiency_params.p1);

    auto expect_fr_efp_p0 = std::array<float, 6> { -2.210081, 6.0228114,    -0.52524257,
                                                   2.3434312, -0.073382795, 3.5008383 };
    auto expect_fr_efp_p1 = std::array<float, 6> { 1.9994605, -4.4488373, 0.47518694,
                                                   2.3434312, 0.06638942, 3.5008383 };

    CHECK(aeff.front.efficiency_params.p0 == expect_fr_efp_p0);
    CHECK(aeff.front.efficiency_params.p1 == expect_fr_efp_p1);

    REQUIRE(aeff.front.effective_area.params.dimension(0) == 1);
    REQUIRE(aeff.front.effective_area.params.dimension(1) == 76); // Me
    REQUIRE(aeff.front.effective_area.params.dimension(2) == 34); // Mc

    // Test effective_area params are what we expect
    //
    TensorMap<Tensor2d const> expect_fr_effarea(
        FermiTest::Aeff::Front::effarea_raw_params.data(), 74, 32);

    // Middle of block Body
    Tensor2d middle_ea
        = aeff.front.effective_area.params.slice(Idx3 { 0, 1, 1 }, Idx3 { 1, 74, 32 })
              .reshape(Idx2 { 74, 32 });

    for (long j = 0; j < middle_ea.dimension(1); ++j)
    {
        for (long i = 0; i < middle_ea.dimension(0); ++i)
        {
            REQUIRE(doctest::Approx(middle_ea(i, j)) == expect_fr_effarea(i, j));
        }
    }
    // Sides
    Tensor1d top
        = aeff.front.effective_area.params.slice(Idx3 { 0, 0, 1 }, Idx3 { 1, 1, 32 })
              .reshape(Idx1 { 32 });
    Tensor1d bottom
        = aeff.front.effective_area.params.slice(Idx3 { 0, 75, 1 }, Idx3 { 1, 1, 32 })
              .reshape(Idx1 { 32 });
    Tensor1d left
        = aeff.front.effective_area.params.slice(Idx3 { 0, 1, 0 }, Idx3 { 1, 74, 1 })
              .reshape(Idx1 { 74 });
    Tensor1d right
        = aeff.front.effective_area.params.slice(Idx3 { 0, 1, 33 }, Idx3 { 1, 74, 1 })
              .reshape(Idx1 { 74 });
    for (long i = 0; i < 32; ++i)
    {
        REQUIRE(doctest::Approx(top(i)) == expect_fr_effarea(0, i));
        REQUIRE(doctest::Approx(bottom(i)) == expect_fr_effarea(73, i));
    }
    for (long i = 0; i < 74; ++i)
    {
        REQUIRE(doctest::Approx(left(i)) == expect_fr_effarea(i, 0));
        REQUIRE(doctest::Approx(right(i)) == expect_fr_effarea(i, 31));
    }
    // Corners
    REQUIRE(doctest::Approx(aeff.front.effective_area.params(0, 0, 0))
            == expect_fr_effarea(0, 0));
    REQUIRE(doctest::Approx(aeff.front.effective_area.params(0, 75, 0))
            == expect_fr_effarea(73, 0));
    REQUIRE(doctest::Approx(aeff.front.effective_area.params(0, 0, 33))
            == expect_fr_effarea(0, 31));
    REQUIRE(doctest::Approx(aeff.front.effective_area.params(0, 75, 33))
            == expect_fr_effarea(73, 31));
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
