#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/misc.hxx"

#include "fmt/format.h"


TEST_CASE("Load AEFF IRF")
{
    auto cfg   = Fermi::XtCfg();
    auto oaeff = Fermi::load_aeff(cfg.aeff_name);
    REQUIRE(oaeff);
    auto aeff             = oaeff.value();

    auto f_effarea_cosths = std::vector<double> {
        // clang-format off
          -1, 0.2125000059604644775390625, 0.2374999970197677612304688, 
          0.262499988079071044921875, 0.28750002384185791015625, 0.3125, 
          0.33749997615814208984375, 0.362500011920928955078125, 
          0.387499988079071044921875, 0.41250002384185791015625, 0.4375, 
          0.46249997615814208984375, 0.487500011920928955078125, 
          0.512499988079071044921875, 0.53750002384185791015625, 0.5625, 
          0.58749997615814208984375, 0.612500011920928955078125, 
          0.637499988079071044921875, 0.66250002384185791015625, 0.6875, 
          0.71249997615814208984375, 0.737500011920928955078125, 
          0.762499988079071044921875, 0.78750002384185791015625, 0.8125, 
          0.83749997615814208984375, 0.862500011920928955078125, 
          0.887499988079071044921875, 0.91250002384185791015625, 0.9375, 
          0.96249997615814208984375, 0.987500011920928955078125, 1
        // clang-format on
    };

    CHECK(aeff.front.effective_area.cosths.size() == 34);
    CHECK(aeff.front.effective_area.cosths == f_effarea_cosths);
    for (size_t i = 0; i < f_effarea_cosths.size(); ++i)
        CHECK(aeff.front.effective_area.cosths[i] == f_effarea_cosths[i]);

    auto f_effarea_loge = std::vector<double> {
        0,       0.78125, 0.84375, 0.90625, 0.96875, 1.03125, 1.09375, 1.15625, 1.21875,
        1.28125, 1.34375, 1.40625, 1.46875, 1.53125, 1.59375, 1.65625, 1.71875, 1.78125,
        1.84375, 1.90625, 1.96875, 2.03125, 2.09375, 2.15625, 2.21875, 2.28125, 2.34375,
        2.40625, 2.46875, 2.53125, 2.59375, 2.65625, 2.71875, 2.78125, 2.84375, 2.90625,
        2.96875, 3.03125, 3.09375, 3.15625, 3.21875, 3.28125, 3.34375, 3.40625, 3.46875,
        3.53125, 3.59375, 3.65625, 3.71875, 3.78125, 3.84375, 3.90625, 3.96875, 4.03125,
        4.09375, 4.15625, 4.21875, 4.3125,  4.4375,  4.5625,  4.6875,  4.8125,  4.9375,
        5.0625,  5.1875,  5.3125,  5.4375,  5.5625,  5.6875,  5.8125,  5.9375,  6.0625,
        6.1875,  6.3125,  6.4375,  10
    };
    CHECK(aeff.front.effective_area.logEs.size() == 76);
    CHECK(aeff.front.effective_area.logEs == f_effarea_loge);

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
    REQUIRE(oaeff);
    auto const aeff = oaeff.value();
    auto const oen  = Fermi::fits::ccube_energies(cfg.cmap);
    REQUIRE(oen);
    auto const en    = oen.value();
    auto const logE  = Fermi::log10_v(en);

    auto const& efp  = aeff.front.efficiency_params;
    auto        LTFs = Fermi::lt_effic_factors(logE, efp);

    auto expected0   = std::vector<double> {
          -0.449916, -0.250026, -0.0501356, 0.149755, 0.263589, 0.311094, 0.3586,
          0.406105,  0.453611,  0.501116,   0.548621, 0.596127, 0.643632, 0.691138,
          0.738643,  0.786149,  0.793299,   0.799936, 0.806573, 0.813211, 0.819848,
          0.826485,  0.833122,  0.839759,   0.846396, 0.853033, 0.85967,  0.866307,
          0.872944,  0.879581,  0.886219,   0.892856, 0.899493, 0.90613,  0.912767,
          0.919404,  0.926041,  0.932678
    };

    auto expected1 = std::vector<double> {
        1.60265,   1.3817,    1.16076,  0.939809, 0.813984, 0.761475, 0.708965,
        0.656456,  0.603946,  0.551436, 0.498927, 0.446417, 0.393907, 0.341398,
        0.288888,  0.236379,  0.228475, 0.221139, 0.213802, 0.206466, 0.19913,
        0.191794,  0.184457,  0.177121, 0.169785, 0.162449, 0.155112, 0.147776,
        0.14044,   0.133104,  0.125767, 0.118431, 0.111095, 0.103759, 0.0964225,
        0.0890863, 0.0817501, 0.0744138
    };

    REQUIRE(LTFs.size() == expected0.size());
    REQUIRE(LTFs.size() == expected1.size());

    for (size_t i = 0; i < LTFs.size(); ++i)
    {
        CHECK(LTFs[i].first == doctest::Approx(expected0[i]));
        CHECK(LTFs[i].second == doctest::Approx(expected1[i]));
    }
}
