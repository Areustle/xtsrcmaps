#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/tests/fits/aeff_expected.hxx"
#include "xtsrcmaps/tests/fits/ccube_energies_expected.hxx"
#include "xtsrcmaps/tests/fits/psf_expected.hxx"

TEST_CASE("Fermi FITS ccube_energies")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::ccube_energies(cfg.cmap);
    REQUIRE(ov);

    auto v = ov.value();
    CHECK(v.size() == FermiTest::CCubeEnergies::expected.size());
    // CHECK(v == FermiTest::CCubeEnergies::expected);
    for (size_t i = 0; i < v.size(); ++i)
    {
        CHECK(v[i] == doctest::Approx(FermiTest::CCubeEnergies::expected[i]));
    }
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB")
{
    auto f = [](std::string tablename, auto expected) -> void {
        auto cfg = Fermi::XtCfg();
        auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, tablename);
        REQUIRE(ov);

        auto v = ov.value();

        // size data
        CHECK(v.extents.size() == expected.extents.size());
        CHECK(v.offsets.size() == expected.offsets.size());
        CHECK(v.rowdata.size() == 1);
        CHECK(v.rowdata[0].size() == expected.rowdata.size());
        // vector values
        CHECK(v.extents == expected.extents);
        CHECK(v.offsets == expected.offsets);
        CHECK(v.rowdata[0] == expected.rowdata);
    };

    // FRONT
    f("RPSF_FRONT", FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_FRONT);
    f("PSF_SCALING_PARAMS_FRONT",
      FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_FRONT);
    f("FISHEYE_CORRECTION_FRONT",
      FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_FRONT);
    // BACK
    f("RPSF_BACK", FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_BACK);
    f("PSF_SCALING_PARAMS_BACK",
      FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_BACK);
    f("FISHEYE_CORRECTION_BACK",
      FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_BACK);
}

TEST_CASE("Fermi FITS read_irf_pars aeff_P8R3_SOURCE_V2_FB EFFECTIVE AREA_FRONT")
{
    auto testAEFF1 = [](std::string tablename, auto expected) -> void {
        auto cfg = Fermi::XtCfg();
        auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, tablename);
        REQUIRE(ov);

        auto v = ov.value();

        // size data
        CHECK(v.extents.size() == expected.extents.size());
        CHECK(v.offsets.size() == expected.offsets.size());
        CHECK(v.rowdata.size() == 1);
        CHECK(v.rowdata[0].size() == expected.rowdata.size());
        // vector values
        CHECK(v.extents == expected.extents);
        CHECK(v.offsets == expected.offsets);
        CHECK(v.rowdata[0] == expected.rowdata);
    };

    auto testAEFF2 = [](std::string tablename, auto expected) -> void {
        auto cfg = Fermi::XtCfg();
        auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, tablename);
        REQUIRE(ov);

        auto v = ov.value();

        // size data
        CHECK(v.extents.size() == expected.extents.size());
        CHECK(v.offsets.size() == expected.offsets.size());
        CHECK(v.rowdata.size() == 2);
        CHECK(v.rowdata[0].size() == expected.rowdata0.size());
        CHECK(v.rowdata[1].size() == expected.rowdata1.size());
        // vector values
        CHECK(v.extents == expected.extents);
        CHECK(v.offsets == expected.offsets);
        CHECK(v.rowdata[0] == expected.rowdata0);
        CHECK(v.rowdata[1] == expected.rowdata1);
    };

    // FRONT
    testAEFF1("EFFECTIVE AREA_FRONT",
              FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFECTIVE_AREA_FRONT);
    testAEFF1("PHI_DEPENDENCE_FRONT",
              FermiTest::aeff_P8R3_SOURCE_V2_FB::PHI_DEPENDENCE_FRONT);
    testAEFF2("EFFICIENCY_PARAMS_FRONT",
              FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_FRONT);
    // BACK
    testAEFF1("EFFECTIVE AREA_BACK",
              FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFECTIVE_AREA_BACK);
    testAEFF1("PHI_DEPENDENCE_BACK",
              FermiTest::aeff_P8R3_SOURCE_V2_FB::PHI_DEPENDENCE_BACK);
    testAEFF2("EFFICIENCY_PARAMS_BACK",
              FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_BACK);
}
