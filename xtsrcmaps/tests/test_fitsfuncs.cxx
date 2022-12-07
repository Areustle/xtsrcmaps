#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/tests/fits/aeff_expected.hxx"
#include "xtsrcmaps/tests/fits/ccube_energies_expected.hxx"
#include "xtsrcmaps/tests/fits/psf_expected.hxx"

TEST_CASE("Fermi FITS ccube_energies.")
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

TEST_CASE("Fermi FITS ccube_pixels.")
{
    auto cfg = Fermi::XtCfg();
    // auto ov  = //Fermi::fits::ccube_energies(cfg.cmap);
    auto ov  = Fermi::fits::ccube_pixels(cfg.cmap);
    REQUIRE(ov);

    auto v = ov.value();
    REQUIRE(v.naxes[0] == 100);
    REQUIRE(v.naxes[1] == 100);
    REQUIRE(v.naxes[2] == 37);
    REQUIRE(v.crpix[0] == 50.5);
    REQUIRE(v.crpix[1] == 50.5);
    REQUIRE(v.crpix[2] == 1.0);
    REQUIRE(v.crval[0] == 193.98);
    REQUIRE(v.crval[1] == -5.82);
    REQUIRE(v.crval[2] == 100.);
    REQUIRE(v.cdelt[0] == -0.2);
    REQUIRE(v.cdelt[1] == 0.2);
    REQUIRE(v.cdelt[2] == 25.8844718872141);
    REQUIRE(v.axis_rot == 0.0);
    REQUIRE(v.proj_name == "AIT");
    REQUIRE(v.is_galactic == false);
}

TEST_CASE("Fermi FITS read_expcube.")
{
    auto cfg  = Fermi::XtCfg();
    auto oecd = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    REQUIRE(oecd);

    auto ecd = oecd.value();
    CHECK(ecd.cosbins.size() == 12 * 64 * 64 * 40);
    CHECK(ecd.nside == 64u);
    CHECK(ecd.nbrbins == 40u);
    CHECK(ecd.cosmin == 0.0);
    CHECK(ecd.ordering == "NESTED");
    CHECK(ecd.coordsys == "EQU");
    CHECK(ecd.thetabin == true);
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
        CHECK(v.rowdata.dimension(1) == 1);
        CHECK(v.rowdata.dimension(0) == expected.rowdata.size());
        // vector values
        CHECK(v.extents == expected.extents);
        CHECK(v.offsets == expected.offsets);
        std::vector<float> vv(&v.rowdata(0, 0), &v.rowdata(0, 1));
        CHECK(vv == expected.rowdata);
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
        CHECK(v.rowdata.dimension(1) == 1);
        CHECK(v.rowdata.dimension(0) == expected.rowdata.size());
        // vector values
        CHECK(v.extents == expected.extents);
        CHECK(v.offsets == expected.offsets);
        // CHECK(v.rowdata[0] == expected.rowdata);
        std::vector<float> vv(&v.rowdata(0, 0), &v.rowdata(0, 1));
        CHECK(vv == expected.rowdata);
    };

    auto testAEFF2 = [](std::string tablename, auto expected) -> void {
        auto cfg = Fermi::XtCfg();
        auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, tablename);
        REQUIRE(ov);

        auto v = ov.value();

        // size data
        CHECK(v.extents.size() == expected.extents.size());
        CHECK(v.offsets.size() == expected.offsets.size());
        CHECK(v.rowdata.dimension(1) == 2);
        CHECK(v.rowdata.dimension(0) == expected.rowdata0.size());
        CHECK(v.rowdata.dimension(0) == expected.rowdata1.size());
        // vector values
        CHECK(v.extents == expected.extents);
        CHECK(v.offsets == expected.offsets);
        std::vector<float> v0(&v.rowdata(0, 0), &v.rowdata(0, 1));
        CHECK(v0 == expected.rowdata0);
        std::vector<float> v1(&v.rowdata(0, 1), &v.rowdata(0, 2));
        CHECK(v1 == expected.rowdata1);
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
