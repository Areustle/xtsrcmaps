#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/tests/fits/aeff_expected.hxx"
#include "xtsrcmaps/tests/fits/psf_expected.hxx"

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB RPSF_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "RPSF_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 10);
    CHECK(v.offsets.size() == 10);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 1166);

    // vector values
    CHECK(v.extents == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_FRONT::extents);
    CHECK(v.offsets == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_FRONT::offsets);
    CHECK(v.rowdata[0] == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_FRONT::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB PSF_SCALING_PARAMS_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "PSF_SCALING_PARAMS_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 1);
    CHECK(v.offsets.size() == 1);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 3);

    // vector values
    CHECK(v.extents
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_FRONT::extents);
    CHECK(v.offsets
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_FRONT::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_FRONT::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB FISHEYE_CORRECTION_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "FISHEYE_CORRECTION_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 7);
    CHECK(v.offsets.size() == 7);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 2332);

    // vector values
    CHECK(v.extents
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_FRONT::extents);
    CHECK(v.offsets
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_FRONT::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_FRONT::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB RPSF_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "RPSF_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 10);
    CHECK(v.offsets.size() == 10);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 1166);

    // vector values
    CHECK(v.extents == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_BACK::extents);
    CHECK(v.offsets == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_BACK::offsets);
    CHECK(v.rowdata[0] == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_BACK::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB PSF_SCALING_PARAMS_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "PSF_SCALING_PARAMS_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 1);
    CHECK(v.offsets.size() == 1);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 3);

    // vector values
    CHECK(v.extents
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_BACK::extents);
    CHECK(v.offsets
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_BACK::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_BACK::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB FISHEYE_CORRECTION_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "FISHEYE_CORRECTION_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 7);
    CHECK(v.offsets.size() == 7);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 2332);

    // vector values
    CHECK(v.extents
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_BACK::extents);
    CHECK(v.offsets
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_BACK::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_BACK::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars aeff_P8R3_SOURCE_V2_FB EFFECTIVE AREA_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, "EFFECTIVE AREA_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 5);
    CHECK(v.offsets.size() == 5);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 2580);

    // vector values
    CHECK(v.extents
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFECTIVE_AREA_FRONT::extents);
    CHECK(v.offsets
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFECTIVE_AREA_FRONT::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFECTIVE_AREA_FRONT::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars aeff_P8R3_SOURCE_V2_FB PHI_DEPENDENCE_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, "PHI_DEPENDENCE_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 6);
    CHECK(v.offsets.size() == 6);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 430);

    // vector values
    CHECK(v.extents
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::PHI_DEPENDENCE_FRONT::extents);
    CHECK(v.offsets
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::PHI_DEPENDENCE_FRONT::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::PHI_DEPENDENCE_FRONT::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars aeff_P8R3_SOURCE_V2_FB EFFICIENCY_PARAMS_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, "EFFICIENCY_PARAMS_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 1);
    CHECK(v.offsets.size() == 1);
    CHECK(v.rowdata.size() == 2);
    CHECK(v.rowdata[0].size() == 6);

    // vector values
    CHECK(v.extents
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_FRONT::extents);
    CHECK(v.offsets
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_FRONT::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_FRONT::rowdata0);
    CHECK(v.rowdata[1]
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_FRONT::rowdata1);
}

TEST_CASE("Fermi FITS read_irf_pars aeff_P8R3_SOURCE_V2_FB EFFECTIVE AREA_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, "EFFECTIVE AREA_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 5);
    CHECK(v.offsets.size() == 5);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 2580);

    // vector values
    CHECK(v.extents
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFECTIVE_AREA_BACK::extents);
    CHECK(v.offsets
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFECTIVE_AREA_BACK::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFECTIVE_AREA_BACK::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars aeff_P8R3_SOURCE_V2_FB PHI_DEPENDENCE_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, "PHI_DEPENDENCE_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 6);
    CHECK(v.offsets.size() == 6);
    CHECK(v.rowdata.size() == 1);
    CHECK(v.rowdata[0].size() == 430);

    // vector values
    CHECK(v.extents
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::PHI_DEPENDENCE_BACK::extents);
    CHECK(v.offsets
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::PHI_DEPENDENCE_BACK::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::PHI_DEPENDENCE_BACK::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars aeff_P8R3_SOURCE_V2_FB EFFICIENCY_PARAMS_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.aeff_name, "EFFICIENCY_PARAMS_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    CHECK(v.extents.size() == 1);
    CHECK(v.offsets.size() == 1);
    CHECK(v.rowdata.size() == 2);
    CHECK(v.rowdata[0].size() == 6);

    // vector values
    CHECK(v.extents
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_BACK::extents);
    CHECK(v.offsets
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_BACK::offsets);
    CHECK(v.rowdata[0]
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_BACK::rowdata0);
    CHECK(v.rowdata[1]
            == FermiTest::aeff_P8R3_SOURCE_V2_FB::EFFICIENCY_PARAMS_BACK::rowdata1);
}
