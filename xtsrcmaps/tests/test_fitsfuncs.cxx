#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/tests/fits/psf_expected.hxx"

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB RPSF_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "RPSF_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    REQUIRE(v.extents.size() == 10);
    REQUIRE(v.offsets.size() == 10);
    REQUIRE(v.rowdata.size() == 1);
    REQUIRE(v.rowdata[0].size() == 1166);

    // vector values
    REQUIRE(v.extents == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_FRONT::extents);
    REQUIRE(v.offsets == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_FRONT::offsets);
    REQUIRE(v.rowdata[0] == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_FRONT::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB PSF_SCALING_PARAMS_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "PSF_SCALING_PARAMS_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    REQUIRE(v.extents.size() == 1);
    REQUIRE(v.offsets.size() == 1);
    REQUIRE(v.rowdata.size() == 1);
    REQUIRE(v.rowdata[0].size() == 3);

    // vector values
    REQUIRE(v.extents
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_FRONT::extents);
    REQUIRE(v.offsets
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_FRONT::offsets);
    REQUIRE(v.rowdata[0]
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_FRONT::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB FISHEYE_CORRECTION_FRONT")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "FISHEYE_CORRECTION_FRONT");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    REQUIRE(v.extents.size() == 7);
    REQUIRE(v.offsets.size() == 7);
    REQUIRE(v.rowdata.size() == 1);
    REQUIRE(v.rowdata[0].size() == 2332);

    // vector values
    REQUIRE(v.extents
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_FRONT::extents);
    REQUIRE(v.offsets
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_FRONT::offsets);
    REQUIRE(v.rowdata[0]
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_FRONT::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB RPSF_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "RPSF_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    REQUIRE(v.extents.size() == 10);
    REQUIRE(v.offsets.size() == 10);
    REQUIRE(v.rowdata.size() == 1);
    REQUIRE(v.rowdata[0].size() == 1166);

    // vector values
    REQUIRE(v.extents == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_BACK::extents);
    REQUIRE(v.offsets == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_BACK::offsets);
    REQUIRE(v.rowdata[0] == FermiTest::psf_P8R3_SOURCE_V2_FB::RPSF_BACK::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB PSF_SCALING_PARAMS_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "PSF_SCALING_PARAMS_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    REQUIRE(v.extents.size() == 1);
    REQUIRE(v.offsets.size() == 1);
    REQUIRE(v.rowdata.size() == 1);
    REQUIRE(v.rowdata[0].size() == 3);

    // vector values
    REQUIRE(v.extents
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_BACK::extents);
    REQUIRE(v.offsets
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_BACK::offsets);
    REQUIRE(v.rowdata[0]
            == FermiTest::psf_P8R3_SOURCE_V2_FB::PSF_SCALING_PARAMS_BACK::rowdata);
}

TEST_CASE("Fermi FITS read_irf_pars psf_P8R3_SOURCE_V2_FB FISHEYE_CORRECTION_BACK")
{
    auto cfg = Fermi::XtCfg();
    auto ov  = Fermi::fits::read_irf_pars(cfg.psf_name, "FISHEYE_CORRECTION_BACK");
    REQUIRE(ov);

    auto v = ov.value();

    // size data
    REQUIRE(v.extents.size() == 7);
    REQUIRE(v.offsets.size() == 7);
    REQUIRE(v.rowdata.size() == 1);
    REQUIRE(v.rowdata[0].size() == 2332);

    // vector values
    REQUIRE(v.extents
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_BACK::extents);
    REQUIRE(v.offsets
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_BACK::offsets);
    REQUIRE(v.rowdata[0]
            == FermiTest::psf_P8R3_SOURCE_V2_FB::FISHEYE_CORRECTION_BACK::rowdata);
}
