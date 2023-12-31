#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include <fstream>

#include "xtsrcmaps/tests/fermi_tests.hxx"
#include "xtsrcmaps/tests/fits/exposure_cosbins_expected.hxx"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/exposure.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/source_utils.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

TEST_CASE("PSF SEPARATION")
{
    auto tsep = Fermi::PSF::separations();
    REQUIRE(sep_step == doctest::Approx(std::log(70. / 1e-4) / (400. - 1.)));
    for (size_t j = 0; j < tsep.size(); ++j)
    {
        REQUIRE_MESSAGE(tsep[j] == doctest::Approx(Fermi::PSF::separation(j)), j);
        REQUIRE_MESSAGE(j
                            == doctest::Approx(Fermi::PSF::inverse_separation(
                                Fermi::PSF::separation(j))),
                        Fermi::PSF::separation(j));
    }

    for (int j = 0; j < int(1e6); ++j)
    {
        double d  = j * (3.99e-8);
        double s  = Fermi::PSF::separation(d);
        double dp = Fermi::PSF::inverse_separation(s);
        REQUIRE(doctest::Approx(d) == dp);
    }
}


TEST_CASE("Test uPsf")
{
    auto cfg                = Fermi::XtCfg();

    auto const opt_energies = Fermi::fits::ccube_energies(cfg.cmap);
    auto const energies     = good(opt_energies, "Cannot read ccube_energies file!");
    auto const logEs        = Fermi::log10_v(energies);

    auto const srcs         = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs         = Fermi::directions_from_point_sources(srcs);

    // skipping ROI cuts.
    // skipping edisp_bin expansion.

    //********************************************************************************
    // Read IRF Fits Files.
    //********************************************************************************
    auto const opt_aeff     = Fermi::load_aeff(cfg.aeff_name);
    auto const opt_psf      = Fermi::load_psf(cfg.psf_name);
    auto const aeff_irf     = good(opt_aeff, "Cannot read AEFF Irf FITS file!");
    auto const psf_irf      = good(opt_psf, "Cannot read PSF Irf FITS file!");

    auto const front_LTF
        = Fermi::lt_effic_factors(logEs, aeff_irf.front.efficiency_params);

    //********************************************************************************
    // Read Exposure Cube Fits File.
    //********************************************************************************
    auto opt_exp_map     = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto opt_wexp_map    = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    auto opt_ccube       = Fermi::fits::ccube_pixels(cfg.cmap);
    auto const exp_cube  = good(opt_exp_map, "Cannot read exposure cube map file!");
    auto const wexp_cube = good(opt_wexp_map, "Cannot read exposure cube map file!");
    auto const ccube     = good(opt_ccube, "Cannot read counts cube map file!");
    auto const exp_costhetas                 = Fermi::exp_costhetas(exp_cube);
    auto const exp_map                       = Fermi::exp_map(exp_cube);
    auto const wexp_map                      = Fermi::exp_map(wexp_cube);
    auto const src_exposure_cosbins          = Fermi::src_exp_cosbins(dirs, exp_map);
    auto const src_weighted_exposure_cosbins = Fermi::src_exp_cosbins(dirs, wexp_map);

    for (size_t i = 0; i < FermiTest::exp_costhetas.size(); ++i)
    {
        REQUIRE(exp_costhetas[i] == doctest::Approx(FermiTest::exp_costhetas[i]));
    }
    //********************************************************************************
    // Effective Area Computations.
    //********************************************************************************
    auto const front_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff_irf.front.effective_area);
    auto const back_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff_irf.back.effective_area);
    SUBCASE("EXPECTED AEFF")
    {
        et2comp_exprm(front_aeff, FermiTest::aeff_front_c_e);
        et2comp_exprm(back_aeff, FermiTest::aeff_back_c_e);
    }


    //********************************************************************************
    // Exposure
    //********************************************************************************
    auto const exposure = Fermi::exposure(src_exposure_cosbins,
                                          src_weighted_exposure_cosbins,
                                          front_aeff,
                                          back_aeff,
                                          front_LTF);
    SUBCASE("EXPECTED EXPOSURE!") { filecomp2(exposure, "exposure"); }

    //********************************************************************************
    // Mean PSF Computations
    //********************************************************************************
    // D,Mc,Me
    // Me,Mc,D
    // [E C D]
    auto const front_kings = Fermi::PSF::king(psf_irf.front);
    auto const back_kings  = Fermi::PSF::king(psf_irf.back);
    SUBCASE("Kings") { filecomp3(front_kings, "king_front"); }

    // D E C
    auto const front_obs_psf = Fermi::PSF::bilerp(exp_costhetas,
                                                  logEs,
                                                  psf_irf.front.rpsf.cosths,
                                                  psf_irf.front.rpsf.logEs,
                                                  front_kings);
    SUBCASE("OBS Front") { filecomp3(front_obs_psf, "obs_psf_front_CED"); }

    auto const back_obs_psf = Fermi::PSF::bilerp(exp_costhetas,
                                                 logEs,
                                                 psf_irf.back.rpsf.cosths,
                                                 psf_irf.back.rpsf.logEs,
                                                 back_kings);
    SUBCASE("OBS Back") { filecomp3(back_obs_psf, "obs_psf_back_CED"); }

    auto const front_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(front_obs_psf,
                                             front_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             front_LTF);
    SUBCASE("Corr Front") { filecomp3(front_corr_exp_psf, "corr_psf_front"); }

    auto const back_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(back_obs_psf,
                                             back_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             /*Stays front for now.*/ front_LTF);
    SUBCASE("Corr Back") { filecomp3(back_corr_exp_psf, "corr_psf_back"); }


    // [Nd, Ne, Ns]
    auto uPsf = Fermi::PSF::mean_psf(front_corr_exp_psf, back_corr_exp_psf, exposure);
    SUBCASE("u PSF") { filecomp3(uPsf, "mean_psf"); }

    auto [part_psf_integ, psf_integ] = Fermi::PSF::partial_total_integral(uPsf);
    SUBCASE("uPsf_part_int_SED") { filecomp3(part_psf_integ, "uPsf_part_int_SED"); }

    Fermi::PSF::normalize(uPsf, psf_integ);
    SUBCASE("uPsf_normalized") { filecomp3(uPsf, "uPsf_normalized_SED"); }

    auto peak = Fermi::PSF::peak_psf(uPsf);
    SUBCASE("peak") { filecomp2(peak, "uPsf_peak_SE"); }
}
