#include "gtest/gtest.h"

#include "tests/fermi_tests.hxx"
#include "tests/fits/exposure_cosbins_expected.hxx"

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/source/source.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

TEST(TestPSFComponent, PSF_SEPARATION) {
    auto tsep = Fermi::PSF::separations();
    ASSERT_TRUE(NearRelative(sep_step, std::log(70. / 1e-4) / (400. - 1.)));
    for (size_t j = 0; j < tsep.size(); ++j) {
        ASSERT_TRUE(NearRelative(tsep[j], Fermi::PSF::separation(j)) << j);
        ASSERT_TRUE(
            NearRelative(
                j, Fermi::PSF::inverse_separation(Fermi::PSF::separation(j)))
            << Fermi::PSF::separation(j));
    }

    for (int j = 0; j < int(1e6); ++j) {
        double d  = j * (3.99e-8);
        double s  = Fermi::PSF::separation(d);
        double dp = Fermi::PSF::inverse_separation(s);
        ASSERT_TRUE(NearRelative(d, dp));
    }
}


TEST(TestPSFComponent, MeanPSF) {
    auto cfg                = Fermi::XtCfg();

    auto const opt_energies = Fermi::fits::ccube_energies(cfg.cmap);
    auto const energies
        = good(opt_energies, "Cannot read ccube_energies file!");
    auto const logEs     = Fermi::log10_v(energies);

    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs      = Fermi::spherical_coords_from_point_sources(srcs);

    // skipping ROI cuts.
    // skipping edisp_bin expansion.

    //********************************************************************************
    // Read IRF Fits Files.
    //********************************************************************************
    auto const opt_aeff  = Fermi::load_aeff(cfg.aeff_file);
    auto const opt_psf   = Fermi::load_psf(cfg.psf_file);
    auto const aeff_irf  = good(opt_aeff, "Cannot read AEFF Irf FITS file!");
    auto const psf_irf   = good(opt_psf, "Cannot read PSF Irf FITS file!");

    auto const front_LTF = Fermi::livetime_efficiency_factors(
        logEs, aeff_irf.front.efficiency_params);

    //********************************************************************************
    // Read Observations and Exposure Cube Fits File.
    //********************************************************************************
    auto opt_exp_map = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
    auto opt_wexp_map
        = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
    auto       opt_ccube = Fermi::fits::ccube_pixels(cfg.cmap);
    auto const exp_cube
        = good(opt_exp_map, "Cannot read exposure cube map file!");
    auto const wexp_cube
        = good(opt_wexp_map, "Cannot read exposure cube map file!");
    auto const ccube = good(opt_ccube, "Cannot read counts cube map file!");
    auto const exp_costhetas        = Fermi::exp_costhetas(exp_cube);
    auto const exp_map              = Fermi::exp_map(exp_cube);
    auto const wexp_map             = Fermi::exp_map(wexp_cube);
    auto const src_exposure_cosbins = Fermi::src_exp_cosbins(dirs, exp_map);
    auto const src_weighted_exposure_cosbins
        = Fermi::src_exp_cosbins(dirs, wexp_map);

    for (size_t i = 0; i < FermiTest::exp_costhetas.size(); ++i) {
        ASSERT_TRUE(
            NearRelative(exp_costhetas[i], (FermiTest::exp_costhetas[i])));
    }
    //********************************************************************************
    // Effective Area Computations.
    //********************************************************************************
    auto const front_aeff = Fermi::aeff_value(
        exp_costhetas, logEs, aeff_irf.front.effective_area);
    auto const back_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff_irf.back.effective_area);
    // SUBCASE("EXPECTED AEFF") {
    et2comp(front_aeff, FermiTest::aeff_front_c_e);
    et2comp(back_aeff, FermiTest::aeff_back_c_e);


    //********************************************************************************
    // Exposure
    //********************************************************************************
    auto const exposure = Fermi::exposure(src_exposure_cosbins,
                                          src_weighted_exposure_cosbins,
                                          front_aeff,
                                          back_aeff,
                                          front_LTF);
    // SUBCASE("EXPECTED EXPOSURE!")
    filecomp(exposure, "exposure");

    //********************************************************************************
    // Mean PSF Computations
    //********************************************************************************
    // [D C E]
    auto front_kings         = Fermi::PSF::king(psf_irf.front);
    auto back_kings          = Fermi::PSF::king(psf_irf.back);
    // SUBCASE("Kings")
    /* filecomp<double>(front_kings, "king_front");  */

    // C D E
    auto const front_obs_psf = Fermi::PSF::bilerp(exp_costhetas,
                                                  logEs,
                                                  psf_irf.front.rpsf.cosths,
                                                  psf_irf.front.rpsf.logEs,
                                                  front_kings);

    size_t const Nc          = front_obs_psf.extent(0);
    size_t const Nd          = front_obs_psf.extent(1);
    size_t const Ne          = front_obs_psf.extent(2);

    /* SUBCASE("OBS Front")  */
    ASSERT_TRUE(exp_costhetas.size() == 40);
    ASSERT_TRUE(logEs.size() == 38);

    ASSERT_TRUE(front_obs_psf.extent(0) == 40);
    ASSERT_TRUE(front_obs_psf.extent(1) == 401);
    ASSERT_TRUE(front_obs_psf.extent(2) == 38);

    /* filecomp(front_obs_psf, "obs_psf_front_CDE"); */

    auto const back_obs_psf = Fermi::PSF::bilerp(exp_costhetas,
                                                 logEs,
                                                 psf_irf.back.rpsf.cosths,
                                                 psf_irf.back.rpsf.logEs,
                                                 back_kings);
    // SUBCASE("OBS Back")
    /* filecomp(back_obs_psf, "obs_psf_back_CDE");  */

    auto const front_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(front_obs_psf,
                                             front_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             front_LTF);

    // [Ns, Nd, Ne]
    size_t const Ns = src_exposure_cosbins.extent(0);
    // SUBCASE("Corr Front")
    filecomp(front_corr_exp_psf, "corr_psf_front_SDE");

    // SUBCASE("Corr Back")
    auto const back_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(back_obs_psf,
                                             back_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             front_LTF);
    filecomp(back_corr_exp_psf, "corr_psf_back_SDE");

    // [Ns, Nd, Ne]
    // SUBCASE("u PSF")
    auto uPsf
        = Fermi::PSF::mean_psf(front_corr_exp_psf, back_corr_exp_psf, exposure);
    filecomp<double, double, 3>(uPsf, "mean_psf_SDE");

    // SUBCASE("uPsf_part_int_SED")
    auto [part_psf_integ, psf_integ] = Fermi::PSF::partial_total_integral(uPsf);
    filecomp(part_psf_integ, "uPsf_part_int_SDE");

    // SUBCASE("uPsf_normalized")
    Fermi::PSF::normalize(uPsf, psf_integ);

    filecomp(uPsf, "uPsf_normalized_SDE");
}
