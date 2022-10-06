#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"

#include <fstream>

#include "xtsrcmaps/tests/fermi_tests.hxx"

#include "xtsrcmaps/config.hxx"
#include "xtsrcmaps/exposure.hxx"
#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/parse_src_mdl.hxx"
#include "xtsrcmaps/psf.hxx"
#include "xtsrcmaps/source_utils.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

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
    auto const exp_cube  = good(opt_exp_map, "Cannot read exposure cube map file!");
    auto const wexp_cube = good(opt_wexp_map, "Cannot read exposure cube map file!");
    auto const exp_costhetas                 = Fermi::exp_costhetas(exp_cube);
    auto const exp_map                       = Fermi::exp_map(exp_cube);
    auto const wexp_map                      = Fermi::exp_map(wexp_cube);
    auto const src_exposure_cosbins          = Fermi::src_exp_cosbins(dirs, exp_map);
    auto const src_weighted_exposure_cosbins = Fermi::src_exp_cosbins(dirs, wexp_map);

    //********************************************************************************
    // Effective Area Computations.
    //********************************************************************************
    auto const front_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff_irf.front.effective_area);
    auto const back_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff_irf.back.effective_area);


    //********************************************************************************
    // Exposure
    //********************************************************************************
    auto const exposure      = Fermi::exposure(src_exposure_cosbins,
                                          src_weighted_exposure_cosbins,
                                          front_aeff,
                                          back_aeff,
                                          front_LTF);

    //********************************************************************************
    // Mean PSF Computations
    //********************************************************************************
    auto const separations   = Fermi::PSF::separations(1e-4, 70., 400);
    auto const front_kings   = Fermi::PSF::king(separations, psf_irf.front);
    auto const back_kings    = Fermi::PSF::king(separations, psf_irf.back);
    auto const front_obs_psf = Fermi::PSF::bilerp(exp_costhetas,
                                                  logEs,
                                                  psf_irf.front.rpsf.cosths,
                                                  psf_irf.front.rpsf.logEs,
                                                  front_kings);
    auto const back_obs_psf  = Fermi::PSF::bilerp(exp_costhetas,
                                                 logEs,
                                                 psf_irf.back.rpsf.cosths,
                                                 psf_irf.back.rpsf.logEs,
                                                 back_kings);
    auto const front_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(front_obs_psf,
                                             front_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             front_LTF);
    auto const back_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(back_obs_psf,
                                             back_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             /*Stays front for now.*/ front_LTF);

    auto uPsf = Fermi::PSF::mean_psf(front_corr_exp_psf, back_corr_exp_psf, exposure);

    SUBCASE("u PSF") { filecomp3(uPsf, "mean_psf"); }
}

// TEST_CASE("Test PSF")
// {
// auto const cfg      = Fermi::XtCfg();
// auto const opt_aeff = Fermi::load_aeff(cfg.aeff_name);
// auto const opt_psf  = Fermi::load_psf(cfg.psf_name);
// auto const oexpcb   = Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE");
// auto const owexpmap = Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE");
// auto const oen      = Fermi::fits::ccube_energies(cfg.cmap);
// auto const srcs     = Fermi::parse_src_xml(cfg.srcmdl);
// auto const dirs     = Fermi::directions_from_point_sources(srcs);
// REQUIRE(opt_aeff);
// REQUIRE(opt_psf);
// REQUIRE(oexpcb);
// REQUIRE(oen);
// auto const aeff_irf = opt_aeff.value();
// auto const psf_irf  = opt_psf.value();
// auto const logEs    = Fermi::log10_v(oen.value());
// auto const front_LTF
//     = Fermi::lt_effic_factors(logEs, aeff_irf.front.efficiency_params);
// auto const exp_costhetas                 = Fermi::exp_costhetas(oexpcb.value());
// auto const exp_map                       = Fermi::exp_map(oexpcb.value());
// auto const wexp_map                      = Fermi::exp_map(owexpmap.value());
// auto const src_exposure_cosbins          = Fermi::src_exp_cosbins(dirs, exp_map);
// auto const src_weighted_exposure_cosbins = Fermi::src_exp_cosbins(dirs, wexp_map);
// auto const separations                   = Fermi::PSF::separations(1e-4, 70., 400);
// auto const front_kings   = Fermi::PSF::king(separations, psf_irf.front);
//
// auto const front_psf_val = Fermi::PSF::bilerp(exp_costhetas,
//                                               logEs,
//                                               psf_irf.front.rpsf.cosths,
//                                               psf_irf.front.rpsf.logEs,
//                                               front_kings);
// REQUIRE(front_psf_val.extent(0) == 401); // D
// REQUIRE(front_psf_val.extent(1) == 40);  // C
// REQUIRE(front_psf_val.extent(2) == 38);  // E
// SUBCASE("Golden PSF") { filecomp3(front_psf_val, "psf_val_front"); }
//
// // :-(
// // REQUIRE(exp_costhetas != wexp_costhetas);
// // REQUIRE(front_psf_exp.container() != front_psf_wexp.container());
//
// // C,E
// auto const front_aeff
//     = Fermi::aeff_value(exp_costhetas, logEs, aeff_irf.front.effective_area);
// SUBCASE("Golden PSF_RAW_AEFF") { filecomp2(front_aeff, "psf_raw_aeff_front"); }
// REQUIRE(front_aeff.extent(0) == 40); // C
// REQUIRE(front_aeff.extent(1) == 38); // E
//
// auto const front_psf_aeff = Fermi::mul322(front_psf_val, front_aeff);
// SUBCASE("Golden PSF_AEFF") { filecomp3(front_psf_aeff, "psf_aeff_front"); }
// REQUIRE(front_psf_aeff.extent(0) == 401); // D
// REQUIRE(front_psf_aeff.extent(1) == 40);  // C
// REQUIRE(front_psf_aeff.extent(2) == 38);  // E
//
// auto const psf_expc_f = Fermi::contract3210(front_psf_aeff, src_exposure_cosbins);
// auto const psf_wexpc_f
//     = Fermi::contract3210(front_psf_aeff, src_weighted_exposure_cosbins);
//
// REQUIRE(psf_expc_f.extent(0) == 263);
// REQUIRE(psf_expc_f.extent(1) == 401);
// REQUIRE(psf_expc_f.extent(2) == 38);
//
// auto const lef = Fermi::mul310(psf_expc_f, front_LTF.first);
// auto const lwf = Fermi::mul310(psf_wexpc_f, front_LTF.second);
// REQUIRE(lef.extent(0) == 263);
// REQUIRE(lef.extent(1) == 401);
// REQUIRE(lef.extent(2) == 38);
// REQUIRE(lwf.extent(0) == 263);
// REQUIRE(lwf.extent(1) == 401);
// REQUIRE(lwf.extent(2) == 38);
// auto const front_psf_exposure = Fermi::sum3_3(lef, lwf);
//
// SUBCASE("PSF_EXPOSURE") { filecomp3(front_psf_exposure, "psf_exposure_front"); }
// }
