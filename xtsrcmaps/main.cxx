#include "xtsrcmaps/cli/cli.hxx"
#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/misc/misc.hxx"
#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/source/source.hxx"

#include <fmt/format.h>

int
main(int const argc, char** argv) {

    auto const cfg       = Fermi::parse_cli_to_cfg(argc, argv);

    auto const energies  = good(Fermi::fits::ccube_energies(cfg.cmap),
                               "Cannot read ccube_energies file!");
    auto const ccube     = good(Fermi::fits::ccube_pixels(cfg.cmap),
                            "Cannot read counts cube map file!");

    auto const srcs      = Fermi::parse_src_xml(cfg.srcmdl);
    auto const src_sph   = Fermi::spherical_coords_from_point_sources(srcs);
    auto const logEs     = Fermi::log10_v(energies);

    // skipping ROI cuts.
    // skipping edisp_bin expansion.

    //**************************************************************************
    // Read IRF Fits Files.
    //**************************************************************************
    auto const opt_aeff  = Fermi::load_aeff(cfg.aeff_file);
    auto const opt_psf   = Fermi::load_psf(cfg.psf_file);
    auto const aeff_irf  = good(opt_aeff, "Cannot read AEFF Irf FITS file!");
    auto const psf_irf   = good(opt_psf, "Cannot read PSF Irf FITS file!");

    auto const front_LTF = Fermi::livetime_efficiency_factors(
        logEs, aeff_irf.front.efficiency_params);

    //**************************************************************************
    // Read Exposure Cube Fits File.
    //**************************************************************************
    auto const exp_cube
        = good(Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE"),
               "Cannot read exposure cube map file!");
    auto const wexp_cube
        = good(Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE"),
               "Cannot read exposure cube map file!");

    auto const exp_costhetas        = Fermi::exp_costhetas(exp_cube);
    auto const exp_map              = Fermi::exp_map(exp_cube);
    auto const wexp_map             = Fermi::exp_map(wexp_cube);
    auto const src_exposure_cosbins = Fermi::src_exp_cosbins(src_sph, exp_map);
    auto const src_weighted_exposure_cosbins
        = Fermi::src_exp_cosbins(src_sph, wexp_map);

    //**************************************************************************
    // Effective Area Computations.
    //**************************************************************************
    auto const front_aeff = Fermi::aeff_value(
        exp_costhetas, logEs, aeff_irf.front.effective_area);
    auto const back_aeff
        = Fermi::aeff_value(exp_costhetas, logEs, aeff_irf.back.effective_area);


    //**************************************************************************
    // Exposure
    //**************************************************************************
    auto const exposures = Fermi::exposure(src_exposure_cosbins,
                                           src_weighted_exposure_cosbins,
                                           front_aeff,
                                           back_aeff,
                                           front_LTF);

    //**************************************************************************
    // Mean PSF Computations
    //**************************************************************************
    auto [uPsf, part_psf_integ]
        = Fermi::PSF::psf_lookup_table_and_partial_integrals(
            psf_irf,
            exp_costhetas,
            logEs,
            /* Used To Compute Corrected PSF */
            front_aeff,
            back_aeff,
            src_exposure_cosbins,
            src_weighted_exposure_cosbins,
            front_LTF,
            /* Exposures */
            exposures);

    auto model_map = Fermi::ModelMap::point_src_model_map_wcs(
        100, 100, src_sph, uPsf, { ccube }, exposures, part_psf_integ, 1e-3);

    Fermi::fits::write_src_model("!test2.fits", model_map, srcs);
}
