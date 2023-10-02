#include <xtsrcmaps/config.hxx>
#include <xtsrcmaps/exposure.hxx>
#include <xtsrcmaps/fitsfuncs.hxx>
#include <xtsrcmaps/irf.hxx>
#include <xtsrcmaps/misc.hxx>
#include <xtsrcmaps/model_map.hxx>
#include <xtsrcmaps/parse_src_mdl.hxx>
#include <xtsrcmaps/psf/psf.hxx>
#include <xtsrcmaps/source.hxx>
#include <xtsrcmaps/source_utils.hxx>
#include <xtsrcmaps/tensor_ops.hxx>

#include <fmt/format.h>
#include <xtsrcmaps/fmt_source.hxx>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <vector>

// using std::vector;

int
main() {
    // : parse-cli : Parse the command line arguments, parfiles, etc.
    // Just use st_app? hoops? ape?
    // rewrite hoops ape locally?
    // No. Goal is to prototype a faster srcmaps, not rewrite stapp+hoops+ape.
    // Just use command line parameters and a pre-defined struct.
    auto cfg            = Fermi::XtCfg();

    auto const energies = good(Fermi::fits::ccube_energies(cfg.cmap),
                               "Cannot read ccube_energies file!");
    auto const ccube    = good(Fermi::fits::ccube_pixels(cfg.cmap),
                            "Cannot read counts cube map file!");

    auto const srcs     = Fermi::parse_src_xml(cfg.srcmdl);
    auto const src_sph  = Fermi::spherical_coords_from_point_sources(srcs);
    auto const logEs    = Fermi::log10_v(energies);

    // skipping ROI cuts.
    // skipping edisp_bin expansion.

    //********************************************************************************
    // Read IRF Fits Files.
    //********************************************************************************
    auto const opt_aeff = Fermi::load_aeff(cfg.aeff_name);
    auto const opt_psf  = Fermi::load_psf(cfg.psf_name);
    auto const aeff_irf = good(opt_aeff, "Cannot read AEFF Irf FITS file!");
    auto const psf_irf  = good(opt_psf, "Cannot read PSF Irf FITS file!");

    auto const front_LTF
        = Fermi::livetime_efficiency_factors(logEs, aeff_irf.front.efficiency_params);

    //********************************************************************************
    // Read Exposure Cube Fits File.
    //********************************************************************************
    auto const exp_cube = good(Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE"),
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
    auto const exposures        = Fermi::exposure(src_exposure_cosbins,
                                           src_weighted_exposure_cosbins,
                                           front_aeff,
                                           back_aeff,
                                           front_LTF);

    //********************************************************************************
    // Mean PSF Computations
    //********************************************************************************
    auto [uPsf, part_psf_integ] = Fermi::PSF::psf_lookup_table_and_partial_integrals(
        psf_irf,
        exp_costhetas,
        logEs,
        /* Used To Compute Exposures */
        front_aeff,
        back_aeff,
        src_exposure_cosbins,
        src_weighted_exposure_cosbins,
        front_LTF,
        /* Exposures */
        exposures);

    auto model_map = Fermi::ModelMap::point_src_model_map_wcs(
        100, 100, src_sph, uPsf, { ccube }, exposures, part_psf_integ, 1e-3);
}
