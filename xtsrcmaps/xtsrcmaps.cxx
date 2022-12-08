#include <xtsrcmaps/config.hxx>
#include <xtsrcmaps/exposure.hxx>
#include <xtsrcmaps/fitsfuncs.hxx>
#include <xtsrcmaps/irf.hxx>
#include <xtsrcmaps/misc.hxx>
#include <xtsrcmaps/model_map.hxx>
#include <xtsrcmaps/parse_src_mdl.hxx>
#include <xtsrcmaps/psf.hxx>
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
main()
{
    // : parse-cli : Parse the command line arguments, parfiles, etc.
    // Just use st_app? hoops? ape?
    // rewrite hoops ape locally?
    // No. Goal is to prototype a faster srcmaps, not rewrite stapp+hoops+ape.
    // Just use command line parameters and a pre-defined struct.
    auto cfg                = Fermi::XtCfg();

    auto const opt_energies = Fermi::fits::ccube_energies(cfg.cmap);
    auto const opt_ccube    = Fermi::fits::ccube_pixels(cfg.cmap);
    auto const energies     = good(opt_energies, "Cannot read ccube_energies file!");
    auto const logEs        = Fermi::log10_v(energies);

    auto const srcs         = Fermi::parse_src_xml(cfg.srcmdl);
    auto const dirs         = Fermi::directions_from_point_sources(srcs);
    auto const ccube        = good(opt_ccube, "Cannot read counts cube map file!");

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
    // auto const separations   = Fermi::PSF::separations();
    auto const front_kings   = Fermi::PSF::king(psf_irf.front);
    auto const back_kings    = Fermi::PSF::king(psf_irf.back);
    auto const front_psf_val = Fermi::PSF::bilerp(exp_costhetas,
                                                  logEs,
                                                  psf_irf.front.rpsf.cosths,
                                                  psf_irf.front.rpsf.logEs,
                                                  front_kings);
    auto const back_psf_val  = Fermi::PSF::bilerp(exp_costhetas,
                                                 logEs,
                                                 psf_irf.back.rpsf.cosths,
                                                 psf_irf.back.rpsf.logEs,
                                                 back_kings);
    auto const front_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(front_psf_val,
                                             front_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             front_LTF);
    auto const back_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(back_psf_val,
                                             back_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             /*Stays front for now.*/ front_LTF);

    auto MDuPsf = Fermi::PSF::mean_psf(front_corr_exp_psf, back_corr_exp_psf, exposure);
    auto MDuPeak   = Fermi::PSF::peak_psf(MDuPsf);


    // long Ns             = 263;
    // long Ne             = 38;
    // long Nd             = 401;
    //
    // Tensor3d const uPsf = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/uPsf_normalized_SED.bin", Ns, Ne, Nd);
    // assert(uPsf.dimension(0) == Nd);
    // assert(uPsf.dimension(1) == Ne);
    // assert(uPsf.dimension(2) == Ns);

    // Tensor2d const uPeak = Fermi::row_major_file_to_col_major_tensor(
    //     "./xtsrcmaps/tests/expected/uPsf_peak_SE.bin", Ns, Ne);
    // assert(uPeak.dimension(0) == Ne);
    // assert(uPeak.dimension(1) == Ns);

    auto model_map = Fermi::ModelMap::point_src_model_map_wcs(
        100, 100, dirs, MDuPsf, { ccube }, 1e-3);
}
