#include "xtsrcmaps/irf/irf.hxx"

auto
Fermi::Irf::collect_irf_data(Config::XtCfg const& cfg,
                             Obs::XtObs const&    obs) -> XtIrf {

    //**************************************************************************
    // Read IRF Fits Files.
    //**************************************************************************
    auto opt_aeff  = Fermi::Irf::load_aeff(cfg.aeff_file);
    auto opt_psf   = Fermi::Irf::load_psf(cfg.psf_file);
    auto aeff_irf  = good(opt_aeff, "Cannot read AEFF Irf FITS file!");
    auto psf_irf   = good(opt_psf, "Cannot read PSF Irf FITS file!");

    auto front_LTF = Fermi::Irf::livetime_efficiency_factors(
        obs.logEs, aeff_irf.front.efficiency_params);

    return { .aeff_irf = aeff_irf, .psf_irf = psf_irf, .front_LTF = front_LTF };
}
