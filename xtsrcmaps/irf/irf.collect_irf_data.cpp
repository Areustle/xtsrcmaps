#include "xtsrcmaps/irf/irf.hxx"

#include "fmt/color.h"

auto
Fermi::collect_irf_data(XtCfg const& cfg, XtObs const& obs) -> XtIrf {

    fmt::print(fg(fmt::color::light_pink),
               "Collecting Instrument Response Functions.\n");

    //**************************************************************************
    // Read IRF Fits Files.
    //**************************************************************************
    auto opt_aeff  = Fermi::load_aeff(cfg.aeff_file);
    auto opt_psf   = Fermi::load_psf(cfg.psf_file);
    auto aeff_irf  = good(opt_aeff, "Cannot read AEFF Irf FITS file!");
    auto psf_irf   = good(opt_psf, "Cannot read PSF Irf FITS file!");

    auto front_LTF = Fermi::livetime_efficiency_factors(
        obs.logEs, aeff_irf.front.efficiency_params);

    return { .aeff_irf = aeff_irf, .psf_irf = psf_irf, .front_LTF = front_LTF };
}
