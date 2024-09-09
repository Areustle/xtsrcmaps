#include "xtsrcmaps/observation/observation.hxx"
#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/misc/misc.hxx"


auto
Fermi::Obs::collect_observation_data(Fermi::Config::XtCfg const& cfg)
    -> Obs::XtObs {

    auto const energies = good(Fermi::fits::read_energies(cfg.cmap),
                               "Cannot read ccube_energies file!");

    auto const pars     = good(Fermi::fits::read_image_meta(cfg.cmap),
                           "Cannot read counts cube map file!");

    //**************************************************************************
    // Read Exposure Cube Fits File.
    //**************************************************************************
    auto const exp_cube
        = good(Fermi::fits::read_expcube(cfg.expcube, "EXPOSURE"),
               "Cannot read exposure cube map file!");
    auto const wexp_cube
        = good(Fermi::fits::read_expcube(cfg.expcube, "WEIGHTED_EXPOSURE"),
               "Cannot read exposure cube map file!");

    return {
        .Nh                = pars.naxes[0],
        .Nw                = pars.naxes[1],
        .energies          = energies,
        .logEs             = Fermi::log10_v(energies),
        .exp_cube          = exp_cube,
        .weighted_exp_cube = wexp_cube,
        .skygeom           = SkyGeom<double> { pars },
    };
};
