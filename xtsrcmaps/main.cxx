#include "xtsrcmaps/cli/cli.hxx"
#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/observation/observation.hxx"
#include "xtsrcmaps/psf/psf.hxx"

#include <fmt/format.h>

int
main(int const argc, char** argv) {

    auto const cfg = Fermi::parse_cli_to_cfg(argc, argv);
    auto const obs = Fermi::collect_observation_data(cfg);
    auto const irf = Fermi::collect_irf_data(cfg, obs);
    auto const exp = Fermi::compute_exposure_data(cfg, obs, irf);

    auto const psf = Fermi::PSF::compute_psf_data(obs, irf, exp);

    auto model_map
        = Fermi::ModelMap::point_src_model_map_wcs(100,
                                                   100,
                                                   obs.src_sph,
                                                   psf.uPsf,
                                                   { obs.ccube },
                                                   exp.exposure,
                                                   psf.partial_psf_integral,
                                                   1e-3);

    Fermi::fits::write_src_model(cfg.outfile, model_map, obs.srcs);
}
