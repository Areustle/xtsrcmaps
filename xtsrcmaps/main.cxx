#include "xtsrcmaps/cli/cli.hxx"
#include "xtsrcmaps/exposure/exposure.hxx"
/* #include "xtsrcmaps/fits/fits.hxx" */
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/observation/observation.hxx"
#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/source/source.hxx"

using namespace Fermi;

int
main(int const argc, char** argv) {

    struct ModelMaps {
        Tensor<double, 4> pt = {};
        Tensor<double, 4> ds = {};
    };

    auto const cfg  = Config::parse_cli_to_cfg(argc, argv);
    auto const obs  = Obs::collect_observation_data(cfg);
    auto const src  = Source::collect_source_model(cfg, obs);
    auto const irf  = Irf::collect_irf_data(cfg, obs);
    auto       maps = ModelMaps {};

    // Point sources
    if (src.point.srcs.size()) {
        auto ptexp = Exposure::compute_exposure(cfg, obs, src.point, irf);
        auto ptpsf = Psf::compute_psf_data(obs, irf, ptexp);
        maps.pt    = ModelMap::compute_srcmaps(obs, src.point, ptexp, ptpsf);
    }

    // Diffuse sources
    /* if (src.diffuse.srcs.size()) { */
    /*     auto dsexp = Exposure::compute_exposure(cfg, obs, src.diffuse, irf); */
    /*     auto dspsf = Psf::compute_psf_data(obs, irf, dsexp); */
    /*     maps.ds    = ModelMap::compute_srcmaps(obs, src.diffuse, dsexp, dspsf); */
    /* } */

    /* fits::write_src_model(cfg.outfile, model_map, src); */
}
