#include "xtsrcmaps/model_map/model_map.hxx"

using namespace Fermi;
using namespace Fermi::Source;

template <>
auto
ModelMap::compute_srcmaps<PointSource>(Obs::XtObs const&       obs,
                                       SourceData<PointSource> src,
                                       Exposure::XtExp const&  exp,
                                       Psf::XtPsf const&       psf)
    -> Tensor<double, 4> {

    return ModelMap::point_src_model_map_wcs(obs.Nh,
                                             obs.Nw,
                                             obs.skygeom,
                                             src.sph_locs,
                                             src.names,
                                             exp.exposure,
                                             psf.uPsf,
                                             psf.partial_psf_integral);
};

// template <>
// auto
// ModelMap::compute_srcmaps<DiffuseSource>(Obs::XtObs const&         obs,
//                                          SourceData<DiffuseSource> src,
//                                          Exposure::XtExp const&    exp,
//                                          Psf::XtPsf const&         psf)
//     -> Tensor<double, 4> {
//
//     return ModelMap::diffuse_src_model_map_wcs(obs.Nh,
//                                                obs.Nw,
//                                                obs.skygeom,
//                                                src.sph_locs,
//                                                src.names,
//                                                exp.exposure,
//                                                psf.uPsf,
//                                                psf.partial_psf_integral);
// };
