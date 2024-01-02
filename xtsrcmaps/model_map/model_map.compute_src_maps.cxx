#include "xtsrcmaps/model_map/model_map.hxx"


auto
Fermi::ModelMap::compute_srcmaps(XtObs const& obs,
                                 XtExp const& exp,
                                 XtPsf const& psf) -> Tensor4f {

    // If wcs
    return Fermi::ModelMap::point_src_model_map_wcs(obs.Nh,
                                                    obs.Nw,
                                                    obs.src_sph,
                                                    psf.uPsf,
                                                    { obs.ccube },
                                                    exp.exposure,
                                                    psf.partial_psf_integral,
                                                    1e-3);
}
