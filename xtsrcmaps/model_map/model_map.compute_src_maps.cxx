#include "xtsrcmaps/model_map/model_map.hxx"


auto
Fermi::ModelMap::compute_srcmaps(XtObs const& obs,
                                 XtExp const& exp,
                                 XtPsf const& psf) -> Tensor<double, 4> {

    // If wcs
    return Fermi::ModelMap::point_src_model_map_wcs(obs.Nh,
                                                    obs.Nw,
                                                    obs.src_sph,
                                                    obs.src_names,
                                                    psf.uPsf,
                                                    { obs.ccube },
                                                    exp.exposure,
                                                    psf.partial_psf_integral);
}
