#include "xtsrcmaps/model_map/model_map.hxx"


void
Fermi::ModelMap::scale_map_by_exposure(Tensor<float, 4>&        model_map,
                                       Tensor<double, 2> const& exposures) {
    size_t const Ns = model_map.extent(0);
    size_t const Nh = model_map.extent(1);
    size_t const Nw = model_map.extent(2);
    size_t const Ne = model_map.extent(3);

    assert(Ns == exposures.extent(0));
    assert(Ne == exposures.extent(1));

    /* model_map */
    /*     *= exposures.reshape<4>({ Ns, 1, 1, Ne }).broadcast({ Ns, Nh, Nw, Ne }); */
#pragma omp parallel for schedule(static, 16)
    for (size_t s = 0; s < Ns; ++s) {
        for (size_t h = 0; h < Nh; ++h) {
            for (size_t w = 0; w < Nw; ++w) {
                for (size_t e = 0; e < Ne; ++e) {
                    model_map[s, h, w, e] *= exposures[s, e];
                }
            }
        }
    }
}
