#include "xtsrcmaps/model_map/model_map.hxx"


void
Fermi::ModelMap::scale_map_by_correction_factors(
    Tensor<float, 4>& model_map, Tensor<float, 2> const& factor) {
    size_t const Ns = model_map.extent(0);
    size_t const Nh = model_map.extent(1);
    size_t const Nw = model_map.extent(2);
    size_t const Ne = model_map.extent(3);

    assert(Ns == factor.extent(0));
    assert(Ne == factor.extent(1));

#pragma omp parallel for schedule(static, 16)
    for (size_t s = 0; s < Ns; ++s) {
        for (size_t h = 0; h < Nh; ++h) {
            for (size_t w = 0; w < Nw; ++w) {
                for (size_t e = 0; e < Ne; ++e) {
                    model_map[s, h, w, e] *= factor[s, e];
                }
            }
        }
    }
    /* model_map *= factor.reshape(Idx4 { Ne, 1, 1, Ns }) */
    /*                  .broadcast(Idx4 { 1, Nh, Nw, 1 }); */
}
