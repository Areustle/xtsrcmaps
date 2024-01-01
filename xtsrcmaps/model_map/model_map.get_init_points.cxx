#include "xtsrcmaps/model_map/model_map.hxx"


auto
Fermi::ModelMap::get_init_points(long const Nh, long const Nw) -> Tensor3d {
    Tensor3d init_points(2, Nh, Nw);
    for (long w = 0; w < Nw; ++w) {
        for (long h = 0; h < Nh; ++h) {
            init_points(0, h, w) = 1. + h;
            init_points(1, h, w) = 1. + w;
        }
    }

    return init_points;
}
