#include "xtsrcmaps/model_map/model_map.hxx"


void
Fermi::ModelMap::scale_map_by_correction_factors(
    Tensor4d&       model_map, /*[E,H,W,S]*/
    Tensor2d const& factor /*[E,S]*/) {
    long const Ne = model_map.dimension(0);
    long const Nh = model_map.dimension(1);
    long const Nw = model_map.dimension(2);
    long const Ns = model_map.dimension(3);

    assert(Ne == factor.dimension(0));
    assert(Ns == factor.dimension(1));

    model_map *= factor.reshape(Idx4 { Ne, 1, 1, Ns })
                     .broadcast(Idx4 { 1, Nh, Nw, 1 });
}
