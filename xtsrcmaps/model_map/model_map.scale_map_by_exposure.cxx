#include "xtsrcmaps/model_map/model_map.hxx"


void
Fermi::ModelMap::scale_map_by_exposure(Tensor4d&       model_map,
                                       Tensor2d const& exposures) {
    long const Ne = model_map.dimension(0);
    long const Nh = model_map.dimension(1);
    long const Nw = model_map.dimension(2);
    long const Ns = model_map.dimension(3);

    assert(Ne == exposures.dimension(0));
    assert(Ns == exposures.dimension(1));

    model_map *= exposures.reshape(Idx4 { Ne, 1, 1, Ns })
                     .broadcast(Idx4 { 1, Nh, Nw, 1 });
}
