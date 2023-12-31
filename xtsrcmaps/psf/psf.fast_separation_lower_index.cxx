#include "xtsrcmaps/psf/psf.hxx"

#include "xtsrcmaps/math/tensor_types.hxx"

auto
Fermi::PSF::fast_separation_lower_index(Tensor1d seps) -> Tensor1i {
    seps           = 1e4 * seps;
    Tensor1d Mseps = 1. + (seps.log() / sep_step);
    Tensor1i index = (seps < 1.).select(seps, Mseps).floor().cast<Eigen::DenseIndex>();
    return index;
}
