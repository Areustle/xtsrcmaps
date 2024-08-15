#include "xtsrcmaps/psf/psf.hxx"

#include <ranges>

auto
Fermi::PSF::fast_separation_lower_index(Tensor<double, 1> seps)
    -> Tensor<int, 1> {
    /* std::transform(seps.begin(),  */
    /* seps           = 1e4 * seps; */
    /* Tensor<double, 1> Mseps = 1. + (seps.log() / sep_step); */
    /* Tensor1i index */
    /*     = (seps < 1.).select(seps, Mseps).floor().cast<Eigen::DenseIndex>();
     */
    Tensor<int, 1> index(seps.extent(0));
    std::ranges::transform(seps, index.begin(), [](double const& v) {
        return v < 1. ? 1
                      : static_cast<int>(
                            std::floor(1. + std::log(1e4 * v) * recipstep));
    });

    return index;
}
