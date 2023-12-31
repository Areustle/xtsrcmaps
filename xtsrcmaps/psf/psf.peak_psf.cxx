#include "xtsrcmaps/psf/psf.hxx"

#include "xtsrcmaps/math/tensor_types.hxx"

#include <fmt/format.h>



auto
Fermi::PSF::peak_psf(Tensor3d const& mean_psf /* [D, E, S] */) -> Tensor2d {
    long const Ne = mean_psf.dimension(1);
    long const Ns = mean_psf.dimension(2);
    return mean_psf.slice(Idx3 { 0, 0, 0 }, Idx3 { 1, Ne, Ns })
        .reshape(Idx2 { Ne, Ns });
}
