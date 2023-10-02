#include "xtsrcmaps/psf/psf.hxx"

#include "xtsrcmaps/utils/bilerp.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <vector>


auto
Fermi::PSF::peak_psf(Tensor3d const& mean_psf /* [D, E, S] */) -> Tensor2d {
    long const Ne = mean_psf.dimension(1);
    long const Ns = mean_psf.dimension(2);
    return mean_psf.slice(Idx3 { 0, 0, 0 }, Idx3 { 1, Ne, Ns })
        .reshape(Idx2 { Ne, Ns });
}
