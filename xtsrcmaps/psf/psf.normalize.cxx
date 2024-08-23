#include "xtsrcmaps/psf/psf.hxx"

using Tensor2d = Fermi::Tensor<double, 2>;
using Tensor3d = Fermi::Tensor<double, 3>;
using Tensor3f = Fermi::Tensor<float, 3>;

auto
Fermi::PSF::normalize(Tensor3f&       uPsf,           /* [Ns, Nd, Ne] */
                      Tensor2d const& total_integrals /*     [Ns, Ne] */
                      ) -> void {
    size_t const Ns = uPsf.extent(0);
    size_t const Nd = uPsf.extent(1);
    size_t const Ne = uPsf.extent(2);
    assert(total_integrals.extent(1) == Ne);
    assert(total_integrals.extent(0) == Ns);

    for (size_t s = 0; s < Ns; ++s) {
        for (size_t e = 0; e < Ne; ++e) {
            for (size_t d = 0; d < Nd; ++d) {
                uPsf[s, d, e] /= total_integrals[s, e];
            }
        }
    }
}
