#include "xtsrcmaps/psf/psf.hxx"

auto
Fermi::PSF::normalize(Tensor3d&       upsf,           /* [Nd, Ne, Ns] */
                      Tensor2d const& total_integrals /*     [Ne, Ns] */
                      ) -> void {
    long const Nd = upsf.dimension(0);
    long const Ne = upsf.dimension(1);
    long const Ns = upsf.dimension(2);
    assert(total_integrals.dimension(0) == Ne);
    assert(total_integrals.dimension(1) == Ns);
    upsf /= total_integrals.reshape(Idx3 { 1, Ne, Ns })
                .broadcast(Idx3 { Nd, 1, 1 });
}
