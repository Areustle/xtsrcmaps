#include "xtsrcmaps/psf/psf.hxx"

using Tensor2d = Fermi::Tensor<double, 2>;
using Tensor3d = Fermi::Tensor<double, 3>;

auto
Fermi::PSF::mean_psf(Tensor3d const& front_corrected_psf, /*[Nsrc, Nd, Ne]*/
                     Tensor3d const& back_corrected_psf,  /*[Nsrc, Nd, Ne]*/
                     Tensor2d const& exposures            /*[Nsrc, Ne]*/
                     ) -> Tensor<float, 3> /* [Nsrc, Nd, Ne] */ {
    size_t const Ns = front_corrected_psf.extent(0);
    size_t const Nd = front_corrected_psf.extent(1);
    size_t const Ne = front_corrected_psf.extent(2);

    Tensor<float, 3> psf(Ns, Nd, Ne);
    /* Tensor3d psf          = front_corrected_psf + back_corrected_psf; */
    /* Tensor3d inv_exposure = exposures.inverse().reshape(Idx3 { 1, Ne, Ns });
     */
    /* return psf * inv_exposure.broadcast(Idx3 { Nd, 1, 1 }); */
    std::transform(front_corrected_psf.begin(),
                   front_corrected_psf.end(),
                   back_corrected_psf.begin(),
                   psf.begin(),
                   std::plus {});

    for (size_t s = 0; s < Ns; ++s) {
        for (size_t d = 0; d < Nd; ++d) {
            for (size_t e = 0; e < Ne; ++e) { psf[s, d, e] /= exposures[s, e]; }
        }
    }

    return psf;
}
