#include "xtsrcmaps/psf/psf.hxx"


auto
Fermi::PSF::mean_psf(Tensor3d const& front_corrected_psf, /*[Nd, Ne, Nsrc]*/
                     Tensor3d const& back_corrected_psf,  /*[Nd, Ne, Nsrc]*/
                     Tensor2d const& exposures             /*[Ne, Nsrc]*/
                     ) -> Tensor3d {
    long const Nd         = front_corrected_psf.dimension(0);
    long const Ne         = front_corrected_psf.dimension(1);
    long const Ns         = front_corrected_psf.dimension(2);

    Tensor3d psf          = front_corrected_psf + back_corrected_psf;
    Tensor3d inv_exposure = exposures.inverse().reshape(Idx3 { 1, Ne, Ns });
    return psf * inv_exposure.broadcast(Idx3 { Nd, 1, 1 });
}
