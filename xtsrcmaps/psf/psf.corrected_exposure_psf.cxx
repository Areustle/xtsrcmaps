#include "xtsrcmaps/psf/psf.hxx"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

using Tensor2d = Fermi::Tensor<double, 2>;
using Tensor3d = Fermi::Tensor<double, 3>;

auto
Fermi::PSF::corrected_exposure_psf(
    Tensor3d const& obs_psf,                       /*[CDE]*/
    Tensor2d const& obs_aeff,                      /*[CE]*/
    Tensor2d const& src_exposure_cosbins,          /*[SC]*/
    Tensor2d const& src_weighted_exposure_cosbins, /*[SC]*/
    Tensor2d const& front_LTF                      /*[2E]*/
    ) -> Tensor3d {
    size_t const Ns = src_exposure_cosbins.extent(0);
    size_t const Nc = obs_psf.extent(0);
    size_t const Nd = obs_psf.extent(1);
    size_t const Ne = obs_psf.extent(2);


    assert(Nc == obs_aeff.extent(0));
    assert(Ne == obs_aeff.extent(1));
    assert(Ne == front_LTF.extent(1));

    // auto psf_aeff     = Fermi::mul322(obs_psf, obs_aeff); // [D, E, C] . [E,
    // C] = [D, E, C]
    /* Tensor3d psf_aeff */
    /*     = obs_psf */
    /*       * obs_aeff.reshape(Idx3 { 1, Ne, Nc }).broadcast(Idx3 { Nd, 1, 1
     * }); */
    // CED = CED * CE
    Tensor3d psf_aeff(Nc, Nd, Ne);
    for (size_t c = 0; c < Nc; ++c) {
        for (size_t d = 0; d < Nd; ++d) {
            for (size_t e = 0; e < Ne; ++e) {
                psf_aeff[c, d, e] = obs_psf[c, d, e] * obs_aeff[c, e];
            }
        }
    }

    /* // auto exposure_psf = Fermi::contract3210(psf_aeff,
     * src_exposure_cosbins); */
    /* Tensor3d exposure_psf */
    /*     = psf_aeff.contract(src_exposure_cosbins, IdxPair1 { { { 2, 0 } } });
     */
    Tensor3d exposure_psf(Ns, Nd, Ne);
    // SDE = SC.CDE
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                Ns,
                Nd * Ne,
                Nc,
                1.0,
                src_exposure_cosbins.data(),
                Nc,
                psf_aeff.data(),
                Nd * Ne,
                0.0,
                exposure_psf.data(),
                Nd * Ne);
    /* TensorMap<Tensor3d const> fLTR1(front_LTF.first.data(), 1, Ne, 1); */
    /* Tensor3d corrected_exp_psf */
    /*     = exposure_psf * fLTR1.broadcast(Idx3 { Ns, 1, Nd }); */
    for (size_t s = 0; s < Ns; ++s) {
        for (size_t d = 0; d < Nd; ++d) {
            for (size_t e = 0; e < Ne; ++e) {
                exposure_psf[s, d, e] *= front_LTF[0, e];
            }
        }
    }

    /* // auto wexp_psf = Fermi::contract3210(psf_aeff, */
    /* // src_weighted_exposure_cosbins); */
    Tensor3d wexp_psf(Ns, Nd, Ne);
    // SDE = SC.CDE
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                Ns,
                Ne * Nd,
                Nc,
                1.0,
                src_weighted_exposure_cosbins.data(),
                Nc,
                psf_aeff.data(),
                Ne * Nd,
                0.0,
                wexp_psf.data(),
                Ne * Nd);
    /* TensorMap<Tensor3d const> fLTR2(front_LTF.second.data(), 1, Ne, 1); */
    /* Tensor3d wexp_psf = psf_aeff.contract(src_weighted_exposure_cosbins, */
    /*                                       IdxPair1 { { { 2, 0 } } }); */
    // auto corrected_weighted_exp_psf = Fermi::mul310(wexp_psf,
    // front_LTF.second);
    /* Tensor3d corrected_weighted_exp_psf */
    /*     = wexp_psf * fLTR2.broadcast(Idx3 { Ns, 1, Nd }); */
    for (size_t s = 0; s < Ns; ++s) {
        for (size_t d = 0; d < Nd; ++d) {
            for (size_t e = 0; e < Ne; ++e) {
                wexp_psf[s, d, e] *= front_LTF[1, e];
            }
        }
    }

    std::transform(exposure_psf.begin(),
                   exposure_psf.end(),
                   wexp_psf.begin(),
                   exposure_psf.begin(),
                   std::plus {});

    // return Fermi::sum3_3(corrected_exp_psf, corrected_weighted_exp_psf);
    /* return corrected_exp_psf + corrected_weighted_exp_psf; */
    return exposure_psf;
}
