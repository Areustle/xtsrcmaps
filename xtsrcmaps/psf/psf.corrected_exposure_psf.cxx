#include "xtsrcmaps/psf/psf.hxx"

auto
Fermi::PSF::corrected_exposure_psf(
    Tensor3d const& obs_psf,                       /*[C, E, D] -> [D, E, C]*/
    Tensor2d const& obs_aeff,                      /*[C, E] -> [E, C]*/
    Tensor2d const& src_exposure_cosbins,          /*[S, C] -> [C, S]*/
    Tensor2d const& src_weighted_exposure_cosbins, /*[S, C] -> [C, S]*/
    std::pair<std::vector<double>, std::vector<double>> const& front_LTF /*[E]*/
    ) -> Tensor3d {
    long const Nd = obs_psf.dimension(0);
    long const Ne = obs_psf.dimension(1);
    long const Nc = obs_psf.dimension(2);
    long const Ns = src_exposure_cosbins.dimension(1);

    assert(Ne == obs_aeff.dimension(0));
    assert(Nc == obs_aeff.dimension(1));

    TensorMap<Tensor3d const> fLTR1(front_LTF.first.data(), 1, Ne, 1);
    TensorMap<Tensor3d const> fLTR2(front_LTF.second.data(), 1, Ne, 1);

    // auto psf_aeff     = Fermi::mul322(obs_psf, obs_aeff); // [D, E, C] . [E,
    // C] [D, E, C]
    Tensor3d psf_aeff
        = obs_psf
          * obs_aeff.reshape(Idx3 { 1, Ne, Nc }).broadcast(Idx3 { Nd, 1, 1 });

    // [D, E, S] = SUM_c ([D, E, C] * [C, S])
    // auto exposure_psf = Fermi::contract3210(psf_aeff, src_exposure_cosbins);
    Tensor3d exposure_psf
        = psf_aeff.contract(src_exposure_cosbins, IdxPair1 { { { 2, 0 } } });
    // auto wexp_psf = Fermi::contract3210(psf_aeff,
    // src_weighted_exposure_cosbins);
    Tensor3d wexp_psf = psf_aeff.contract(src_weighted_exposure_cosbins,
                                          IdxPair1 { { { 2, 0 } } });

    // [D, E, S]
    // auto corrected_exp_psf          = Fermi::mul310(exposure_psf,
    // front_LTF.first);
    Tensor3d corrected_exp_psf
        = exposure_psf * fLTR1.broadcast(Idx3 { Nd, 1, Ns });
    // auto corrected_weighted_exp_psf = Fermi::mul310(wexp_psf,
    // front_LTF.second);
    Tensor3d corrected_weighted_exp_psf
        = wexp_psf * fLTR2.broadcast(Idx3 { Nd, 1, Ns });

    // return Fermi::sum3_3(corrected_exp_psf, corrected_weighted_exp_psf);
    return corrected_exp_psf + corrected_weighted_exp_psf;
}
