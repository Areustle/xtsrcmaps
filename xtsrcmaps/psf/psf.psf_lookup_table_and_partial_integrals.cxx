#include "xtsrcmaps/psf/psf.hxx"

auto
Fermi::PSF::psf_lookup_table_and_partial_integrals(
    irf::psf::Pass8FB const&   psf_irf,
    std::vector<double> const& exp_costhetas,
    std::vector<double> const& logEs,
    Tensor2d const&            front_aeff,
    Tensor2d const&            back_aeff,
    Tensor2d const&            src_exposure_cosbins,
    Tensor2d const&            src_weighted_exposure_cosbins,
    std::pair<std::vector<double>, std::vector<double>> const& front_LTF, /*[Ne]*/
    Tensor2d const&                                            exposures   /*[Ne, Nsrc]*/
    ) -> std::tuple<Tensor3d, Tensor3d> {
    auto const front_kings   = Fermi::PSF::king(psf_irf.front);
    auto const back_kings    = Fermi::PSF::king(psf_irf.back);
    auto const front_psf_val = Fermi::PSF::bilerp(exp_costhetas,
                                                  logEs,
                                                  psf_irf.front.rpsf.cosths,
                                                  psf_irf.front.rpsf.logEs,
                                                  front_kings);
    auto const back_psf_val  = Fermi::PSF::bilerp(exp_costhetas,
                                                 logEs,
                                                 psf_irf.back.rpsf.cosths,
                                                 psf_irf.back.rpsf.logEs,
                                                 back_kings);
    auto const front_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(front_psf_val,
                                             front_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             front_LTF);
    auto const back_corr_exp_psf
        = Fermi::PSF::corrected_exposure_psf(back_psf_val,
                                             back_aeff,
                                             src_exposure_cosbins,
                                             src_weighted_exposure_cosbins,
                                             /*Stays front for now.*/ front_LTF);

    Tensor3d uPsf
        = Fermi::PSF::mean_psf(front_corr_exp_psf, back_corr_exp_psf, exposures);
    // auto uPeak                       = Fermi::PSF::peak_psf(uPsf);
    auto [part_psf_integ, psf_integ] = Fermi::PSF::partial_total_integral(uPsf);

    Fermi::PSF::normalize(uPsf, psf_integ);

    return { uPsf, part_psf_integ };
}
