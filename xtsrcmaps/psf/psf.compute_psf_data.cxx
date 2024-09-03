#include "xtsrcmaps/psf/psf.hxx"

#include "xtsrcmaps/exposure/exposure.hxx"
#include "xtsrcmaps/irf/irf_types.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"

#include "fmt/color.h"

auto
Fermi::PSF::compute_psf_data(XtObs const& obs,
                             XtIrf const& irf,
                             XtExp const  exp) -> XtPsf {

    /* fmt::print(fg(fmt::color::light_pink), "Computing Point Spread Function.\n"); */
    //**************************************************************************
    // Mean PSF Computations
    //**************************************************************************
    auto [uPsf, part_psf_integ]
        = Fermi::PSF::psf_lookup_table_and_partial_integrals(
            irf.psf_irf,
            exp.exp_costhetas,
            obs.logEs,
            /* Used To Compute Corrected PSF */
            exp.front_aeff,
            exp.back_aeff,
            exp.src_exposure_cosbins,
            exp.src_weighted_exposure_cosbins,
            irf.front_LTF,
            /* Exposures */
            exp.exposure);

    return {
        .uPsf                 = uPsf,
        .partial_psf_integral = part_psf_integ,
    };
}
