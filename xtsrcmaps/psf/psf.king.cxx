#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/misc/misc.hxx"

#include <span>

/* inline auto */
/* king_single(double const                    sep, */
/*             Fermi::Tensor<double, 1> const& pars) noexcept -> double { */
/*     assert(pars.extent(0) == 6); */
/*     double const& ncore = pars[0]; */
/*     double const& ntail = pars[1]; */
/*     double const& score = pars[2]; */
/*     double const& stail = pars[3]; */
/*     double const& gcore = pars[4]; // assured not to be 1.0 */
/*     double const& gtail = pars[5]; // assured not to be 1.0 */
/**/
/*     double rc           = sep / score; */
/*     double uc           = rc * rc / 2.; */
/**/
/*     double rt           = sep / stail; */
/*     double ut           = rt * rt / 2.; */
/**/
/*     // scaled king function */
/*     return (ncore * (1. - 1. / gcore) * std::pow(1. + uc / gcore, -gcore) */
/*             + ntail * ncore * (1. - 1. / gtail) */
/*                   * std::pow(1. + ut / gtail, -gtail)); */
/*     // If perfomance is limited by this function call it may be improved by
 */
/*     // computing x ^ -g as exp(-g * ln(x)) with SIMD log and exp. */
/* } */

//[Nd, Nc, Ne] -> [Ne, Nc, Nd]
auto
Fermi::PSF::king(irf::psf::Data const& psfdata) -> Tensor<double, 3> {
    Fermi::IrfData3 const& psf_grid = psfdata.rpsf;
    assert(psf_grid.params.extent(0) == psf_grid.cosths.extent(0)); // Nc
    assert(psf_grid.params.extent(1) == psf_grid.logEs.extent(0));  // Ne
    assert(psf_grid.params.extent(2) == 6);                         // 6
    //
    size_t const Ne = psf_grid.logEs.extent(0);
    size_t const Nc = psf_grid.cosths.extent(0);
    size_t const Nd = sep_arr_len;

    SepArr seps     = separations();
    assert(seps.size() == Nd);

    /* Tensor3d Kings(Nc, Ne, Nd); */
    Tensor<double, 3> Kings(Nc, Ne, Nd);

    // // co_king_base(Kings, delta, psf_grid.params);
    // Kings               [Nd, Mc, Me] -> [Ne, Nc, Nd]
    // delta (Separations) [Nd]
    // psf_grid.params (IRF Params)  [Mc, Me, 6] -> [6, Ne, Nc]
    /* auto const& P = psf_grid.params; */
    assert(psf_grid.params.extent(0) == Nc);
    assert(psf_grid.params.extent(1) == Ne);
    assert(psf_grid.params.extent(2) == 6);

    for (size_t c = 0; c < Nc; ++c) {
        for (size_t e = 0; e < Ne; ++e) {
            for (size_t d = 0; d < Nd; ++d) {
                Kings[c, e, d] = Fermi::evaluate_king(
                    seps[d] * deg2rad,
                    1.0,
                    std::span { &psf_grid.params[c, e, 0], 6 });
                /* P.slice(Idx3 { 0, e, c }, Idx3 { 6, 1, 1 }) */
                /*     .reshape(Idx1 { 6 })); */
                /* Kings[] */
            }
        }
    }

    // [E C D]
    return Kings;
}
