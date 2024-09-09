#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/misc/misc.hxx"

#include <span>


// [Mc, Me, Nd]
auto
Fermi::Psf::king(Irf::psf::Data const& psfdata) -> Tensor<double, 3> {
    Fermi::Irf::IrfData3 const& psf_grid = psfdata.rpsf;
    assert(psf_grid.params.extent(0) == psf_grid.cosths.extent(0)); // Nc
    assert(psf_grid.params.extent(1) == psf_grid.logEs.extent(0));  // Ne
    assert(psf_grid.params.extent(2) == 6);                         // 6
    //
    size_t const Mc = psf_grid.cosths.extent(0);
    size_t const Me = psf_grid.logEs.extent(0);
    size_t const Nd = sep_arr_len;

    SepArr seps     = separations();
    assert(seps.size() == Nd);

    Tensor<double, 3> Kings(Nd, Mc, Me);

    // psf_grid.params (IRF Params)  [Mc, Me, 6]
    assert(psf_grid.params.extent(0) == Mc);
    assert(psf_grid.params.extent(1) == Me);
    assert(psf_grid.params.extent(2) == 6);

    for (long d = 0; d < Nd; ++d) {
        for (long c = 0; c < Mc; ++c) {
            for (long e = 0; e < Me; ++e) {
                Kings[d, c, e] = Fermi::Irf::evaluate_king(
                    seps[d] * deg2rad,
                    1.0,
                    std::span { &psf_grid.params[c, e, 0], 6 });
            }
        }
    }

    // [D C E]
    return Kings;
}

/* #include "xtsrcmaps/psf/psf.hxx" */
/**/
/* #include "xtsrcmaps/misc.hxx" */
/**/
/**/
/* inline auto */
/* king_single(double const sep, Tensor1d const& pars) noexcept -> double { */
/*     assert(pars.dimension(0) == 6); */
/*     double const& ncore = pars(0); */
/*     double const& ntail = pars(1); */
/*     double const& score = pars(2); */
/*     double const& stail = pars(3); */
/*     double const& gcore = pars(4); // assured not to be 1.0 */
/*     double const& gtail = pars(5); // assured not to be 1.0 */
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
/*             * std::pow(1. + ut / gtail, -gtail)); */
/*     // If perfomance is limited by this function */
/*     // call it may be improved by computing */
/*     // x ^ -g as exp(-g * ln(x)) with SIMD log and exp. */
/* } */
/**/
/* //[Nd, Nc, Ne] -> [Ne, Nc, Nd] */
/* auto */
/* Fermi::PSF::king(irf::psf::Data const& psfdata) -> Tensor3d { */
/*     Fermi::IrfData3 const& psf_grid = psfdata.rpsf; */
/*     assert(psf_grid.params.dimension(0) == 6);                      // 6 */
/*     assert(psf_grid.params.dimension(1) == psf_grid.logEs.size());  // Ne */
/*     assert(psf_grid.params.dimension(2) == psf_grid.cosths.size()); // Nc */
/*     // */
/*     long const Ne = psf_grid.logEs.size(); */
/*     long const Nc = psf_grid.cosths.size(); */
/*     long const Nd = sep_arr_len; */
/**/
/*     SepArr seps   = separations(); */
/*     assert(seps.size() == Nd); */
/*     TensorMap<Tensor1d> delta(seps.data(), Nd); */
/*     delta = deg2rad * delta; */
/**/
/*     Tensor3d Kings(Ne, Nc, Nd); */
/**/
/*     // // co_king_base(Kings, delta, psf_grid.params); */
/*     // Kings               [Nd, Mc, Me] -> [Ne, Nc, Nd] */
/*     // delta (Separations) [Nd] */
/*     // psf_grid.params (IRF Params)  [Mc, Me, 6] -> [6, Ne, Nc] */
/*     auto const& P = psf_grid.params; */
/*     assert(P.dimension(0) == 6); */
/*     assert(P.dimension(1) == Ne); */
/*     assert(P.dimension(2) == Nc); */
/**/
/*     for (long d = 0; d < Nd; ++d) { */
/*         for (long c = 0; c < Nc; ++c) { */
/*             for (long e = 0; e < Ne; ++e) { */
/*                 Kings(e, c, d) = king_single( */
/*                     delta(d), */
/*                     P.slice(Idx3 { 0, e, c }, Idx3 { 6, 1, 1 }).reshape( */
/*                     Idx1 { 6 }) */
/*                     ); */
/*             } */
/*         } */
/*     } */
/**/
/*     // [E C D] */
/*     return Kings; */
/* } */
