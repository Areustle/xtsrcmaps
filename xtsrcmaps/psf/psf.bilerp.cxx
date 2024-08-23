#include "xtsrcmaps/psf/psf.hxx"

#include "xtsrcmaps/math/bilerp.hxx"


auto
Fermi::PSF::bilerp(std::vector<double> const& costhetas,  // [Nc]
                   std::vector<double> const& logEs,      // [Ne]
                   Tensor<double, 1> const&   par_cosths, // [Mc]
                   Tensor<double, 1> const&   par_logEs,  // [Me]
                   Tensor<double, 3> const&   kings       /*[Nd, Mc, Me]*/
                   ) -> Tensor<double, 3> /* [C D E] */ {

    size_t const Nc = costhetas.size();
    size_t const Nd = kings.extent(0);
    size_t const Ne = logEs.size();

    /* assert(par_cosths.extent(0) == kings.extent(0)); */
    assert(par_cosths.extent(0) == kings.extent(1));
    assert(par_logEs.extent(0) == kings.extent(2));

    Tensor<double, 3> Bilerps(Nc, Nd, Ne); // [C D E]

    // Sample the Look Up Table's axes parameters with the supplied sample
    // points
    auto const clerps = Fermi::lerp_pars(par_cosths, costhetas);
    auto const elerps = Fermi::lerp_pars(par_logEs, logEs);

    // biLerp the [E,C] slice of the Kings lookup table for each psf separation
    // (D)
    for (size_t c = 0; c < Nc; ++c) {
        // auto const& [c_wgt, c_cmplm, c_index] = clerps[c];
        double const c_wgt   = std::get<0>(clerps[c]);
        double const c_cmplm = std::get<1>(clerps[c]);
        size_t const c_index = std::get<2>(clerps[c]);
        for (size_t e = 0; e < Ne; ++e) {
            // auto const& [e_wgt, e_cmplm, e_index] = elerps[e];
            double const e_wgt   = std::get<0>(elerps[e]);
            double const e_cmplm = std::get<1>(elerps[e]);
            size_t const e_index = std::get<2>(elerps[e]);
            for (size_t d = 0; d < Nd; ++d) {
                Bilerps[c, d, e]
                    = c_cmplm * e_cmplm * kings[d, c_index - 1, e_index - 1]
                      + c_cmplm * e_wgt * kings[d, c_index - 1, e_index]
                      + c_wgt * e_cmplm * kings[d, c_index, e_index - 1]
                      + c_wgt * e_wgt * kings[d, c_index, e_index];
            }
        }
    }

    return Bilerps;
}
/* #include "xtsrcmaps/psf/psf.hxx" */
/**/

/* #include "xtsrcmaps/utils/bilerp.hxx" */
/**/
/**/
/* auto */
/* Fermi::PSF::bilerp(std::vector<double> const& costhetas,  // [Nc] */
/*                    std::vector<double> const& logEs,      // [Ne] */
/*                    Tensor1d const&            par_cosths, // [Mc] */
/*                    Tensor1d const&            par_logEs,  // [Me] */
/*                    Tensor3d const&            kings       //[Me, Mc, Nd] */
/*                    ) -> Tensor3d { */
/*     long const Nd = kings.dimension(2); */
/*     long const Nc = costhetas.size(); */
/*     long const Ne = logEs.size(); */
/*     assert(par_logEs.size() == kings.dimension(0)); */
/*     assert(par_cosths.size() == kings.dimension(1)); */
/**/
/*     Tensor3d Bilerps(Nd, Ne, Nc); */
/*     Bilerps.setZero(); */
/**/
/*     // Sample the Look Up Table's axes parameters with the supplied sample
 * points */
/*     auto const clerps = Fermi::lerp_pars(par_cosths, costhetas); */
/*     auto const elerps = Fermi::lerp_pars(par_logEs, logEs); */
/**/
/*     // biLerp the [E,C] slice of the Kings lookup table for each psf
 * separation (D) */
/*     for (long c = 0; c < Bilerps.dimension(2); ++c) { */
/*         auto ct = clerps[c]; */
/*         for (long e = 0; e < Bilerps.dimension(1); ++e) { */
/*             auto et = elerps[e]; */
/*             for (long d = 0; d < Bilerps.dimension(0); ++d) { */
/**/
/*                 Bilerps(d, e, c) = Fermi::bilerp( */
/*                     et, */
/*                     ct, */
/*                     kings */
/*                         .slice(Idx3 { 0, 0, d }, */
/*                                Idx3 { kings.dimension(0), kings.dimension(1),
 * 1 }) */
/*                         .reshape(Idx2 { kings.dimension(0),
 * kings.dimension(1) })); */
/*             } */
/*         } */
/*     } */
/**/
/*     return Bilerps; */
/* } */
