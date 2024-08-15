#include "xtsrcmaps/psf/psf.hxx"

#include "xtsrcmaps/math/bilerp.hxx"


auto
Fermi::PSF::bilerp(std::vector<double> const& costhetas,  // [Nc]
                   std::vector<double> const& logEs,      // [Ne]
                   Tensor<double, 1> const&   par_cosths, // [Mc]
                   Tensor<double, 1> const&   par_logEs,  // [Me]
                   Tensor<double, 3> const&   kings       /*[Mc, Me, Nd]*/
                   ) -> Tensor<double, 3> /* [C, D, E] */ {
    size_t const Nc = costhetas.size();
    size_t const Nd = kings.extent(2);
    size_t const Ne = logEs.size();
    assert(par_cosths.extent(0) == kings.extent(0));
    assert(par_logEs.extent(0) == kings.extent(1));

    Tensor<double, 3> Bilerps(Nc, Nd, Ne);

    // Sample the Look Up Table's axes parameters with the supplied sample
    // points
    auto const clerps = Fermi::lerp_pars(par_cosths, costhetas);
    auto const elerps = Fermi::lerp_pars(par_logEs, logEs);

    // biLerp the [E,C] slice of the Kings lookup table for each psf separation
    // (D)
    for (size_t c = 0; c < Bilerps.extent(0); ++c) {
        auto const& [c_wgt, c_cmplm, c_index] = clerps[c];
        for (size_t e = 0; e < Bilerps.extent(2); ++e) {
            auto const& [e_wgt, e_cmplm, e_index] = elerps[e];
            for (size_t d = 0; d < Bilerps.extent(1); ++d) {
                Bilerps[c, d, e]
                    = c_cmplm * e_cmplm * kings[c_index - 1, e_index - 1, d]
                      + c_cmplm * e_wgt * kings[c_index, e_index - 1, d]
                      + c_wgt * e_cmplm * kings[c_index - 1, e_index, d]
                      + c_wgt * e_wgt * kings[c_index, e_index, d];
            }
        }
    }

    return Bilerps;
}
