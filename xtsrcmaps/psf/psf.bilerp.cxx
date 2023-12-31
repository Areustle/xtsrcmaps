#include "xtsrcmaps/psf/psf.hxx"

#include "xtsrcmaps/math/bilerp.hxx"


auto
Fermi::PSF::bilerp(std::vector<double> const& costhetas,  // [Nc]
                   std::vector<double> const& logEs,      // [Ne]
                   Tensor1d const&            par_cosths, // [Mc]
                   Tensor1d const&            par_logEs,  // [Me]
                   Tensor3d const&            kings       /*[Me, Mc, Nd]*/
                   ) -> Tensor3d {
    long const Nd = kings.dimension(2);
    long const Nc = costhetas.size();
    long const Ne = logEs.size();
    assert(par_logEs.size() == kings.dimension(0));
    assert(par_cosths.size() == kings.dimension(1));

    Tensor3d Bilerps(Nd, Ne, Nc);
    Bilerps.setZero();

    // Sample the Look Up Table's axes parameters with the supplied sample points
    auto const clerps = Fermi::lerp_pars(par_cosths, costhetas);
    auto const elerps = Fermi::lerp_pars(par_logEs, logEs);

    // biLerp the [E,C] slice of the Kings lookup table for each psf separation (D)
    for (long c = 0; c < Bilerps.dimension(2); ++c) {
        auto ct = clerps[c];
        for (long e = 0; e < Bilerps.dimension(1); ++e) {
            auto et = elerps[e];
            for (long d = 0; d < Bilerps.dimension(0); ++d) {

                Bilerps(d, e, c) = Fermi::bilerp(
                    et,
                    ct,
                    kings
                        .slice(Idx3 { 0, 0, d },
                               Idx3 { kings.dimension(0), kings.dimension(1), 1 })
                        .reshape(Idx2 { kings.dimension(0), kings.dimension(1) }));
            }
        }
    }

    return Bilerps;
}
