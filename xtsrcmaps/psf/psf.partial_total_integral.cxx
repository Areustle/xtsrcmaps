#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/misc/misc.hxx"

using Tensor2f = Fermi::Tensor<float, 2>;
using Tensor3f = Fermi::Tensor<float, 3>;

auto
Fermi::PSF::partial_total_integral(Tensor3f const& uPsf /* [Ns, Nd, Ne] */
                                   ) -> std::pair<Tensor3f, Tensor2f> {
    size_t const Ns   = uPsf.extent(0);
    size_t const Nd   = uPsf.extent(1);
    size_t const Ne   = uPsf.extent(2);
    SepArr       seps = separations();
    assert(Nd == seps.size());

    Tensor3f parint(Ns, Nd, Ne);
    Tensor2f totint(Ns, Ne);
    std::fill(totint.begin(), totint.end(), 1.0);

    // Use Midpoint Rule to compute approximate sum of psf from each separation
    // entry over the lookup table.
    for (size_t s = 0; s < Ns; ++s) {
        for (size_t e = 0; e < Ne; ++e) {
            // Cumulative psf integral across separation from point source (d)
            parint[s, 0, e] = 0.0f;
            for (size_t d = 1uz; d < Nd; ++d) {
                float const theta1 = seps[d - 1] * deg2rad;
                float const theta2 = seps[d] * deg2rad;
                float const y1     = twopi * uPsf[s, d - 1, e] * theta1;
                float const y2     = twopi * uPsf[s, d, e] * theta2;
                float const m      = (y2 - y1) / (theta2 - theta1);
                float const b      = y1 - theta1 * m;
                float const v = 0.5f * m * (theta2 * theta2 - theta1 * theta1)
                                + b * (theta2 - theta1);
                parint[s, d, e] = v + parint[s, d - 1, e];
            }
            // Ensure normalizations of differential and integral arrays.
            // We only need to do the normilization if the sum in non-zero
            totint[s, e] = parint[s, Nd - 1, e];
            if (totint[s, e]) {
                for (size_t d = 0; d < Nd; ++d) {
                    parint[s, d, e] /= totint[s, e];
                }
            }
        }
    }

    return { parint, totint };
}
