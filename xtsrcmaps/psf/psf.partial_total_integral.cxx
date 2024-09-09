#include "xtsrcmaps/psf/psf.hxx"
#include "xtsrcmaps/misc/misc.hxx"

using Tensor2d = Fermi::Tensor<double, 2>;
using Tensor3d = Fermi::Tensor<double, 3>;
using Tensor3f = Fermi::Tensor<float, 3>;

auto
Fermi::Psf::partial_total_integral(Tensor3d const& uPsf /* [Ns, Nd, Ne] */
                                   ) -> std::pair<Tensor3d, Tensor2d> {
    size_t const Ns   = uPsf.extent(0);
    size_t const Nd   = uPsf.extent(1);
    size_t const Ne   = uPsf.extent(2);
    SepArr const seps = separations();
    assert(Nd == seps.size());

    Tensor3d parint(Ns, Nd, Ne);
    Tensor2d totint(Ns, Ne);
    /* std::fill(totint.begin(), totint.end(), 1.0f); */

    // Use Midpoint Rule to compute approximate sum of psf from each separation
    // entry over the lookup table.
    for (size_t s = 0; s < Ns; ++s) {
        for (size_t e = 0; e < Ne; ++e) {
            // Cumulative psf integral across separation from point source (d)
            double cumulative = 0.0;
            parint[s, 0, e]   = 0.0;
            for (size_t d = 1; d < Nd; ++d) {
                double const theta1 = deg2rad * seps[d - 1];
                double const theta2 = deg2rad * seps[d];
                double const y1 = twopi * uPsf[s, d - 1, e] * std::sin(theta1);
                double const y2 = twopi * uPsf[s, d, e] * std::sin(theta2);
                double const m  = (y2 - y1) / (theta2 - theta1);
                double const b  = y1 - theta1 * m;
                double const v  = 0.5 * m * (theta2 * theta2 - theta1 * theta1)
                                 + b * (theta2 - theta1);
                cumulative += v;
                parint[s, d, e] = cumulative;
            }
            // Ensure normalizations of differential and integral arrays.
            // We only need to do the normilization if the sum in non-zero
            totint[s, e] = cumulative;
            if (totint[s, e]) {
                for (size_t d = 0; d < Nd; ++d) {
                    parint[s, d, e] /= cumulative;
                }
            }
        }
    }

    return { parint, totint };
}
