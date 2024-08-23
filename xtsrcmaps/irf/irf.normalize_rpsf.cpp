#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/irf/_irf_private.hpp"

#include "xtsrcmaps/math/gauss_legendre.hxx"
#include "xtsrcmaps/misc/misc.hxx"

#include <span>

namespace irf_private {

auto
normalize_rpsf(Fermi::irf::psf::Data& psfdata) -> void {

    Fermi::IrfData3&       data  = psfdata.rpsf;
    Fermi::IrfScale const& scale = psfdata.psf_scaling_params;
    // Next normalize and scale.

    auto scaleFactor             = [sp0 = (scale.scale0 * scale.scale0),
                        sp1 = (scale.scale1 * scale.scale1),
                        si  = scale.scale_index](double const energy) {
        double const tt = std::pow(energy * 1.e-2, si);
        return std::sqrt(sp0 * tt * tt + sp1);
    };

    // An integration is required below, so let's precompute the orthogonal
    // legendre polynomials here for future use.
    auto const polypars = Fermi::legendre_poly_rw<64>(1e-15);

    for (size_t c = 0; c < data.params.extent(0); ++c) // costheta
    {
        for (size_t e = 0; e < data.params.extent(1); ++e) // energy
        {
            double const energy = std::pow(10.0, data.logEs[e]);
            double const sf     = scaleFactor(energy);
            double const norm
                = energy < 120. //
                      ? Fermi::gauss_legendre_integral(
                            0.0,
                            90.,
                            polypars,
                            [&](auto const& v) -> double {
                                double x = Fermi::evaluate_king(
                                    v * deg2rad,
                                    sf,
                                    std::span { &data.params[c, e, 0], 6 });
                                double y = sin(v * deg2rad) * twopi * deg2rad;
                                return x * y;
                            })
                      : Fermi::psf3_psf_base_integral(
                            90.0, sf, std::span { &data.params[c, e, 0], 6 });

            data.params[c, e, 0] /= norm;
            data.params[c, e, 2] *= sf;
            data.params[c, e, 3] *= sf;
            data.params[c, e, 4]
                = data.params[c, e, 4] == 1. ? 1.001 : data.params[c, e, 4];
            data.params[c, e, 5]
                = data.params[c, e, 5] == 1. ? 1.001 : data.params[c, e, 5];
        }
    }
};
} // namespace irf_private
