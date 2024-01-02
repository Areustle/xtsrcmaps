#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/math/bilerp.hxx"
#include "xtsrcmaps/math/tensor_types.hxx"

using std::vector;

inline void
co_aeff_value_base(Tensor2d&       R,
                   auto const&     C,
                   auto const&     E,
                   auto const&     IC,
                   auto const&     IE,
                   Tensor2d const& IP,
                   double const    minCosTheta) noexcept {
    auto elerps = Fermi::lerp_pars(IE, E);
    auto clerps = Fermi::lerp_pars(IC, C, minCosTheta);

    for (long e = 0; e < R.dimension(0); ++e) {
        for (long c = 0; c < R.dimension(1); ++c) {
            R(e, c) = 1e4 * Fermi::bilerp(elerps[e], clerps[c], IP);
        }
    }
}

auto
Fermi::aeff_value(vector<double> const& costhet,
                  vector<double> const& logEs,
                  IrfData3 const&       AeffData) -> Tensor2d {
    // auto        aeff = vector<double>(costhet.size() * logEs.size(), 0.0);
    // auto        R    = mdspan(aeff.data(), costhet.size(), logEs.size());
    Tensor2d R(logEs.size(), costhet.size());
    R.setZero();
    auto const& C  = costhet;
    auto const& E  = logEs;
    auto const& IC = AeffData.cosths;
    auto const& IE = AeffData.logEs;

    assert(AeffData.params.dimension(0) == 1);
    TensorMap<Tensor2d const> IP(AeffData.params.data(),
                                 AeffData.params.dimension(1),
                                 AeffData.params.dimension(2));

    co_aeff_value_base(R, C, E, IC, IE, IP, AeffData.minCosTheta);

    // [E,C]
    return R;
}
