#include "xtsrcmaps/irf/irf.hxx"

#include "xtsrcmaps/math/bilerp.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <cassert>
#include <mdspan>


using std::vector;
auto
Fermi::aeff_value(vector<double> const& costhet, // M_t
                  vector<double> const& logEs,   // M_e
                  IrfData3 const&       AeffData // M_t, M_e, Ngrids
                  ) -> Tensor<double, 2> {
    Tensor<double, 2> R(costhet.size(), logEs.size());
    R.clear();
    auto const& C  = costhet;
    auto const& E  = logEs;
    auto const& IC = AeffData.cosths;
    auto const& IE = AeffData.logEs;

    assert(AeffData.params.extent(0) == IC.extent(0));
    assert(AeffData.params.extent(1) == IE.extent(0));
    assert(AeffData.params.extent(2) == 1);
    /* TensorMap<Tensor2d const> IP(AeffData.params.data(), */
    std::mdspan IP { AeffData.params.data(),
                     AeffData.params.extent(0),
                     AeffData.params.extent(1) };

    /* co_aeff_value_base(R, C, E, IC, IE, IP, AeffData.minCosTheta); */
    auto elerps = Fermi::lerp_pars(IE, E);
    auto clerps = Fermi::lerp_pars(IC, C, AeffData.minCosTheta);

    for (size_t c = 0; c < R.extent(0); ++c) {
        std::tuple<double, double, size_t> const& ct = clerps[c];
        auto const& [c_wgt, c_cmpl, c_idx]           = ct;
        for (size_t e = 0; e < R.extent(1); ++e) {
            /* R[e, c] = 1e4 * Fermi::bilerp(elerps[e], clerps[c], IP); */
            std::tuple<double, double, size_t> const& et = elerps[e];
            auto const& [e_wgt, e_cmpl, e_idx]           = et;

            /////////
            double xx = c_cmpl * e_cmpl * IP[c_idx - 1, e_idx - 1];
            double xy = c_cmpl * e_wgt * IP[c_idx-1 , e_idx];
            double yx = c_wgt * e_cmpl * IP[c_idx, e_idx - 1];
            double yy = c_wgt * e_wgt * IP[c_idx, e_idx];

            R[c, e]   = 1e4 * (xx + xy + yx + yy);
        }
    }

    // [C, E]
    return R;
}
