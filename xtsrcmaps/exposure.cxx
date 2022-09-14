#include "xtsrcmaps/exposure.hxx"

#include "xtsrcmaps/bilerp.hxx"

#include "experimental/mdspan"
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <span>
#include <utility>
#include <vector>

using std::pair;
using std::span;
using std::vector;
// using std::experimental::full_extent;
using std::experimental::mdspan;
// using std::experimental::submdspan;


// B                   [Nc, Ne]
// C  (costheta)       [Nc]
// E  (Energies)       [Ne]
// IC (IRF costheta)   [Me]
// IE (IRF energies)   [Mc]
// IP (IRF Params)     [Mc, Me]
inline void
co_aeff_value_base(auto        R,
                   auto const& C,
                   auto const& E,
                   auto const& IC,
                   auto const& IE,
                   auto const& IP) noexcept
{

    auto clerps = vector<pair<double, size_t>>(C.extent(0));
    for (size_t c = 0; c < C.extent(0); ++c) { clerps[c] = Fermi::lerp(IC, C(c)); }

    auto elerps = vector<pair<double, size_t>>(E.extent(0));
    for (size_t e = 0; e < E.extent(0); ++e) { elerps[e] = Fermi::lerp(IE, E(e)); }

    for (size_t c = 0; c < R.extent(0); ++c)
    {
        for (size_t e = 0; e < R.extent(1); ++e)
        {
            auto const& uu   = std::get<0>(clerps[c]);
            auto const& cidx = std::get<1>(clerps[c]);
            auto const& tt   = std::get<0>(elerps[e]);
            auto const& eidx = std::get<1>(elerps[e]);

            R(c, e)          = 1e4 * Fermi::bilerp(tt, uu, cidx, eidx, IP);
        }
    }
}

auto
Fermi::aeff_value(vector<double> const& costhet,
                  vector<double> const& logEs,
                  IrfData3 const&        pars) -> mdarray2
{

    auto aeff = vector<double>(costhet.size() * logEs.size(), 0.0);
    auto R    = mdspan(aeff.data(), costhet.size(), logEs.size());
    auto C    = mdspan(costhet.data(), costhet.size());
    auto E    = mdspan(logEs.data(), logEs.size());
    auto IC   = span(pars.cosths.cbegin(), pars.cosths.cend());
    auto IE   = span(pars.logEs.cbegin(), pars.logEs.cend());
    assert(pars.params.extent(2) == 1);
    auto IP = mdspan(pars.params.data(),
                     pars.params.extent(0),
                     pars.params.extent(1)); //, pars.params.extent(2));

    co_aeff_value_base(R, C, E, IC, IE, IP);

    return mdarray2(std::move(aeff), R.extent(0), R.extent(1));
}

inline auto
phi_modulation(double const& par0, double const& par1 /*double phi = 0*/) -> double
{
    // if (phi < 0) { phi += 360.; }
    double norm(1. / (1. + par0 / (1. + par1)));
    // double phi_pv(std::fmod(phi * M_PI / 180., M_PI) - M_PI / 2.); // == -pi/2
    // double phi_pv(-M_PI / 2.); == -pi/2
    // double xx(2. * std::fabs(2. / M_PI * std::fabs(phi_pv) - 0.5));
    // // 2 * |((2/pi)*(pi/2))-0.5| = 2/2 = 1
    // return norm * (1. + par0 * std::pow(xx, par1));
    return norm * (1. + par0 /* * 1.0^par1 */);
}

inline void
co_phi_mod_base(auto        R,
                auto const& C,
                auto const& E,
                auto const& IC,
                auto const& IE,
                auto const& IP) noexcept
{

    //
    auto cgl = vector<size_t>(C.extent(0));
    for (size_t c = 0; c < C.extent(0); ++c)
    {
        cgl[c] = Fermi::greatest_lower(IC, C(c));
    }

    auto egl = vector<size_t>(E.extent(0));
    for (size_t e = 0; e < E.extent(0); ++e)
    {
        egl[e] = Fermi::greatest_lower(IE, E(e));
    }

    for (size_t c = 0; c < R.extent(0); ++c)
    {
        for (size_t e = 0; e < R.extent(1); ++e)
        {
            R(c, e) = phi_modulation(IP(cgl[c], egl[e], 0), IP(cgl[c], egl[e], 1));
        }
    }
}

auto
Fermi::phi_mod(vector<double> const& cosBins,
               vector<double> const& logEs,
               IrfData3 const&        pars,
               bool                  phi_dep = false) -> mdarray2
{

    auto phi = vector<double>(cosBins.size() * logEs.size(), 1.0);
    auto R   = mdspan(phi.data(), cosBins.size(), logEs.size());

    if (phi_dep)
    {
        auto C  = mdspan(cosBins.data(), cosBins.size());
        auto E  = mdspan(logEs.data(), logEs.size());
        auto IC = span(pars.cosths.cbegin(), pars.cosths.cend());
        auto IE = span(pars.logEs.cbegin(), pars.logEs.cend());
        auto IP = mdspan(pars.params.data(),
                         pars.params.extent(0),
                         pars.params.extent(1),
                         pars.params.extent(2));

        co_phi_mod_base(R, C, E, IC, IE, IP);
    }

    return mdarray2(phi, R.extent(0), R.extent(1));
}

auto inline expcontract(mdspan2 const Ap, mdspan1 const costhe) -> mdarray1
{
    assert(Ap.extent(0) == costhe.extent(0));
    auto rv = std::vector<double>(Ap.extent(1), 0.0);
    auto R  = mdarray1(rv, Ap.extent(1));
    for (size_t c = 0; c < Ap.extent(0); ++c)
    {
        for (size_t e = 0; e < Ap.extent(1); ++e) { R(e) += Ap(c, e) * costhe[c]; }
    }
    return R;
}

auto
Fermi::exposure(mdarray2 const& aeff, mdarray2 const& phi, std::vector<double> costhe)
    -> mdarray1
{
    // Scale aeff * phi
    assert(aeff.rank() == 2);
    assert(aeff.extents() == phi.extents());
    assert(aeff.stride(0) == phi.stride(0));
    assert(aeff.stride(1) == phi.stride(1));

    auto Ap_v = std::vector<double>(aeff.size());
    std::transform(std::cbegin(aeff.container()),
                   std::cend(aeff.container()),
                   std::cbegin(phi.container()),
                   std::begin(Ap_v),
                   std::multiplies<> {});

    // Contract by cosine binner <-- Should be separate function
    auto Ap   = mdspan(Ap_v.data(), aeff.extent(0), aeff.extent(1));
    auto Ct   = mdspan(costhe.data(), aeff.extent(0));
    auto Expo = expcontract(Ap, Ct);
    return Expo;
};
