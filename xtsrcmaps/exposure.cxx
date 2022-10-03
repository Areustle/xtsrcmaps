#include "xtsrcmaps/exposure.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/healpix.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

#include "experimental/mdspan"
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

using std::pair;
using std::vector;
using std::experimental::mdspan;

auto
Fermi::exp_map(fits::ExposureCubeData const& data) -> ExposureMap
{
    size_t const& nside = data.nside;
    size_t const  npix  = 12 * nside * nside;
    size_t const  nbins = data.nbrbins;
    auto          v     = vector<double>(data.cosbins.size());
    std::copy(data.cosbins.begin(), data.cosbins.end(), v.begin());

    return { nside, nbins, mdarray2(v, npix, nbins) };
}

auto
Fermi::exp_costhetas(fits::ExposureCubeData const& data) -> vector<double>
{
    auto v = std::vector<double>(data.nbrbins);
    std::iota(v.begin(), v.end(), 0.0);
    std::transform(v.begin(), v.end(), v.begin(), [&](auto x) {
        double const f = (x + 0.5) / data.nbrbins;
        return data.thetabin ? f * f : f;
    });
    std::transform(v.begin(), v.end(), v.begin(), [&](auto f) {
        return 1. - f * (1. - data.cosmin);
    });
    return v;
}

auto
Fermi::src_exp_cosbins(vector<pair<double, double>> const& dirs,
                       ExposureMap const&                  expmap) -> mdarray2
{

    using pd_t   = std::pair<double, double>;
    // get theta, phi (radians) in appropriate coordinate system
    auto convert = [](pd_t const p) -> pd_t {
        return { halfpi - radians(p.second), pi_180 * p.first };
    };

    auto theta_phi_dirs = vector<pd_t>(dirs.size(), { 0., 0. });
    std::transform(dirs.cbegin(), dirs.cend(), theta_phi_dirs.begin(), convert);

    auto const nbins = expmap.nbins;
    auto const pixs  = Fermi::Healpix::ang2pix(theta_phi_dirs, expmap.nside);
    auto       data  = vector<double>(theta_phi_dirs.size() * nbins, 0.0);
    auto       A     = mdarray2(data, theta_phi_dirs.size(), nbins);

    for (size_t i = 0; i < pixs.size(); ++i)
    {
        auto const& pix = pixs[i];
        std::copy(&expmap.params(pix, 0), &expmap.params(pix, nbins), &A(i, 0));
    }

    return A;
}

// B                   [Nc, Ne]
// C  (costheta)       [Nc]
// E  (Energies)       [Ne]
// IC (IRF costheta)   [Me]
// IE (IRF energies)   [Mc]
// IP (IRF Params)     [Mc, Me]
inline void
co_aeff_value_base(auto         R,
                   auto const&  C,
                   auto const&  E,
                   auto const&  IC,
                   auto const&  IE,
                   auto const&  IP,
                   double const minCosTheta) noexcept
{
    auto clerps = Fermi::lerp_pars(IC, C, minCosTheta);
    auto elerps = Fermi::lerp_pars(IE, E);

    for (size_t c = 0; c < R.extent(0); ++c)
        for (size_t e = 0; e < R.extent(1); ++e)
            R(c, e) = 1e4 * Fermi::bilerp(clerps[c], elerps[e], IP);
}

auto
Fermi::aeff_value(vector<double> const& costhet,
                  vector<double> const& logEs,
                  IrfData3 const&       AeffData) -> mdarray2
{
    auto        aeff = vector<double>(costhet.size() * logEs.size(), 0.0);
    auto        R    = mdspan(aeff.data(), costhet.size(), logEs.size());
    auto const& C    = costhet;
    auto const& E    = logEs;
    auto const& IC   = AeffData.cosths;
    auto const& IE   = AeffData.logEs;

    assert(AeffData.params.extent(2) == 1);
    auto IP = mdspan(AeffData.params.data(),
                     AeffData.params.extent(0),
                     AeffData.params.extent(1)); //, pars.params.extent(2));

    co_aeff_value_base(R, C, E, IC, IE, IP, AeffData.minCosTheta);

    // [C,E]
    return mdarray2(aeff, R.extent(0), R.extent(1));
}

// Compute the exposure by operating on all pre-generated tensors as necessary.
auto
Fermi::exposure(mdarray2 const& src_exposure_cosbins,                 /*[Nsrc, Nc]*/
                mdarray2 const& src_weighted_exposure_cosbins,        /*[Nsrc, Nc]*/
                mdarray2 const& front_aeff,                           /*[Nc, Ne]*/
                mdarray2 const& back_aeff,                            /*[Nc, Ne]*/
                pair<vector<double>, vector<double>> const& front_LTF /*[Ne]*/
                // pair<vector<double>, vector<double>> const& back_LTF   /*[Ne]*/
                ) -> mdarray2
{

    // Nsrc
    assert(src_exposure_cosbins.extent(0) == src_weighted_exposure_cosbins.extent(0));
    assert(src_exposure_cosbins.extent(1) == src_weighted_exposure_cosbins.extent(1));
    // Nc
    assert(src_exposure_cosbins.extent(1) == front_aeff.extent(0));
    assert(src_exposure_cosbins.extent(1) == back_aeff.extent(0));
    // Ne
    assert(back_aeff.extent(1) == front_LTF.first.size());
    assert(back_aeff.extent(1) == front_LTF.second.size());
    // assert(back_aeff.extent(1) == back_LTF.first.size());
    // assert(back_aeff.extent(1) == back_LTF.second.size());

    // [Ne]
    auto const& LTFe      = front_LTF.first;
    auto const& LTFw      = front_LTF.second;
    // auto const& LTFeb     = back_LTF.first;
    // auto const& LTFwb     = back_LTF.second;

    // [Nsrc, Ne]
    // ExpC[n, e] = Sum_c (ECB[n, c] * Aeff[c, e])
    auto const exp_aeff_f = Fermi::contract210(src_exposure_cosbins, front_aeff);
    auto const exp_aeff_b = Fermi::contract210(src_exposure_cosbins, back_aeff);
    auto const wexp_aeff_f
        = Fermi::contract210(src_weighted_exposure_cosbins, front_aeff);
    auto const wexp_aeff_b
        = Fermi::contract210(src_weighted_exposure_cosbins, back_aeff);

    // Response_front = (LTF1 * ExpC) + (LTF2 * WexpC)
    auto const lef        = Fermi::mul210(exp_aeff_f, LTFe);
    auto const lwf        = Fermi::mul210(wexp_aeff_f, LTFw);
    auto const leb        = Fermi::mul210(exp_aeff_b, LTFe);
    auto const lwb        = Fermi::mul210(wexp_aeff_b, LTFw);
    auto const response_f = Fermi::sum2_2(lef, lwf);
    auto const response_b = Fermi::sum2_2(leb, lwb);

    auto const exposure   = Fermi::sum2_2(response_f, response_b);

    return exposure;
};

//
//
// inline auto
// phi_modulation(double const& par0, double const& par1 /*double phi = 0*/) ->
// double
// {
//     // if (phi < 0) { phi += 360.; }
//     double norm(1. / (1. + par0 / (1. + par1)));
//     // double phi_pv(std::fmod(phi * M_PI / 180., M_PI) - M_PI / 2.); // == -pi/2
//     // double phi_pv(-M_PI / 2.); == -pi/2
//     // double xx(2. * std::fabs(2. / M_PI * std::fabs(phi_pv) - 0.5));
//     // // 2 * |((2/pi)*(pi/2))-0.5| = 2/2 = 1
//     // return norm * (1. + par0 * std::pow(xx, par1));
//     return norm * (1. + par0 /* * 1.0^par1 */);
// }
//
// inline void
// co_phi_mod_base(auto        R,
//                 auto const& C,
//                 auto const& E,
//                 auto const& IC,
//                 auto const& IE,
//                 auto const& IP) noexcept
// {
//
//     //
//     auto cgl = vector<size_t>(C.extent(0));
//     for (size_t c = 0; c < C.extent(0); ++c)
//     {
//         cgl[c] = Fermi::greatest_lower(IC, C(c));
//     }
//
//     auto egl = vector<size_t>(E.extent(0));
//     for (size_t e = 0; e < E.extent(0); ++e)
//     {
//         egl[e] = Fermi::greatest_lower(IE, E(e));
//     }
//
//     for (size_t c = 0; c < R.extent(0); ++c)
//     {
//         for (size_t e = 0; e < R.extent(1); ++e)
//         {
//             R(c, e) = phi_modulation(IP(cgl[c], egl[e], 0), IP(cgl[c], egl[e],
//             1));
//         }
//     }
// }
//
// auto
// Fermi::phi_mod(vector<double> const& cosBins,
//                vector<double> const& logEs,
//                IrfData3 const&       pars,
//                bool                  phi_dep = false) -> mdarray2
// {
//
//     auto phi = vector<double>(cosBins.size() * logEs.size(), 1.0);
//     auto R   = mdspan(phi.data(), cosBins.size(), logEs.size());
//
//     if (phi_dep)
//     {
//         auto C  = mdspan(cosBins.data(), cosBins.size());
//         auto E  = mdspan(logEs.data(), logEs.size());
//         auto IC = span(pars.cosths);
//         auto IE = span(pars.logEs);
//         auto IP = mdspan(pars.params.data(),
//                          pars.params.extent(0),
//                          pars.params.extent(1),
//                          pars.params.extent(2));
//
//         co_phi_mod_base(R, C, E, IC, IE, IP);
//     }
//
//     return mdarray2(phi, R.extent(0), R.extent(1));
// }
