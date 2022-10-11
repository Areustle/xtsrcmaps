#include "xtsrcmaps/psf.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

#include "experimental/mdspan"
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <vector>

// using std::pair;
using std::vector;
using std::experimental::full_extent;
using std::experimental::mdspan;
using std::experimental::submdspan;

inline auto
king_single(double const sep, auto const& pars) noexcept -> double
{
    assert(pars.extent(0) == 6);
    double const& ncore = pars[0];
    double const& ntail = pars[1];
    double const& score = pars[2];
    double const& stail = pars[3];
    double const& gcore = pars[4]; // assured not to be 1.0
    double const& gtail = pars[5]; // assured not to be 1.0

    double rc           = sep / score;
    double uc           = rc * rc / 2.;

    double rt           = sep / stail;
    double ut           = rt * rt / 2.;

    // scaled king function
    return (ncore * (1. - 1. / gcore) * std::pow(1. + uc / gcore, -gcore)
            + ntail * ncore * (1. - 1. / gtail) * std::pow(1. + ut / gtail, -gtail));
    // should be able to compute x ^ -g as exp(-g * ln(x)) with SIMD log and exp.
    // return (ncore * psf_base_function(uc, gcore)
    //         + ntail * ncore * psf_base_function(ut, gtail));
}

// A               [Nd, Mc, Me]
// D (Separations) [Nd]
// P (IRF Params)  [Mc, Me, 6]
inline void
co_king_base(auto A, auto D, auto P) noexcept
{
    assert(P.extent(2) == 6);

    for (size_t d = 0; d < A.extent(0); ++d)
    {
        for (size_t c = 0; c < A.extent(1); ++c)
        {
            for (size_t e = 0; e < A.extent(2); ++e)
            {
                A(d, c, e) = king_single(D[d], submdspan(P, c, e, full_extent));
            }
        }
    }
}

//[Nd, Mc, Me]
auto
Fermi::PSF::king(vector<double> const& deltas, irf::psf::Data const& psfdata)
    -> mdarray3
{

    Fermi::IrfData3 const& psf_grid = psfdata.rpsf;
    // Fermi::IrfScale const& scale    = psfdata.psf_scaling_params;
    assert(psf_grid.params.extent(0) == psf_grid.cosths.size());
    assert(psf_grid.params.extent(1) == psf_grid.logEs.size());

    auto seps = vector<double>(deltas.size());
    std::transform(deltas.cbegin(), deltas.cend(), seps.begin(), radians);


    auto kings = vector<double>(
        deltas.size() * psf_grid.params.extent(0) * psf_grid.params.extent(1), 0.0);

    auto       A = mdspan(kings.data(),
                    deltas.size(),
                    psf_grid.params.extent(0),
                    psf_grid.params.extent(1));
    auto const D = mdspan(seps.data(), seps.size());
    auto const P = mdspan(psf_grid.params.data(),
                          psf_grid.params.extent(0),
                          psf_grid.params.extent(1),
                          psf_grid.params.extent(2));

    co_king_base(A, D, P);

    return mdarray3(kings, A.extent(0), A.extent(1), A.extent(2));
}

//////
auto
Fermi::PSF::separations(double const xmin, double const xmax, size_t const N)
    -> std::vector<double>
{
    auto   sep   = std::vector<double>(N + 1, 0.0);
    double xstep = std::log(xmax / xmin) / (N - 1.);
    for (size_t i = 0; i < N; ++i) sep[i + 1] = xmin * std::exp(i * xstep);
    return sep;
}

auto
Fermi::PSF::inverse_separations(double const s) -> double
{
    return std::clamp(1 + std::log(s * 1e4) / sep_step, 0.0, 399.);
}

// R  (psf_bilerp result)      [Nc, Ne, Nd]
// C  (costhetas)         [Nc]
// E  (Energies)        [Ne]
// IP (IRF Params)      [Nd, Me, Mc]
// IC (IRF costheta)    [Mc]
// IE (IRF energies)    [Me]
inline void
co_psf_bilerp(auto        R,
              auto const& C,
              auto const& E,
              auto const& IP,
              auto const& IC,
              auto const& IE) noexcept
{
    auto const clerps = Fermi::lerp_pars(IC, C);
    auto const elerps = Fermi::lerp_pars(IE, E);

    // for (size_t d = 0; d < R.extent(0); ++d)
    // {
    //     auto const sIP = submdspan(IP, d, full_extent, full_extent);
    //     for (size_t c = 0; c < R.extent(1); ++c)
    //         for (size_t e = 0; e < R.extent(2); ++e)
    //             R(d, c, e) = Fermi::bilerp(clerps[c], elerps[e], sIP);
    // }

    // C, E, D
    for (size_t c = 0; c < R.extent(0); ++c)
    {
        auto const cps = clerps[c];
        for (size_t e = 0; e < R.extent(1); ++e)
        {
            auto const eps = elerps[e];
            for (size_t d = 0; d < R.extent(2); ++d)
            {
                auto const sIP = submdspan(IP, d, full_extent, full_extent);
                R(c, e, d)     = Fermi::bilerp(cps, eps, sIP);
            }
        }
    }
}


auto
Fermi::PSF::bilerp(std::vector<double> const& costhetas,
                   std::vector<double> const& logEs,
                   std::vector<double> const& par_cosths,
                   std::vector<double> const& par_logEs,
                   mdarray3 const&            kings) -> mdarray3
{

    // Nd, Ne, Nc, Mc, Me
    size_t const Nd = kings.extent(0);
    size_t const Mc = kings.extent(1);
    size_t const Me = kings.extent(2);
    size_t const Nc = costhetas.size();
    size_t const Ne = logEs.size();
    assert(par_cosths.size() == Mc);
    assert(par_logEs.size() == Me);

    auto        bilerps = vector<double>(Nc * Ne * Nd, 0.0);
    auto        R       = mdspan(bilerps.data(), Nc, Ne, Nd);
    auto const& E       = logEs;
    auto const& C       = costhetas;
    auto const  IP      = mdspan(kings.data(), Nd, Mc, Me);
    auto const& IC      = par_cosths;
    auto const& IE      = par_logEs;

    co_psf_bilerp(R, C, E, IP, IC, IE);

    return mdarray3(bilerps, Nc, Ne, Nd);
}


auto
Fermi::PSF::corrected_exposure_psf(
    mdarray3 const& obs_psf,                                             /*[C, E, D]*/
    mdarray2 const& obs_aeff,                                            /*[C, E]*/
    mdarray2 const& src_exposure_cosbins,                                /*[S, C]*/
    mdarray2 const& src_weighted_exposure_cosbins,                       /*[S, C]*/
    std::pair<std::vector<double>, std::vector<double>> const& front_LTF /*[E]*/
    ) -> mdarray3
{

    auto psf_aeff     = Fermi::mul322(obs_psf, obs_aeff); // [C, E, D]

    // [S, E, D] = SUM_c ([S, C] * [C, E, D])
    auto exposure_psf = Fermi::contract3210(psf_aeff, src_exposure_cosbins);
    auto wexp_psf     = Fermi::contract3210(psf_aeff, src_weighted_exposure_cosbins);

    // [S, E, D]
    auto corrected_exp_psf          = Fermi::mul310(exposure_psf, front_LTF.first);
    auto corrected_weighted_exp_psf = Fermi::mul310(wexp_psf, front_LTF.second);

    return Fermi::sum3_3(corrected_exp_psf, corrected_weighted_exp_psf);
}

auto
Fermi::PSF::mean_psf(                    //
    mdarray3 const& front_corrected_psf, /*[Nsrc, Ne, Nd]*/
    mdarray3 const& back_corrected_psf,  /*[Nsrc, Ne, Nd]*/
    mdarray2 const& exposure) -> mdarray3
{
    auto psf          = Fermi::sum3_3(front_corrected_psf, back_corrected_psf);
    auto inv_exposure = Fermi::safe_reciprocal(exposure);
    return Fermi::mul32_1(psf, inv_exposure);
}

auto
Fermi::PSF::partial_total_integral(std::vector<double> const& deltas,
                                   mdarray3 const&            mean_psf)
    -> std::pair<mdarray3, mdarray2>
{
    auto Ns = mean_psf.extent(0);
    auto Ne = mean_psf.extent(1);
    auto Nd = mean_psf.extent(2);
    assert(Ns == 263);
    assert(Ne == 38);
    assert(Nd == deltas.size());
    auto v1  = vector<double>(Ns * Ne * Nd, 0.0);
    auto v2  = vector<double>(Ns * Ne, 1.0);
    auto sp1 = mdspan(v1.data(), Ns, Ne, Nd);
    auto sp2 = mdspan(v2.data(), Ns, Ne);
    for (size_t s(0); s < Ns; ++s)
    {
        for (size_t e(0); e < Ne; ++e)
        {
            sp1(s, e, 0) = 0.;
            for (size_t d(1); d < Nd; ++d)
            {
                double theta1(deltas.at(d - 1) * M_PI / 180.);
                double theta2(deltas.at(d) * M_PI / 180.);
                double y1(2. * M_PI * mean_psf(s, e, d - 1) * std::sin(theta1));
                double y2(2. * M_PI * mean_psf(s, e, d) * std::sin(theta2));
                double slope((y2 - y1) / (theta2 - theta1));
                double intercept(y1 - theta1 * slope);
                double value(slope * (theta2 * theta2 - theta1 * theta1) / 2.
                             + intercept * (theta2 - theta1));
                sp1(s, e, d) = sp1(s, e, d - 1) + value;
            }
            // Ensure normalizations of differential and integral arrays.
            // We only need to do the normilization if the sum in non-zero
            if (sp1(s, e, Nd - 1) != 0.0)
            {
                for (size_t d(1); d < Nd - 1; ++d)
                {
                    // m_psfValues.at(index) /= partialIntegral.back();
                    sp1(s, e, d) /= sp1(s, e, Nd - 1);
                }
                sp2(s, e)         = sp1(s, e, Nd - 1);
                sp1(s, e, Nd - 1) = 1.0;
            }
        }
    }
    return {
        mdarray3(v1, Ns, Ne, Nd),
        mdarray2(v2, Ns, Ne),
    };
}

auto
Fermi::PSF::integral(std::vector<double> deltas,
                     mdarray3 const&     partial_integrals,
                     mdarray3 const&     mean_psf) -> mdarray3;

auto
Fermi::PSF::normalize(mdarray3& mean_psf, mdarray2 const& total_integrals) -> void
{
    for (size_t s = 0; s < mean_psf.extent(0); ++s)
    {
        for (size_t e = 0; e < mean_psf.extent(1); ++e)
        {
            for (size_t d = 0; d < mean_psf.extent(2); ++d)
            {
                mean_psf(s, e, d) /= total_integrals(s, e);
            }
        }
    }
}

auto
Fermi::PSF::peak_psf(mdarray3 const& mean_psf) -> mdarray2
{
    auto const& Ns = mean_psf.extent(0);
    auto const& Ne = mean_psf.extent(1);
    auto const& Nd = mean_psf.extent(2);
    auto        v  = vector<double>(Ns * Ne);
    for (size_t i = 0; i < Ns * Ne; ++i) { v[i] = mean_psf.container()[i * Nd]; }

    return mdarray2(v, Ns, Ne);
}

// auto
// Fermi::PSF::makePointSourceMap_wcs( // const PointSource&          pointSrc,
//                                     //  const CountsMap&            dataMap,
//     const std::vector<double>& energies,
//     // const PsfIntegConfig&       config,
//     const mdarray3& meanpsf,
//     // const BinnedExposureBase*   bexpmap,
//     // st_stream::StreamFormatter& formatter,
//     std::vector<float>& modelmap,
//     // FileUtils::SrcMapType&      mapType,
//     int kmin,
//     int kmax) -> std::vector<double>
// {
// }
