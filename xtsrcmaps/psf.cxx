#include "xtsrcmaps/psf.hxx"

#include "xtsrcmaps/bilerp.hxx"
#include "xtsrcmaps/misc.hxx"
#include "xtsrcmaps/tensor_ops.hxx"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <vector>

// using std::pair;
using std::vector;

inline auto
king_single(double const sep, Tensor1d const& pars) noexcept -> double
{
    assert(pars.dimension(0) == 6);
    double const& ncore = pars(0);
    double const& ntail = pars(1);
    double const& score = pars(2);
    double const& stail = pars(3);
    double const& gcore = pars(4); // assured not to be 1.0
    double const& gtail = pars(5); // assured not to be 1.0

    double rc           = sep / score;
    double uc           = rc * rc / 2.;

    double rt           = sep / stail;
    double ut           = rt * rt / 2.;

    // scaled king function
    return (ncore * (1. - 1. / gcore) * std::pow(1. + uc / gcore, -gcore)
            + ntail * ncore * (1. - 1. / gtail) * std::pow(1. + ut / gtail, -gtail));
    // If perfomance is limited by this function call it may be improved by computing
    // x ^ -g as exp(-g * ln(x)) with SIMD log and exp.
}

//[Nd, Nc, Ne] -> [Ne, Nc, Nd]
auto
Fermi::PSF::king(irf::psf::Data const& psfdata) -> Tensor3d
{
    Fermi::IrfData3 const& psf_grid = psfdata.rpsf;
    assert(psf_grid.params.dimension(0) == 6);                      // 6
    assert(psf_grid.params.dimension(1) == psf_grid.logEs.size());  // Ne
    assert(psf_grid.params.dimension(2) == psf_grid.cosths.size()); // Nc
    //
    long const Ne = psf_grid.logEs.size();
    long const Nc = psf_grid.cosths.size();
    long const Nd = sep_arr_len;

    SepArr seps   = separations();
    assert(seps.size() == Nd);
    TensorMap<Tensor1d> delta(seps.data(), Nd);
    delta = D2R * delta;

    Tensor3d Kings(Ne, Nc, Nd);

    // // co_king_base(Kings, delta, psf_grid.params);
    // Kings               [Nd, Mc, Me] -> [Ne, Nc, Nd]
    // delta (Separations) [Nd]
    // psf_grid.params (IRF Params)  [Mc, Me, 6] -> [6, Ne, Nc]
    auto const& P = psf_grid.params;
    assert(P.dimension(0) == 6);
    assert(P.dimension(1) == Ne);
    assert(P.dimension(2) == Nc);

    for (long d = 0; d < Nd; ++d)
    {
        for (long c = 0; c < Nc; ++c)
        {
            for (long e = 0; e < Ne; ++e)
            {
                Kings(e, c, d) = king_single(
                    delta(d),
                    P.slice(Idx3 { 0, e, c }, Idx3 { 6, 1, 1 }).reshape(Idx1 { 6 }));
            }
        }
    }

    // [E C D]
    return Kings;
}


auto
Fermi::PSF::bilerp(std::vector<double> const& costhetas,  // [Nc]
                   std::vector<double> const& logEs,      // [Ne]
                   Tensor1d const&            par_cosths, // [Mc]
                   Tensor1d const&            par_logEs,  // [Me]
                   Tensor3d const&            kings       /*[Me, Mc, Nd]*/
                   ) -> Tensor3d
{
    long const Nd = kings.dimension(2);
    long const Nc = costhetas.size();
    long const Ne = logEs.size();
    assert(par_logEs.size() == kings.dimension(0));
    assert(par_cosths.size() == kings.dimension(1));

    Tensor3d Bilerps(Nd, Ne, Nc);
    Bilerps.setZero();

    // co_psf_bilerp(Bilerps, costhetas, logEs, kings, par_cosths, par_logEs);
    //
    // Sample the Look Up Table's axes parameters with the supplied sample points
    auto const clerps = Fermi::lerp_pars(par_cosths, costhetas);
    auto const elerps = Fermi::lerp_pars(par_logEs, logEs);

    // biLerp the [E,C] slice of the Kings lookup table for each psf separation (D)
    for (long c = 0; c < Bilerps.dimension(2); ++c)
    {
        auto ct = clerps[c];
        for (long e = 0; e < Bilerps.dimension(1); ++e)
        {
            auto et = elerps[e];
            for (long d = 0; d < Bilerps.dimension(0); ++d)
            {

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
//

auto
Fermi::PSF::corrected_exposure_psf(
    Tensor3d const& obs_psf,                       /*[C, E, D] -> [D, E, C]*/
    Tensor2d const& obs_aeff,                      /*[C, E] -> [E, C]*/
    Tensor2d const& src_exposure_cosbins,          /*[S, C] -> [C, S]*/
    Tensor2d const& src_weighted_exposure_cosbins, /*[S, C] -> [C, S]*/
    std::pair<std::vector<double>, std::vector<double>> const& front_LTF /*[E]*/
    ) -> Tensor3d
{
    long const Nd = obs_psf.dimension(0);
    long const Ne = obs_psf.dimension(1);
    long const Nc = obs_psf.dimension(2);
    long const Ns = src_exposure_cosbins.dimension(1);

    assert(Ne == obs_aeff.dimension(0));
    assert(Nc == obs_aeff.dimension(1));

    TensorMap<Tensor3d const> fLTR1(front_LTF.first.data(), 1, Ne, 1);
    TensorMap<Tensor3d const> fLTR2(front_LTF.second.data(), 1, Ne, 1);

    // auto psf_aeff     = Fermi::mul322(obs_psf, obs_aeff); // [D, E, C] . [E, C]
    // [D, E, C]
    Tensor3d psf_aeff
        = obs_psf * obs_aeff.reshape(Idx3 { 1, Ne, Nc }).broadcast(Idx3 { Nd, 1, 1 });

    // [D, E, S] = SUM_c ([D, E, C] * [C, S])
    // auto exposure_psf = Fermi::contract3210(psf_aeff, src_exposure_cosbins);
    Tensor3d exposure_psf
        = psf_aeff.contract(src_exposure_cosbins, IdxPair1 { { { 2, 0 } } });
    // auto wexp_psf = Fermi::contract3210(psf_aeff, src_weighted_exposure_cosbins);
    Tensor3d wexp_psf
        = psf_aeff.contract(src_weighted_exposure_cosbins, IdxPair1 { { { 2, 0 } } });

    // [D, E, S]
    // auto corrected_exp_psf          = Fermi::mul310(exposure_psf, front_LTF.first);
    Tensor3d corrected_exp_psf = exposure_psf * fLTR1.broadcast(Idx3 { Nd, 1, Ns });
    // auto corrected_weighted_exp_psf = Fermi::mul310(wexp_psf, front_LTF.second);
    Tensor3d corrected_weighted_exp_psf
        = wexp_psf * fLTR2.broadcast(Idx3 { Nd, 1, Ns });

    // return Fermi::sum3_3(corrected_exp_psf, corrected_weighted_exp_psf);
    return corrected_exp_psf + corrected_weighted_exp_psf;
}

auto
Fermi::PSF::mean_psf(                    //
    Tensor3d const& front_corrected_psf, /*[Nd, Ne, Nsrc]*/
    Tensor3d const& back_corrected_psf,  /*[Nd, Ne, Nsrc]*/
    Tensor2d const& exposure /*[Ne, Nsrc]*/) -> Tensor3d
{
    long const Nd         = front_corrected_psf.dimension(0);
    long const Ne         = front_corrected_psf.dimension(1);
    long const Ns         = front_corrected_psf.dimension(2);

    // auto psf          = Fermi::sum3_3(front_corrected_psf, back_corrected_psf);
    Tensor3d psf          = front_corrected_psf + back_corrected_psf;
    // auto inv_exposure = Fermi::safe_reciprocal(exposure);
    // Tensor3d inv_exposure = (exposure == 0.)
    //                             .select(exposure.constant(0.), exposure.inverse())
    //                             .reshape(Idx3 { 1, Ne, Ns });
    Tensor3d inv_exposure = exposure.inverse().reshape(Idx3 { 1, Ne, Ns });
    // inv_exposure = inv_exposure.isnan().select(inv_exposure.constant(0.),
    // inv_exposure); return Fermi::mul32_1(psf, inv_exposure); [Nd, Ne, Ns]
    return psf * inv_exposure.broadcast(Idx3 { Nd, 1, 1 });
}

auto
Fermi::PSF::partial_total_integral(Tensor3d const& mean_psf /* [Nd, Ne, Ns] */
                                   ) -> std::pair<Tensor3d, Tensor2d>
{
    auto   Nd   = mean_psf.dimension(0);
    auto   Ne   = mean_psf.dimension(1);
    auto   Ns   = mean_psf.dimension(2);
    SepArr seps = separations();
    assert(seps.size() == Nd);
    TensorMap<Tensor1d> delta(seps.data(), Nd);
    assert(Nd == delta.size());
    delta = D2R * delta;
    //
    // auto v1  = vector<double>(Ns * Ne * Nd, 0.0);
    // auto v2  = vector<double>(Ns * Ne, 1.0);
    // auto sp1 = mdspan(v1.data(), Ns, Ne, Nd);
    // auto sp2 = mdspan(v2.data(), Ns, Ne);

    Tensor3d parint(Nd, Ne, Ns);
    parint.setZero();
    Tensor2d totint(Ne, Ns);
    totint.setConstant(1.);

    for (long s = 0; s < Ns; ++s)
    {
        for (long e = 0; e < Ne; ++e)
        {
            // sp1(s, e, 0) = 0.;
            parint(0, e, s) = 0.;
            for (long d = 1; d < Nd; ++d)
            {
                double theta1(delta(d - 1));
                double theta2(delta(d));
                double y1(2. * M_PI * mean_psf(d - 1, e, s) * std::sin(theta1));
                double y2(2. * M_PI * mean_psf(d, e, s) * std::sin(theta2));
                double slope((y2 - y1) / (theta2 - theta1));
                double intercept(y1 - theta1 * slope);
                double value(slope * (theta2 * theta2 - theta1 * theta1) / 2.
                             + intercept * (theta2 - theta1));
                parint(d, e, s) = parint(d - 1, e, s) + value;
            }
            // Ensure normalizations of differential and integral arrays.
            // We only need to do the normilization if the sum in non-zero
            if (parint(Nd - 1, e, s) != 0.0)
            {
                for (long d = 1; d < Nd - 1; ++d)
                {
                    // m_psfValues.at(index) /= partialIntegral.back();
                    parint(d, e, s) /= parint(Nd - 1, e, s);
                }
                totint(e, s)         = parint(Nd - 1, e, s);
                parint(Nd - 1, e, s) = 1.0;
            }
        }
    }
    return { parint, totint };
}

auto
Fermi::PSF::normalize(Tensor3d&       mean_psf,       /* [Nd, Ne, Ns] */
                      Tensor2d const& total_integrals /*     [Ne, Ns] */
                      ) -> void
{
    for (long s = 0; s < mean_psf.dimension(2); ++s)
    {
        for (long e = 0; e < mean_psf.dimension(1); ++e)
        {
            for (long d = 0; d < mean_psf.dimension(0); ++d)
            {
                mean_psf(d, e, s) /= total_integrals(e, s);
            }
        }
    }
}

auto
Fermi::PSF::peak_psf(Tensor3d const& mean_psf /* [Nd, Ne, Ns] */) -> Tensor2d
{
    // auto Nd = mean_psf.dimension(0);
    auto Ne = mean_psf.dimension(1);
    auto Ns = mean_psf.dimension(2);
    // auto const& Ns = mean_psf.extent(0);
    // auto const& Ne = mean_psf.extent(1);
    // auto const& Nd = mean_psf.extent(2);
    // auto v  = vector<double>(Ns * Ne);
    // for (long i = 0; i < Ns * Ne; ++i) { v[i] = mean_psf.container()[i * Nd]; }
    //
    // return mdarray2(v, Ns, Ne);
    return mean_psf.slice(Idx3 { 0, 0, 0 }, Idx3 { 1, Ne, Ns })
        .reshape(Idx2 { Ne, Ns });
}
//
// auto
// Fermi::PSF::integral(std::vector<double> deltas,
//                      Tensor3d const&     partial_integrals,
//                      Tensor3d const&     mean_psf) -> Tensor3d;
