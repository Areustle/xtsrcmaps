#include "xtsrcmaps/exposure/exposure.hxx"

#include "xtsrcmaps/healpix/healpix.hxx"
#include "xtsrcmaps/math/tensor_types.hxx"
#include "xtsrcmaps/misc/misc.hxx"

#include "fmt/color.h"

#include <algorithm>
#include <utility>
#include <vector>

using std::pair;
using std::vector;

auto
Fermi::exp_map(Obs::ExposureCubeData const& data) -> ExposureMap {
    size_t const& nside = data.nside;
    size_t const  npix  = 12 * nside * nside;
    size_t const  nbins = data.nbrbins;

    assert(data.cosbins.size() == npix * nbins);

    Eigen::Tensor<double, 2, Eigen::RowMajor> rm_cosbins(npix, nbins);
    std::copy(data.cosbins.begin(), data.cosbins.end(), &rm_cosbins(0, 0));

    return { nside, nbins, rm_cosbins.swap_layout() };
}

auto
Fermi::exp_costhetas(Obs::ExposureCubeData const& data) -> vector<double> {
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
Fermi::src_exp_cosbins(vector<pair<double, double>> const& src_sph,
                       ExposureMap const&                  expmap) -> Tensor2d {

    using pd_t   = std::pair<double, double>;
    // get theta, phi (radians) in appropriate coordinate system
    auto convert = [](pd_t const p) -> pd_t {
        return { halfpi - to_radians(p.second), pi_180 * p.first };
    };

    auto theta_phi_dirs = vector<pd_t>(src_sph.size(), { 0., 0. });
    std::transform(
        src_sph.cbegin(), src_sph.cend(), theta_phi_dirs.begin(), convert);

    auto const nbins = expmap.nbins;
    auto const pixs  = Fermi::Healpix::ang2pix(theta_phi_dirs, expmap.nside);
    Tensor2d   A(nbins, theta_phi_dirs.size());

    for (size_t i = 0; i < pixs.size(); ++i) {
        auto const& pix = pixs[i];
        std::copy(&expmap.params(0, pix), &expmap.params(nbins, pix), &A(0, i));
    }

    return A;
}


// Compute the exposure by operating on all pre-generated tensors as necessary.
// [Ne, Nsrc]
auto
Fermi::exposure(
    Tensor2d const& src_exposure_cosbins,          /*[Nsrc, Nc] -> [Nc, Nsrc]*/
    Tensor2d const& src_weighted_exposure_cosbins, /*[Nsrc, Nc] -> [Nc, Nsrc]*/
    Tensor2d const& front_aeff,                    /*[Nc, Ne] -> [Ne, Nc]*/
    Tensor2d const& back_aeff,                     /*[Nc, Ne] -> [Ne, Nc]*/
    pair<vector<double>, vector<double>> const& front_LTF /*[Ne]*/
    ) -> Tensor2d {

    // Nsrc
    assert(src_exposure_cosbins.dimension(1)
           == src_weighted_exposure_cosbins.dimension(1));
    long const Nsrc = src_exposure_cosbins.dimension(1);
    // Nc
    assert(src_exposure_cosbins.dimension(0)
           == src_weighted_exposure_cosbins.dimension(0));
    assert(src_exposure_cosbins.dimension(0) == front_aeff.dimension(1));
    assert(src_exposure_cosbins.dimension(0) == back_aeff.dimension(1));
    // Ne
    assert(front_aeff.dimension(0) == long(front_LTF.first.size()));
    assert(front_aeff.dimension(0) == long(front_LTF.second.size()));
    long const Ne = front_LTF.first.size();

    // [Ne]
    TensorMap<Tensor2d const> LTFe(front_LTF.first.data(), Ne, 1);
    TensorMap<Tensor2d const> LTFw(front_LTF.second.data(), Ne, 1);

    // [Ne, Nsrc]
    // ExpC[s, e] = Sum_c (ECB[c, s] * Aeff[e, c])
    Tensor2d const exp_aeff_f
        = front_aeff.contract(src_exposure_cosbins, IdxPair1 { { { 1, 0 } } });
    Tensor2d const wexp_aeff_f = front_aeff.contract(
        src_weighted_exposure_cosbins, IdxPair1 { { { 1, 0 } } });

    Tensor2d const exp_aeff_b
        = back_aeff.contract(src_exposure_cosbins, IdxPair1 { { { 1, 0 } } });
    Tensor2d const wexp_aeff_b = back_aeff.contract(
        src_weighted_exposure_cosbins, IdxPair1 { { { 1, 0 } } });

    // [Ne, Nsrc]
    // Response_front = (LTF1 * ExpC) + (LTF2 * WexpC)
    Tensor2d const lef        = exp_aeff_f * LTFe.broadcast(Idx2 { 1, Nsrc });
    Tensor2d const lwf        = wexp_aeff_f * LTFw.broadcast(Idx2 { 1, Nsrc });
    Tensor2d const leb        = exp_aeff_b * LTFe.broadcast(Idx2 { 1, Nsrc });
    Tensor2d const lwb        = wexp_aeff_b * LTFw.broadcast(Idx2 { 1, Nsrc });
    Tensor2d const response_f = lef + lwf;
    Tensor2d const response_b = leb + lwb;
    Tensor2d       exposure   = response_f + response_b;

    // [Ne, Nsrc]
    return exposure;
};

auto
Fermi::compute_exposure_data(XtCfg const& cfg,
                             XtObs const& obs,
                             XtIrf const& irf) -> XtExp {

    fmt::print(fg(fmt::color::magenta), "Computing Exposure.\n");

    //**************************************************************************
    // Exposure Cube Obsdata transformations
    //**************************************************************************
    auto const exp_costhetas = Fermi::exp_costhetas(obs.exp_cube);
    auto const exp_map       = Fermi::exp_map(obs.exp_cube);
    auto const wexp_map      = Fermi::exp_map(obs.weighted_exp_cube);
    auto const src_exposure_cosbins
        = Fermi::src_exp_cosbins(obs.src_sph, exp_map);
    auto const src_weighted_exposure_cosbins
        = Fermi::src_exp_cosbins(obs.src_sph, wexp_map);

    //**************************************************************************
    // Effective Area Computations.
    //**************************************************************************
    auto const front_aeff = Fermi::aeff_value(
        exp_costhetas, obs.logEs, irf.aeff_irf.front.effective_area);
    auto const back_aeff = Fermi::aeff_value(
        exp_costhetas, obs.logEs, irf.aeff_irf.back.effective_area);


    //**************************************************************************
    // Exposure
    //**************************************************************************
    auto const exposures = Fermi::exposure(src_exposure_cosbins,
                                           src_weighted_exposure_cosbins,
                                           front_aeff,
                                           back_aeff,
                                           irf.front_LTF);

    return {
        .exp_costhetas                 = exp_costhetas,
        .exp_map                       = exp_map,
        .wexp_map                      = wexp_map,
        .front_aeff                    = front_aeff,
        .back_aeff                     = back_aeff,
        .src_exposure_cosbins          = src_exposure_cosbins,
        .src_weighted_exposure_cosbins = src_weighted_exposure_cosbins,
        .exposure                      = exposures,
    };
}
