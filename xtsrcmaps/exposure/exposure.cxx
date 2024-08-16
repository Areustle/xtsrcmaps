#include "xtsrcmaps/exposure/exposure.hxx"

#include "xtsrcmaps/healpix/healpix.hxx"
#include "xtsrcmaps/misc/misc.hxx"

#include "fmt/color.h"

#include <algorithm>
#include <cassert>
#include <span>
#include <vector>

using std::vector;
using Tensor2d = Fermi::Tensor<double, 2>;

auto
Fermi::exp_map(Obs::ExposureCubeData const& data) -> ExposureMap {
    size_t const& nside = data.nside;
    size_t const  npix  = 12 * nside * nside;
    size_t const  nbins = data.nbrbins;

    assert(data.cosbins.size() == npix * nbins);

    /* Eigen::Tensor<double, 2, Eigen::RowMajor> rm_cosbins(npix, nbins); */
    Tensor<double, 2> rm_cosbins(npix, nbins);
    std::copy(data.cosbins.begin(), data.cosbins.end(), rm_cosbins.data());

    return { nside, nbins, rm_cosbins }; // swap_layout() };
}

auto
Fermi::exp_costhetas(Obs::ExposureCubeData const& data)
    -> vector<double> /*[Nbins]*/ {
    auto v = std::vector<double>(data.nbrbins);
    std::iota(v.begin(), v.end(), 0.0);
    std::transform(v.begin(), v.end(), v.begin(), [&](auto& x) {
        double f = (x + 0.5) / data.nbrbins;
        f        = data.thetabin ? f * f : f;
        return 1. - f * (1. - data.cosmin);
    });
    return v;
}

auto
Fermi::src_exp_cosbins(Tensor<double, 2> const& src_sph, // [Nsrc, 2]
                       ExposureMap const&       expmap   //
                       ) -> Tensor<double, 2> /*[Nsrc, Nc]*/ {

    size_t const Nsrc = src_sph.extent(0);
    /*[Nsrc, nbins]*/
    Tensor<double, 2> A(Nsrc, expmap.nbins);

    for (size_t s = 0UZ; s < Nsrc; ++s) {
        // get theta, phi (radians) in appropriate coordinate system
        double theta = halfpi - to_radians(src_sph[s, 1]);
        double phi   = pi_180 * src_sph[s, 0];
        // compute resulting healpix pixel
        uint64_t pix = Fermi::Healpix::ang2pix(theta, phi, expmap.nside);
        std::copy(expmap.params.begin_at(pix, 0),
                  expmap.params.end_at(pix, expmap.nbins),
                  A.begin_at(s, 0));
    }

    return A;
}

// Double check order of rank. Now row major.

// Compute the exposure by operating on all pre-generated tensors as necessary.
// [Ne, Nsrc]
auto
Fermi::exposure(Tensor2d const& src_exposure_cosbins,          /*[Nsrc, Nc]*/
                Tensor2d const& src_weighted_exposure_cosbins, /*[Nsrc, Nc]*/
                Tensor2d const& front_aeff,                    /*[Nc, Ne]*/
                Tensor2d const& back_aeff,                     /*[Nc, Ne]*/
                Tensor2d const& front_LTF                      /*[2, Ne]*/
                ) -> Tensor<double, 2> /* [Nsrc, Ne] */ {

    // Nsrc
    assert(src_exposure_cosbins.extent(0)
           == src_weighted_exposure_cosbins.extent(0));
    /* size_t const Nsrc = src_exposure_cosbins.extent(0); */

    // Nc
    assert(src_exposure_cosbins.extent(1)
           == src_weighted_exposure_cosbins.extent(1));
    assert(src_exposure_cosbins.extent(1) == front_aeff.extent(0));
    assert(src_exposure_cosbins.extent(1) == back_aeff.extent(0));
    /* size_t const Nc = front_aeff.extent(0); */

    // Ne
    assert(front_aeff.extent(1) == front_LTF.extent(1));
    size_t const Ne = front_LTF.extent(1);

    // xxxxxxxx[Ne, Nsrc]
    // [Nsrc, Ne]
    /* ===========================================================
     * Tensor Contractions as DGEMM Matrix Multiplies
     */
    // ExpC[s, e] = Sum_c (ECB[s, c] * Aeff[c, e])
    Tensor2d exp_aeff_f
        = exp_contract(src_exposure_cosbins, front_aeff); //(Nsrc, Ne);

    Tensor2d wexp_aeff_f
        = exp_contract(src_weighted_exposure_cosbins, front_aeff); //(Nsrc, Ne);

    Tensor2d exp_aeff_b
        = exp_contract(src_exposure_cosbins, back_aeff); //(Nsrc, Ne);

    Tensor2d wexp_aeff_b
        = exp_contract(src_weighted_exposure_cosbins, back_aeff); //(Nsrc, Ne);

    auto const LTFe = std::span { &front_LTF[0, 0], Ne };
    auto const LTFw = std::span { &front_LTF[1, 0], Ne };
    /* TensorMap<Tensor2d const> LTFe(front_LTF.first.data(), Ne, 1); */
    /* TensorMap<Tensor2d const> LTFw(front_LTF.second.data(), Ne, 1); */

    // [Ne, Nsrc]
    // Response_front = (LTF1 * ExpC) + (LTF2 * WexpC)
    /* Tensor2d const lef = exp_aeff_f * LTFe.broadcast(Idx2 { 1, Nsrc }); */
    for (size_t j = 0; j < exp_aeff_f.extent(0); /*Nsrc*/ ++j) {
        std::transform(exp_aeff_f.begin_at(j, 0),
                       exp_aeff_f.end_at(j, Ne),
                       LTFe.begin(),
                       exp_aeff_f.begin_at(j, 0),
                       std::multiplies {});
    }
    /* Tensor2d const lwf = wexp_aeff_f * LTFw.broadcast(Idx2 { 1, Nsrc }); */
    for (size_t j = 0; j < wexp_aeff_f.extent(0); /*Nsrc*/ ++j) {
        std::transform(wexp_aeff_f.begin_at(j, 0),
                       wexp_aeff_f.end_at(j, Ne),
                       LTFw.begin(),
                       wexp_aeff_f.begin_at(j, 0),
                       std::multiplies {});
    }
    /* Tensor2d const leb = exp_aeff_b * LTFe.broadcast(Idx2 { 1, Nsrc }); */
    for (size_t j = 0; j < exp_aeff_b.extent(0); /*Nsrc*/ ++j) {
        std::transform(exp_aeff_b.begin_at(j, 0),
                       exp_aeff_b.end_at(j, Ne),
                       LTFe.begin(),
                       exp_aeff_b.begin_at(j, 0),
                       std::multiplies {});
    }
    /* Tensor2d const lwb = wexp_aeff_b * LTFw.broadcast(Idx2 { 1, Nsrc }); */
    for (size_t j = 0; j < wexp_aeff_b.extent(0); /*Nsrc*/ ++j) {
        std::transform(wexp_aeff_b.begin_at(j, 0),
                       wexp_aeff_b.end_at(j, Ne),
                       LTFw.begin(),
                       wexp_aeff_b.begin_at(j, 0),
                       std::multiplies {});
    }
    /* Tensor2d const response_f = lef + lwf; */
    /* Tensor2d const response_b = leb + lwb; */
    /* Tensor2d       exposure   = response_f + response_b; */
    /* Tensor2d exposure (Nsrc, Ne); */
    std::transform(exp_aeff_f.begin(),
                   exp_aeff_f.end(),
                   exp_aeff_b.begin(),
                   exp_aeff_f.begin(),
                   std::plus {});
    std::transform(exp_aeff_f.begin(),
                   exp_aeff_f.end(),
                   wexp_aeff_f.begin(),
                   exp_aeff_f.begin(),
                   std::plus {});
    std::transform(exp_aeff_f.begin(),
                   exp_aeff_f.end(),
                   wexp_aeff_b.begin(),
                   exp_aeff_f.begin(),
                   std::plus {});

    // [Nsrc, Ne]
    return exp_aeff_f;
};

auto
Fermi::compute_exposure_data(XtCfg const& cfg,
                             XtObs const& obs,
                             XtIrf const& irf) -> XtExp {

    fmt::print(fg(fmt::color::light_pink), "Computing Exposure.\n");

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
