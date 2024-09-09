#pragma once

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/source/source.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <vector>

namespace Fermi::Exposure {

struct ExposureMap {
    std::size_t       nside;
    std::size_t       nbins;
    Tensor<double, 2> params;
};

struct XtExp {
    std::vector<double> exp_costhetas;
    /* ExposureMap         exp_map; */
    /* ExposureMap         wexp_map; */
    Tensor<double, 2>   front_aeff;
    Tensor<double, 2>   back_aeff;
    Tensor<double, 2>   src_exposure_cosbins;
    Tensor<double, 2>   src_weighted_exposure_cosbins;
    Tensor<double, 2>   exposure;
};

auto map(Obs::ExposureCubeData const&) -> ExposureMap;

auto costhetas(Obs::ExposureCubeData const&) -> std::vector<double>;

auto contract(Fermi::Tensor<double, 2> const& A,
              Fermi::Tensor<double, 2> const& B) -> Fermi::Tensor<double, 2>;

auto
src_cosbins(Tensor<double, 2> const&, ExposureMap const&) -> Tensor<double, 2>;

auto exposure(Tensor<double, 2> const& src_exposure_cosbins,
              Tensor<double, 2> const& src_weighted_exposure_cosbins,
              Tensor<double, 2> const& front_aeff,
              Tensor<double, 2> const& back_aeff,
              Tensor<double, 2> const& front_ltfs) -> Tensor<double, 2>;

template <Source::SourceConcept T>
auto
compute_exposure(Config::XtCfg const&         cfg,
                 Obs::XtObs const&            obs,
                 Source::SourceData<T> const& src,
                 Irf::XtIrf const&            irf) -> XtExp {


    //**************************************************************************
    // Exposure Cube Obsdata transformations
    //**************************************************************************
    auto const exp_costhetas        = costhetas(obs.exp_cube);
    auto const exp_map              = map(obs.exp_cube);
    auto const wexp_map             = map(obs.weighted_exp_cube);
    auto const src_exposure_cosbins = src_cosbins(src.sph_locs, exp_map);
    auto const src_weighted_exposure_cosbins
        = src_cosbins(src.sph_locs, wexp_map);

    //**************************************************************************
    // Effective Area Computations.
    //**************************************************************************
    auto const front_aeff = Irf::aeff_value(
        exp_costhetas, obs.logEs, irf.aeff_irf.front.effective_area);
    auto const back_aeff = Irf::aeff_value(
        exp_costhetas, obs.logEs, irf.aeff_irf.back.effective_area);


    //**************************************************************************
    // Exposure
    //**************************************************************************
    auto const exposures = exposure(src_exposure_cosbins,
                                    src_weighted_exposure_cosbins,
                                    front_aeff,
                                    back_aeff,
                                    irf.front_LTF);

    return {
        .exp_costhetas                 = exp_costhetas,
        /* .exp_map                       = exp_map, */
        /* .wexp_map                      = wexp_map, */
        .front_aeff                    = front_aeff,
        .back_aeff                     = back_aeff,
        .src_exposure_cosbins          = src_exposure_cosbins,
        .src_weighted_exposure_cosbins = src_weighted_exposure_cosbins,
        .exposure                      = exposures,
    };
}

} // namespace Fermi::Exp
