#pragma once

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/irf/irf_types.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"
#include "xtsrcmaps/tensor/tensor.hpp"

#include <vector>

namespace Fermi {

struct ExposureMap {
    std::size_t       nside;
    std::size_t       nbins;
    Tensor<double, 2> params;
};

struct XtExp {
    std::vector<double> exp_costhetas;
    ExposureMap         exp_map;
    ExposureMap         wexp_map;
    Tensor<double, 2>   front_aeff;
    Tensor<double, 2>   back_aeff;
    Tensor<double, 2>   src_exposure_cosbins;
    Tensor<double, 2>   src_weighted_exposure_cosbins;
    Tensor<double, 2>   exposure;
};

auto exp_map(Obs::ExposureCubeData const&) -> ExposureMap;

auto exp_costhetas(Obs::ExposureCubeData const&) -> std::vector<double>;

auto
exp_contract(Fermi::Tensor<double, 2> const& A,
             Fermi::Tensor<double, 2> const& B) -> Fermi::Tensor<double, 2>;

auto src_exp_cosbins(Tensor<double, 2> const&,
                     ExposureMap const&) -> Tensor<double, 2>;

auto aeff_value(std::vector<double> const&,
                std::vector<double> const&,
                IrfData3 const&) -> Tensor<double, 2>;

auto exposure(Tensor<double, 2> const& src_exposure_cosbins,
              Tensor<double, 2> const& src_weighted_exposure_cosbins,
              Tensor<double, 2> const& front_aeff,
              Tensor<double, 2> const& back_aeff,
              Tensor<double, 2> const& front_ltfs) -> Tensor<double, 2>;

auto compute_exposure_data(XtCfg const& cfg,
                           XtObs const& obs,
                           XtIrf const& irf) -> XtExp;

} // namespace Fermi
