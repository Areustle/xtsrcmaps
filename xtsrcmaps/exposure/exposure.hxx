#pragma once

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/irf/irf_types.hxx"
#include "xtsrcmaps/math/tensor_types.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"

#include <vector>

namespace Fermi {

struct ExposureMap {
    std::size_t nside;
    std::size_t nbins;
    Tensor2d    params;
};

struct XtExp {
    std::vector<double> exp_costhetas;
    ExposureMap         exp_map;
    ExposureMap         wexp_map;
    Tensor2d            front_aeff;
    Tensor2d            back_aeff;
    Tensor2d            src_exposure_cosbins;
    Tensor2d            src_weighted_exposure_cosbins;
    Tensor2d            exposure;
};

auto exp_map(Obs::ExposureCubeData const&) -> ExposureMap;

auto exp_costhetas(Obs::ExposureCubeData const&) -> std::vector<double>;

auto src_exp_cosbins(std::vector<std::pair<double, double>> const&,
                     ExposureMap const&) -> Tensor2d;

auto aeff_value(std::vector<double> const&,
                std::vector<double> const&,
                IrfData3 const&) -> Tensor2d;

auto
exposure(Tensor2d const& src_exposure_cosbins,
         Tensor2d const& src_weighted_exposure_cosbins,
         Tensor2d const& front_aeff,
         Tensor2d const& back_aeff,
         std::pair<std::vector<double>, std::vector<double>> const& front_ltfs)
    -> Tensor2d;

auto compute_exposure_data(XtCfg const& cfg, XtObs const& obs, XtIrf const& irf)
    -> XtExp;

} // namespace Fermi
