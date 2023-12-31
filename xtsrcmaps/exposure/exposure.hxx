#pragma once

#include "xtsrcmaps/fits/fits.hxx"
#include "xtsrcmaps/irf/irf.hxx"
#include "xtsrcmaps/math/tensor_types.hxx"

#include <vector>

namespace Fermi {

auto exp_map(fits::ExposureCubeData const&) -> ExposureMap;

auto exp_costhetas(fits::ExposureCubeData const&) -> std::vector<double>;

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


} // namespace Fermi
