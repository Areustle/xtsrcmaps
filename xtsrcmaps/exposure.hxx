#pragma once

#include "xtsrcmaps/fitsfuncs.hxx"
#include "xtsrcmaps/irf.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include <vector>

namespace Fermi
{

auto
exp_map(fits::ExposureCubeData const&) -> ExposureMap;

auto
exp_costhetas(fits::ExposureCubeData const&) -> std::vector<double>;

auto
src_exp_cosbins(std::vector<std::pair<double, double>> const&, ExposureMap const&)
    -> mdarray2;

auto
aeff_value(std::vector<double> const&, std::vector<double> const&, IrfData3 const&)
    -> mdarray2;

auto
exposure(mdarray2 const& src_exposure_cosbins,          /*[Nsrc, Nc]*/
         mdarray2 const& src_weighted_exposure_cosbins, /*[Nsrc, Nc]*/
         mdarray2 const& front_aeff,                    /*[Nc, Ne]*/
         mdarray2 const& back_aeff,                     /*[Nc, Ne]*/
         std::pair<std::vector<double>, std::vector<double>> const& front_ltfs /*[Ne]*/
         // std::pair<std::vector<double>, std::vector<double>> const& back_ltfs   /*[Ne]*/
         ) -> mdarray2;


} // namespace Fermi
