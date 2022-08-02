#pragma once

#include <vector>

#include "xtsrcmaps/fitsfuncs.hxx"

namespace Fermi
{

struct PsfData
{
    std::vector<double> logEs;
    std::vector<double> Energies;
    std::vector<double> cosths;
    std::vector<double> params;
};


auto
prepare_psf_data(fits::PsfParamData const& pars) -> PsfData;

auto
psf_fixed_grid(PsfData const& pars) -> std::vector<double>;

// auto
// psf_fixed_grid(Fermi::fits::PsfParamData const& pars,
//                std::vector<double> const&       energies,
//                std::vector<double> const&       costheta) -> std::vector<double>;

} // namespace Fermi
