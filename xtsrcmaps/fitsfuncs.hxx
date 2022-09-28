#pragma once

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Fermi
{
namespace fits
{

auto
ccube_energies(std::string const&) noexcept -> std::optional<std::vector<double>>;


struct ExposureCubeData
{
    std::vector<float> cosbins;
    std::vector<float> ra;
    std::vector<float> dec;
    unsigned int       nside;    // = 0;
    unsigned int       nbrbins;  // = 40;
    double             cosmin;   // = 0.0;
    std::string        ordering; // = "NESTED";
    std::string        coordsys; // = "EQU";
    bool               thetabin; // = false;
};

auto
read_expcube(std::string const&, std::string const&) -> std::optional<ExposureCubeData>;

struct TablePars
{
    std::vector<size_t>             extents;
    std::vector<size_t>             offsets;
    std::vector<std::vector<float>> rowdata;
};

auto
read_irf_pars(std::string const&, std::string const&) -> std::optional<TablePars>;


} // namespace fits
} // namespace Fermi
