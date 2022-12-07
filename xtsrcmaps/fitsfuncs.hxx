#pragma once

#include "xtsrcmaps/tensor_types.hxx"

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

struct CCubePixels
{
    std::array<size_t, 3> naxes;
    std::array<double, 3> crpix;
    std::array<double, 3> crval;
    std::array<double, 3> cdelt;
    double                axis_rot;
    std::string           proj_name;
    bool                  is_galactic;
};

auto
ccube_pixels(std::string const&) noexcept -> std::optional<CCubePixels>;


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
    std::vector<size_t> extents;
    std::vector<size_t> offsets;
    Tensor2f            rowdata;
};

auto
read_irf_pars(std::string const&, std::string const&) -> std::optional<TablePars>;


} // namespace fits
} // namespace Fermi
