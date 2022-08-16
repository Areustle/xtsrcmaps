#pragma once

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


struct LiveTimeCubeData
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
read_ltcube(std::string const&) -> std::optional<LiveTimeCubeData>;

struct PsfParamData
{
    std::vector<float> energy_lo;
    std::vector<float> energy_hi;
    std::vector<float> costhe_lo;
    std::vector<float> costhe_hi;
    std::vector<float> ncore;
    std::vector<float> ntail;
    std::vector<float> score;
    std::vector<float> stail;
    std::vector<float> gcore;
    std::vector<float> gtail;
    float              scale0;
    float              scale1;
    float              scale_index;
};

auto
read_psf(std::string const&) -> std::optional<PsfParamData>;

} // namespace fits
} // namespace Fermi
