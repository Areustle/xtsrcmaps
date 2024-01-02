#pragma once

#include "xtsrcmaps/source/source.hxx"

#include <array>
#include <vector>


namespace Fermi {
namespace Obs {

using sphcrd_t   = std::pair<double, double>;
using sphcrd_v_t = std::vector<sphcrd_t>;
using src_v_t    = std::vector<Fermi::Source>;



struct CCubePixels {
    std::array<long, 3>   naxes;
    std::array<double, 3> crpix;
    std::array<double, 3> crval;
    std::array<double, 3> cdelt;
    double                axis_rot;
    std::string           proj_name;
    bool                  is_galactic;
};


struct ExposureCubeData {
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


} // namespace Obs
//
struct XtObs {
    std::vector<double>      energies;
    std::vector<double>      logEs;
    Obs::CCubePixels         ccube;
    Obs::src_v_t             srcs;
    Obs::sphcrd_v_t          src_sph;
    std::vector<std::string> src_names;
    Obs::ExposureCubeData    exp_cube;
    Obs::ExposureCubeData    weighted_exp_cube;
    long                     Nh;
    long                     Nw;
};

} // namespace Fermi
