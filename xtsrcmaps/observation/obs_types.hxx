#pragma once

#include "xtsrcmaps/sky_geom/sky_geom.hxx"

#include <vector>


namespace Fermi {
namespace Obs {

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

//
struct XtObs {
    long                Nh;
    long                Nw;
    std::vector<double> energies;
    std::vector<double> logEs;
    ExposureCubeData    exp_cube;
    ExposureCubeData    weighted_exp_cube;
    SkyGeom<double>     skygeom;
};

} // namespace Obs
} // namespace Fermi
