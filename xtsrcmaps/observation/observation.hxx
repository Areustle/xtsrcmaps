#pragma once

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"

namespace Fermi {

auto collect_observation_data(Fermi::XtCfg const&) -> XtObs;

}
