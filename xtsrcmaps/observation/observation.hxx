#pragma once

#include "xtsrcmaps/config/config.hxx"
#include "xtsrcmaps/observation/obs_types.hxx"

namespace Fermi::Obs {

auto collect_observation_data(Fermi::Config::XtCfg const&) -> Obs::XtObs;

}
