#pragma once

#include "xtsrcmaps/irf_types.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include <vector>

namespace Fermi
{

///////////////////////////////////////////////////////////////////////////////////////
///Given a PSF IRF grid and a set of separations, compute the King/Moffat results for
///every entry in the table and every separation.
///////////////////////////////////////////////////////////////////////////////////////
auto
king(std::vector<double> const& deltas, Psf::Data const& data)
    -> mdarray3; //[Nd, Me, Mc]

}
