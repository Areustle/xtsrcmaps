#pragma once

#include "xtsrcmaps/tensor_types.hxx"

namespace Fermi
{

auto
point_src_model_map_wcs(std::vector<double> const&                    Px,
                        std::vector<double> const&                    Py,
                        std::vector<std::pair<double, double>> const& dirs,
                        mdarray3 const& uPsf
                        ) -> mdarray3;

}
