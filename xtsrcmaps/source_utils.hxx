#include "xtsrcmaps/source.hxx"

#include <utility>
#include <vector>

namespace Fermi {

auto spherical_coords_from_point_sources(std::vector<Source> const&)
    -> std::vector<std::pair<double, double>>;

}
