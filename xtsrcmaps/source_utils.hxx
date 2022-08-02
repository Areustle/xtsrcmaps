#include "xtsrcmaps/source.hxx"

#include <utility>
#include <vector>

namespace Fermi
{

auto
directions_from_point_sources(std::vector<Source> const&)
    -> std::vector<std::pair<double, double>>;

}
