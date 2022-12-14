#include "xtsrcmaps/source_utils.hxx"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

auto
Fermi::directions_from_point_sources(std::vector<Fermi::Source> const& srcs)
    -> std::vector<std::pair<double, double>>
{
    auto dirs = std::vector<std::pair<double, double>>();
    std::transform(
        srcs.cbegin(),
        srcs.cend(),
        std::back_inserter(dirs),
        [](auto const& s) -> std::pair<double, double> {
            auto is_ra  = [](auto const& p) -> bool { return p.name == "RA"; };
            auto is_dec = [](auto const& p) -> bool { return p.name == "DEC"; };
            auto params = std::get<SkyDirFunctionSpatialModel>(
                              std::get<PointSource>(s).spatial_model)
                              .params;
            auto ra_it  = std::find_if(params.cbegin(), params.cend(), is_ra);
            auto dec_it = std::find_if(params.cbegin(), params.cend(), is_dec);
            return { ra_it->value, dec_it->value };
        });
    // std::sort(dirs.begin(), dirs.end());
    return dirs;
}
