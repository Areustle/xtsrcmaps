#include "xtsrcmaps/source/source.hxx"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

auto
Fermi::spherical_coords_from_point_sources(
    std::vector<Fermi::Source> const& srcs)
    -> std::vector<std::pair<double, double>> {
    auto dirs = std::vector<std::pair<double, double>>();
    std::transform(srcs.cbegin(),
                   srcs.cend(),
                   std::back_inserter(dirs),
                   [](auto const& s) -> std::pair<double, double> {
                       auto params = std::get<SkyDirFunctionSpatialModel>(
                                         std::get<PointSource>(s).spatial_model)
                                         .params;

                       auto ra_it  = std::find_if(params.cbegin(),
                                                 params.cend(),
                                                 [](auto const& p) -> bool {
                                                     return p.name == "RA";
                                                 });

                       auto dec_it = std::find_if(params.cbegin(),
                                                  params.cend(),
                                                  [](auto const& p) -> bool {
                                                      return p.name == "DEC";
                                                  });

                       return { ra_it->value, dec_it->value };
                   });
    return dirs;
}

auto
Fermi::names_from_point_sources(std::vector<Fermi::Source> const& srcs)
    -> std::vector<std::string> {

    auto names = std::vector<std::string>();
    std::transform(srcs.cbegin(),
                   srcs.cend(),
                   std::back_inserter(names),
                   [](auto const& s) -> std::string {
                       return std::get<PointSource>(s).name;
                   });
    return names;
}
