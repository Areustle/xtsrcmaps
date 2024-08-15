#include "xtsrcmaps/source/source.hxx"

#include <algorithm>
#include <string>
#include <vector>

auto
Fermi::spherical_coords_from_point_sources(
    std::vector<Fermi::Source> const& srcs) -> Tensor<double, 2> {

    size_t const Nsrc = srcs.size();

    Tensor<double, 2> dirs(Nsrc, 2);

    for (size_t i = 0; i < Nsrc; ++i) {
        auto const& s      = srcs[i];
        auto        params = std::get<SkyDirFunctionSpatialModel>(
                          std::get<PointSource>(s).spatial_model)
                          .params;

        auto ra_it = std::find_if(
            params.cbegin(), params.cend(), [](auto const& p) -> bool {
                return p.name == "RA";
            });

        auto dec_it = std::find_if(
            params.cbegin(), params.cend(), [](auto const& p) -> bool {
                return p.name == "DEC";
            });

        if (ra_it == params.cend()) {
            throw std::runtime_error("RA keyword not found.");
        }
        if (dec_it == params.cend()) {
            throw std::runtime_error("DEC keyword not found.");
        }

        dirs[i, 0] = ra_it->value;
        dirs[i, 1] = dec_it->value;
    }

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
