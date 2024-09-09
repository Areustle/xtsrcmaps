#include "xtsrcmaps/source/source.hxx"

#include <algorithm>
#include <vector>

auto
Fermi::Source::spherical_coords(std::vector<PointSource> const& srcs)
    -> Tensor<double, 2> {

    size_t const Nsrc = srcs.size();

    Tensor<double, 2> dirs(Nsrc, 2);

    for (size_t i = 0; i < Nsrc; ++i) {
        auto const& s = srcs[i];
        auto        params
            = std::get<SkyDirFunctionSpatialModel>(s.spatial_model).params;

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
Fermi::Source::spherical_coords(std::vector<DiffuseSource> const& srcs,
                                std::pair<double, double> const   cmap_ref_dir)
    -> Tensor<double, 2> {

    size_t const Nsrc = srcs.size();

    Tensor<double, 2> dirs(Nsrc, 2);

    for (size_t i = 0; i < Nsrc; ++i) {
        dirs[i, 0] = std::get<0>(cmap_ref_dir);
        dirs[i, 1] = std::get<1>(cmap_ref_dir);
    }

    return dirs;
}
