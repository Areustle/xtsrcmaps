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
Fermi::Source::refsph_coords(size_t const                Nsrc,
                             std::array<double, 2> const ref_sph)
    -> Tensor<double, 2> {
    Tensor<double, 2> dirs(Nsrc, 2);
    for (size_t i = 0; i < Nsrc; ++i) {
        dirs[i, 0] = (ref_sph[0]);
        dirs[i, 1] = (ref_sph[1]);
    }
    return dirs;
}
