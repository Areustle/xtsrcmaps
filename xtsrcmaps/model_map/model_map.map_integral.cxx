#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/misc/misc.hxx"

auto
Fermi::ModelMap::map_integral(Tensor<float, 4> const&  model_map,
                              Tensor<double, 2> const& src_sph,
                              SkyGeom<float> const&    skygeom,
                              Tensor<double, 1> const& psf_radius,
                              std::vector<bool> const& is_in_fov)
    -> Tensor<float, 2> {
    size_t const Ns = model_map.extent(0);
    size_t const Nh = model_map.extent(1);
    size_t const Nw = model_map.extent(2);
    size_t const Ne = model_map.extent(3);
    /* size_t const Nf = psf_radius.extent(0); */

    Tensor<float, 2> MapIntegral(Ns, Ne);
    MapIntegral.clear();

    for (size_t s = 0; s < Ns; ++s) {
        if (!is_in_fov[s]) { continue; }

        auto const ss = std::array<double, 2> { src_sph[s, 0], src_sph[s, 1] };
        double const rad = psf_radius[s];
        for (size_t w = 0; w < Nw; ++w) {
            for (size_t h = 0; h < Nh; ++h) {
                if (R2D
                        * SkyGeom<float>::dir_diff(
                            skygeom.sph2dir(ss),
                            skygeom.pix2dir({ h + 1.0f, w + 1.0f }))
                    <= rad) {
                    for (size_t e = 0; e < Ne; ++e) {
                        MapIntegral[s, e] += model_map[s, h, w, 0];
                    }
                }
            }
        }
    }

    std::transform(MapIntegral.begin(),
                   MapIntegral.end(),
                   MapIntegral.begin(),
                   [](float const& v) { return v ? 1.0f / v : v; });


    return MapIntegral;
}
