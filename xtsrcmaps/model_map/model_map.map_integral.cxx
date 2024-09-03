#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/misc/misc.hxx"

auto
Fermi::ModelMap::map_integral(Tensor<double, 4> const& model_map,
                              Tensor<double, 2> const& src_sph,
                              SkyGeom<double> const&   skygeom,
                              Tensor<double, 1> const& psf_radius,
                              std::vector<bool> const& is_in_fov)
    -> Tensor<double, 2> {
    size_t const Ns = model_map.extent(0);
    size_t const Nh = model_map.extent(1);
    size_t const Nw = model_map.extent(2);
    size_t const Ne = model_map.extent(3);
    /* size_t const Nf = psf_radius.extent(0); */

    Tensor<double, 2> MapIntegral(Ns, Ne);
    MapIntegral.clear();


#pragma omp parallel for schedule(static, 16)
    for (size_t s = 0; s < Ns; ++s) {
        if (!is_in_fov[s]) { continue; }

        auto const ss = skygeom.sph2dir({ src_sph[s, 0], src_sph[s, 1] });
        for (size_t h = 0; h < Nh; ++h) {
            for (size_t w = 0; w < Nw; ++w) {

                double const rad = SkyGeom<double>::dir_diff(
                    ss, skygeom.pix2dir({ h + 1.0f, w + 1.0f }));

                if (rad <= deg2rad * psf_radius[s]) {
                    for (size_t e = 0; e < Ne; ++e) {
                        MapIntegral[s, e] += model_map[s, h, w, e];
                    }
                }
            }
        }
    }

    std::transform(MapIntegral.begin(),
                   MapIntegral.end(),
                   MapIntegral.begin(),
                   [](auto v) { return v ? 1.0f / v : v; });


    return MapIntegral;
}
