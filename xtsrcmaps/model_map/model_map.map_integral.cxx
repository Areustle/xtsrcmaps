#include "xtsrcmaps/model_map/model_map.hxx"
#include "xtsrcmaps/misc/misc.hxx"

auto
Fermi::ModelMap::map_integral(Tensor4d const& model_map,
                              vpd const&      src_dirs,
                              SkyGeom const&  skygeom,
                              Tensor1d const& psf_radius,
                              Tensor1b const& is_in_fov) -> Tensor2d {
    long const Ne = model_map.dimension(0);
    long const Nh = model_map.dimension(1);
    long const Nw = model_map.dimension(2);
    long const Ns = model_map.dimension(3);
    long const Nf = psf_radius.dimension(0);

    Tensor2d MapIntegral(Ne, Nf);
    MapIntegral.setZero();

    // Annoyingly nested, but hard to declarize because of the skygeom
    // dependency.
    long i = 0;
    for (long s = 0; s < Ns; ++s) {
        if (!is_in_fov(s)) { continue; }

        double const rad = psf_radius(i);
        for (long w = 0; w < Nw; ++w) {
            for (long h = 0; h < Nh; ++h) {
                if (sph_pix_diff(src_dirs[s], Vector2d(h + 1., w + 1.), skygeom)
                        * R2D
                    <= rad) {
                    MapIntegral.slice(Idx2 { 0, i }, Idx2 { Ne, 1 })
                        += model_map
                               .slice(Idx4 { 0, h, w, s }, Idx4 { Ne, 1, 1, 1 })
                               .reshape(Idx2 { Ne, 1 });
                }
            }
        }
        ++i;
    }

    Tensor2d zeros          = MapIntegral.constant(0.0);
    Tensor2d invMapIntegral = MapIntegral.inverse();
    MapIntegral             = (MapIntegral == 0.).select(zeros, invMapIntegral);

    return MapIntegral;
}
