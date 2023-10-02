#include "xtsrcmaps/model_map/grid_operations.hxx"

auto
foo(long const            Nh,
    long const            Nw,
    vpd const&            src_sphcrds,
    Tensor3d const&       uPsf,
    Fermi::SkyGeom const& skygeom) {

    long const Ns             = src_sphcrds.size();

    Tensor3d const gridverts  = pixel_grid_vertices(Nh, Nw, skygeom);
    auto const [NormH, NormW] = pixel_grid_normals(gridverts);

    for (long s = 0; s < Ns; ++s) {
        // get pixel.
        /* Vector2d const p0 { src_sphcrds[s] }; */
        for (long w = 0; w < Nw; ++w) {
            for (long h = 0; h < Nh; ++h) {}
        }
    }
}
