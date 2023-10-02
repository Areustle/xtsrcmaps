#include "xtsrcmaps/model_map/grid_operations.hxx"
#include "xtsrcmaps/sky_geom.hxx"
#include "xtsrcmaps/tensor_types.hxx"

#include "unsupported/Eigen/CXX11/Tensor"
#include <Eigen/Dense>

auto
pixel_grid_centers(long const Nh, long const Nw) -> Tensor3d {
    Tensor3d p(2, Nh, Nw);
    for (long w = 0; w < Nw; ++w) {
        for (long h = 0; h < Nh; ++h) {
            p(0, h, w) = 1 + h;
            p(1, h, w) = 1 + w;
        }
    }
    return p;
}

auto
pixel_grid_vertices(long const Nh, long const Nw, Fermi::SkyGeom const& skygeom)
    -> Tensor3d {
    Tensor3d v(3, Nh + 1, Nw + 1);
    for (long w = 0; w < Nw + 1; ++w) {
        for (long h = 0; h < Nh + 1; ++h) {
            Vector3d const p = skygeom.pix2dir({ 0.5 + h, 0.5 + w });
            v(0, h, w)       = p(0);
            v(1, h, w)       = p(1);
            v(2, h, w)       = p(2);
        }
    }
    return v;
}

auto
pixel_grid_normals(Tensor3d const& vertices) -> std::pair<Tensor2d, Tensor2d> {
    long const Nh1 = vertices.dimension(0);
    long const Nw1 = vertices.dimension(1);

    Tensor2d normalsH(3, Nh1);
    Tensor2d normalsW(3, Nw1);

    for (long h = 0; h < Nh1; ++h) {
        Map<Vector3d const> u(&vertices(0, h, 0), 3);
        Map<Vector3d const> v(&vertices(0, h, Nw1 - 1), 3);
        Vector3d const      n = u.cross(v);
        normalsH(0, h)        = n(0);
        normalsH(1, h)        = n(1);
        normalsH(2, h)        = n(2);
    }

    for (long w = 0; w < Nw1; ++w) {
        Map<Vector3d const> u(&vertices(0, 0, w), 3);
        Map<Vector3d const> v(&vertices(0, Nh1 - 1, w), 3);
        Vector3d const      n = u.cross(v);
        normalsW(0, w)        = n(0);
        normalsW(1, w)        = n(1);
        normalsW(2, w)        = n(2);
    }

    return { normalsH, normalsW };
}
