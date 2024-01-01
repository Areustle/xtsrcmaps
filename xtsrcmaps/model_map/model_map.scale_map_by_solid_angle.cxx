#include "xtsrcmaps/model_map/model_map.hxx"

#include <Eigen/Dense>


auto
Fermi::ModelMap::solid_angle(Tensor3d const&       points,
                             Fermi::SkyGeom const& skygeom) -> Tensor2d {
    Tensor2d phi(points.dimension(1), points.dimension(2));
    for (int w = 0; w < points.dimension(2); ++w) {
        for (int h = 0; h < points.dimension(1); ++h) {
            // Adapted from FermiTools CountsMap.cxx:612 and FitsImage.cxx:108
            Vector3d const A
                = skygeom.pix2dir({ points(0, h, w), points(1, h, w) });
            Vector3d const B
                = skygeom.pix2dir({ points(0, h, w), points(1, h, w) + 1. });
            Vector3d const C = skygeom.pix2dir(
                { points(0, h, w) + 1., points(1, h, w) + 1. });
            Vector3d const D
                = skygeom.pix2dir({ points(0, h, w) + 1., points(1, h, w) });

            double dOmega1
                = dir_diff(A, B) * dir_diff(A, D)
                  * (A - B).normalized().cross((A - D).normalized()).norm();

            double dOmega2
                = dir_diff(C, B) * dir_diff(C, D)
                  * (C - B).normalized().cross((C - D).normalized()).norm();
            phi(h, w) = 0.5 * (dOmega1 + dOmega2);
        }
    }
    return phi;
}

void
Fermi::ModelMap::scale_map_by_solid_angle(Tensor4d&      model_map,
                                          SkyGeom const& skygeom) {
    long const Ne              = model_map.dimension(0);
    long const Nh              = model_map.dimension(1);
    long const Nw              = model_map.dimension(2);
    long const Ns              = model_map.dimension(3);

    Tensor3d const init_points = get_init_points(Nh, Nw);
    // Compute solid angle for the pixel center points and scale PSF by it.
    model_map *= solid_angle(init_points, skygeom)
                     .reshape(Idx4 { 1, Nh, Nw, 1 })
                     .broadcast(Idx4 { Ne, 1, 1, Ns });
}
