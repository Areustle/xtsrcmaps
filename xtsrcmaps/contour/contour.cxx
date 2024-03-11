#include "xtsrcmaps/model_map/model_map.hxx"

auto
foo(Fermi::SkyGeom const& skygeom) -> auto {

    long const Ne = 1;
    long const Nh = 100;
    long const Nw = 100;
    long const Ns = 1;


    Tensor4d model_map(Ne, Nh, Nw, Ns);
    model_map.setZero();

    for (long h = 0; h < Nh; ++h) {
        for (long w = 0; w < Nw; ++w) {
            // Get the 4 corner points for this (h,w) pixel.
            // Recall wcs arrays start at 1.
            // ..................
            // ..p0----------p3....
            // ...|     ↓     |....
            // ...|     n3    |....
            // ...|→n0     n2←|....
            // ...|    n1     |....
            // ...|     ↑     |....
            // ..p1----------p2....
            // ..................
            // Note the redundant recomputation. Optimization at this stage
            // may be premature, but there should be a way to pre-compute this
            // for each pixel with only O(H+W) work.
            Vector3d p0 = skygeom.pix2dir({ h + 0.5, w + 0.5 });
            Vector3d p1 = skygeom.pix2dir({ h + 1.5, w + 0.5 });
            Vector3d p2 = skygeom.pix2dir({ h + 1.5, w + 1.5 });
            Vector3d p3 = skygeom.pix2dir({ h + 0.5, w + 1.5 });

            Vector3d n0 = p0.cross(p1).normalized();
            Vector3d n1 = p1.cross(p2).normalized();
            Vector3d n2 = p2.cross(p3).normalized();
            Vector3d n3 = p3.cross(p0).normalized();

            for (long s = 0; s < Ns; ++s) {}
        }
    }
}
